"""
06_finetune_roberta.py
======================
Fine-tunes roberta-base on the StyleGuard binary dataset.
Uses the EXACT same GroupShuffleSplit (SEED=42, TEST_SIZE=0.20) as script 05
so train/test sets are identical and comparison is fair.

Outputs (saved to data/outputs/):
  roberta_proba_meta.csv          — P(AI) for the meta-train fold (for ensemble)
  roberta_proba_test.csv          — P(AI) for the held-out test set
  roberta_standalone_metrics.json — accuracy, precision, recall, f1, fpr
  fig_roberta_confusion_matrix.png

Run on Kaggle GPU T4 (~25 min).  CPU fallback works but takes ~2-3 hrs.
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── paths ──────────────────────────────────────────────────────────────────────
# Works both locally (config.py present) and on Kaggle (flat upload).
try:
    from config import PROCESSED_DIR, OUTPUT_DIR, SEED, TEST_SIZE
    FINAL_CSV = PROCESSED_DIR / "final_binary_dataset.csv"
except ImportError:
    from pathlib import Path
    BASE = Path("/kaggle/working")
    PROCESSED_DIR = BASE / "data" / "processed"
    OUTPUT_DIR = BASE / "data" / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_CSV = PROCESSED_DIR / "final_binary_dataset.csv"
    SEED = 42
    TEST_SIZE = 0.20

# ── hyper-parameters ───────────────────────────────────────────────────────────
MODEL_NAME   = "roberta-base"
MAX_LEN      = 256          # abstracts fit comfortably; keeps VRAM low
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
META_FRAC    = 0.25         # fraction of train set held back for ensemble meta-train
WARMUP_RATIO = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")


# ── dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]


# ── helpers ───────────────────────────────────────────────────────────────────
def get_probabilities(model, loader):
    """Return P(AI) for every sample in loader."""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch, labels in loader:
        batch   = {k: v.to(DEVICE) for k, v in batch.items()}
        labels  = labels.to(DEVICE)
        outputs = model(**batch, labels=labels)
        loss    = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. load data
    print("[1/7] Loading dataset …")
    df = pd.read_csv(FINAL_CSV)
    df.columns = df.columns.astype(str).str.strip()
    df["pair_group"] = df["source_human_id"].astype(str)
    df["label_int"]  = (df["label"] == "ai").astype(int)

    texts  = df["text"].values
    labels = df["label_int"].values
    groups = df["pair_group"].values

    # 2. same primary split as script 05
    print("[2/7] Splitting (same GroupShuffleSplit as script 05) …")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(texts, labels, groups=groups))

    X_train_text, X_test_text = texts[train_idx], texts[test_idx]
    y_train,       y_test      = labels[train_idx], labels[test_idx]
    g_train                    = groups[train_idx]

    # 3. carve out meta-train fold (no group leakage)
    print("[3/7] Carving meta-train fold …")
    gss2 = GroupShuffleSplit(n_splits=1, test_size=META_FRAC, random_state=SEED)
    finetune_idx, meta_idx = next(gss2.split(X_train_text, y_train, groups=g_train))

    X_ft,   y_ft   = X_train_text[finetune_idx], y_train[finetune_idx]
    X_meta, y_meta = X_train_text[meta_idx],     y_train[meta_idx]

    print(f"   fine-tune: {len(y_ft)}  |  meta-train: {len(y_meta)}  |  test: {len(y_test)}")

    # 4. tokeniser + datasets
    print("[4/7] Tokenising …")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    ft_ds    = TextDataset(X_ft,    y_ft,    tokenizer)
    meta_ds  = TextDataset(X_meta,  y_meta,  tokenizer)
    test_ds  = TextDataset(X_test_text, y_test, tokenizer)

    ft_loader    = DataLoader(ft_ds,   batch_size=BATCH_SIZE, shuffle=True)
    meta_loader  = DataLoader(meta_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5. model + optimiser
    print("[5/7] Loading roberta-base …")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(ft_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 6. training loop
    print("[6/7] Training …")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, ft_loader, optimizer, scheduler)
        print(f"   epoch {epoch}/{EPOCHS}  loss={loss:.4f}")

    # 7. evaluate + save outputs
    print("[7/7] Evaluating and saving outputs …")

    proba_meta = get_probabilities(model, meta_loader)
    proba_test = get_probabilities(model, test_loader)

    y_pred_test = (proba_test >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec  = recall_score(y_test, y_pred_test, zero_division=0)
    f1   = f1_score(y_test, y_pred_test, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"   RoBERTa standalone → acc={acc:.4f}  f1={f1:.4f}  fpr={fpr:.4f}")

    # probability CSVs (used by script 07)
    pd.DataFrame({"proba_ai": proba_meta, "label": y_meta}).to_csv(
        OUTPUT_DIR / "roberta_proba_meta.csv", index=False
    )
    pd.DataFrame({"proba_ai": proba_test, "label": y_test}).to_csv(
        OUTPUT_DIR / "roberta_proba_test.csv", index=False
    )

    # metrics JSON
    metrics = {
        "model": "RoBERTa (fine-tuned)",
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "fpr":       round(fpr,  4),
    }
    with open(OUTPUT_DIR / "roberta_standalone_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix PNG
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["human", "ai"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    disp.plot(ax=ax, values_format="d")
    plt.title("Figure — Confusion Matrix: RoBERTa (fine-tuned)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_roberta_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # append to existing model comparison table if present
    comp_csv = OUTPUT_DIR / "table_model_comparison.csv"
    if comp_csv.exists():
        comp_df = pd.read_csv(comp_csv)
        new_row = pd.DataFrame([{
            "model":     "RoBERTa (fine-tuned)",
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        }])
        comp_df = pd.concat([comp_df, new_row], ignore_index=True).sort_values("f1", ascending=False)
        comp_df.to_csv(OUTPUT_DIR / "table_model_comparison_with_roberta.csv", index=False)
        print("   Updated model comparison table saved.")

    print("\nDone. Outputs in:", OUTPUT_DIR)
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
