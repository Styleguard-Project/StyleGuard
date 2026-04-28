"""
07_ensemble_and_shap.py
=======================
Stage 3 of StyleGuard:
  • Loads stylometric features (from script 05) + RoBERTa probabilities (from script 06)
  • Trains an XGBoost meta-classifier on 9 inputs:
        8 stylometric features  +  RoBERTa P(AI)
  • Produces final model comparison table (all models)
  • Runs SHAP TreeExplainer on the ensemble and saves:
        - beeswarm summary plot  (global feature impact)
        - bar chart              (mean |SHAP|)
        - 4 waterfall plots      (correct human / correct AI / FP / FN)
        - shap_values_test.csv

Run locally on your Mac after downloading the 3 files from Kaggle.
No GPU needed — finishes in ~5-10 minutes.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import shap

from config import PROCESSED_DIR, OUTPUT_DIR, SEED, TEST_SIZE
from utils import save_dataframe_as_png, stylometric_features

# ── paths ──────────────────────────────────────────────────────────────────────
FINAL_CSV    = PROCESSED_DIR / "final_binary_dataset.csv"
FEATURE_CSV  = PROCESSED_DIR / "stylometric_features.csv"
SHAP_DIR     = OUTPUT_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "avg_sentence_length",
    "sentence_length_variance",
    "type_token_ratio",
    "punctuation_ratio",
    "stopword_ratio",
    "avg_word_length",
    "paragraph_length_variance",
    "discourse_ratio",
]
META_FRAC = 0.25   # must match script 06


# ── helpers ───────────────────────────────────────────────────────────────────
def load_stylometric(final_df, train_idx, test_idx):
    """Re-use cached stylometric_features.csv if present, else recompute."""
    from tqdm.auto import tqdm

    if FEATURE_CSV.exists():
        print("   Loading cached stylometric features …")
        feat_df = pd.read_csv(FEATURE_CSV)
        feat_df["pair_group"] = feat_df["pair_group"].astype(str)
    else:
        print("   Computing stylometric features (this takes a few minutes) …")
        rows = []
        for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
            feats = stylometric_features(row["text"])
            feats.update({
                "label":          row["label"],
                "category":       row["category"],
                "source":         row["source"],
                "generator_name": row["generator_name"],
                "word_count":     row["word_count"],
                "pair_group":     str(row["source_human_id"]),
            })
            rows.append(feats)
        feat_df = pd.DataFrame(rows)
        feat_df.to_csv(FEATURE_CSV, index=False)

    X  = feat_df[FEATURE_COLS].copy()
    y  = feat_df["label"].map({"human": 0, "ai": 1}).astype(int)
    g  = feat_df["pair_group"].values
    return X, y, g, feat_df


def waterfall(shap_exp, idx, title, path):
    """Save a single SHAP waterfall plot."""
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_exp[idx], max_display=9, show=False)
    plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── 1. load RoBERTa probabilities ─────────────────────────────────────────
    print("[1/8] Loading RoBERTa probability files …")
    meta_proba_path = OUTPUT_DIR / "roberta_proba_meta.csv"
    test_proba_path = OUTPUT_DIR / "roberta_proba_test.csv"

    if not meta_proba_path.exists() or not test_proba_path.exists():
        raise FileNotFoundError(
            "roberta_proba_meta.csv / roberta_proba_test.csv not found in data/outputs/.\n"
            "Run 06_finetune_roberta.py on Kaggle first, then download those files."
        )

    meta_proba_df = pd.read_csv(meta_proba_path)   # columns: proba_ai, label
    test_proba_df = pd.read_csv(test_proba_path)

    # ── 2. rebuild the same splits as script 05 ───────────────────────────────
    print("[2/8] Rebuilding primary train/test split …")
    final_df = pd.read_csv(FINAL_CSV)
    final_df.columns = final_df.columns.astype(str).str.strip()
    final_df["pair_group"] = final_df["source_human_id"].astype(str)

    texts  = final_df["text"].values
    groups = final_df["pair_group"].values
    y_all  = (final_df["label"] == "ai").astype(int).values

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(texts, y_all, groups=groups))

    g_train = groups[train_idx]
    y_train_all = y_all[train_idx]
    y_test  = y_all[test_idx]

    # ── 3. load / compute stylometric features ────────────────────────────────
    print("[3/8] Loading stylometric features …")
    X_all, _, _, feat_df = load_stylometric(final_df, train_idx, test_idx)

    X_train_all = X_all.iloc[train_idx]
    X_test      = X_all.iloc[test_idx]

    # ── 4. carve meta-train fold (same split as script 06) ───────────────────
    print("[4/8] Carving meta-train fold …")
    gss2 = GroupShuffleSplit(n_splits=1, test_size=META_FRAC, random_state=SEED)
    finetune_rel_idx, meta_rel_idx = next(
        gss2.split(X_train_all, y_train_all, groups=g_train)
    )

    X_meta_stylo = X_train_all.iloc[meta_rel_idx].reset_index(drop=True)
    y_meta       = y_train_all[meta_rel_idx]
    X_ft_stylo   = X_train_all.iloc[finetune_rel_idx]
    y_ft         = y_train_all[finetune_rel_idx]

    # sanity-check alignment with RoBERTa probability files
    assert len(meta_proba_df) == len(X_meta_stylo), (
        f"meta fold size mismatch: stylo={len(X_meta_stylo)}, roberta={len(meta_proba_df)}"
    )
    assert len(test_proba_df) == len(X_test), (
        f"test set size mismatch: stylo={len(X_test)}, roberta={len(test_proba_df)}"
    )

    # ── 5. assemble ensemble feature matrices ─────────────────────────────────
    print("[5/8] Assembling ensemble feature matrices …")
    X_meta_ensemble = X_meta_stylo.copy()
    X_meta_ensemble["roberta_proba_ai"] = meta_proba_df["proba_ai"].values

    X_test_ensemble = X_test.reset_index(drop=True).copy()
    X_test_ensemble["roberta_proba_ai"] = test_proba_df["proba_ai"].values

    ENSEMBLE_COLS = FEATURE_COLS + ["roberta_proba_ai"]

    # ── 6. train XGBoost ensemble ─────────────────────────────────────────────
    print("[6/8] Training XGBoost ensemble …")
    ensemble = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
    )
    ensemble.fit(X_meta_ensemble[ENSEMBLE_COLS], y_meta)

    y_pred   = ensemble.predict(X_test_ensemble[ENSEMBLE_COLS])
    proba_e  = ensemble.predict_proba(X_test_ensemble[ENSEMBLE_COLS])[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"   Ensemble → acc={acc:.4f}  f1={f1:.4f}  fpr={fpr:.4f}")

    ensemble_metrics = {
        "model": "Ensemble (XGBoost: stylometry + RoBERTa)",
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "fpr":       round(fpr,  4),
    }
    with open(OUTPUT_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(ensemble_metrics, f, indent=2)

    # ensemble confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["human", "ai"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    disp.plot(ax=ax, values_format="d")
    plt.title("Figure — Confusion Matrix: Ensemble (XGBoost)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_ensemble_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── 7. build full model comparison table ──────────────────────────────────
    print("[7/8] Building final model comparison table …")
    comp_csv = OUTPUT_DIR / "table_model_comparison.csv"
    if comp_csv.exists():
        base_df = pd.read_csv(comp_csv)
    else:
        base_df = pd.DataFrame(columns=["model","accuracy","precision","recall","f1"])

    # load RoBERTa standalone metrics if available
    rob_json = OUTPUT_DIR / "roberta_standalone_metrics.json"
    new_rows = []
    if rob_json.exists():
        with open(rob_json) as f:
            rob = json.load(f)
        new_rows.append({
            "model":     rob["model"],
            "accuracy":  rob["accuracy"],
            "precision": rob["precision"],
            "recall":    rob["recall"],
            "f1":        rob["f1"],
        })

    new_rows.append({
        "model":     "Ensemble (XGBoost: stylometry + RoBERTa)",
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    })

    full_df = pd.concat([base_df, pd.DataFrame(new_rows)], ignore_index=True)
    full_df = full_df.drop_duplicates(subset="model").sort_values("f1", ascending=False).reset_index(drop=True)
    full_df.to_csv(OUTPUT_DIR / "table_model_comparison_full.csv", index=False)
    save_dataframe_as_png(
        full_df.round(4),
        OUTPUT_DIR / "table_model_comparison_full.png",
        "Table 2 (Final). Model Comparison — All Methods"
    )
    print("   Full comparison table saved.")

    # ── 8. SHAP explainability ────────────────────────────────────────────────
    print("[8/8] Running SHAP …")
    explainer  = shap.TreeExplainer(ensemble)
    shap_vals  = explainer(X_test_ensemble[ENSEMBLE_COLS])

    # save raw SHAP values
    shap_df = pd.DataFrame(
        shap_vals.values,
        columns=[f"shap_{c}" for c in ENSEMBLE_COLS]
    )
    shap_df["label"]    = y_test
    shap_df["pred"]     = y_pred
    shap_df["proba_ai"] = proba_e
    shap_df.to_csv(SHAP_DIR / "shap_values_test.csv", index=False)

    # ── beeswarm (global summary)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_vals, max_display=9, show=False)
    plt.title("Figure 3a. SHAP Beeswarm — Ensemble (all test samples)")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "fig_shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── bar (mean |SHAP|)
    plt.figure(figsize=(9, 5))
    shap.plots.bar(shap_vals, max_display=9, show=False)
    plt.title("Figure 3b. Mean |SHAP| — Ensemble Feature Importance")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "fig_shap_bar_global.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── 4 waterfall plots ─────────────────────────────────────────────────────
    labels_arr = np.array(y_test)
    preds_arr  = np.array(y_pred)

    correct_human = np.where((labels_arr == 0) & (preds_arr == 0))[0]
    correct_ai    = np.where((labels_arr == 1) & (preds_arr == 1))[0]
    false_pos     = np.where((labels_arr == 0) & (preds_arr == 1))[0]   # human → AI
    false_neg     = np.where((labels_arr == 1) & (preds_arr == 0))[0]   # AI → human

    def pick(arr):
        return arr[0] if len(arr) > 0 else None

    cases = [
        (pick(correct_human), "Correct — Human",     "fig_shap_waterfall_correct_human.png"),
        (pick(correct_ai),    "Correct — AI",         "fig_shap_waterfall_correct_ai.png"),
        (pick(false_pos),     "False Positive (Human flagged as AI)", "fig_shap_waterfall_false_positive.png"),
        (pick(false_neg),     "False Negative (AI slipped through)",  "fig_shap_waterfall_false_negative.png"),
    ]

    for idx, title, fname in cases:
        if idx is None:
            print(f"   Skipping '{title}' — no sample in test set for this case.")
            continue
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            shap.plots.waterfall(shap_vals[int(idx)], max_display=9, show=False)
            plt.title(f"Figure 3. SHAP Waterfall — {title}", pad=12)
            plt.tight_layout()
            plt.savefig(SHAP_DIR / fname, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"   Saved: {fname}")
        except Exception as e:
            print(f"   Warning: could not save {fname}: {e}")

    print("\nAll done. Outputs:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(OUTPUT_DIR))


if __name__ == "__main__":
    main()
