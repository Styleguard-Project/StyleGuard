"""
predict.py
==========
StyleGuard — predict whether a text is Human or AI-written.

Uses Random Forest trained on 8 stylometric features (Layer 1).
Fast: runs in under 10 seconds, no GPU needed, no internet needed.

Usage:
    python3.11 predict.py --file test_abstracts.json
    python3.11 predict.py --file test_abstracts.json --explain
    python3.11 predict.py --text "Your abstract text here..."
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import shap

from config import PROCESSED_DIR, OUTPUT_DIR, SEED, TEST_SIZE
from utils import stylometric_features

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

(OUTPUT_DIR / "shap").mkdir(parents=True, exist_ok=True)


def load_model():
    print("[setup] Loading features and re-training Random Forest ...")

    feat_df  = pd.read_csv(PROCESSED_DIR / "stylometric_features.csv")
    feat_df["pair_group"] = feat_df["pair_group"].astype(str)

    final_df = pd.read_csv(PROCESSED_DIR / "final_binary_dataset.csv")
    final_df.columns = final_df.columns.astype(str).str.strip()
    final_df["pair_group"] = final_df["source_human_id"].astype(str)

    y_all  = (final_df["label"] == "ai").astype(int).values
    groups = final_df["pair_group"].values
    texts  = final_df["text"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, _ = next(gss.split(texts, y_all, groups=groups))

    X_train = feat_df[FEATURE_COLS].iloc[train_idx]
    y_train = y_all[train_idx]

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)
    print("[setup] Model ready.\n")
    return rf, explainer


def predict_one(text, model, explainer, explain=False, label=None, idx=1):
    stylo  = stylometric_features(text)
    values = [stylo[c] for c in FEATURE_COLS]
    X      = pd.DataFrame([values], columns=FEATURE_COLS)

    pred      = model.predict(X)[0]
    proba_ai  = model.predict_proba(X)[0, 1]
    proba_hum = 1 - proba_ai
    confidence = proba_ai if pred == 1 else proba_hum
    label_str  = "AI-GENERATED" if pred == 1 else "HUMAN-WRITTEN"

    print(f"{'─'*58}")
    print(f"  Sample {idx}")
    if label:
        correct = "CORRECT" if (pred == 1) == (label.lower() == "ai") else "WRONG"
        print(f"  True label   : {label.upper()}  [{correct}]")
    print(f"  Prediction   : {label_str}")
    print(f"  Confidence   : {confidence*100:.1f}%")
    print(f"  P(AI): {proba_ai*100:.1f}%   P(Human): {proba_hum*100:.1f}%")
    print(f"\n  Feature values:")
    for fname, fval in zip(FEATURE_COLS, values):
        print(f"    {fname:<35} {fval:.4f}")

    if explain:
        shap_vals = explainer(X)
        fig, ax = plt.subplots(figsize=(10, 5))
        # fix for shap 0.51+ with multi-output Random Forest
        sv = shap_vals[:, 1] if hasattr(shap_vals, '__getitem__') else shap_vals
        shap.plots.waterfall(sv[0], max_display=9, show=False)
        true_tag = f" | true={label}" if label else ""
        plt.title(f"SHAP — Sample {idx}: {label_str}{true_tag}", pad=12)
        plt.tight_layout()
        out = OUTPUT_DIR / f"shap/fig_predict_sample_{idx}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  SHAP saved → {out}")

    print()
    return pred, proba_ai


def main():
    parser = argparse.ArgumentParser(description="StyleGuard predictor")
    parser.add_argument("--text",    type=str, help="Single text to classify")
    parser.add_argument("--file",    type=str, help="Path to test_abstracts.json")
    parser.add_argument("--explain", action="store_true",
                        help="Save SHAP waterfall plot for each prediction")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.print_help()
        sys.exit(1)

    model, explainer = load_model()

    if args.text:
        predict_one(args.text, model, explainer, explain=args.explain, idx=1)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        print(f"Running StyleGuard on {len(samples)} samples ...\n")
        correct = 0
        for i, sample in enumerate(samples, 1):
            pred, _ = predict_one(
                sample["text"], model, explainer,
                explain=args.explain,
                label=sample.get("label"),
                idx=i
            )
            if sample.get("label"):
                if (pred == 1) == (sample["label"].lower() == "ai"):
                    correct += 1

        if any(s.get("label") for s in samples):
            acc = correct / len(samples) * 100
            print(f"{'─'*58}")
            print(f"  Final accuracy: {correct}/{len(samples)} = {acc:.1f}%")
            print(f"{'─'*58}")


if __name__ == "__main__":
    main()
