import json

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from config import PROCESSED_DIR, OUTPUT_DIR, TEST_SIZE, SEED, TARGET_PER_CLASS
from utils import save_dataframe_as_png, stylometric_features

FINAL_DATASET_CSV = PROCESSED_DIR / "final_binary_dataset.csv"
FEATURE_CSV = PROCESSED_DIR / "stylometric_features.csv"

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

def main():
    if not FINAL_DATASET_CSV.exists():
        raise FileNotFoundError(f"Run 04_build_final_dataset.py first. Missing: {FINAL_DATASET_CSV}")

    final_df = pd.read_csv(FINAL_DATASET_CSV)
    final_df.columns = final_df.columns.astype(str).str.strip()

    # Important: source_human_id is the natural pair key in this project.
    # Human rows already carry their own source_human_id, and AI rows point back
    # to the human abstract they were generated from.
    if "source_human_id" not in final_df.columns:
        raise ValueError(
            "final_binary_dataset.csv does not contain source_human_id, "
            "so grouped pair-aware splitting cannot be applied."
        )

    final_df["pair_group"] = final_df["source_human_id"].astype(str)
    print(f"[group check] unique groups: {final_df['pair_group'].nunique()}, rows: {len(final_df)}")

    rows = []
    for _, row in tqdm(final_df.iterrows(), total=len(final_df), desc="Extracting stylometric features"):
        feats = stylometric_features(row["text"])
        feats.update({
            "label": row["label"],
            "category": row["category"],
            "source": row["source"],
            "generator_name": row["generator_name"],
            "word_count": row["word_count"],
            "pair_group": row["pair_group"],
        })
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(FEATURE_CSV, index=False)

    X = feat_df[FEATURE_COLS].copy()
    y = feat_df["label"].map({"human": 0, "ai": 1}).astype(int)

    groups = feat_df["pair_group"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"[split] train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"[split] train class balance (AI fraction): {y_train.mean():.4f}")
    print(f"[split] test  class balance (AI fraction): {y_test.mean():.4f}")

    train_groups = set(feat_df.iloc[train_idx]["pair_group"])
    test_groups = set(feat_df.iloc[test_idx]["pair_group"])
    overlap = train_groups & test_groups
    print(f"[split] group overlap between train and test: {len(overlap)} (should be 0)")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(random_state=SEED, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, class_weight="balanced"
        ),
    }

    results = []
    fitted = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })
        fitted[model_name] = model

    # Length-only baseline
    wc_train = feat_df.iloc[train_idx][["word_count"]].values
    wc_test = feat_df.iloc[test_idx][["word_count"]].values
    length_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")),
    ])
    length_pipe.fit(wc_train, y_train)
    wc_preds = length_pipe.predict(wc_test)
    results.append({
        "model": "Length-only (word_count)",
        "accuracy": accuracy_score(y_test, wc_preds),
        "precision": precision_score(y_test, wc_preds, zero_division=0),
        "recall": recall_score(y_test, wc_preds, zero_division=0),
        "f1": f1_score(y_test, wc_preds, zero_division=0),
    })

    # Top-3 feature ablation
    top3_cols = ["avg_word_length", "stopword_ratio", "type_token_ratio"]
    rf_top3 = RandomForestClassifier(
        n_estimators=300, random_state=SEED, class_weight="balanced"
    )
    rf_top3.fit(X_train[top3_cols], y_train)
    top3_preds = rf_top3.predict(X_test[top3_cols])
    results.append({
        "model": "Random Forest (top-3 features)",
        "accuracy": accuracy_score(y_test, top3_preds),
        "precision": precision_score(y_test, top3_preds, zero_division=0),
        "recall": recall_score(y_test, top3_preds, zero_division=0),
        "f1": f1_score(y_test, top3_preds, zero_division=0),
    })

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)

    # Keep the paper discussion anchored to the full-feature Random Forest
    best_model_name = "Random Forest"
    best_model = fitted[best_model_name]
    best_preds = best_model.predict(X_test)

    table1 = (
        final_df.groupby(["label", "category"])
        .agg(samples=("text", "size"), avg_word_count=("word_count", "mean"))
        .reset_index()
    )
    table1["avg_word_count"] = table1["avg_word_count"].round(1)

    table1.to_csv(OUTPUT_DIR / "table_dataset_summary.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "table_model_comparison.csv", index=False)

    save_dataframe_as_png(table1, OUTPUT_DIR / "table_dataset_summary.png", "Table 1. Dataset Summary")
    save_dataframe_as_png(results_df.round(4), OUTPUT_DIR / "table_model_comparison.png", "Table 2. Model Comparison")

    cm = confusion_matrix(y_test, best_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["human", "ai"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    disp.plot(ax=ax, values_format="d")
    plt.title(f"Figure 1. Confusion Matrix — {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    rf_model = fitted["Random Forest"]
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=True)
    fi_df.to_csv(OUTPUT_DIR / "feature_importance_values.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(fi_df["feature"], fi_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Figure 2. Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    config = {
        "seed": SEED,
        "target_per_class": TARGET_PER_CLASS,
        "test_size": TEST_SIZE,
        "feature_columns": FEATURE_COLS,
        "split_strategy": "GroupShuffleSplit by source_human_id",
        "models": {
            "Logistic Regression": {"max_iter": 2000, "class_weight": "balanced"},
            "Linear SVM": {"class_weight": "balanced"},
            "Random Forest": {"n_estimators": 300, "class_weight": "balanced"},
            "Length-only (word_count)": {"base_model": "LogisticRegression"},
            "Random Forest (top-3 features)": {"features": top3_cols},
        },
        "best_model": best_model_name,
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Saved outputs to:", OUTPUT_DIR)
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
