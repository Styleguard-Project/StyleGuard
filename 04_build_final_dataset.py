import pandas as pd

from config import PROCESSED_DIR, TARGET_PER_CLASS
from utils import clean_text_basic, is_english, md5_hash, word_count

SAMPLED_HUMAN_CSV = PROCESSED_DIR / "sampled_human.csv"
AI_RAW_CSV = PROCESSED_DIR / "ai_generated_raw.csv"
AI_CLEAN_CSV = PROCESSED_DIR / "clean_ai_generated.csv"
FINAL_DATASET_CSV = PROCESSED_DIR / "final_binary_dataset.csv"

def main():
    if not SAMPLED_HUMAN_CSV.exists():
        raise FileNotFoundError(f"Run 02_prepare_prompt_pack.py first. Missing: {SAMPLED_HUMAN_CSV}")
    if not AI_RAW_CSV.exists():
        raise FileNotFoundError(f"Run 03_openai_direct_generate_ai.py first. Missing: {AI_RAW_CSV}")

    human_df = pd.read_csv(SAMPLED_HUMAN_CSV)
    ai_df = pd.read_csv(AI_RAW_CSV)

    # Clean up column names just in case
    human_df.columns = human_df.columns.astype(str).str.strip()
    ai_df.columns = ai_df.columns.astype(str).str.strip()

    required_ai = {"prompt_id", "source_human_id", "category", "generator_name", "generated_text"}
    missing_ai = required_ai - set(ai_df.columns)
    if missing_ai:
        raise ValueError(f"AI CSV is missing required columns: {sorted(missing_ai)}")

    required_human = {"prompt_id", "arxiv_id", "category", "text", "label", "source", "word_count"}
    missing_human = required_human - set(human_df.columns)
    if missing_human:
        raise ValueError(
            f"sampled_human.csv is missing required columns: {sorted(missing_human)}\n"
            f"Available columns: {list(human_df.columns)}"
        )

    ai_df["text"] = ai_df["generated_text"].astype(str).map(clean_text_basic)
    ai_df["word_count"] = ai_df["text"].map(word_count)
    ai_df["is_english"] = ai_df["text"].map(is_english)

    ai_df = ai_df[
        (ai_df["word_count"] >= 80) &
        (ai_df["word_count"] <= 300) &
        (ai_df["is_english"])
    ].copy()

    ai_df["label"] = "ai"
    ai_df["source"] = "openai_batch"
    ai_df["text_hash"] = ai_df["text"].map(md5_hash)
    ai_df = ai_df.drop_duplicates(subset=["text_hash"]).copy()

    human_hashes = set(human_df["text"].map(md5_hash))
    ai_df = ai_df[~ai_df["text_hash"].isin(human_hashes)].copy()

    ai_df = ai_df[["prompt_id", "source_human_id", "category", "generator_name", "text", "label", "source", "word_count"]].reset_index(drop=True)
    ai_df.to_csv(AI_CLEAN_CSV, index=False)

    human_subset = human_df[human_df["arxiv_id"].isin(ai_df["source_human_id"])].copy()
    human_subset["source_human_id"] = human_subset["arxiv_id"]
    human_subset["generator_name"] = None   # human rows do not naturally have a generator name

    human_subset = human_subset[[
        "prompt_id",
        "source_human_id",
        "category",
        "generator_name",
        "text",
        "label",
        "source",
        "word_count"
    ]].copy()

    n_final = min(len(human_subset), len(ai_df), TARGET_PER_CLASS)
    human_final = human_subset.sample(n_final, random_state=42).reset_index(drop=True)
    ai_final = ai_df.sample(n_final, random_state=42).reset_index(drop=True)

    final_df = pd.concat([human_final, ai_final], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df.to_csv(FINAL_DATASET_CSV, index=False)

    print("Saved:")
    print(" -", AI_CLEAN_CSV)
    print(" -", FINAL_DATASET_CSV)
    print("Final label counts:")
    print(final_df["label"].value_counts())

if __name__ == "__main__":
    main()
