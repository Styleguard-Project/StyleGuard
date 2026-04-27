import time
import random

import arxiv
import pandas as pd
from tqdm.auto import tqdm

from config import RAW_DIR, PROCESSED_DIR, ARXIV_CATEGORIES, SEED
from utils import clean_text_basic, is_english, md5_hash, save_json, word_count

random.seed(SEED)

RAW_JSON = RAW_DIR / "raw_arxiv_human.json"
FINAL_JSON = PROCESSED_DIR / "final_arxiv_human.json"
FINAL_CSV = PROCESSED_DIR / "final_arxiv_human.csv"
SUMMARY_CSV = PROCESSED_DIR / "human_summary.csv"

def main():
    if FINAL_JSON.exists() and FINAL_CSV.exists():
        print("Found existing human dataset. Nothing to recollect.")
        print(FINAL_JSON)
        print(FINAL_CSV)
        return

    client = arxiv.Client()
    papers = []

    for cat_code, cat_name in ARXIV_CATEGORIES.items():
        print(f"\nCollecting from: {cat_name} ({cat_code})")
        search = arxiv.Search(
            query=f"cat:{cat_code}",
            max_results=1000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Ascending,
        )

        count = 0
        for result in tqdm(client.results(search)):
            year = result.published.year
            if year >= 2021:
                continue

            papers.append({
                "arxiv_id": result.entry_id,
                "title": result.title,
                "abstract": result.summary,
                "year": year,
                "category": cat_code,
                "label": "human",
                "source": "arxiv",
            })
            count += 1
            time.sleep(0.3)

        print(f"  Collected {count} papers from {cat_code}")

    save_json(papers, RAW_JSON)
    print(f"\nSaved raw file to: {RAW_JSON}")

    cleaned = []
    skipped = 0
    for row in tqdm(papers, desc="Cleaning human abstracts"):
        text = clean_text_basic(row["abstract"])
        wc = word_count(text)

        if not is_english(text):
            skipped += 1
            continue
        if not (80 <= wc <= 300):
            skipped += 1
            continue

        new_row = row.copy()
        new_row["text"] = text
        new_row["word_count"] = wc
        cleaned.append(new_row)

    print(f"Kept: {len(cleaned)} | Skipped: {skipped}")

    seen = set()
    final_rows = []
    for row in cleaned:
        h = md5_hash(row["text"])
        if h not in seen:
            seen.add(h)
            final_rows.append(row)

    df = pd.DataFrame(final_rows)
    df.to_csv(FINAL_CSV, index=False)
    save_json(final_rows, FINAL_JSON)

    summary = (
        df.groupby("category")
        .agg(samples=("text", "size"), avg_word_count=("word_count", "mean"))
        .reset_index()
    )
    summary["avg_word_count"] = summary["avg_word_count"].round(1)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nDone.")
    print("Saved:")
    print(" -", FINAL_JSON)
    print(" -", FINAL_CSV)
    print(" -", SUMMARY_CSV)

if __name__ == "__main__":
    main()
