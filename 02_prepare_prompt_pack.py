import random

import pandas as pd

from config import PROCESSED_DIR, TARGET_PER_CLASS, CATEGORY_MAP, SEED

random.seed(SEED)

HUMAN_CSV = PROCESSED_DIR / "final_arxiv_human.csv"
SAMPLED_HUMAN_CSV = PROCESSED_DIR / "sampled_human.csv"
PROMPTS_CSV = PROCESSED_DIR / "ai_generation_prompts.csv"
PILOT_PROMPTS_CSV = PROCESSED_DIR / "ai_generation_prompts_pilot25.csv"
TEMPLATE_CSV = PROCESSED_DIR / "ai_generated_raw_template.csv"

def make_prompt(title: str, category_code: str) -> str:
    area = CATEGORY_MAP.get(category_code, category_code)
    title = str(title).strip()
    return (
        f"Write a formal academic research abstract of about 120 to 180 words for a single hypothetical research paper. "
        f"Topic title: {title}. "
        f"Research area: {area}. "
        f"Include a clear objective, method, main result, and conclusion. "
        f"Do not write a review, tutorial, or general overview. "
        f"Do not use headings, bullet points, or markdown. "
        f"Use plain text only and a scholarly tone."
    )

def build_pilot_prompt_pack(sampled: pd.DataFrame, total_rows: int = 25, min_per_category: int = 5) -> pd.DataFrame:
    categories = sorted(sampled["category"].dropna().unique().tolist())
    pilot_parts = []

    for cat in categories:
        cat_df = sampled[sampled["category"] == cat].copy()
        if len(cat_df) < min_per_category:
            raise ValueError(
                f"Category {cat} has only {len(cat_df)} rows in sampled_human.csv, "
                f"but pilot requires at least {min_per_category}."
            )
        pilot_parts.append(cat_df.sample(min_per_category, random_state=SEED))

    pilot = pd.concat(pilot_parts, ignore_index=True)

    if len(pilot) < total_rows:
        need = total_rows - len(pilot)
        remaining_pool = sampled.loc[~sampled["arxiv_id"].isin(pilot["arxiv_id"])].copy()
        extra = remaining_pool.sample(need, random_state=SEED)
        pilot = pd.concat([pilot, extra], ignore_index=True)

    pilot = pilot.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return pilot

def main():
    if not HUMAN_CSV.exists():
        raise FileNotFoundError(f"Run 01_collect_human_data.py first. Missing: {HUMAN_CSV}")

    human_df = pd.read_csv(HUMAN_CSV)

    # Make the script more robust to CSV formatting quirks
    human_df.columns = human_df.columns.astype(str).str.strip()

    required_cols = {"arxiv_id", "title", "category"}
    missing = required_cols - set(human_df.columns)
    if missing:
        raise ValueError(
            f"final_arxiv_human.csv is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(human_df.columns)}"
        )

    if len(human_df) < TARGET_PER_CLASS:
        raise ValueError(
            f"Not enough human rows. Found {len(human_df)}, but TARGET_PER_CLASS is {TARGET_PER_CLASS}."
        )

    category_counts = human_df["category"].value_counts()
    total_rows = len(human_df)

    sampled_parts = []
    for category, count in category_counts.items():
        target_n = max(1, round(TARGET_PER_CLASS * count / total_rows))
        part = human_df[human_df["category"] == category].sample(
            min(target_n, count),
            random_state=SEED
        )
        sampled_parts.append(part)

    sampled = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled) > TARGET_PER_CLASS:
        sampled = sampled.sample(TARGET_PER_CLASS, random_state=SEED).reset_index(drop=True)
    elif len(sampled) < TARGET_PER_CLASS:
        need = TARGET_PER_CLASS - len(sampled)
        remaining_pool = human_df.loc[~human_df["arxiv_id"].isin(sampled["arxiv_id"])].copy()
        extra = remaining_pool.sample(need, random_state=SEED)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.reset_index(drop=True)
    sampled["prompt_id"] = [f"prompt_{i:04d}" for i in range(1, len(sampled) + 1)]
    sampled.to_csv(SAMPLED_HUMAN_CSV, index=False)

    prompt_pack = sampled[["prompt_id", "arxiv_id", "title", "category"]].copy()
    prompt_pack["prompt"] = [make_prompt(t, c) for t, c in zip(prompt_pack["title"], prompt_pack["category"])]
    prompt_pack = prompt_pack.rename(columns={"arxiv_id": "source_human_id"})
    prompt_pack.to_csv(PROMPTS_CSV, index=False)

    pilot_source = sampled[["prompt_id", "arxiv_id", "title", "category"]].copy()
    pilot_source = pilot_source.rename(columns={"arxiv_id": "source_human_id"})
    pilot = build_pilot_prompt_pack(
        pilot_source.rename(columns={"source_human_id": "arxiv_id"}),  # temporary for helper
        total_rows=25,
        min_per_category=5
    ).rename(columns={"arxiv_id": "source_human_id"})
    pilot["prompt"] = [make_prompt(t, c) for t, c in zip(pilot["title"], pilot["category"])]
    pilot = pilot[["prompt_id", "source_human_id", "title", "category", "prompt"]]
    pilot.to_csv(PILOT_PROMPTS_CSV, index=False)

    template = prompt_pack[["prompt_id", "source_human_id", "category"]].copy()
    template["generator_name"] = ""
    template["generated_text"] = ""
    template.to_csv(TEMPLATE_CSV, index=False)

    print("Saved:")
    print(" -", SAMPLED_HUMAN_CSV)
    print(" -", PROMPTS_CSV)
    print(" -", PILOT_PROMPTS_CSV)
    print(" -", TEMPLATE_CSV)

if __name__ == "__main__":
    main()
