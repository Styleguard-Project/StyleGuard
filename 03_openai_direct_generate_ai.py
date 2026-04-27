from __future__ import annotations

import argparse
import getpass
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

from config import DEFAULT_OPENAI_MODEL, PROCESSED_DIR

load_dotenv()

RESPONSES_URL = "https://api.openai.com/v1/responses"
PROMPTS_CSV = PROCESSED_DIR / "ai_generation_prompts.csv"
FINAL_AI_CSV = PROCESSED_DIR / "ai_generated_raw.csv"


def get_api_key(passed_key: Optional[str]) -> str:
    if passed_key:
        return passed_key.strip()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    return getpass.getpass("Enter your OpenAI API key: ").strip()


def read_prompt_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"prompt_id", "source_human_id", "category", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prompt CSV is missing required columns: {sorted(missing)}")
    return df.copy()


def extract_text_from_response_body(body: Dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = body.get("output", [])
    chunks = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content_item in item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())

    return "\n".join(chunks).strip()


def call_openai_once(api_key: str, model: str, prompt: str, max_output_tokens: int = 220, timeout: int = 180) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    resp = requests.post(RESPONSES_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    text = extract_text_from_response_body(body)
    if not text:
        raise RuntimeError("Empty response text returned from Responses API.")
    return text


def generate_with_retries(api_key: str, model: str, prompt: str, max_output_tokens: int, max_retries: int, sleep_seconds: float) -> str:
    last_error = None
    for attempt in range(max_retries):
        try:
            return call_openai_once(api_key, model, prompt, max_output_tokens=max_output_tokens)
        except Exception as e:
            last_error = e
            wait = sleep_seconds * (2 ** attempt)
            print(f"Warning: request failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait:.1f}s.")
            time.sleep(wait)
    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ai_generated_raw.csv from ai_generation_prompts.csv using normal OpenAI API calls.")
    parser.add_argument("--prompts-csv", default=str(PROMPTS_CSV), help="Path to ai_generation_prompts.csv")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="OpenAI model name to use")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (optional if OPENAI_API_KEY is set)")
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt limit for a pilot run")
    parser.add_argument("--max-output-tokens", type=int, default=220, help="Max output tokens per request")
    parser.add_argument("--max-retries", type=int, default=5, help="Retry attempts per request")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Base wait time between retries")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing ai_generated_raw.csv if present")
    args = parser.parse_args()

    if not args.model:
        raise ValueError("No model provided. Set OPENAI_MODEL in .env or pass --model MODEL_NAME")

    api_key = get_api_key(args.api_key)
    prompt_df = read_prompt_csv(Path(args.prompts_csv))

    if args.limit is not None:
        prompt_df = prompt_df.head(args.limit).copy()
        print(f"Pilot mode: using first {len(prompt_df)} prompts")

    done_prompt_ids = set()
    rows = []

    if args.resume and FINAL_AI_CSV.exists():
        existing = pd.read_csv(FINAL_AI_CSV)
        required_existing = {"prompt_id", "source_human_id", "category", "generator_name", "generated_text"}
        if required_existing.issubset(existing.columns):
            done_prompt_ids = set(existing["prompt_id"].astype(str))
            rows = existing.to_dict(orient="records")
            print(f"Resume mode: found {len(done_prompt_ids)} existing rows in {FINAL_AI_CSV}")

    remaining = prompt_df[~prompt_df["prompt_id"].astype(str).isin(done_prompt_ids)].copy().reset_index(drop=True)
    print(f"Prompts to generate now: {len(remaining)}")

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Generating AI abstracts"):
        generated_text = generate_with_retries(
            api_key=api_key,
            model=args.model,
            prompt=str(row["prompt"]),
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )

        rows.append({
            "prompt_id": str(row["prompt_id"]),
            "source_human_id": row["source_human_id"],
            "category": row["category"],
            "generator_name": args.model,
            "generated_text": generated_text,
        })

        # Save incrementally after every request so progress is not lost.
        pd.DataFrame(rows).to_csv(FINAL_AI_CSV, index=False)

    print("Done.")
    print("Saved final AI CSV to:", FINAL_AI_CSV)
    print("Rows written:", len(rows))


if __name__ == "__main__":
    main()
