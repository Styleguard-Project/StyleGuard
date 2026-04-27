from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from config import BATCH_DIR, DEFAULT_OPENAI_MODEL, PROCESSED_DIR

load_dotenv()

FILES_URL = "https://api.openai.com/v1/files"
BATCHES_URL = "https://api.openai.com/v1/batches"
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

def build_batch_jsonl(df: pd.DataFrame, model: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "custom_id": str(row["prompt_id"]),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": str(row["prompt"]),
                    "max_output_tokens": 220,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def upload_batch_file(api_key: str, batch_input_jsonl: Path) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    with batch_input_jsonl.open("rb") as f:
        files = {"file": (batch_input_jsonl.name, f, "application/jsonl")}
        data = {"purpose": "batch"}
        resp = requests.post(FILES_URL, headers=headers, files=files, data=data, timeout=300)
    resp.raise_for_status()
    return resp.json()["id"]

def create_batch(api_key: str, input_file_id: str, metadata: Optional[Dict[str, str]] = None) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }
    if metadata:
        body["metadata"] = metadata
    resp = requests.post(BATCHES_URL, headers=headers, json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["id"]

def retrieve_batch(api_key: str, batch_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{BATCHES_URL}/{batch_id}", headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()

def wait_for_batch(api_key: str, batch_id: str, poll_seconds: int) -> Dict[str, Any]:
    while True:
        payload = retrieve_batch(api_key, batch_id)
        status = payload.get("status")
        print(f"Batch status: {status}")
        if status in {"completed", "failed", "expired", "cancelled"}:
            return payload
        time.sleep(poll_seconds)

def download_file_content(api_key: str, file_id: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{FILES_URL}/{file_id}/content", headers=headers, timeout=600)
    resp.raise_for_status()
    return resp.text

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

def parse_batch_output_to_csv(prompt_df: pd.DataFrame, output_jsonl_text: str, model_name: str, out_csv: Path) -> None:
    prompt_lookup = {
        str(row["prompt_id"]): {
            "prompt_id": str(row["prompt_id"]),
            "source_human_id": row["source_human_id"],
            "category": row["category"],
        }
        for _, row in prompt_df.iterrows()
    }

    rows = []
    for line in output_jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        custom_id = str(obj.get("custom_id", ""))
        if custom_id not in prompt_lookup:
            continue

        error_obj = obj.get("error")
        if error_obj:
            print(f"Warning: request {custom_id} returned an error: {error_obj}")
            continue

        response = obj.get("response", {})
        body = response.get("body", {})
        generated_text = extract_text_from_response_body(body)

        if not generated_text:
            print(f"Warning: request {custom_id} produced empty text.")
            continue

        meta = prompt_lookup[custom_id]
        rows.append({
            "prompt_id": meta["prompt_id"],
            "source_human_id": meta["source_human_id"],
            "category": meta["category"],
            "generator_name": model_name,
            "generated_text": generated_text,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Generate ai_generated_raw.csv from ai_generation_prompts.csv using OpenAI Batch API.")
    parser.add_argument("--prompts-csv", default=str(PROMPTS_CSV), help="Path to ai_generation_prompts.csv")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="OpenAI model name to use")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (optional if OPENAI_API_KEY is set)")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--resume-batch-id", default=None, help="Resume an existing batch ID")
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt limit for a small pilot run")
    args = parser.parse_args()

    if not args.model:
        raise ValueError("No model provided. Set OPENAI_MODEL in .env or pass --model MODEL_NAME")

    api_key = get_api_key(args.api_key)
    prompt_df = read_prompt_csv(Path(args.prompts_csv))
    if args.limit is not None:
        prompt_df = prompt_df.head(args.limit).copy()
        print(f"Pilot mode: using first {len(prompt_df)} prompts")

    batch_input_path = BATCH_DIR / "batch_input.jsonl"
    batch_info_path = BATCH_DIR / "batch_info.json"
    batch_output_path = BATCH_DIR / "batch_output.jsonl"

    if args.resume_batch_id:
        batch_payload = wait_for_batch(api_key, args.resume_batch_id.strip(), args.poll_seconds)
    else:
        print("Building batch input JSONL...")
        build_batch_jsonl(prompt_df, args.model, batch_input_path)
        print("Uploading batch input...")
        input_file_id = upload_batch_file(api_key, batch_input_path)
        print("Creating batch job...")
        batch_id = create_batch(api_key, input_file_id, metadata={"project": "styleguard-minimum", "source": Path(args.prompts_csv).name})
        print(f"Created batch ID: {batch_id}")
        batch_payload = wait_for_batch(api_key, batch_id, args.poll_seconds)

    save_text(batch_info_path, json.dumps(batch_payload, indent=2))
    print(f"Saved batch info: {batch_info_path}")

    status = batch_payload.get("status")
    if status != "completed":
        print(f"Batch did not complete successfully. Final status: {status}")
        print("output_file_id:", batch_payload.get("output_file_id"))
        print("error_file_id:", batch_payload.get("error_file_id"))
        sys.exit(1)

    output_file_id = batch_payload.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id was returned.")

    print("Downloading batch output...")
    output_text = download_file_content(api_key, output_file_id)
    save_text(batch_output_path, output_text)
    print(f"Saved raw output: {batch_output_path}")

    print("Converting batch output to ai_generated_raw.csv ...")
    parse_batch_output_to_csv(prompt_df, output_text, args.model, FINAL_AI_CSV)
    print(f"Saved final AI CSV: {FINAL_AI_CSV}")

if __name__ == "__main__":
    main()
