#!/usr/bin/env python3
"""Phase 1B: Classify incorrect pool examples by error type using GPT-4o.

For each incorrect pool example, GPT-4o identifies ALL errors with granular detail:
- Which specific claim in the answer is wrong
- What error type it is
- What the correct information should be (from the notes)

Uses the biomistral incorrect pool as the shared error pool for all models.

Usage:
    python classify_errors.py --folds 0 1 2 3 4
    python classify_errors.py --folds 0   # single fold for testing
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent.parent
BIO_INDEX_DIR = PROJECT_ROOT / "output" / "fullscale_4_biomistral" / "indices"
NOTES_FILE = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output" / "step8" / "error_classification"

ERROR_TYPES = [
    "omission",
    "hallucination",
    "reasoning_failure",
    "specificity",
    "context_confusion",
    "temporal_error",
]

ERROR_INSTRUCTIONS = {
    "omission": (
        "Ensure you include ALL relevant details from the discharge notes. "
        "Do not leave out diagnoses, medications, procedures, or test results."
    ),
    "hallucination": (
        "Do NOT fabricate any information. If something is not stated in the notes, "
        "say 'not specified'. Never invent dates, measurements, or procedures."
    ),
    "reasoning_failure": (
        "Consider the patient's full medical history when answering. Make reasonable "
        "clinical inferences from documented diagnoses, but distinguish inference from fact."
    ),
    "specificity": (
        "Provide specific names, values, and details from the notes. "
        "Avoid vague or general statements when specific information is available."
    ),
    "context_confusion": (
        "When multiple discharge summaries are provided, carefully distinguish which "
        "information comes from which admission. Do not conflate findings across notes."
    ),
    "temporal_error": (
        "Pay careful attention to the chronological sequence of events. "
        "Verify dates and timelines against the discharge notes before answering."
    ),
}

CLASSIFY_PROMPT = """This AI answer has been identified as INCORRECT by medical evaluation.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

AI's INCORRECT ANSWER:
{model_answer}

CORRECT ANSWER (Ground Truth):
{ground_truth}

Analyze the AI's answer line by line. For EACH specific error, identify:
1. The exact wrong claim from the AI's answer (quote it)
2. The error type (one of: omission, hallucination, reasoning_failure, specificity, context_confusion, temporal_error)
3. What the correct information is (from the discharge notes or ground truth)

Error type definitions:
- omission: Key information from the notes is missing from the answer
- hallucination: The answer states something not found in the discharge notes
- reasoning_failure: The answer misinterprets or draws wrong conclusions from the notes
- specificity: The answer is vague when specific details are available in the notes
- context_confusion: The answer mixes up information from different notes or admissions
- temporal_error: The answer gets dates, timeline, or sequence of events wrong

Respond in this exact format:

PRIMARY_ERROR: <the single most important error type>
NUM_ERRORS: <total number of distinct errors found>

ERROR_1:
QUOTE: "<exact text from the AI answer that is wrong>"
TYPE: <error_type>
CORRECT: "<what should have been said, based on the notes>"

ERROR_2:
QUOTE: "<exact text>"
TYPE: <error_type>
CORRECT: "<correction>"

(continue for all errors found)

SUMMARY: <one sentence overall description of what went wrong>"""


def get_note_for_patient(patient_id, notes_df):
    """Get assembled discharge note for a patient."""
    rows = notes_df[notes_df["patient_id"] == patient_id]
    if len(rows) == 0:
        return ""
    row = rows.iloc[0]
    parts = []
    for i in [1, 2, 3]:
        col = f"note_{i}"
        if col in row and pd.notna(row[col]):
            t = str(row[col]).strip()
            if t and t.lower() != "nan":
                parts.append(f"[Note {i}]\n{t}")
    return "\n\n".join(parts)


def classify_one(client, note, question, model_answer, ground_truth):
    """Classify errors using GPT-4o."""
    user_content = CLASSIFY_PROMPT.format(
        note=note, question=question,
        model_answer=model_answer, ground_truth=ground_truth,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical expert analyzing errors in AI-generated clinical answers. Be precise and thorough.",
                    },
                    {"role": "user", "content": user_content},
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            return parse_classification(raw)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {
                    "primary_error": None,
                    "num_errors": 0,
                    "errors": [],
                    "summary": f"API error: {e}",
                    "raw": "",
                }


def parse_classification(raw):
    """Parse GPT-4o classification output."""
    import re

    result = {
        "primary_error": None,
        "num_errors": 0,
        "errors": [],
        "summary": "",
        "raw": raw[:1000],
    }

    # Extract primary error
    m = re.search(r"PRIMARY_ERROR:\s*(\w+)", raw)
    if m:
        pe = m.group(1).lower()
        if pe in ERROR_TYPES:
            result["primary_error"] = pe

    # Extract num errors
    m = re.search(r"NUM_ERRORS:\s*(\d+)", raw)
    if m:
        result["num_errors"] = int(m.group(1))

    # Extract individual errors
    error_blocks = re.findall(
        r"ERROR_\d+:\s*\n"
        r"QUOTE:\s*\"?(.*?)\"?\s*\n"
        r"TYPE:\s*(\w+)\s*\n"
        r"CORRECT:\s*\"?(.*?)\"?\s*(?=\n\nERROR_|\nSUMMARY:|\Z)",
        raw, re.DOTALL
    )
    for quote, etype, correct in error_blocks:
        etype_clean = etype.strip().lower()
        if etype_clean not in ERROR_TYPES:
            # Try fuzzy match
            for et in ERROR_TYPES:
                if et.startswith(etype_clean[:4]):
                    etype_clean = et
                    break
        result["errors"].append({
            "quote": quote.strip(),
            "type": etype_clean,
            "correct": correct.strip(),
        })

    # Extract summary
    m = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", raw)
    if m:
        result["summary"] = m.group(1).strip()

    # If no primary error was parsed but we have errors, use the first one
    if not result["primary_error"] and result["errors"]:
        result["primary_error"] = result["errors"][0]["type"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1B: Classify errors with GPT-4o")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    # Setup API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    client = OpenAI(api_key=api_key)

    # Load notes
    notes_df = pd.read_json(NOTES_FILE, lines=True)
    print(f"Loaded notes for {len(notes_df)} patients")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Deduplicate: collect unique patient_ids across all folds ---
    all_pools = {}  # fold_id -> pool
    unique_examples = {}  # patient_id -> example (first occurrence)

    for fold_id in args.folds:
        fold_dir = BIO_INDEX_DIR / f"fold_{fold_id}"
        pool_file = fold_dir / "incorrect_pool.json"
        if not pool_file.exists():
            print(f"Fold {fold_id}: incorrect_pool.json not found, skipping")
            continue
        with open(pool_file) as f:
            pool = json.load(f)
        all_pools[fold_id] = pool
        for ex in pool:
            pid = int(ex["patient_id"])
            if pid not in unique_examples:
                unique_examples[pid] = ex

    total_entries = sum(len(p) for p in all_pools.values())
    print(f"Total entries across {len(all_pools)} folds: {total_entries}")
    print(f"Unique patients to classify: {len(unique_examples)} (saving {total_entries - len(unique_examples)} duplicate calls)")

    # --- Resume support: load existing classifications ---
    master_file = OUTPUT_DIR / "all_errors_by_patient.json"
    classified = {}
    if master_file.exists():
        with open(master_file) as f:
            existing = json.load(f)
        classified = {int(ex["patient_id"]): ex for ex in existing}
        print(f"Already classified: {len(classified)} patients")

    # --- Classify unique patients ---
    to_classify = [pid for pid in unique_examples if pid not in classified]
    print(f"Remaining to classify: {len(to_classify)}")

    start_time = time.time()
    new_count = 0

    for i, pid in enumerate(to_classify):
        ex = unique_examples[pid]
        note = get_note_for_patient(pid, notes_df)
        classification = classify_one(
            client, note, ex["question"],
            ex["openended_answer"], ex["ground_truth"],
        )

        result = {
            "patient_id": pid,
            "question": ex["question"],
            "openended_answer": ex["openended_answer"],
            "ground_truth": ex["ground_truth"],
            "primary_error": classification["primary_error"],
            "num_errors": classification["num_errors"],
            "errors": classification["errors"],
            "error_summary": classification["summary"],
        }
        classified[pid] = result
        new_count += 1

        if new_count % 20 == 0:
            elapsed = time.time() - start_time
            rate = new_count / elapsed if elapsed > 0 else 0
            remaining = (len(to_classify) - new_count) / rate if rate > 0 else 0
            print(f"  [{new_count}/{len(to_classify)}] "
                  f"{rate:.1f} ex/s, ~{remaining/60:.0f}min remaining | "
                  f"last: {classification['primary_error']}")
            # Checkpoint
            with open(master_file, "w") as f:
                json.dump(list(classified.values()), f)

        time.sleep(0.2)  # Rate limiting

    # Final save of master file
    with open(master_file, "w") as f:
        json.dump(list(classified.values()), f, indent=2)
    print(f"\nMaster classification saved: {master_file} ({len(classified)} patients)")

    # --- Map back to per-fold files ---
    for fold_id, pool in all_pools.items():
        fold_results = []
        for ex in pool:
            pid = int(ex["patient_id"])
            if pid in classified:
                c = classified[pid]
                fold_results.append({
                    **ex,
                    "primary_error": c["primary_error"],
                    "num_errors": c["num_errors"],
                    "errors": c["errors"],
                    "error_summary": c["error_summary"],
                })
            else:
                fold_results.append({**ex, "primary_error": None, "num_errors": 0, "errors": [], "error_summary": ""})

        fold_file = OUTPUT_DIR / f"fold_{fold_id}_errors.json"
        with open(fold_file, "w") as f:
            json.dump(fold_results, f, indent=2)
        print(f"  Fold {fold_id}: {len(fold_results)} examples -> {fold_file.name}")

    # --- Overall summary ---
    all_type_counts = Counter(c["primary_error"] for c in classified.values())
    print(f"\n{'='*60}")
    print(f"ERROR DISTRIBUTION ({len(classified)} unique patients)")
    print(f"{'='*60}")
    for et in ERROR_TYPES:
        count = all_type_counts.get(et, 0)
        pct = count / len(classified) * 100 if classified else 0
        print(f"  {et}: {count} ({pct:.1f}%)")
    null_count = all_type_counts.get(None, 0)
    if null_count:
        print(f"  unclassified: {null_count}")

    # Multi-error stats
    multi = sum(1 for c in classified.values() if c["num_errors"] > 1)
    avg_errors = sum(c["num_errors"] for c in classified.values()) / len(classified) if classified else 0
    print(f"\n  Avg errors/example: {avg_errors:.1f}")
    print(f"  Multi-error: {multi} ({multi/len(classified)*100:.0f}%)")

    print(f"\nClassification saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
