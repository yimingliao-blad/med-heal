#!/usr/bin/env python3
"""
Phase 3: Evaluate Composite ICL Pilot Predictions with GPT-4o

Usage:
    python evaluate_pilot12_phase3.py
    python evaluate_pilot12_phase3.py --methods gtr_note_guideline_pos_k1
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "pilot_phase3" / "fold_0"

AVAILABLE_METHODS = [
    "gtr_note_guideline_pos_k1",
    "gtr_note_guideline_pos_annotated_k1",
    "gtr_note_neg_annotated_k1",
    "gtr_note_neg_guideline_k1",
    "gtr_note_neg_full_k1",
    "gtr_note_posneg_annotated_k1",
    "gtr_note_posneg_guideline_k1",
]

EVAL_PROMPT = """You are a medical expert evaluating an AI model's answer to a clinical question.

DISCHARGE SUMMARY:
{note}

QUESTION: {question}

GROUND TRUTH (Correct Answer):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

Task: Determine if the model's answer is correct compared to the ground truth.
The answer doesn't need to be word-for-word identical, but should capture the key medical facts.

Respond in this EXACT format:
CORRECT: [yes/no]
REASON: [Brief explanation in 1-2 sentences]"""


def evaluate_one(client, note, question, ground_truth, model_answer, model="gpt-4o"):
    prompt = EVAL_PROMPT.format(
        note=note, question=question, ground_truth=ground_truth, model_answer=model_answer,
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=150,
            )
            content = resp.choices[0].message.content.strip()
            correct_m = re.search(r"CORRECT:\s*(yes|no)", content, re.IGNORECASE)
            reason_m = re.search(r"REASON:\s*(.+?)(?:\n|$)", content, re.IGNORECASE | re.DOTALL)
            return {
                "correct": correct_m.group(1).lower() if correct_m else None,
                "reason": reason_m.group(1).strip() if reason_m else None,
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return {"correct": None, "reason": str(e)}
    return {"correct": None, "reason": "max retries"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 pilot predictions")
    parser.add_argument("--methods", nargs="+", default=AVAILABLE_METHODS)
    parser.add_argument("--gpt4_model", default="gpt-4o")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    client = OpenAI(api_key=api_key)

    # Load notes for evaluation context
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    notes_lookup = {}
    for _, row in notes_df.iterrows():
        pid = str(row["patient_id"])
        parts = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in row and pd.notna(row[col]):
                t = str(row[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(t)
        notes_lookup[pid] = "\n\n".join(parts)
    print(f"Loaded notes for {len(notes_lookup)} patients")

    total_methods = len(args.methods)
    for method_idx, method in enumerate(args.methods, 1):
        gen_file = OUTPUT_DIR / f"{method}_generated.csv"
        eval_file = OUTPUT_DIR / f"{method}_evaluated.csv"

        if not gen_file.exists():
            print(f"\n  [{method_idx}/{total_methods}] {method}: No generated file, skipping")
            continue

        df = pd.read_csv(gen_file)

        # Check if already evaluated
        if eval_file.exists():
            existing = pd.read_csv(eval_file)
            if "openended_correct" in existing.columns and existing["openended_correct"].notna().sum() >= len(df):
                acc = (existing["openended_correct"] == "yes").mean() * 100
                print(f"\n  [{method_idx}/{total_methods}] {method}: Already evaluated ({acc:.1f}%)")
                continue

        # Resume support
        if eval_file.exists():
            df = pd.read_csv(eval_file)
            start_idx = df["openended_correct"].notna().sum()
            print(f"\n  [{method_idx}/{total_methods}] {method}: Resuming from {start_idx}/{len(df)}")
        else:
            df["openended_correct"] = None
            df["openended_reason"] = None
            start_idx = 0
            print(f"\n  [{method_idx}/{total_methods}] {method}: Evaluating {len(df)} predictions")

        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            pid = str(row.get("patient_id", ""))
            note = notes_lookup.get(pid, "")
            result = evaluate_one(
                client, note, row["question"], row["ground_truth"],
                row["model_answer"], model=args.gpt4_model,
            )
            df.at[i, "openended_correct"] = result["correct"]
            df.at[i, "openended_reason"] = result["reason"]

            if (i + 1) % 10 == 0:
                df.to_csv(eval_file, index=False)
                done = i + 1
                correct = (df["openended_correct"].iloc[:done] == "yes").sum()
                print(f"    Progress: {done}/{len(df)} ({correct}/{done} correct)")

            time.sleep(0.2)

        df.to_csv(eval_file, index=False)
        acc = (df["openended_correct"] == "yes").mean() * 100
        print(f"    Done: {method} -> {acc:.1f}%")

    print(f"\nAll evaluation complete. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
