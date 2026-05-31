#!/usr/bin/env python3
"""
Phase 1: Evaluate Pilot 12 Predictions with GPT-4o

Evaluates all generated conditions from generate_pilot12.py using the same
GPT-4o evaluation prompt as fullscale_3.

Usage:
    python evaluate_pilot12.py                              # All conditions
    python evaluate_pilot12.py --methods gtr_pos_k1         # Single condition
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "pilot" / "fold_0"

AVAILABLE_METHODS = [
    "bm25_pos_k1", "gtr_pos_k1", "kate_pos_k1",
    "gtr_pos_k2", "gtr_pos_k3",
    "gtr_type_pos_k1", "gtr_guideline_pos_k1",
    "gtr_note_pos_k1", "gtr_type_note_pos_k1",
    "gtr_note_context_pos_k1", "gtr_note_fullctx_pos_k1",
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
    parser = argparse.ArgumentParser(description="Evaluate Pilot 12 predictions")
    parser.add_argument("--methods", nargs="+", default=AVAILABLE_METHODS)
    parser.add_argument("--gpt4_model", default="gpt-4o")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    client = OpenAI(api_key=api_key)

    # Load notes
    notes_file = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
    notes_df = pd.read_json(notes_file, lines=True)
    notes_lookup = {str(r.get("patient_id", "")): r.to_dict() for _, r in notes_df.iterrows()}
    print(f"Loaded notes for {len(notes_lookup)} patients")

    total_methods = len(args.methods)
    for method_idx, method in enumerate(args.methods, 1):
        gen_file = OUTPUT_DIR / f"{method}_generated.csv"
        eval_file = OUTPUT_DIR / f"{method}_evaluated.csv"

        if not gen_file.exists():
            print(f"  [{method_idx}/{total_methods}] SKIP {method}: no generated file")
            continue

        gen_df = pd.read_csv(gen_file)

        if eval_file.exists():
            existing = pd.read_csv(eval_file)
            if len(existing) >= len(gen_df):
                correct = (existing["openended_correct"] == "yes").sum()
                print(f"  [{method_idx}/{total_methods}] DONE {method}: {correct}/{len(existing)} = {correct/len(existing):.0%}")
                continue

        print(f"\n  [{method_idx}/{total_methods}] Evaluating {method} ({len(gen_df)} samples)")

        results = []
        done_ids = set()
        if eval_file.exists():
            existing = pd.read_csv(eval_file)
            done_ids = set(existing["idx"].tolist())
            results = existing.to_dict("records")
            print(f"    Resuming from {len(results)}")

        for _, row in gen_df.iterrows():
            idx = row["idx"]
            if idx in done_ids:
                continue

            pid = str(row.get("patient_id", ""))
            note = ""
            if pid in notes_lookup:
                note_row = notes_lookup[pid]
                parts = []
                for i in [1, 2, 3]:
                    k = f"note_{i}"
                    if k in note_row and note_row[k] and str(note_row[k]).strip().lower() != "nan":
                        parts.append(f"[Note {i}]\n{str(note_row[k]).strip()}")
                note = "\n\n".join(parts)

            ev = evaluate_one(
                client, note, str(row["question"]),
                str(row["ground_truth"]), str(row["model_answer"]),
                model=args.gpt4_model,
            )

            results.append({
                "idx": idx,
                "patient_id": pid,
                "fold_id": 0,
                "question": row["question"],
                "question_type": row.get("question_type", ""),
                "ground_truth": row["ground_truth"],
                "model_answer": row["model_answer"],
                "method": method,
                "retrieval_sim_score": row.get("retrieval_sim_score", ""),
                "openended_correct": ev["correct"],
                "openended_reason": ev.get("reason", ""),
            })

            time.sleep(0.3)
            if len(results) % 20 == 0:
                pd.DataFrame(results).to_csv(eval_file, index=False)
                print(f"    Progress: {len(results)}/{len(gen_df)}")

        result_df = pd.DataFrame(results)
        result_df.to_csv(eval_file, index=False)
        correct = (result_df["openended_correct"] == "yes").sum()
        print(f"    {method}: {correct}/{len(result_df)} = {correct/len(result_df):.0%}")

    # Print summary
    print(f"\n{'='*60}")
    print("PILOT 12 EVALUATION SUMMARY")
    print(f"{'='*60}")
    for method in args.methods:
        ef = OUTPUT_DIR / f"{method}_evaluated.csv"
        if ef.exists():
            df = pd.read_csv(ef)
            correct = (df["openended_correct"] == "yes").sum()
            total = len(df)
            print(f"  {method:<30} {correct}/{total} = {correct/total:.0%}")


if __name__ == "__main__":
    main()
