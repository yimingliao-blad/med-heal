#!/usr/bin/env python3
"""
Phase 2: Evaluate Fullscale Pilot 12 Predictions with GPT-4o

Usage:
    python evaluate_fullscale_pilot12.py
    python evaluate_fullscale_pilot12.py --folds 0 --methods gtr_note_pos_k1
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "fullscale"

AVAILABLE_METHODS = [
    "gtr_note_pos_k1", "gtr_note_fullctx_pos_k1",
    "gtr_note_neg_k1", "gtr_note_posneg_k1",
    "gtr_type_note_pos_k1",
    "gtr_note_any_unlabeled_k1", "gtr_note_any_labeled_k1",
    "gtr_note_pos_k2", "gtr_note_pos_k3", "gtr_note_pos_k4", "gtr_note_pos_k5",
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
    parser = argparse.ArgumentParser(description="Evaluate Fullscale Pilot 12 predictions")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=AVAILABLE_METHODS)
    parser.add_argument("--gpt4_model", default="gpt-4o")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    client = OpenAI(api_key=api_key)

    notes_file = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
    notes_df = pd.read_json(notes_file, lines=True)
    notes_lookup = {str(r.get("patient_id", "")): r.to_dict() for _, r in notes_df.iterrows()}
    print(f"Loaded notes for {len(notes_lookup)} patients")

    total_combos = len(args.methods) * len(args.folds)
    completed = 0

    for fold_id in args.folds:
        fold_dir = OUTPUT_DIR / f"fold_{fold_id}"

        for method in args.methods:
            completed += 1
            gen_file = fold_dir / f"{method}_generated.csv"
            eval_file = fold_dir / f"{method}_evaluated.csv"

            if not gen_file.exists():
                print(f"  [{completed}/{total_combos}] SKIP fold_{fold_id}/{method}: no generated file")
                continue

            gen_df = pd.read_csv(gen_file)

            if eval_file.exists():
                existing = pd.read_csv(eval_file)
                if len(existing) >= len(gen_df):
                    correct = (existing["openended_correct"] == "yes").sum()
                    print(f"  [{completed}/{total_combos}] DONE fold_{fold_id}/{method}: {correct}/{len(existing)} = {correct/len(existing):.2%}")
                    continue

            print(f"\n  [{completed}/{total_combos}] Evaluating fold_{fold_id}/{method} ({len(gen_df)} samples)")

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
                    "fold_id": fold_id,
                    "question": row["question"],
                    "question_type": row.get("question_type", ""),
                    "ground_truth": row["ground_truth"],
                    "model_answer": row["model_answer"],
                    "method": method,
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
            print(f"    {method}: {correct}/{len(result_df)} = {correct/len(result_df):.2%}")

    # Summary
    print(f"\n{'='*60}")
    print("FULLSCALE PILOT 12 EVALUATION SUMMARY")
    print(f"{'='*60}")
    for method in args.methods:
        all_correct = 0
        all_total = 0
        for fold_id in args.folds:
            ef = OUTPUT_DIR / f"fold_{fold_id}" / f"{method}_evaluated.csv"
            if ef.exists():
                df = pd.read_csv(ef)
                all_correct += (df["openended_correct"] == "yes").sum()
                all_total += len(df)
        if all_total > 0:
            print(f"  {method:<35} {all_correct}/{all_total} = {all_correct/all_total:.2%}")


if __name__ == "__main__":
    main()
