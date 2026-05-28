#!/usr/bin/env python3
"""
Step 8: Evaluate generated predictions with GPT-4o

Usage:
    python evaluate_step8.py --model biomistral-7b
    python evaluate_step8.py --model qwen3-8b --folds 0 1 --conditions zeroshot gtr_note_pos_k1
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "step8"

ALL_CONDITIONS = [
    "zeroshot",
    "gtr_note_pos_k1", "gtr_note_neg_k1", "gtr_note_posneg_k1",
    "cot_evidence", "cot_conclusion", "multiturn",
    "gtr_note_any_unlabeled_k1",
    "gtr_note_neg_k2", "gtr_note_neg_k3", "gtr_note_neg_k4", "gtr_note_neg_k5",
]

ALL_MODELS = [
    "biomistral-7b",
    "deepseek-r1-distill-llama-8b",
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "qwen3-8b",
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
                "openended_correct": correct_m.group(1).lower() if correct_m else None,
                "openended_reason": reason_m.group(1).strip() if reason_m else None,
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return {"openended_correct": None, "openended_reason": str(e)}
    return {"openended_correct": None, "openended_reason": "max retries"}


def main():
    parser = argparse.ArgumentParser(description="Step 8: GPT-4o Evaluation")
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions to evaluate (default: all found)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpt4_model", default="gpt-4o")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    client = OpenAI(api_key=api_key)

    # Load notes for discharge summary context in evaluation
    notes_file = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
    notes_df = pd.read_json(notes_file, lines=True)
    notes_lookup = {}
    for _, r in notes_df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(f"[Note {i}]\n{t}")
        notes_lookup[pid] = "\n\n".join(parts)
    print(f"Loaded notes for {len(notes_lookup)} patients")

    # Determine conditions to evaluate
    conditions = args.conditions if args.conditions else ALL_CONDITIONS

    model_dir = OUTPUT_DIR / args.model
    total_evaluated = 0
    total_skipped = 0

    for fold_id in args.folds:
        fold_dir = model_dir / f"fold_{fold_id}"
        if not fold_dir.exists():
            continue

        for condition in conditions:
            gen_file = fold_dir / f"{condition}_generated.csv"
            eval_file = fold_dir / f"{condition}_evaluated.csv"

            if not gen_file.exists():
                continue

            gen_df = pd.read_csv(gen_file)

            # Skip if already evaluated
            if eval_file.exists():
                existing = pd.read_csv(eval_file)
                if len(existing) >= len(gen_df):
                    correct = (existing["openended_correct"] == "yes").sum()
                    total = len(existing)
                    print(f"  DONE fold_{fold_id}/{condition}: {correct}/{total} = {correct/total:.1%}")
                    total_skipped += 1
                    continue

            print(f"\n  Evaluating fold_{fold_id}/{condition} ({len(gen_df)} samples)")

            results = []
            # Resume support for partial evaluations
            done_ids = set()
            if eval_file.exists():
                existing = pd.read_csv(eval_file)
                done_ids = set(existing["idx"].tolist())
                results = existing.to_dict("records")
                print(f"    Resuming from {len(results)}")

            for _, row in gen_df.iterrows():
                idx = row.get("idx", 0)
                if idx in done_ids:
                    continue

                pid = str(row.get("patient_id", ""))
                note = notes_lookup.get(pid, "")
                question = str(row.get("question", ""))
                gt = str(row.get("ground_truth", ""))
                model_answer = str(row.get("model_answer", ""))

                eval_result = evaluate_one(
                    client, note, question, gt, model_answer, model=args.gpt4_model
                )

                result_row = row.to_dict()
                result_row.update(eval_result)
                results.append(result_row)

                if len(results) % 20 == 0:
                    pd.DataFrame(results).to_csv(eval_file, index=False)
                    correct_so_far = sum(1 for r in results if r.get("openended_correct") == "yes")
                    print(f"    Progress: {len(results)}/{len(gen_df)} "
                          f"({correct_so_far}/{len(results)} correct)")

                time.sleep(0.3)

            pd.DataFrame(results).to_csv(eval_file, index=False)
            correct = sum(1 for r in results if r.get("openended_correct") == "yes")
            print(f"    Done: {correct}/{len(results)} = {correct/len(results):.1%}")
            total_evaluated += 1

    print(f"\nEvaluation complete for {args.model}.")
    print(f"  Evaluated: {total_evaluated} condition-folds")
    print(f"  Skipped (already done): {total_skipped}")


if __name__ == "__main__":
    main()
