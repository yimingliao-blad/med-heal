#!/usr/bin/env python3
"""Compare pilot results across all conditions for all models.

Extracts the same random 10 samples per fold (seed=42) from existing evaluated
results and new contrastive conditions, producing a comparison table.

Usage:
    python pilot_comparison.py
    python pilot_comparison.py --model biomistral-7b
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
FOLDS_DIR = PROJECT_ROOT / "output" / "folds"
STEP8_DIR = PROJECT_ROOT / "output" / "step8"

EXISTING_CONDITIONS = [
    "zeroshot", "gtr_note_pos_k1", "gtr_note_neg_k1", "gtr_note_posneg_k1",
    "cot_evidence", "cot_conclusion", "multiturn", "gtr_note_any_unlabeled_k1",
]

NEW_CONDITIONS = [
    "contrastive_random", "contrastive_targeted",
]

MODELS = [
    "biomistral-7b", "qwen2.5-7b-instruct", "llama-3.1-8b-instruct",
    "deepseek-r1-distill-llama-8b", "qwen3-8b",
]

SEED = 42


def get_pilot_idx(seed=SEED, n_per_fold=10):
    """Get random pilot sample indices per fold."""
    pilot_idx = {}
    for fold_id in range(5):
        test_file = FOLDS_DIR / f"fold_{fold_id}" / "test.jsonl"
        with open(test_file) as f:
            samples = [json.loads(line) for line in f]
        for i, s in enumerate(samples):
            if "idx" not in s:
                s["idx"] = i
        rng = random.Random(seed + fold_id)
        selected = rng.sample(samples, min(n_per_fold, len(samples)))
        pilot_idx[fold_id] = [s["idx"] for s in selected]
    return pilot_idx


def get_condition_accuracy(model, condition, pilot_idx, source="evaluated"):
    """Get accuracy for a condition on pilot samples."""
    correct = 0
    total = 0
    for fold_id in range(5):
        if source == "evaluated":
            f = STEP8_DIR / model / f"fold_{fold_id}" / f"{condition}_evaluated_binary.csv"
        else:
            f = STEP8_DIR / model / f"fold_{fold_id}" / f"{condition}_evaluated_binary.csv"

        if not f.exists():
            continue
        df = pd.read_csv(f)
        target_idx = set(pilot_idx[fold_id])
        pilot_df = df[df["idx"].isin(target_idx)]
        correct += pilot_df["binary_correct"].sum()
        total += len(pilot_df)
    return correct, total


def get_critic_stats(model, pilot_idx):
    """Get critic flagging stats for pilot samples."""
    flagged = 0
    total = 0
    null = 0
    error_types = {}
    for fold_id in range(5):
        f = STEP8_DIR / model / f"fold_{fold_id}" / "critic_results.json"
        if not f.exists():
            continue
        with open(f) as fp:
            data = json.load(fp)
        data_by_idx = {int(r["idx"]): r for r in data}
        for idx in pilot_idx[fold_id]:
            if idx in data_by_idx:
                r = data_by_idx[idx]
                total += 1
                if r.get("verdict") == 0:
                    flagged += 1
                    et = r.get("error_type", "unknown")
                    error_types[et] = error_types.get(et, 0) + 1
                elif r.get("verdict") is None:
                    null += 1
    return flagged, total, null, error_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=MODELS)
    args = parser.parse_args()

    pilot_idx = get_pilot_idx()

    all_conditions = EXISTING_CONDITIONS + NEW_CONDITIONS

    for model in args.model:
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print(f"{'='*70}")

        # Critic stats
        flagged, total, null, error_types = get_critic_stats(model, pilot_idx)
        if total > 0:
            print(f"Critic: {flagged}/{total} flagged as wrong ({flagged/total*100:.0f}%), {null} null")
            if error_types:
                print(f"  Error types: {error_types}")

        # Condition comparison
        print(f"\n{'Condition':<30} {'Correct':>7} {'Total':>5} {'Accuracy':>8}")
        print("-" * 55)
        for cond in all_conditions:
            correct, total = get_condition_accuracy(model, cond, pilot_idx)
            if total > 0:
                print(f"{cond:<30} {correct:>7.0f} {total:>5} {correct/total*100:>7.1f}%")
            else:
                print(f"{cond:<30} {'—':>7} {'—':>5} {'N/A':>8}")

    # Cross-model summary table
    print(f"\n\n{'='*80}")
    print("CROSS-MODEL SUMMARY (Pilot: 10 random/fold × 5 folds = 50 samples)")
    print(f"{'='*80}")

    header = f"{'Model':<25}"
    for cond in ["zeroshot", "gtr_note_pos_k1", "contrastive_random", "contrastive_targeted"]:
        header += f" {cond:>12}"
    print(header)
    print("-" * 75)

    for model in args.model:
        row = f"{model:<25}"
        for cond in ["zeroshot", "gtr_note_pos_k1", "contrastive_random", "contrastive_targeted"]:
            correct, total = get_condition_accuracy(model, cond, pilot_idx)
            if total > 0:
                row += f" {correct/total*100:>11.1f}%"
            else:
                row += f" {'N/A':>12}"
        print(row)


if __name__ == "__main__":
    main()
