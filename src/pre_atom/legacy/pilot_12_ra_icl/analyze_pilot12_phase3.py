#!/usr/bin/env python3
"""
Phase 3: Analyze Composite ICL Pilot Results

Compares 7 composite conditions against Phase 1/2 baselines on fold 0 (50 samples).

Usage:
    python analyze_pilot12_phase3.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "pilot_phase3" / "fold_0"

PHASE3_METHODS = [
    "gtr_note_guideline_pos_k1",
    "gtr_note_guideline_pos_annotated_k1",
    "gtr_note_neg_annotated_k1",
    "gtr_note_neg_guideline_k1",
    "gtr_note_neg_full_k1",
    "gtr_note_posneg_annotated_k1",
    "gtr_note_posneg_guideline_k1",
]

# Phase 1 pilot baselines (fold 0, first 50 samples) — from pilot results
# These are loaded dynamically if available
PILOT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "pilot" / "fold_0"


def load_pilot_baseline(method):
    """Load a Phase 1 pilot result for comparison."""
    eval_file = PILOT_DIR / f"{method}_evaluated.csv"
    if eval_file.exists():
        df = pd.read_csv(eval_file)
        if "openended_correct" in df.columns:
            acc = (df["openended_correct"] == "yes").mean() * 100
            return acc
    gen_file = PILOT_DIR / f"{method}_generated.csv"
    if gen_file.exists():
        return None  # generated but not evaluated
    return None


def main():
    print("=" * 70)
    print("PHASE 3: COMPOSITE ICL PILOT RESULTS (Fold 0, 50 samples)")
    print("=" * 70)

    # Load Phase 3 results
    results = {}
    for method in PHASE3_METHODS:
        eval_file = OUTPUT_DIR / f"{method}_evaluated.csv"
        if not eval_file.exists():
            print(f"  WARNING: {method} not evaluated yet")
            continue
        df = pd.read_csv(eval_file)
        acc = (df["openended_correct"] == "yes").mean() * 100
        n = len(df)
        results[method] = {"acc": round(acc, 1), "n": n}

    # Load Phase 1 baselines for comparison
    phase1_baselines = {}
    for method in ["gtr_note_pos_k1", "gtr_note_fullctx_pos_k1"]:
        acc = load_pilot_baseline(method)
        if acc is not None:
            phase1_baselines[method] = round(acc, 1)

    # Print results table
    print("\n--- PHASE 3 COMPOSITE CONDITIONS ---\n")
    print(f"{'Method':<45} {'Acc%':>6} {'N':>4}")
    print("-" * 58)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["acc"], reverse=True)
    for method, r in sorted_results:
        print(f"  {method:<43} {r['acc']:>5.1f}% {r['n']:>4}")

    # Print Phase 1 baselines
    if phase1_baselines:
        print(f"\n--- PHASE 1/2 BASELINES (fold 0, 50 samples) ---\n")
        for method, acc in sorted(phase1_baselines.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method:<43} {acc:>5.1f}%")

    # Fullscale baselines for reference
    print(f"\n--- FULLSCALE BASELINES (5-fold CV, for reference) ---\n")
    fullscale_baselines = {
        "zeroshot": 77.13,
        "type_positive": 78.80,
        "guideline_pos_annotated": 79.21,
        "gtr_note_pos_k1": 80.14,
        "gtr_note_neg_k1": 76.71,
        "gtr_note_posneg_k1": 78.79,
    }
    for method, acc in sorted(fullscale_baselines.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:<43} {acc:>5.1f}%")

    # Question type breakdown
    print(f"\n--- QUESTION TYPE BREAKDOWN ---\n")
    print(f"{'Method':<45} {'Select':>7} {'Temp':>7} {'Doc':>7} {'Prec':>7}")
    print("-" * 80)
    for method in PHASE3_METHODS:
        eval_file = OUTPUT_DIR / f"{method}_evaluated.csv"
        if not eval_file.exists():
            continue
        df = pd.read_csv(eval_file)
        type_accs = []
        for qtype in ["selectivity", "temporal", "documentation", "precision"]:
            subset = df[df["question_type"] == qtype]
            if len(subset) > 0:
                acc = (subset["openended_correct"] == "yes").mean() * 100
                type_accs.append(f"{acc:>5.1f}%")
            else:
                type_accs.append("  N/A")
        print(f"  {method:<43} {type_accs[0]:>7} {type_accs[1]:>7} {type_accs[2]:>7} {type_accs[3]:>7}")

    # Save summary
    summary = {
        "phase3_results": results,
        "phase1_baselines": phase1_baselines,
        "fullscale_baselines": fullscale_baselines,
    }
    summary_file = OUTPUT_DIR / "phase3_pilot_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
