#!/usr/bin/env python3
"""
Phase 1: Analyze Pilot 12 Results — RA-ICL Retriever Comparison

Produces:
  1. Accuracy table for all conditions (7 retrieval + 2 controls)
  2. Retrieval similarity score analysis
  3. Per-question-type breakdown
  4. Qualitative retrieval analysis
  5. Recommendation for Phase 2

Usage:
    python analyze_pilot12.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "pilot" / "fold_0"
RESULTS_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "results"
PILOT7_DIR = PROJECT_ROOT / "output" / "pilot_7_fullscale"

RETRIEVAL_METHODS = [
    "bm25_pos_k1", "gtr_pos_k1", "kate_pos_k1",
    "gtr_pos_k2", "gtr_pos_k3",
    "gtr_type_pos_k1", "gtr_guideline_pos_k1",
    "gtr_note_pos_k1", "gtr_type_note_pos_k1",
    "gtr_note_context_pos_k1", "gtr_note_fullctx_pos_k1",
]

CONTROL_METHODS = {
    "type_pos_k1": "type_positive",
    "random_pos_k1": "random_positive",
}


def load_pilot7_controls(n_samples=50):
    """Load control results from Pilot 7 fold 0 (first n_samples)."""
    controls = {}
    for control_name, pilot7_name in CONTROL_METHODS.items():
        eval_file = PILOT7_DIR / "fold_0" / f"{pilot7_name}_evaluated.csv"
        if eval_file.exists():
            df = pd.read_csv(eval_file)
            df = df.head(n_samples)
            correct = (df["openended_correct"] == "yes").sum()
            controls[control_name] = {
                "correct": correct,
                "total": len(df),
                "accuracy": correct / len(df) * 100 if len(df) > 0 else 0,
                "data": df,
            }
    return controls


def load_retrieval_results():
    """Load all retrieval method results."""
    results = {}
    for method in RETRIEVAL_METHODS:
        eval_file = OUTPUT_DIR / f"{method}_evaluated.csv"
        if eval_file.exists():
            df = pd.read_csv(eval_file)
            correct = (df["openended_correct"] == "yes").sum()
            results[method] = {
                "correct": correct,
                "total": len(df),
                "accuracy": correct / len(df) * 100 if len(df) > 0 else 0,
                "data": df,
            }
    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retrieval_results = load_retrieval_results()
    if not retrieval_results:
        print("No evaluated results found. Run evaluate_pilot12.py first.")
        return

    n_samples = list(retrieval_results.values())[0]["total"]
    controls = load_pilot7_controls(n_samples)

    # Combine all results
    all_results = {}
    all_results.update(controls)
    all_results.update(retrieval_results)

    # ================================================================
    # 1. OVERALL ACCURACY TABLE
    # ================================================================
    print("=" * 80)
    print(f"PILOT 12: RA-ICL RETRIEVER COMPARISON (Fold 0, {n_samples} samples)")
    print("=" * 80)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    best_control = max(
        [(k, v) for k, v in controls.items()],
        key=lambda x: x[1]["accuracy"],
        default=(None, {"accuracy": 0}),
    )

    print(f"\n{'Rank':<6}{'Method':<35}{'Correct':>8}{'Accuracy':>10}{'vs Control':>12}  {'Type'}")
    print("-" * 80)
    for rank, (method, data) in enumerate(sorted_results, 1):
        is_control = method in CONTROL_METHODS
        diff = data["accuracy"] - best_control[1]["accuracy"] if best_control[0] else 0
        mtype = "CONTROL" if is_control else "RETRIEVAL"
        diff_str = f"{diff:+.1f}pp" if not is_control else "---"
        print(f"  {rank:<4} {method:<33} {data['correct']:>6}/{data['total']:<3} {data['accuracy']:>8.1f}%  {diff_str:>10}  ({mtype})")

    # ================================================================
    # 2. RETRIEVER COMPARISON (k=1 only)
    # ================================================================
    print(f"\n{'='*80}")
    print("RETRIEVER COMPARISON (k=1 only)")
    print(f"{'='*80}")

    k1_methods = ["bm25_pos_k1", "gtr_pos_k1", "kate_pos_k1", "gtr_type_pos_k1"]
    for method in k1_methods:
        if method in all_results:
            data = all_results[method]
            print(f"  {method:<30} {data['accuracy']:>6.1f}%")

    # ================================================================
    # 3. k ABLATION (GTR)
    # ================================================================
    print(f"\n{'='*80}")
    print("k ABLATION (GTR retrieval)")
    print(f"{'='*80}")

    k_methods = ["gtr_pos_k1", "gtr_pos_k2", "gtr_pos_k3"]
    for method in k_methods:
        if method in all_results:
            data = all_results[method]
            print(f"  {method:<30} {data['accuracy']:>6.1f}%")

    # ================================================================
    # 4. RETRIEVAL SIMILARITY SCORES
    # ================================================================
    print(f"\n{'='*80}")
    print("RETRIEVAL SIMILARITY SCORES")
    print(f"{'='*80}")

    for method in RETRIEVAL_METHODS:
        if method not in retrieval_results:
            continue
        df = retrieval_results[method]["data"]
        if "retrieval_sim_score" in df.columns:
            scores = pd.to_numeric(df["retrieval_sim_score"], errors="coerce").dropna()
            if len(scores) > 0:
                print(f"  {method:<30} mean={scores.mean():.4f}  std={scores.std():.4f}  min={scores.min():.4f}  max={scores.max():.4f}")

    # ================================================================
    # 5. PER-QUESTION-TYPE BREAKDOWN
    # ================================================================
    print(f"\n{'='*80}")
    print("PER-QUESTION-TYPE ACCURACY")
    print(f"{'='*80}")

    types = ["selectivity", "temporal", "documentation", "precision"]
    header = f"{'Method':<35}" + "".join(f"{t:>16}" for t in types)
    print(header)
    print("-" * len(header))

    for method, data in sorted_results:
        df = data["data"]
        row = f"  {method:<33}"
        for t in types:
            if "question_type" in df.columns:
                typed = df[df["question_type"] == t]
            else:
                typed = pd.DataFrame()
            if len(typed) > 0:
                acc = (typed["openended_correct"] == "yes").sum() / len(typed) * 100
                row += f"  {acc:>5.1f}% (n={len(typed):<3})"
            else:
                row += f"{'---':>16}"
        print(row)

    # ================================================================
    # 6. PER-QUESTION COMPARISON (retrieval vs control)
    # ================================================================
    print(f"\n{'='*80}")
    print("PER-QUESTION FLIP ANALYSIS (vs type_pos_k1 control)")
    print(f"{'='*80}")

    if "type_pos_k1" in controls:
        control_df = controls["type_pos_k1"]["data"]
        control_correct_ids = set(control_df[control_df["openended_correct"] == "yes"]["idx"])

        print(f"\n  {'Method':<35}{'Gained':>8}{'Lost':>8}{'Net':>8}")
        print(f"  {'-'*60}")

        for method in RETRIEVAL_METHODS:
            if method not in retrieval_results:
                continue
            df = retrieval_results[method]["data"]
            method_correct_ids = set(df[df["openended_correct"] == "yes"]["idx"])
            gained = len(method_correct_ids - control_correct_ids)
            lost = len(control_correct_ids - method_correct_ids)
            net = gained - lost
            print(f"    {method:<33} {gained:>6} {lost:>6} {net:>+6}")

    # ================================================================
    # 7. RECOMMENDATION
    # ================================================================
    print(f"\n{'='*80}")
    print("RECOMMENDATION FOR PHASE 2")
    print(f"{'='*80}")

    best_retrieval = max(
        [(k, v) for k, v in retrieval_results.items()],
        key=lambda x: x[1]["accuracy"],
    )
    best_control_acc = best_control[1]["accuracy"] if best_control[0] else 0
    diff = best_retrieval[1]["accuracy"] - best_control_acc

    print(f"\n  Best retrieval: {best_retrieval[0]} ({best_retrieval[1]['accuracy']:.1f}%)")
    print(f"  Best control:   {best_control[0]} ({best_control_acc:.1f}%)")
    print(f"  Difference:     {diff:+.1f}pp")

    if diff >= 2:
        print(f"\n  RECOMMENDATION: Promote {best_retrieval[0]} to fullscale (clear improvement)")
    elif diff >= -1:
        print(f"\n  RECOMMENDATION: Promote top retrievers to fullscale (marginal, needs scale validation)")
    else:
        print(f"\n  RECOMMENDATION: Retrieval does not improve over type-matching. Investigate quality.")

    # Suggest top methods for Phase 2
    sorted_retrieval = sorted(retrieval_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    print(f"\n  Suggested Phase 2 candidates (top 4):")
    for i, (method, data) in enumerate(sorted_retrieval[:4], 1):
        print(f"    {i}. {method} ({data['accuracy']:.1f}%)")

    # ================================================================
    # SAVE SUMMARY
    # ================================================================
    summary = {
        "pilot": "pilot_12_ra_icl",
        "phase": "phase_1",
        "fold": 0,
        "n_samples": int(n_samples),
        "results": {
            method: {
                "correct": int(data["correct"]),
                "total": int(data["total"]),
                "accuracy": round(float(data["accuracy"]), 2),
            }
            for method, data in all_results.items()
        },
        "ranking": [
            {"rank": i + 1, "method": method, "accuracy": round(data["accuracy"], 2)}
            for i, (method, data) in enumerate(sorted_results)
        ],
        "recommendation": {
            "best_retrieval": best_retrieval[0],
            "best_retrieval_acc": round(best_retrieval[1]["accuracy"], 2),
            "best_control": best_control[0],
            "best_control_acc": round(best_control_acc, 2),
            "diff_pp": round(diff, 2),
        },
    }

    summary_file = RESULTS_DIR / "pilot12_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")

    # Update experiment log
    log_file = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "experiment_log.json"
    with open(log_file) as f:
        log = json.load(f)
    log["phases"]["phase_1"]["status"] = "completed"
    log["phases"]["phase_1"]["completed"] = "2026-02-11"
    log["phases"]["phase_1"]["notes"] = f"Best retrieval: {best_retrieval[0]} ({best_retrieval[1]['accuracy']:.1f}%), best control: {best_control[0]} ({best_control_acc:.1f}%)"
    log["current_phase"] = "awaiting_review"
    log["status"] = "awaiting_review"
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
