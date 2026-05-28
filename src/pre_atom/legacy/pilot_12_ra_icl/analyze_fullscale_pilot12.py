#!/usr/bin/env python3
"""
Phase 2: Analyze Fullscale Pilot 12 Results — Note-Retrieval ICL (5-Fold CV)

Compares 4 note-retrieval conditions against existing baselines:
  Baselines (reused):
    - zeroshot: 77.13% +/- 1.51%
    - type_positive (Pilot 7): 78.80% +/- 1.42%
    - type_negative (Fullscale-2): 75.68% +/- 2.11%
    - type_both (Fullscale-2): 78.07% +/- 2.52%

  New conditions:
    - gtr_note_pos_k1: Note-retrieved positive example
    - gtr_note_fullctx_pos_k1: Note-retrieved positive + full example notes
    - gtr_note_neg_k1: Note-retrieved negative example
    - gtr_note_posneg_k1: Note-retrieved contrastive (neg + pos)

Usage:
    python analyze_fullscale_pilot12.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "fullscale"
RESULTS_DIR = OUTPUT_DIR / "results"

NEW_METHODS = [
    "gtr_note_pos_k1", "gtr_note_fullctx_pos_k1",
    "gtr_note_neg_k1", "gtr_note_posneg_k1",
]

# Existing fullscale baselines
FULLSCALE_BASELINES = {
    "zeroshot": {"mean": 77.13, "std": 1.51, "per_fold": [74.61, 78.24, 78.65, 76.04, 78.13]},
    "type_positive": {"mean": 78.80, "std": 1.42, "per_fold": [78.24, 80.31, 79.69, 77.60, 78.13]},
    "type_negative": {"mean": 75.68, "std": 2.11, "per_fold": [75.13, 75.13, 79.69, 73.44, 75.0]},
    "type_both": {"mean": 78.07, "std": 2.52, "per_fold": [78.24, 80.83, 80.73, 73.96, 76.56]},
    "guideline_pos_ann": {"mean": 79.21, "std": 1.72, "per_fold": [79.27, 80.83, 81.25, 76.56, 77.14]},
}

DISPLAY_NAMES = {
    "zeroshot": "Zero-shot (baseline)",
    "type_positive": "Type-matched Pos",
    "type_negative": "Type-matched Neg",
    "type_both": "Type-matched Contrastive",
    "guideline_pos_ann": "Guideline + Pos (best)",
    "gtr_note_pos_k1": "Note-Ret Pos",
    "gtr_note_fullctx_pos_k1": "Note-Ret Pos + Context",
    "gtr_note_neg_k1": "Note-Ret Neg",
    "gtr_note_posneg_k1": "Note-Ret Contrastive",
}

# Question type classifier
TEMPORAL_KEYWORDS = [
    "before admission", "prior to", "at discharge", "during the first",
    "during the second", "between", "pre-admission", "post-admission",
    "after the", "before the surgery", "before the procedure",
    "upon discharge", "upon admission", "at the time of",
]
DOCUMENTATION_KEYWORDS = [
    "as noted", "as stated", "as documented", "according to",
    "exact cause", "what was the reason", "what reason was stated",
]
PRECISION_KEYWORDS = [
    "dose", "dosage", "mg", "what results", "test results",
    "what level", "what value", "what was prescribed",
]


def classify_question(question):
    q = question.lower()
    for kw in TEMPORAL_KEYWORDS:
        if kw in q:
            return "temporal"
    for kw in DOCUMENTATION_KEYWORDS:
        if kw in q:
            return "documentation"
    for kw in PRECISION_KEYWORDS:
        if kw in q:
            return "precision"
    return "selectivity"


def load_new_results():
    """Load all new fullscale results across folds."""
    results = {}
    for method in NEW_METHODS:
        fold_accs = []
        fold_data = []
        for fold_id in range(5):
            ef = OUTPUT_DIR / f"fold_{fold_id}" / f"{method}_evaluated.csv"
            if not ef.exists():
                continue
            df = pd.read_csv(ef)
            correct = (df["openended_correct"] == "yes").sum()
            acc = correct / len(df) * 100
            fold_accs.append(acc)
            fold_data.append(df)

        if fold_accs:
            results[method] = {
                "mean": np.mean(fold_accs),
                "std": np.std(fold_accs),
                "per_fold": fold_accs,
                "data": fold_data,
            }
    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    new_results = load_new_results()
    if not new_results:
        print("No evaluated results found. Run evaluate_fullscale_pilot12.py first.")
        return

    # Combine all results
    all_methods = {}
    all_methods.update(FULLSCALE_BASELINES)
    all_methods.update(new_results)

    zs_mean = FULLSCALE_BASELINES["zeroshot"]["mean"]
    zs_folds = FULLSCALE_BASELINES["zeroshot"]["per_fold"]

    # ================================================================
    # 1. OVERALL RANKING
    # ================================================================
    print("=" * 80)
    print("FULLSCALE PILOT 12: NOTE-RETRIEVAL ICL (5-FOLD CV, 962 samples)")
    print("=" * 80)

    sorted_methods = sorted(all_methods.items(), key=lambda x: x[1]["mean"], reverse=True)

    print(f"\n{'Rank':<6}{'Method':<30}{'Mean':>8}{'Std':>8}{'vs ZS':>8}  {'Source'}")
    print("-" * 80)
    for rank, (method, data) in enumerate(sorted_methods, 1):
        source = "BASELINE" if method in FULLSCALE_BASELINES else "NEW"
        diff = data["mean"] - zs_mean
        name = DISPLAY_NAMES.get(method, method)
        print(f"  {rank:<4} {name:<28} {data['mean']:>6.2f}% {data['std']:>6.2f}%  {diff:>+6.2f}pp  ({source})")

    # ================================================================
    # 2. NOTE RETRIEVAL vs TYPE MATCHING
    # ================================================================
    print(f"\n{'='*80}")
    print("Q1: DOES NOTE-RETRIEVAL BEAT TYPE-MATCHING?")
    print(f"{'='*80}")

    pairs = [
        ("type_positive", "gtr_note_pos_k1", "Positive: type-match vs note-retrieval"),
        ("type_negative", "gtr_note_neg_k1", "Negative: type-match vs note-retrieval"),
        ("type_both", "gtr_note_posneg_k1", "Contrastive: type-match vs note-retrieval"),
    ]
    for base, enhanced, desc in pairs:
        if base in all_methods and enhanced in all_methods:
            b = all_methods[base]
            e = all_methods[enhanced]
            diff = e["mean"] - b["mean"]
            if len(b["per_fold"]) == 5 and len(e["per_fold"]) == 5:
                t_stat, p_val = stats.ttest_rel(e["per_fold"], b["per_fold"])
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            else:
                t_stat, p_val, sig = 0, 1, "N/A"
            print(f"\n  {desc}")
            print(f"    {DISPLAY_NAMES[base]:<28} {b['mean']:.2f}% +/- {b['std']:.2f}%")
            print(f"    {DISPLAY_NAMES[enhanced]:<28} {e['mean']:.2f}% +/- {e['std']:.2f}%")
            print(f"    Diff: {diff:+.2f}pp  t={t_stat:.3f}  p={p_val:.4f} ({sig})")

    # ================================================================
    # 3. NEGATIVE EXAMPLE HYPOTHESIS
    # ================================================================
    print(f"\n{'='*80}")
    print("Q2: DOES NOTE-RETRIEVAL MAKE NEGATIVES USEFUL?")
    print(f"{'='*80}")

    if "gtr_note_neg_k1" in new_results:
        neg = new_results["gtr_note_neg_k1"]
        print(f"\n  Note-Ret Neg:       {neg['mean']:.2f}% +/- {neg['std']:.2f}%")
        print(f"  Type-matched Neg:   {FULLSCALE_BASELINES['type_negative']['mean']:.2f}% +/- {FULLSCALE_BASELINES['type_negative']['std']:.2f}%")
        print(f"  Zeroshot:           {zs_mean:.2f}%")
        if neg["mean"] >= zs_mean:
            print(f"  => Note-retrieved negatives NO LONGER HURT (>= zeroshot)")
        else:
            print(f"  => Note-retrieved negatives STILL HURT (< zeroshot)")

    # ================================================================
    # 4. FULL CONTEXT EFFECT
    # ================================================================
    print(f"\n{'='*80}")
    print("Q3: DOES FULL CONTEXT HELP?")
    print(f"{'='*80}")

    if "gtr_note_pos_k1" in new_results and "gtr_note_fullctx_pos_k1" in new_results:
        p = new_results["gtr_note_pos_k1"]
        f = new_results["gtr_note_fullctx_pos_k1"]
        diff = f["mean"] - p["mean"]
        if len(p["per_fold"]) == 5 and len(f["per_fold"]) == 5:
            t_stat, p_val = stats.ttest_rel(f["per_fold"], p["per_fold"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        else:
            t_stat, p_val, sig = 0, 1, "N/A"
        print(f"\n  Note-Ret Pos:           {p['mean']:.2f}% +/- {p['std']:.2f}%")
        print(f"  Note-Ret Pos + Context: {f['mean']:.2f}% +/- {f['std']:.2f}%")
        print(f"  Diff: {diff:+.2f}pp  t={t_stat:.3f}  p={p_val:.4f} ({sig})")

    # ================================================================
    # 5. STATISTICAL TESTS (all vs zeroshot)
    # ================================================================
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISONS vs ZEROSHOT (paired t-test, 5 folds)")
    print(f"{'='*80}")

    comparisons = []
    print(f"\n  {'Method':<30}{'Diff':>8}{'t-stat':>10}{'p-value':>10}{'Sig':>6}")
    print(f"  {'-'*65}")

    test_order = ["type_positive", "type_negative", "type_both", "guideline_pos_ann"] + NEW_METHODS
    for method in test_order:
        data = all_methods.get(method)
        if not data or len(data["per_fold"]) != 5:
            continue
        t_stat, p_val = stats.ttest_rel(data["per_fold"], zs_folds)
        diff = data["mean"] - zs_mean
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        name = DISPLAY_NAMES.get(method, method)
        comparisons.append({
            "method": method, "diff": round(diff, 2),
            "t_stat": round(t_stat, 3), "p_value": round(p_val, 4), "sig": sig,
        })
        print(f"    {name:<28} {diff:>+6.2f}pp  {t_stat:>9.3f}  {p_val:>9.4f}  {sig}")

    # ================================================================
    # 6. PER-FOLD BREAKDOWN
    # ================================================================
    print(f"\n{'='*80}")
    print("PER-FOLD ACCURACY BREAKDOWN")
    print(f"{'='*80}")

    header = f"{'Method':<30}" + "".join(f"{'F'+str(i):>8}" for i in range(5)) + f"{'Mean':>9}{'Std':>7}"
    print(header)
    print("-" * len(header))

    display_order = ["zeroshot", "type_positive", "type_negative", "type_both", "guideline_pos_ann"] + NEW_METHODS
    for method in display_order:
        data = all_methods.get(method)
        if not data:
            continue
        name = DISPLAY_NAMES.get(method, method)
        row = f"  {name:<28}"
        for acc in data["per_fold"]:
            row += f"{acc:>7.1f}%"
        row += f"  {data['mean']:>6.2f}% {data['std']:>5.2f}%"
        print(row)

    # ================================================================
    # 7. PER-QUESTION-TYPE BREAKDOWN
    # ================================================================
    print(f"\n{'='*80}")
    print("PER-QUESTION-TYPE ACCURACY (across all folds)")
    print(f"{'='*80}")

    type_accs = {}
    for method in NEW_METHODS:
        data = all_methods.get(method)
        if not data or "data" not in data:
            continue
        all_df = pd.concat(data["data"])
        if "question_type" not in all_df.columns:
            all_df["question_type"] = all_df["question"].apply(classify_question)
        for qtype in ["selectivity", "temporal", "documentation", "precision"]:
            typed = all_df[all_df["question_type"] == qtype]
            if len(typed) > 0:
                acc = (typed["openended_correct"] == "yes").sum() / len(typed) * 100
                type_accs.setdefault(method, {})[qtype] = (acc, len(typed))

    if type_accs:
        types = ["selectivity", "temporal", "documentation", "precision"]
        header = f"{'Method':<30}" + "".join(f"{t:>16}" for t in types)
        print(header)
        print("-" * len(header))
        for method in NEW_METHODS:
            if method not in type_accs:
                continue
            name = DISPLAY_NAMES.get(method, method)
            row = f"  {name:<28}"
            for t in types:
                if t in type_accs[method]:
                    acc, n = type_accs[method][t]
                    row += f"  {acc:>5.1f}% (n={n:<3})"
                else:
                    row += f"{'---':>16}"
            print(row)

    # ================================================================
    # 8. RETRIEVAL SIMILARITY SCORES
    # ================================================================
    print(f"\n{'='*80}")
    print("RETRIEVAL SIMILARITY SCORES (across all folds)")
    print(f"{'='*80}")

    for method in NEW_METHODS:
        data = all_methods.get(method)
        if not data or "data" not in data:
            continue
        all_df = pd.concat(data["data"])
        if "retrieval_sim_score" not in all_df.columns:
            continue
        # Load from generated files (eval files may not have sim_score)
        scores = []
        for fold_id in range(5):
            gf = OUTPUT_DIR / f"fold_{fold_id}" / f"{method}_generated.csv"
            if gf.exists():
                gdf = pd.read_csv(gf)
                if "retrieval_sim_score" in gdf.columns:
                    scores.extend(gdf["retrieval_sim_score"].dropna().tolist())
        if scores:
            scores = np.array(scores)
            name = DISPLAY_NAMES.get(method, method)
            print(f"  {name:<28} mean={scores.mean():.4f}  std={scores.std():.4f}  min={scores.min():.4f}  max={scores.max():.4f}")

    # ================================================================
    # 9. KEY FINDINGS
    # ================================================================
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    best_new = max(new_results.items(), key=lambda x: x[1]["mean"])
    best_name = DISPLAY_NAMES.get(best_new[0], best_new[0])
    print(f"\n  Best note-retrieval method: {best_name} ({best_new[1]['mean']:.2f}% +/- {best_new[1]['std']:.2f}%)")
    print(f"  Best baseline: Guideline + Pos ({FULLSCALE_BASELINES['guideline_pos_ann']['mean']:.2f}%)")
    print(f"  Zeroshot: {zs_mean:.2f}%")

    # ================================================================
    # SAVE SUMMARY
    # ================================================================
    summary = {
        "all_results": {
            k: {"mean": round(float(v["mean"]), 2), "std": round(float(v["std"]), 2),
                "per_fold": [round(float(x), 2) for x in v["per_fold"]]}
            for k, v in all_methods.items()
        },
        "comparisons": comparisons,
        "question_type_accuracy": {
            method: {t: {"acc": round(float(acc), 2), "n": int(n)} for t, (acc, n) in types_dict.items()}
            for method, types_dict in type_accs.items()
        } if type_accs else {},
    }
    with open(RESULTS_DIR / "fullscale_pilot12_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {RESULTS_DIR / 'fullscale_pilot12_summary.json'}")


if __name__ == "__main__":
    main()
