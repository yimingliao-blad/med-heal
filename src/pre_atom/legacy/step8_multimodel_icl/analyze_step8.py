#!/usr/bin/env python3
"""
Step 8: Analysis — Cross-model ICL comparison + Negative k-sweep

Produces:
  1. Per-model table: 8 conditions × 5 folds + mean ± std + Δ + p-value
  2. Cross-model summary table
  3. Negative k-sweep table (4 models × k=0-5)
  4. LaTeX output

Usage:
    python analyze_step8.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[4]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUTPUT_DIR = RUN_ROOT / "output" / "step8"
RESULTS_DIR = OUTPUT_DIR / "results"

ALL_MODELS = [
    "biomistral-7b",
    "deepseek-r1-distill-llama-8b",
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "qwen3-8b",
]

STEP1_CONDITIONS = [
    "zeroshot",
    "gtr_note_pos_k1", "gtr_note_neg_k1", "gtr_note_posneg_k1",
    "cot_evidence", "cot_conclusion", "multiturn",
    "gtr_note_any_unlabeled_k1",
]

NEG_KSWEEP_CONDITIONS = [
    "zeroshot",
    "gtr_note_neg_k1", "gtr_note_neg_k2", "gtr_note_neg_k3",
    "gtr_note_neg_k4", "gtr_note_neg_k5",
]

KSWEEP_MODELS = [
    "deepseek-r1-distill-llama-8b",
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "qwen3-8b",
]

CONDITION_NAMES = {
    "zeroshot": "Zero-shot",
    "gtr_note_pos_k1": "Positive Retrieval",
    "gtr_note_neg_k1": "Negative Retrieval",
    "gtr_note_posneg_k1": "Contrastive Retrieval",
    "cot_evidence": "Evidence-first CoT",
    "cot_conclusion": "Conclusion-first CoT",
    "multiturn": "Multi-turn Few-shot",
    "gtr_note_any_unlabeled_k1": "Unlabeled Retrieval",
    "gtr_note_neg_k2": "Neg k=2",
    "gtr_note_neg_k3": "Neg k=3",
    "gtr_note_neg_k4": "Neg k=4",
    "gtr_note_neg_k5": "Neg k=5",
}

MODEL_SHORT = {
    "biomistral-7b": "BioMistral-7B",
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1-8B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "qwen3-8b": "Qwen3-8B",
}


def load_fold_accuracy(model, condition, fold_id):
    """Load accuracy for a single model/condition/fold."""
    eval_file = OUTPUT_DIR / model / f"fold_{fold_id}" / f"{condition}_evaluated.csv"
    if not eval_file.exists():
        return None
    df = pd.read_csv(eval_file)
    if "openended_correct" not in df.columns:
        return None
    return (df["openended_correct"] == "yes").mean() * 100


def compute_stats(model, condition):
    """Compute mean, std, and per-fold accuracies across 5 folds."""
    fold_accs = []
    for fold_id in range(5):
        acc = load_fold_accuracy(model, condition, fold_id)
        if acc is not None:
            fold_accs.append(acc)
    if not fold_accs:
        return None
    return {
        "fold_accs": fold_accs,
        "mean": np.mean(fold_accs),
        "std": np.std(fold_accs),
        "n_folds": len(fold_accs),
    }


def paired_ttest(model, cond1, cond2):
    """Paired t-test between two conditions across folds."""
    accs1 = []
    accs2 = []
    for fold_id in range(5):
        a1 = load_fold_accuracy(model, cond1, fold_id)
        a2 = load_fold_accuracy(model, cond2, fold_id)
        if a1 is not None and a2 is not None:
            accs1.append(a1)
            accs2.append(a2)
    if len(accs1) < 2:
        return None
    _, p_value = stats.ttest_rel(accs1, accs2)
    return p_value


def print_exp1():
    """Print Experiment 1: 5 models × 8 conditions."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 1: FULL-SCALE 5-FOLD CV — 5 MODELS × 8 CONDITIONS")
    print("=" * 100)

    all_results = {}

    for model in ALL_MODELS:
        print(f"\n{'─' * 80}")
        print(f"  {MODEL_SHORT[model]}")
        print(f"{'─' * 80}")

        header = f"  {'Condition':<25}"
        for fold_id in range(5):
            header += f" {'F'+str(fold_id):>7}"
        header += f" {'Mean':>7} {'±Std':>6} {'Δ':>7} {'p':>7}"
        print(header)
        print("  " + "-" * 90)

        model_results = {}
        zs_stats = compute_stats(model, "zeroshot")

        for condition in STEP1_CONDITIONS:
            s = compute_stats(model, condition)
            if not s:
                row = f"  {CONDITION_NAMES.get(condition, condition):<25}"
                row += "  — (no data)"
                print(row)
                continue

            model_results[condition] = s
            row = f"  {CONDITION_NAMES.get(condition, condition):<25}"
            for acc in s["fold_accs"]:
                row += f" {acc:>6.1f}%"
            # Pad if fewer than 5 folds
            for _ in range(5 - s["n_folds"]):
                row += "       —"
            row += f" {s['mean']:>6.1f}% {s['std']:>5.2f}"

            if condition != "zeroshot" and zs_stats:
                delta = s["mean"] - zs_stats["mean"]
                sign = "+" if delta > 0 else ""
                row += f" {sign}{delta:>+5.1f}"
                p = paired_ttest(model, "zeroshot", condition)
                if p is not None:
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    row += f" {p:>6.3f}{sig}"
            else:
                row += "     ---"

            print(row)

        all_results[model] = model_results

    return all_results


def print_exp2():
    """Print Experiment 2: Negative k-sweep."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: NEGATIVE K-SWEEP — 4 MODELS × k=0-5")
    print("=" * 100)

    all_results = {}

    for model in KSWEEP_MODELS:
        print(f"\n{'─' * 60}")
        print(f"  {MODEL_SHORT[model]}")
        print(f"{'─' * 60}")

        header = f"  {'k':<5} {'Condition':<25} {'Mean':>7} {'±Std':>6} {'Δ':>7}"
        print(header)
        print("  " + "-" * 55)

        model_results = {}
        zs_stats = compute_stats(model, "zeroshot")

        for k, condition in enumerate(NEG_KSWEEP_CONDITIONS):
            s = compute_stats(model, condition)
            if not s:
                print(f"  {k:<5} {CONDITION_NAMES.get(condition, condition):<25}  — (no data)")
                continue

            model_results[condition] = s
            row = f"  {k:<5} {CONDITION_NAMES.get(condition, condition):<25}"
            row += f" {s['mean']:>6.1f}% {s['std']:>5.2f}"

            if condition != "zeroshot" and zs_stats:
                delta = s["mean"] - zs_stats["mean"]
                sign = "+" if delta > 0 else ""
                row += f" {sign}{delta:>+5.1f}"

            print(row)

        all_results[model] = model_results

    return all_results


def print_cross_model_summary(all_results):
    """Print cross-model comparison table."""
    print("\n" + "=" * 100)
    print("CROSS-MODEL SUMMARY")
    print("=" * 100)

    header = f"  {'Condition':<25}"
    for model in ALL_MODELS:
        header += f" {MODEL_SHORT[model]:>16}"
    print(header)
    print("  " + "-" * (25 + 17 * len(ALL_MODELS)))

    for condition in STEP1_CONDITIONS:
        row = f"  {CONDITION_NAMES.get(condition, condition):<25}"
        for model in ALL_MODELS:
            mr = all_results.get(model, {}).get(condition)
            if mr:
                row += f" {mr['mean']:>6.1f}% ±{mr['std']:<4.1f} "
            else:
                row += f" {'—':>16}"
        print(row)


def generate_latex(all_results, ksweep_results):
    """Generate LaTeX tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latex_file = RESULTS_DIR / "latex_tables.tex"

    lines = []

    # Table 1: Cross-model comparison (Experiment 1)
    lines.append("% " + "=" * 75)
    lines.append("% TABLE 1: Cross-Model Performance (5-Fold CV)")
    lines.append("% " + "=" * 75)
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy (\\%) of five models across eight ICL conditions on EHRNoteQA "
                  "(full 5-fold CV, 962 questions). Values are mean $\\pm$ std across folds. "
                  "$\\Delta$ denotes the change relative to zero-shot. "
                  "\\textbf{Bold} = best per model; $^*$ = overall best.}")
    lines.append("\\label{tab:cross-model-fullscale}")

    # Determine which models have data
    active_models = [m for m in ALL_MODELS if m in all_results and all_results[m]]
    n_models = len(active_models)
    col_spec = "l" + " cc" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header = ""
    for i, model in enumerate(active_models):
        header += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{MODEL_SHORT[model]}}}}}"
    lines.append(header + " \\\\")

    # Cmidrules
    cmidrules = ""
    for i, _ in enumerate(active_models):
        col_start = 2 + i * 2
        col_end = col_start + 1
        cmidrules += f" \\cmidrule(lr){{{col_start}-{col_end}}}"
    lines.append(cmidrules)

    lines.append("\\textbf{Method}" + " & Acc & $\\Delta$" * n_models + " \\\\")
    lines.append("\\midrule")

    # Find best per model
    best_per_model = {}
    overall_best = 0
    for model in active_models:
        best_acc = 0
        best_cond = None
        for cond in STEP1_CONDITIONS:
            mr = all_results.get(model, {}).get(cond)
            if mr and mr["mean"] > best_acc:
                best_acc = mr["mean"]
                best_cond = cond
        best_per_model[model] = (best_cond, best_acc)
        if best_acc > overall_best:
            overall_best = best_acc

    # Data rows
    for ci, condition in enumerate(STEP1_CONDITIONS):
        name = CONDITION_NAMES.get(condition, condition)
        row = name

        for model in active_models:
            mr = all_results.get(model, {}).get(condition)
            if not mr:
                row += " & --- & ---"
                continue

            acc_str = f"{mr['mean']:.1f}"
            is_best = best_per_model[model][0] == condition
            is_overall = abs(mr["mean"] - overall_best) < 0.01

            if is_best:
                acc_str = f"\\textbf{{{acc_str}}}"
            if is_overall:
                acc_str += "$^*$"

            if condition == "zeroshot":
                delta_str = "---"
            else:
                zs = all_results.get(model, {}).get("zeroshot")
                if zs:
                    delta = mr["mean"] - zs["mean"]
                    sign = "+" if delta >= 0 else "$-$"
                    delta_str = f"\\small{{{sign}{abs(delta):.1f}}}"
                else:
                    delta_str = "---"

            row += f" & {acc_str} & {delta_str}"

        row += " \\\\"

        # Add spacing between groups
        if condition in ("zeroshot", "gtr_note_posneg_k1", "cot_conclusion"):
            row += "\n\\addlinespace"

        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")

    # Table 2: Negative k-sweep
    lines.append("% " + "=" * 75)
    lines.append("% TABLE 2: Negative K-Sweep")
    lines.append("% " + "=" * 75)
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy (\\%) for negative k-shot retrieval (k=0--5) across four models "
                  "(5-fold CV). k=0 is the zero-shot baseline. $\\Delta$ is relative to k=0.}")
    lines.append("\\label{tab:neg-ksweep}")

    active_ksweep = [m for m in KSWEEP_MODELS if m in ksweep_results and ksweep_results[m]]
    n_km = len(active_ksweep)
    col_spec = "c" + " cc" * n_km
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = ""
    for model in active_ksweep:
        header += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{MODEL_SHORT[model]}}}}}"
    lines.append(header + " \\\\")

    cmidrules = ""
    for i, _ in enumerate(active_ksweep):
        col_start = 2 + i * 2
        col_end = col_start + 1
        cmidrules += f" \\cmidrule(lr){{{col_start}-{col_end}}}"
    lines.append(cmidrules)

    lines.append("\\textbf{k}" + " & Acc & $\\Delta$" * n_km + " \\\\")
    lines.append("\\midrule")

    for k, condition in enumerate(NEG_KSWEEP_CONDITIONS):
        row = f"{k}"
        for model in active_ksweep:
            mr = ksweep_results.get(model, {}).get(condition)
            if not mr:
                row += " & --- & ---"
                continue

            acc_str = f"{mr['mean']:.1f}"

            if condition == "zeroshot":
                delta_str = "---"
            else:
                zs = ksweep_results.get(model, {}).get("zeroshot")
                if zs:
                    delta = mr["mean"] - zs["mean"]
                    sign = "+" if delta >= 0 else "$-$"
                    delta_str = f"\\small{{{sign}{abs(delta):.1f}}}"
                else:
                    delta_str = "---"

            row += f" & {acc_str} & {delta_str}"

        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(latex_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLaTeX tables saved to: {latex_file}")


def save_json_summaries(all_results, ksweep_results):
    """Save JSON summaries."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Experiment 1
    exp1 = {}
    for model in ALL_MODELS:
        exp1[model] = {}
        for condition in STEP1_CONDITIONS:
            mr = all_results.get(model, {}).get(condition)
            if mr:
                exp1[model][condition] = {
                    "mean": round(mr["mean"], 2),
                    "std": round(mr["std"], 2),
                    "fold_accs": [round(a, 2) for a in mr["fold_accs"]],
                    "n_folds": mr["n_folds"],
                }

    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(exp1, f, indent=2)

    # Experiment 2
    exp2 = {}
    for model in KSWEEP_MODELS:
        exp2[model] = {}
        for condition in NEG_KSWEEP_CONDITIONS:
            mr = ksweep_results.get(model, {}).get(condition)
            if mr:
                exp2[model][condition] = {
                    "mean": round(mr["mean"], 2),
                    "std": round(mr["std"], 2),
                    "fold_accs": [round(a, 2) for a in mr["fold_accs"]],
                    "n_folds": mr["n_folds"],
                }

    with open(RESULTS_DIR / "exp2_neg_ksweep_summary.json", "w") as f:
        json.dump(exp2, f, indent=2)

    print(f"JSON summaries saved to: {RESULTS_DIR}")


def main():
    print("Step 8: Full-Scale Multi-Model ICL Analysis")
    print("=" * 100)

    all_results = print_exp1()
    ksweep_results = print_exp2()
    print_cross_model_summary(all_results)
    generate_latex(all_results, ksweep_results)
    save_json_summaries(all_results, ksweep_results)


if __name__ == "__main__":
    main()
