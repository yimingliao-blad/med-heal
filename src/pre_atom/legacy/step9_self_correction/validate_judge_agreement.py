#!/usr/bin/env python3
"""Validate GPT-4o binary judge agreement with human reviewers on step8 BioMistral zeroshot.

Reproduces the inter-rater agreement table (output/final/tables/inter_rater_agreement.tex)
using step8 BioMistral zeroshot answers (which use "medical expert" system prompt)
instead of step2 answers (which used "helpful assistant" system prompt).

Usage:
    python validate_judge_agreement.py
"""

import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
HUMAN_EVAL_CSV = PROJECT_ROOT / "datasets" / "external" / "all_users_openended_BioMistral-7B_latest.csv"
STEP8_DIR = PROJECT_ROOT / "output" / "step8" / "biomistral-7b"
OUTPUT_DIR = PROJECT_ROOT / "output" / "final" / "tables"


def load_human_labels():
    """Load human evaluations, binarize: quality=5 -> 1 (correct), quality=1 -> 0 (incorrect)."""
    df = pd.read_csv(HUMAN_EVAL_CSV)
    # Binarize
    df["human_binary"] = (df["Answer Quality"] == 5).astype(int)
    return df


def load_step8_gpt4o_labels():
    """Load GPT-4o binary labels from step8 BioMistral zeroshot across all folds."""
    all_rows = []
    for fold in range(5):
        fpath = STEP8_DIR / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if fpath.exists():
            fold_df = pd.read_csv(fpath)
            all_rows.append(fold_df)
    if not all_rows:
        raise FileNotFoundError("No step8 BioMistral zeroshot evaluated_binary files found")
    combined = pd.concat(all_rows, ignore_index=True)
    return combined


def compute_agreement(labels_a, labels_b):
    """Compute agreement metrics between two binary label arrays."""
    assert len(labels_a) == len(labels_b), f"Length mismatch: {len(labels_a)} vs {len(labels_b)}"
    n = len(labels_a)
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    pct = agree / n * 100 if n > 0 else 0
    kappa = cohen_kappa_score(labels_a, labels_b) if n > 1 else 0

    # Confusion matrix
    tp = sum(a == 1 and b == 1 for a, b in zip(labels_a, labels_b))
    tn = sum(a == 0 and b == 0 for a, b in zip(labels_a, labels_b))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels_a, labels_b))  # a=wrong, b=correct
    fn = sum(a == 1 and b == 0 for a, b in zip(labels_a, labels_b))  # a=correct, b=wrong

    return {
        "n": n,
        "agreement": pct,
        "kappa": kappa,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def kappa_interpretation(k):
    if k < 0.20:
        return "Poor"
    elif k < 0.40:
        return "Fair"
    elif k < 0.60:
        return "Mod."
    elif k < 0.80:
        return "Subs."
    else:
        return "A.Perf."


def main():
    # --- Load data ---
    human_df = load_human_labels()
    gpt4o_df = load_step8_gpt4o_labels()

    # Build GPT-4o lookup: patient_id -> binary_correct
    gpt4o_lookup = {}
    for _, row in gpt4o_df.iterrows():
        pid = int(row["patient_id"])
        gpt4o_lookup[pid] = int(row["binary_correct"])

    print(f"Human evaluations: {len(human_df)} rows, {human_df['Patient ID'].nunique()} unique questions")
    print(f"GPT-4o step8 labels: {len(gpt4o_df)} questions")
    print()

    # --- Map reviewer names to letters ---
    # Main reviewers: Reviewer A, Reviewer B, Reviewer C
    REVIEWER_MAP = {
        "Reviewer A": "A",
        "Reviewer B": "B",
        "Reviewer C": "C",
    }

    # Filter to main reviewers only
    main_df = human_df[human_df["User Name"].isin(REVIEWER_MAP.keys())].copy()
    main_df["reviewer"] = main_df["User Name"].map(REVIEWER_MAP)
    print(f"Main reviewer evaluations: {len(main_df)}")

    # --- Build per-reviewer label dictionaries ---
    reviewer_labels = defaultdict(dict)  # reviewer -> {patient_id: binary}
    for _, row in main_df.iterrows():
        pid = int(row["Patient ID"])
        rev = row["reviewer"]
        reviewer_labels[rev][pid] = int(row["human_binary"])

    for rev in ["A", "B", "C"]:
        labels = reviewer_labels[rev]
        n_correct = sum(v for v in labels.values())
        n_wrong = sum(1 - v for v in labels.values())
        print(f"  Reviewer {rev}: {len(labels)} evals ({n_correct} correct, {n_wrong} wrong)")
    print()

    # --- Compute all agreement pairs ---
    results = []

    # 1. Human inter-rater agreement
    print("=" * 60)
    print("HUMAN INTER-RATER AGREEMENT")
    print("=" * 60)
    for r1, r2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        # Find overlapping patient_ids
        common_pids = sorted(set(reviewer_labels[r1].keys()) & set(reviewer_labels[r2].keys()))
        if not common_pids:
            print(f"  {r1} vs {r2}: No overlap")
            continue
        l1 = [reviewer_labels[r1][pid] for pid in common_pids]
        l2 = [reviewer_labels[r2][pid] for pid in common_pids]
        metrics = compute_agreement(l1, l2)
        interp = kappa_interpretation(metrics["kappa"])
        print(f"  {r1} vs {r2}: N={metrics['n']}, Agr={metrics['agreement']:.1f}%, "
              f"κ={metrics['kappa']:.2f} ({interp})")
        results.append(("human", f"{r1} vs {r2}", metrics, interp))

    # 2. Individual reviewer vs GPT-4o (step8)
    print()
    print("=" * 60)
    print("INDIVIDUAL REVIEWER vs GPT-4o (STEP8 PROMPT)")
    print("=" * 60)
    for rev in ["A", "B", "C"]:
        common_pids = sorted(set(reviewer_labels[rev].keys()) & set(gpt4o_lookup.keys()))
        if not common_pids:
            print(f"  {rev} vs GPT-4o: No overlap")
            continue
        l_human = [reviewer_labels[rev][pid] for pid in common_pids]
        l_gpt = [gpt4o_lookup[pid] for pid in common_pids]
        metrics = compute_agreement(l_human, l_gpt)
        interp = kappa_interpretation(metrics["kappa"])
        print(f"  {rev} vs GPT-4o: N={metrics['n']}, Agr={metrics['agreement']:.1f}%, "
              f"κ={metrics['kappa']:.2f} ({interp}), FN={metrics['fn']}, FP={metrics['fp']}")
        results.append(("individual", f"{rev} vs GPT-4o", metrics, interp))

    # 3. Gold standard (A ∩ B agreement) vs GPT-4o
    print()
    print("=" * 60)
    print("GOLD STANDARD (A ∩ B AGREEMENT) vs GPT-4o (STEP8 PROMPT)")
    print("=" * 60)
    common_ab = sorted(set(reviewer_labels["A"].keys()) & set(reviewer_labels["B"].keys()))
    # Gold standard = subset where A and B agree
    gold_pids = [pid for pid in common_ab
                 if reviewer_labels["A"][pid] == reviewer_labels["B"][pid]]
    gold_with_gpt = [pid for pid in gold_pids if pid in gpt4o_lookup]

    gold_human = [reviewer_labels["A"][pid] for pid in gold_with_gpt]  # A=B, so either is fine
    gold_gpt = [gpt4o_lookup[pid] for pid in gold_with_gpt]
    metrics = compute_agreement(gold_human, gold_gpt)
    interp = kappa_interpretation(metrics["kappa"])
    print(f"  A ∩ B vs GPT-4o: N={metrics['n']}, Agr={metrics['agreement']:.1f}%, "
          f"κ={metrics['kappa']:.2f} ({interp})")
    print(f"  False Neg (human=correct, GPT=wrong): {metrics['fn']}")
    print(f"  False Pos (human=wrong, GPT=correct): {metrics['fp']}")
    results.append(("gold", "A ∩ B vs GPT-4o", metrics, interp))

    # --- Print comparison with original table ---
    print()
    print("=" * 60)
    print("COMPARISON: ORIGINAL (step2) vs CURRENT (step8)")
    print("=" * 60)
    original = {
        "A vs B": (143, 78.3, 0.44, "Mod."),
        "A vs C": (57, 66.7, 0.23, "Fair"),
        "B vs C": (72, 63.9, 0.25, "Fair"),
        "A vs GPT-4o": (328, 86.6, 0.67, "Subs."),
        "B vs GPT-4o": (300, 81.0, 0.53, "Mod."),
        "C vs GPT-4o": (72, 65.3, 0.28, "Fair"),
        "A ∩ B vs GPT-4o": (112, 92.0, 0.75, "Subs."),
    }
    print(f"{'Pair':<20} {'Orig N':>7} {'Orig Agr':>9} {'Orig κ':>7} | "
          f"{'Step8 N':>8} {'Step8 Agr':>10} {'Step8 κ':>8} {'Δκ':>6}")
    print("-" * 90)
    for _, pair_name, m, interp in results:
        if pair_name in original:
            o_n, o_agr, o_kappa, o_interp = original[pair_name]
            dk = m["kappa"] - o_kappa
            print(f"{pair_name:<20} {o_n:>7} {o_agr:>8.1f}% {o_kappa:>7.2f} | "
                  f"{m['n']:>8} {m['agreement']:>9.1f}% {m['kappa']:>8.2f} {dk:>+6.2f}")
        else:
            print(f"{pair_name:<20} {'N/A':>7} {'N/A':>9} {'N/A':>7} | "
                  f"{m['n']:>8} {m['agreement']:>9.1f}% {m['kappa']:>8.2f}")

    # --- Generate LaTeX table ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tex_path = OUTPUT_DIR / "inter_rater_agreement_step8.tex"

    lines = [
        r"\begin{table}[ht]",
        r"\caption{Inter-rater agreement for BioMistral-7B open-ended evaluation (step8 prompt).}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Pair} & \textbf{N} & \textbf{Agr.} & \textbf{$\kappa$} & \textbf{Level} \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Human inter-rater agreement}} \\",
    ]

    for cat, pair_name, m, interp in results:
        if cat == "human":
            lines.append(
                f"\\quad {pair_name:<20} & {m['n']} & {m['agreement']:.1f}\\% "
                f"& {m['kappa']:.2f} & {interp} \\\\"
            )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Individual reviewer vs GPT-4o (step8 prompt)}} \\")

    for cat, pair_name, m, interp in results:
        if cat == "individual":
            lines.append(
                f"\\quad {pair_name:<20} & {m['n']} & {m['agreement']:.1f}\\% "
                f"& {m['kappa']:.2f} & {interp} \\\\"
            )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Gold standard vs GPT-4o (step8 prompt)}} \\")

    for cat, pair_name, m, interp in results:
        if cat == "gold":
            lines.append(
                f"\\quad A $\\cap$ B vs GPT-4o & {m['n']} & {m['agreement']:.1f}\\% "
                f"& {m['kappa']:.2f} & {interp} \\\\"
            )
            lines.append(
                f"\\quad \\footnotesize{{False Neg / False Pos}} & "
                f"& \\footnotesize{{{m['fn']} / {m['fp']}}} & & \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:openended_agreement_step8}",
        r"\end{table}",
    ])

    tex_content = "\n".join(lines) + "\n"
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"\nLaTeX table written to: {tex_path}")

    # --- Additional analysis: how many questions changed correctness between step2 and step8? ---
    print()
    print("=" * 60)
    print("DETAIL: Gold standard disagreement cases")
    print("=" * 60)
    for pid in gold_with_gpt:
        h = reviewer_labels["A"][pid]
        g = gpt4o_lookup[pid]
        if h != g:
            # Find the question
            row = gpt4o_df[gpt4o_df["patient_id"] == pid].iloc[0]
            h_rows = main_df[main_df["Patient ID"] == pid]
            reasons = h_rows["Reasoning"].values
            label_str = "CORRECT" if h == 1 else "WRONG"
            gpt_str = "CORRECT" if g == 1 else "WRONG"
            print(f"\n  Patient {pid}: Human={label_str}, GPT-4o(step8)={gpt_str}")
            print(f"  Question: {row['question'][:120]}...")
            print(f"  Model answer: {str(row['model_answer'])[:150]}...")
            for r in reasons:
                if pd.notna(r) and str(r).strip():
                    print(f"  Human reasoning: {str(r)[:200]}")


if __name__ == "__main__":
    main()
