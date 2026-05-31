#!/usr/bin/env python3
"""Phase 0 (revised): Blind critic test on BioMistral zeroshot answers.

Qwen3-think evaluates BioMistral answers WITHOUT ground truth (real critic scenario).
Compare against human gold standard (reviewers A∩B agree, N=112).

Usage:
    python test_critic_blind.py --port 8003
"""

import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
HUMAN_EVAL_CSV = PROJECT_ROOT / "datasets" / "external" / "all_users_openended_BioMistral-7B_latest.csv"
STEP2_CSV = PROJECT_ROOT / "output" / "ours_biomistral-7b_EHRNoteQA_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "step8" / "experiments" / "phase0_critic"


# Blind critic prompt — NO ground truth, only discharge notes + question + answer
BLIND_CRITIC_PROMPT = """You are a medical expert reviewing an AI model's answer to a clinical question about a patient's discharge summary.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

AI MODEL'S ANSWER:
{model_answer}

Based ONLY on the discharge summary above, evaluate whether the AI's answer is correct and complete.

Use the following evaluation criteria:

1. **Accuracy**: The answer must accurately reflect what is documented in the discharge summary. Any information that contradicts the notes is incorrect.

2. **Completeness**: The answer should include all critical details needed to fully address the question. If important information from the notes is omitted and this changes the understanding, mark as incorrect.

3. **No Fabrication**: The answer must not include information that is not supported by the discharge summary. Fabricated details (dates, medications, procedures not in the notes) make the answer incorrect.

4. **Specificity**: When the discharge summary contains specific values, names, or details, the answer should reflect them. Vague answers when specific information is available are likely incorrect.

5. **Close Enough vs Wrong**: An answer that captures the essence of the correct information is acceptable even if minor wording differs. But if differences lead to a different clinical interpretation, it is incorrect.

6. **Extra Information**: Additional correct information beyond what the question asks is acceptable and does not make an answer incorrect, as long as it does not introduce inaccuracies.

Provide your assessment, then end with:
VERDICT: 1 (correct) or VERDICT: 0 (incorrect)
If incorrect, also state: ERROR_TYPE: <one of: omission, hallucination, reasoning_failure, specificity, context_confusion, temporal_error>"""


def get_note_for_patient(patient_id, step2_df):
    """Get assembled discharge note from step2 CSV."""
    rows = step2_df[step2_df["patient_id"] == patient_id]
    if len(rows) == 0:
        return ""
    row = rows.iloc[0]
    parts = []
    for i in [1, 2, 3]:
        col = f"note_{i}"
        if col in row and pd.notna(row[col]):
            t = str(row[col]).strip()
            if t and t.lower() != "nan":
                parts.append(f"[Note {i}]\n{t}")
    return "\n\n".join(parts)


def evaluate_qwen3_blind(port, note, question, model_answer):
    """Ask Qwen3-think to evaluate without ground truth."""
    base_url = f"http://localhost:{port}/v1"

    user_content = BLIND_CRITIC_PROMPT.format(
        note=note, question=question, model_answer=model_answer,
    )

    system_msg = "You are a medical expert evaluating an AI model's answer to a clinical question."
    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        model_name = requests.get(f"{base_url}/models", timeout=5).json()["data"][0]["id"]
        resp = requests.post(
            f"{base_url}/completions",
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": 0.1,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return None, None, ""

        raw = resp.json()["choices"][0]["text"].strip()

        # Strip thinking tags
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"</think>", "", clean).strip()

        # Extract verdict
        verdict = None
        verdict_match = re.search(r"VERDICT:\s*([01])", clean)
        if verdict_match:
            verdict = int(verdict_match.group(1))
        elif "1" in clean[-30:] and "0" not in clean[-30:]:
            verdict = 1
        elif "0" in clean[-30:]:
            verdict = 0

        # Extract error type
        error_type = None
        error_match = re.search(r"ERROR_TYPE:\s*(\w+)", clean)
        if error_match:
            error_type = error_match.group(1).lower()

        return verdict, error_type, clean

    except Exception as e:
        return None, None, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n-samples", type=int, default=50, help="Number of gold standard samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Load human labels ---
    human_df = pd.read_csv(HUMAN_EVAL_CSV)
    human_df["human_binary"] = (human_df["Answer Quality"] == 5).astype(int)

    REVIEWER_MAP = {"Reviewer A": "A", "Reviewer B": "B"}
    main_df = human_df[human_df["User Name"].isin(REVIEWER_MAP.keys())].copy()
    main_df["reviewer"] = main_df["User Name"].map(REVIEWER_MAP)

    reviewer_labels = defaultdict(dict)
    for _, row in main_df.iterrows():
        pid = int(row["Patient ID"])
        rev = row["reviewer"]
        reviewer_labels[rev][pid] = int(row["human_binary"])

    # Gold standard: A and B agree
    common_ab = sorted(set(reviewer_labels["A"].keys()) & set(reviewer_labels["B"].keys()))
    gold_pids = [pid for pid in common_ab
                 if reviewer_labels["A"][pid] == reviewer_labels["B"][pid]]
    gold_correct = sum(reviewer_labels["A"][pid] for pid in gold_pids)
    gold_wrong = len(gold_pids) - gold_correct
    print(f"Gold standard (A∩B agree): {len(gold_pids)} questions ({gold_correct} correct, {gold_wrong} wrong)")

    # --- Load step2 BioMistral answers ---
    step2_df = pd.read_csv(STEP2_CSV)
    step2_lookup = {}
    for _, row in step2_df.iterrows():
        pid = int(row["patient_id"])
        step2_lookup[pid] = {
            "question": row["question"],
            "model_answer": str(row.get("openended_answer", "")),
        }

    gold_with_step2 = [pid for pid in gold_pids if pid in step2_lookup]
    print(f"Gold standard with step2 answers: {len(gold_with_step2)}")

    # Sample
    random.seed(args.seed)
    sample_pids = random.sample(gold_with_step2, min(args.n_samples, len(gold_with_step2)))
    sample_correct = sum(reviewer_labels["A"][pid] for pid in sample_pids)
    sample_wrong = len(sample_pids) - sample_correct
    print(f"Sampled {len(sample_pids)}: {sample_correct} human-correct, {sample_wrong} human-wrong")

    # Check vLLM
    try:
        resp = requests.get(f"http://localhost:{args.port}/v1/models", timeout=5)
        model_name = resp.json()["data"][0]["id"]
        print(f"Qwen3 vLLM: {model_name} on port {args.port}")
    except Exception:
        print(f"ERROR: vLLM not available on port {args.port}")
        return

    # --- Run blind evaluation ---
    print(f"\nRunning Qwen3-think blind critic on {len(sample_pids)} BioMistral answers...")
    print("(No ground truth given — critic must judge from discharge notes alone)\n")

    results = []
    for i, pid in enumerate(sample_pids):
        data = step2_lookup[pid]
        note = get_note_for_patient(pid, step2_df)

        verdict, error_type, reasoning = evaluate_qwen3_blind(
            args.port, note, data["question"], data["model_answer"],
        )

        human_label = reviewer_labels["A"][pid]
        marker = "✓" if verdict == human_label else ("✗" if verdict is not None else "?")

        results.append({
            "patient_id": pid,
            "human_label": human_label,
            "qwen3_verdict": verdict,
            "error_type": error_type,
            "reasoning": reasoning[:500],
        })

        if (i + 1) % 5 == 0 or verdict != human_label:
            h_str = "correct" if human_label == 1 else "wrong"
            v_str = str(verdict) if verdict is not None else "None"
            et_str = f" ({error_type})" if error_type else ""
            print(f"  [{i+1}/{len(sample_pids)}] Pt {pid}: Human={h_str}, Qwen3={v_str}{et_str} {marker}")

    # --- Compute metrics ---
    valid = [r for r in results if r["qwen3_verdict"] is not None]
    human_list = [r["human_label"] for r in valid]
    qwen3_list = [r["qwen3_verdict"] for r in valid]

    n = len(valid)
    agree = sum(h == q for h, q in zip(human_list, qwen3_list))
    pct = agree / n * 100 if n > 0 else 0
    kappa = cohen_kappa_score(human_list, qwen3_list) if n > 1 else 0

    fn = sum(h == 1 and q == 0 for h, q in zip(human_list, qwen3_list))
    fp = sum(h == 0 and q == 1 for h, q in zip(human_list, qwen3_list))
    null_count = len(results) - len(valid)

    print(f"\n{'='*60}")
    print("BLIND CRITIC RESULTS: Qwen3-think vs Human Gold Standard")
    print(f"{'='*60}")
    print(f"  N = {n} (null: {null_count})")
    print(f"  Agreement = {pct:.1f}%")
    print(f"  Cohen's κ = {kappa:.3f}")
    print(f"  FN (human=correct, Qwen3=wrong) = {fn}")
    print(f"  FP (human=wrong, Qwen3=correct) = {fp}")
    print()

    # Error type distribution for cases Qwen3 flagged as wrong
    flagged_wrong = [r for r in valid if r["qwen3_verdict"] == 0]
    if flagged_wrong:
        print(f"Error types identified (N={len(flagged_wrong)} flagged wrong):")
        error_counts = defaultdict(int)
        for r in flagged_wrong:
            et = r["error_type"] or "unspecified"
            error_counts[et] += 1
        for et, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            # How many of these were actually wrong (true positives)?
            tp = sum(1 for r in flagged_wrong if (r["error_type"] or "unspecified") == et and r["human_label"] == 0)
            print(f"  {et}: {count} ({tp} true positive, {count - tp} false positive)")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  Phase 0 (with GT):  Qwen3-think-binary vs GPT-4o: κ=0.592, Agr=79.6%")
    print(f"  This test (blind):  Qwen3-think vs Human gold:    κ={kappa:.3f}, Agr={pct:.1f}%")
    print(f"  Original table:     A∩B vs GPT-4o (step2):        κ=0.75,  Agr=92.0%")

    # Save details
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    details_df = pd.DataFrame(results)
    details_df.to_csv(OUTPUT_DIR / "blind_critic_details.csv", index=False)
    print(f"\nDetails saved to {OUTPUT_DIR / 'blind_critic_details.csv'}")

    # Save summary
    summary = {
        "test": "blind_critic_qwen3_think_vs_human_gold",
        "n_valid": n,
        "n_null": null_count,
        "agreement_pct": round(pct, 1),
        "kappa": round(kappa, 3),
        "fn": fn,
        "fp": fp,
        "sample_correct": sample_correct,
        "sample_wrong": sample_wrong,
    }
    with open(OUTPUT_DIR / "blind_critic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
