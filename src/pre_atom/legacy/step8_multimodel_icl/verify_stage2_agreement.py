#!/usr/bin/env python3
"""Verify GPT-4o binary judge agreement on step2 BioMistral answers (original "helpful assistant" prompt).

Pick 30 random gold standard questions, evaluate step2 BioMistral answers with
GPT-4o binary prompt, and check agreement with human labels.

This confirms whether the 92% agreement holds when we use the original prompt.
"""

import os
import random
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
HUMAN_EVAL_CSV = PROJECT_ROOT / "datasets" / "external" / "all_users_openended_BioMistral-7B_latest.csv"
STEP2_CSV = PROJECT_ROOT / "output" / "ours_biomistral-7b_EHRNoteQA_processed.csv"


def evaluate_one_binary(client, note, question, ground_truth, model_answer):
    """Evaluate with the same binary prompt used in stage1 and step8."""
    messages = [
        {
            "role": "system",
            "content": "You are a medical expert evaluating an AI model's answer to a clinical question.",
        },
        {
            "role": "user",
            "content": (
                f"DISCHARGE SUMMARY:\n{note}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
                f"MODEL'S ANSWER:\n{model_answer}\n\n"
                f"Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
                f"Respond with ONLY a single digit:\n"
                f"1 = Correct\n"
                f"0 = Incorrect"
            ),
        },
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=10,
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            if "1" in content and "0" not in content:
                return 1
            elif "0" in content:
                return 0
            else:
                return None
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error: {e}")
                return None
    return None


def main():
    # Load API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    client = OpenAI(api_key=api_key)

    # --- Load human labels ---
    human_df = pd.read_csv(HUMAN_EVAL_CSV)
    human_df["human_binary"] = (human_df["Answer Quality"] == 5).astype(int)

    REVIEWER_MAP = {
        "Reviewer A": "A",
        "Reviewer B": "B",
    }
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
    print(f"Gold standard (A∩B agree): {len(gold_pids)} questions")
    gold_correct = sum(reviewer_labels["A"][pid] for pid in gold_pids)
    gold_wrong = len(gold_pids) - gold_correct
    print(f"  Correct: {gold_correct}, Wrong: {gold_wrong}")

    # --- Load step2 BioMistral answers ---
    step2_df = pd.read_csv(STEP2_CSV)
    step2_lookup = {}
    for _, row in step2_df.iterrows():
        pid = int(row["patient_id"])
        # Assemble notes
        notes = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in row and pd.notna(row[col]):
                t = str(row[col]).strip()
                if t and t.lower() != "nan":
                    notes.append(f"[Note {i}]\n{t}")
        note_text = "\n\n".join(notes)

        # Ground truth: get the answer text from the choice columns
        answer_letter = str(row.get("answer", "")).strip()
        gt_col = f"choice_{answer_letter}" if answer_letter in "ABCDE" else None
        if gt_col and gt_col in row and pd.notna(row[gt_col]):
            ground_truth = f"{answer_letter}: {row[gt_col]}"
        else:
            ground_truth = str(row.get("answer", ""))

        step2_lookup[pid] = {
            "question": row["question"],
            "note": note_text,
            "ground_truth": ground_truth,
            "model_answer": str(row.get("openended_answer", "")),
        }

    # --- Pick 30 random gold standard questions ---
    gold_with_step2 = [pid for pid in gold_pids if pid in step2_lookup]
    print(f"Gold standard with step2 answers: {len(gold_with_step2)}")

    random.seed(42)
    sample_pids = random.sample(gold_with_step2, min(30, len(gold_with_step2)))
    print(f"Sampled {len(sample_pids)} questions for verification")

    # Show human label distribution in sample
    sample_correct = sum(reviewer_labels["A"][pid] for pid in sample_pids)
    sample_wrong = len(sample_pids) - sample_correct
    print(f"  Sample: {sample_correct} human-correct, {sample_wrong} human-wrong")
    print()

    # --- Run GPT-4o binary eval on step2 answers ---
    print("Running GPT-4o binary eval on step2 BioMistral answers...")
    gpt_labels = {}
    for i, pid in enumerate(sample_pids):
        data = step2_lookup[pid]
        result = evaluate_one_binary(
            client, data["note"], data["question"],
            data["ground_truth"], data["model_answer"]
        )
        gpt_labels[pid] = result
        human_label = reviewer_labels["A"][pid]
        agree = "✓" if result == human_label else "✗"
        print(f"  [{i+1}/{len(sample_pids)}] Patient {pid}: Human={human_label}, GPT-4o={result} {agree}")
        time.sleep(0.5)

    # --- Compute agreement ---
    valid_pids = [pid for pid in sample_pids if gpt_labels[pid] is not None]
    human_list = [reviewer_labels["A"][pid] for pid in valid_pids]
    gpt_list = [gpt_labels[pid] for pid in valid_pids]

    n = len(valid_pids)
    agree = sum(h == g for h, g in zip(human_list, gpt_list))
    pct = agree / n * 100 if n > 0 else 0
    kappa = cohen_kappa_score(human_list, gpt_list) if n > 1 else 0

    fn = sum(h == 1 and g == 0 for h, g in zip(human_list, gpt_list))
    fp = sum(h == 0 and g == 1 for h, g in zip(human_list, gpt_list))

    print()
    print("=" * 60)
    print("RESULTS: Step2 BioMistral (helpful assistant) + GPT-4o binary")
    print("=" * 60)
    print(f"  N = {n}")
    print(f"  Agreement = {pct:.1f}%")
    print(f"  Cohen's κ = {kappa:.2f}")
    print(f"  False Neg (human=correct, GPT=wrong): {fn}")
    print(f"  False Pos (human=wrong, GPT=correct): {fp}")
    print()
    print("COMPARISON:")
    print(f"  Original table (full gold, N=112): 92.0%, κ=0.75")
    print(f"  Step8 prompt   (full gold, N=112): 70.5%, κ=0.28")
    print(f"  Step2 prompt   (sample,   N={n}):  {pct:.1f}%, κ={kappa:.2f}")

    # --- Also show step8 agreement on same 30 questions ---
    # Load step8 labels
    step8_lookup = {}
    for fold in range(5):
        fpath = PROJECT_ROOT / "output" / "step8" / "biomistral-7b" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if fpath.exists():
            fold_df = pd.read_csv(fpath)
            for _, row in fold_df.iterrows():
                step8_lookup[int(row["patient_id"])] = int(row["binary_correct"])

    step8_valid = [pid for pid in valid_pids if pid in step8_lookup]
    if step8_valid:
        h8 = [reviewer_labels["A"][pid] for pid in step8_valid]
        g8 = [step8_lookup[pid] for pid in step8_valid]
        n8 = len(step8_valid)
        agree8 = sum(a == b for a, b in zip(h8, g8))
        pct8 = agree8 / n8 * 100
        kappa8 = cohen_kappa_score(h8, g8) if n8 > 1 else 0
        print(f"  Step8 prompt   (same 30, N={n8}):  {pct8:.1f}%, κ={kappa8:.2f}")

    # --- Show disagreement details ---
    print()
    print("=" * 60)
    print("DISAGREEMENT DETAILS (step2 answers)")
    print("=" * 60)
    for pid in valid_pids:
        h = reviewer_labels["A"][pid]
        g = gpt_labels[pid]
        if h != g:
            data = step2_lookup[pid]
            h_str = "CORRECT" if h == 1 else "WRONG"
            g_str = "CORRECT" if g == 1 else "WRONG"
            print(f"\n  Patient {pid}: Human={h_str}, GPT-4o={g_str}")
            print(f"  Q: {data['question'][:120]}...")
            print(f"  Model answer: {data['model_answer'][:150]}...")


if __name__ == "__main__":
    main()
