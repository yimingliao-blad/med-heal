#!/usr/bin/env python3
"""
Test correction prompts on the 49 detected wrong items.
Uses error descriptions from D2d detection + BM error pool.
GPT-4o judges if corrections are correct.

Correction strategies:
  C1: Simple hint — tell model what's wrong, ask to re-answer
  C2: Hint + notes quote — provide the specific notes evidence
  C3: Hint + error pool — provide similar error→correction examples from BM pool
  C4: Hint + notes quote + error pool — all available info
  C5: Full regen — just re-answer with no hint (baseline)

Usage:
    python test_correction.py --port 8003 --n 10
"""
import json, random, re, sys, os, time, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

# API key
api_key = None
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
        api_key = line.split("=", 1)[1]; break

from openai import OpenAI
gpt_client = OpenAI(api_key=api_key)
spending = {"calls": 0, "cost": 0.0}

PORT = 8003

# ============================================================
# CORRECTION PROMPTS
# ============================================================

C1_HINT = """Discharge summary:
{note}

Question: {question}

Your previous answer had an error:
ERROR: {error_statement}

Re-answer the question based on the discharge notes. Fix the identified error.
Answer in 1-3 direct sentences."""

C2_HINT_NOTES = """Discharge summary:
{note}

Question: {question}

Your previous answer had an error:
ERROR: {error_statement}
THE NOTES SAY: {correct_statement}

Re-answer the question based on the discharge notes. Include the information above.
Answer in 1-3 direct sentences."""

C3_HINT_POOL = """Discharge summary:
{note}

Question: {question}

Your previous answer had an error:
ERROR: {error_statement}

Here are examples of similar errors and their corrections from other clinical cases:
{pool_examples}

Re-answer the question based on the discharge notes. Fix the identified error.
Answer in 1-3 direct sentences."""

C4_HINT_NOTES_POOL = """Discharge summary:
{note}

Question: {question}

Your previous answer had an error:
ERROR: {error_statement}
THE NOTES SAY: {correct_statement}

Here are examples of similar errors and their corrections from other clinical cases:
{pool_examples}

Re-answer the question based on the discharge notes. Include the correct information.
Answer in 1-3 direct sentences."""

C5_REGEN = """Discharge summary:
{note}

Question: {question}

Answer this question based on the discharge notes.
Answer in 1-3 direct sentences."""

PROMPTS = {
    "C1_hint": C1_HINT,
    "C2_hint_notes": C2_HINT_NOTES,
    "C3_hint_pool": C3_HINT_POOL,
    "C4_hint_notes_pool": C4_HINT_NOTES_POOL,
    "C5_regen": C5_REGEN,
}

# ============================================================
# HELPERS
# ============================================================

def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def vllm_gen(prompt, max_tokens=512, temperature=1.0):
    model = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{PORT}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=120)
    return resp.json()["choices"][0]["text"].strip()

def gpt4o_eval(note, question, gt, answer):
    time.sleep(1.5)
    try:
        r = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical expert evaluating an AI model's answer to a clinical question."},
                {"role": "user", "content": (
                    f"DISCHARGE SUMMARY:\n{note}\n\nQUESTION:\n{question}\n\n"
                    f"CORRECT ANSWER (Ground Truth):\n{gt}\n\n"
                    f"MODEL'S ANSWER:\n{answer}\n\n"
                    f"Respond with ONLY a single digit: 1 = Correct, 0 = Incorrect"
                )},
            ],
            max_tokens=10, temperature=0.1,
        )
        text = r.choices[0].message.content.strip()
        cost = r.usage.prompt_tokens * 2.5 / 1e6 + r.usage.completion_tokens * 10.0 / 1e6
        spending["calls"] += 1
        spending["cost"] += cost
        return 1 if text.startswith("1") else 0 if text.startswith("0") else -1
    except Exception as e:
        print(f"  GPT-4o error: {e}")
        time.sleep(5)
        return -1

def load_notes():
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    lookup = {}
    for _, r in notes_df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(f"[Note {i}]\n{t}")
        lookup[pid] = "\n\n".join(parts)
    return lookup

def load_error_pool(fold_id):
    """Load BM atomic error pool for this fold (cross-fold safe)."""
    pool_file = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool" / f"fold_{fold_id}_atoms.json"
    if pool_file.exists():
        with open(pool_file) as f:
            return json.load(f)
    return []

def get_pool_examples(error_type, fold_id, k=2):
    """Get k similar error→correction examples from BM pool."""
    pool = load_error_pool(fold_id)
    if not pool:
        return ""

    # Filter by error type
    type_map = {"OMISSION": "omission", "MISREADING": "factual_error", "FABRICATION": "fabrication"}
    target = type_map.get(error_type, "factual_error")
    matching = [a for a in pool if a.get("main_error_type") == target and a.get("gt_atom_raw")]

    if not matching:
        matching = [a for a in pool if a.get("gt_atom_raw")]

    random.shuffle(matching)
    examples = []
    for a in matching[:k]:
        examples.append(f"  Wrong: \"{a['text_raw'][:120]}\"\n  Correct: \"{a['gt_atom_raw'][:120]}\"")

    return "\n".join(examples) if examples else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n", type=int, default=10, help="Number of detected items to test")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()))
    args = parser.parse_args()

    global PORT
    PORT = args.port

    notes = load_notes()

    # Load detected wrong items
    with open(OUTPUT_DIR / "d2d_fullscale_qwen25_progress.json") as f:
        all_results = json.load(f)
    detected_wrong = [r for r in all_results if r["label"] == "wrong" and r["verdict"] == "INCORRECT"]

    # Load original data for GT
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # Sample
    random.seed(42)
    sample = random.sample(detected_wrong, min(args.n, len(detected_wrong)))

    print(f"Correction test: {len(sample)} detected wrong items, {len(args.prompts)} prompts")
    print(f"GPT-4o eval cost estimate: ~${len(sample) * len(args.prompts) * 0.015:.2f}")
    print("=" * 70)

    all_correction_results = {}
    for pkey in args.prompts:
        ptemplate = PROMPTS[pkey]
        print(f"\n--- {pkey} ---")

        results = []
        for r in sample:
            row = all_df[(all_df["fold"] == r["fold"]) & (all_df["idx"] == r["idx"])]
            if len(row) == 0: continue
            row = row.iloc[0]
            note = notes.get(str(row["patient_id"]), "")
            if not note: continue

            # Build pool examples
            pool_examples = get_pool_examples(r["error_type"], r["fold"])

            # Build correction prompt
            fmt_args = {
                "note": note, "question": row["question"],
                "error_statement": r.get("error_statement", "")[:200],
                "correct_statement": r.get("correct_statement", "")[:200],
                "pool_examples": pool_examples,
            }

            # Only include fields the prompt uses
            try:
                msg = ptemplate.format(**fmt_args)
            except KeyError:
                msg = ptemplate.format(**{k: v for k, v in fmt_args.items()
                                         if "{" + k + "}" in ptemplate})

            prompt = build_chatml("You are a medical expert answering questions about discharge summaries.", msg)
            corrected = vllm_gen(prompt, max_tokens=512, temperature=1.0)

            # GPT-4o judge
            gt = row["ground_truth"]
            eval_result = gpt4o_eval(note, row["question"], gt, corrected)

            results.append({
                "idx": r["idx"], "fold": r["fold"],
                "error_type": r["error_type"],
                "eval_corrected": eval_result,
                "corrected_answer": corrected[:300],
            })

            status = "FIX" if eval_result == 1 else "FAIL" if eval_result == 0 else "ERR"
            print(f"  idx={r['idx']} {r['error_type']:>15} → [{status}] ${spending['cost']:.2f}")

        all_correction_results[pkey] = results

        fix = sum(1 for r in results if r["eval_corrected"] == 1)
        fail = sum(1 for r in results if r["eval_corrected"] == 0)
        print(f"  FIX: {fix}/{len(results)} ({100*fix/max(len(results),1):.0f}%)")

    # Save
    out_file = OUTPUT_DIR / "correction_test_results.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_correction_results, "n": len(sample)}, f, indent=2)

    # Final table
    print(f"\n{'='*70}")
    print(f"CORRECTION RESULTS")
    print(f"{'='*70}")
    print(f"{'Prompt':<22} {'FIX':>6} {'FAIL':>6} {'Rate':>8}")
    print("-" * 44)
    for pkey in args.prompts:
        r = all_correction_results[pkey]
        fix = sum(1 for x in r if x["eval_corrected"] == 1)
        fail = sum(1 for x in r if x["eval_corrected"] == 0)
        print(f"  {pkey:<22} {fix:>4}   {fail:>4}   {100*fix/max(len(r),1):>5.0f}%")

    print(f"\nGPT-4o: {spending['calls']} calls, ${spending['cost']:.3f}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
