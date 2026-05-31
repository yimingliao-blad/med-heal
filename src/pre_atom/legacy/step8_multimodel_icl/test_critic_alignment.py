#!/usr/bin/env python3
"""Phase 0: Critic Selection — Alignment Test.

0A: Ask GPT-4o to articulate its binary judging criteria (with a real EHRNoteQA example).
0B: Test 4 Qwen3 configs + 1 GPT-4o config on 50 random zeroshot answers.
     Measure agreement with existing GPT-4o binary labels.

Usage:
    # Full test (0A + 0B)
    python test_critic_alignment.py --port 8003

    # Just extract GPT-4o criteria (0A only)
    python test_critic_alignment.py --criteria-only

    # Just run alignment test (0B only, requires criteria file)
    python test_critic_alignment.py --port 8003 --alignment-only
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent.parent
STEP8_DIR = PROJECT_ROOT / "output" / "step8"
FOLDS_DIR = PROJECT_ROOT / "output" / "folds"
OUTPUT_DIR = PROJECT_ROOT / "output" / "step8" / "experiments" / "phase0_critic"

# All 5 models to sample from
ALL_MODELS = [
    "biomistral-7b",
    "deepseek-r1-distill-llama-8b",
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen3-8b",
]


# =============================================================================
# PHASE 0A: Extract GPT-4o judging criteria
# =============================================================================

CRITERIA_EXTRACTION_PROMPT = """You are used as a binary judge for the EHRNoteQA benchmark. You evaluate if an AI model's answer about a patient's discharge summary is correct compared to the ground truth. You respond with 1 (Correct) or 0 (Incorrect).

Here is a typical evaluation you perform:

DISCHARGE SUMMARY:
{note_excerpt}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER (Correct example):
{correct_answer}

Your verdict: 1

MODEL'S ANSWER (Incorrect example):
{incorrect_answer}

Your verdict: 0

Please articulate the specific criteria you use when deciding 1 vs 0:
- How much detail overlap is needed for '1'?
- Are partial answers acceptable?
- Does extra correct information matter?
- What about information that is correct but not in the ground truth?
- When is an answer 'close enough' vs wrong?
- How do you handle vague vs specific answers?
- What makes you decide '0' (incorrect)?"""


def extract_gpt4o_criteria(client, notes_df, eval_data):
    """Ask GPT-4o to articulate its binary judging criteria."""
    # Pick one correct and one incorrect example from the eval data
    correct_examples = [d for d in eval_data if d["binary_correct"] == 1]
    incorrect_examples = [d for d in eval_data if d["binary_correct"] == 0]

    if not correct_examples or not incorrect_examples:
        raise ValueError("Need both correct and incorrect examples")

    # Use first of each (deterministic)
    correct_ex = correct_examples[0]
    incorrect_ex = incorrect_examples[0]

    # Get a note excerpt (first 500 tokens worth)
    note = get_note_for_patient(correct_ex["patient_id"], notes_df)
    note_excerpt = note[:2000] + "..." if len(note) > 2000 else note

    prompt = CRITERIA_EXTRACTION_PROMPT.format(
        note_excerpt=note_excerpt,
        question=correct_ex["question"],
        ground_truth=correct_ex["ground_truth"],
        correct_answer=correct_ex["model_answer"],
        incorrect_answer=incorrect_ex["model_answer"],
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are reflecting on your own evaluation methodology."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# =============================================================================
# PHASE 0B: Alignment test configs
# =============================================================================

# Binary prompt (simple, matches GPT-4o eval format)
BINARY_PROMPT = """DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect"""

# Reasoning prompt (uses GPT-4o's self-described criteria)
REASONING_PROMPT = """DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

{criteria}

Based on these criteria, evaluate the model's answer step by step.
End your response with a final line: VERDICT: 1 (correct) or VERDICT: 0 (incorrect)"""

# GPT-4o critic prompt (different structure from binary eval to reduce circularity)
GPT4O_CRITIC_PROMPT = """You are reviewing an AI model's answer to a clinical question based on a discharge summary.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

REFERENCE ANSWER:
{ground_truth}

AI MODEL'S ANSWER:
{model_answer}

First, identify any specific errors in the AI's answer:
- Missing information that should be included
- Fabricated/hallucinated information not in the notes
- Incorrect clinical reasoning
- Vague when specific information was available

Then conclude: Is the AI's answer correct overall?
End with: VERDICT: 1 (correct) or VERDICT: 0 (incorrect)"""


CRITIC_CONFIGS = {
    "qwen3-think-binary": {
        "model": "qwen3",
        "thinking": True,
        "prompt_type": "binary",
    },
    "qwen3-think-reason": {
        "model": "qwen3",
        "thinking": True,
        "prompt_type": "reasoning",
    },
    "qwen3-nothink-binary": {
        "model": "qwen3",
        "thinking": False,
        "prompt_type": "binary",
    },
    "qwen3-nothink-reason": {
        "model": "qwen3",
        "thinking": False,
        "prompt_type": "reasoning",
    },
    "gpt4o-critic": {
        "model": "gpt4o",
        "thinking": False,
        "prompt_type": "gpt4o_critic",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def get_note_for_patient(patient_id, notes_df):
    """Get assembled discharge note for a patient."""
    rows = notes_df[notes_df["patient_id"] == patient_id]
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


def load_sample_data(n_per_fold=10, seed=42):
    """Load 50 random zeroshot answers (10/fold) with stratified correct/incorrect."""
    all_samples = []

    for fold in range(5):
        # Load from any model that has full data — use qwen2.5 as reference
        eval_file = STEP8_DIR / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if not eval_file.exists():
            print(f"  WARNING: Missing eval file for fold {fold}")
            continue

        df = pd.read_csv(eval_file)

        # Stratified sample: ~half correct, ~half incorrect
        correct = df[df["binary_correct"] == 1]
        incorrect = df[df["binary_correct"] == 0]

        rng = random.Random(seed + fold)
        n_correct = min(n_per_fold // 2, len(correct))
        n_incorrect = min(n_per_fold - n_correct, len(incorrect))
        # If not enough incorrect, fill with correct
        if n_incorrect < n_per_fold - n_correct:
            n_correct = min(n_per_fold - n_incorrect, len(correct))

        sampled_correct = correct.sample(n=n_correct, random_state=seed + fold)
        sampled_incorrect = incorrect.sample(n=n_incorrect, random_state=seed + fold)
        fold_sample = pd.concat([sampled_correct, sampled_incorrect]).sample(
            frac=1, random_state=seed + fold  # Shuffle
        )

        for _, row in fold_sample.iterrows():
            all_samples.append({
                "patient_id": int(row["patient_id"]),
                "fold_id": fold,
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "model_answer": str(row["model_answer"]),
                "binary_correct": int(row["binary_correct"]),
                "source_model": "qwen2.5-7b-instruct",
            })

    return all_samples


def evaluate_qwen3(port, note, question, ground_truth, model_answer,
                   thinking, prompt_type, criteria_text=""):
    """Evaluate with Qwen3 via vLLM."""
    base_url = f"http://localhost:{port}/v1"

    if prompt_type == "binary":
        user_content = BINARY_PROMPT.format(
            note=note, question=question,
            ground_truth=ground_truth, model_answer=model_answer,
        )
    elif prompt_type == "reasoning":
        user_content = REASONING_PROMPT.format(
            note=note, question=question,
            ground_truth=ground_truth, model_answer=model_answer,
            criteria=criteria_text,
        )
    else:
        raise ValueError(f"Unknown prompt_type for qwen3: {prompt_type}")

    # Build ChatML prompt
    think_tag = "/no_think" if not thinking else ""
    system_msg = "You are a medical expert evaluating an AI model's answer to a clinical question."

    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{think_tag}\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Thinking mode needs more tokens — think block alone can be 500+ tokens
    if thinking:
        max_tokens = 2048
    elif prompt_type == "binary":
        max_tokens = 50
    else:
        max_tokens = 1024

    try:
        resp = requests.post(
            f"{base_url}/completions",
            json={
                "model": requests.get(f"{base_url}/models", timeout=5).json()["data"][0]["id"],
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return None, ""

        raw = resp.json()["choices"][0]["text"].strip()

        # Extract verdict
        import re
        # Strip thinking tags
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"</think>", "", clean).strip()

        if prompt_type == "binary":
            if "1" in clean and "0" not in clean:
                return 1, clean
            elif "0" in clean:
                return 0, clean
            else:
                return None, clean
        else:
            # Look for VERDICT line
            verdict_match = re.search(r"VERDICT:\s*([01])", clean)
            if verdict_match:
                return int(verdict_match.group(1)), clean
            # Fallback: last digit
            if "1" in clean[-20:] and "0" not in clean[-20:]:
                return 1, clean
            elif "0" in clean[-20:]:
                return 0, clean
            return None, clean

    except Exception as e:
        return None, str(e)


def evaluate_gpt4o_critic(client, note, question, ground_truth, model_answer):
    """Evaluate with GPT-4o using the critic prompt (different from binary eval)."""
    user_content = GPT4O_CRITIC_PROMPT.format(
        note=note, question=question,
        ground_truth=ground_truth, model_answer=model_answer,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical expert reviewing AI-generated clinical answers."},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()

            import re
            verdict_match = re.search(r"VERDICT:\s*([01])", raw)
            if verdict_match:
                return int(verdict_match.group(1)), raw
            # Fallback
            if "1" in raw[-20:] and "0" not in raw[-20:]:
                return 1, raw
            elif "0" in raw[-20:]:
                return 0, raw
            return None, raw

        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None, str(e)
    return None, "max retries"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Critic Selection")
    parser.add_argument("--port", type=int, default=8003, help="vLLM port for Qwen3")
    parser.add_argument("--criteria-only", action="store_true", help="Only extract GPT-4o criteria (Phase 0A)")
    parser.add_argument("--alignment-only", action="store_true", help="Only run alignment test (Phase 0B)")
    parser.add_argument("--configs", nargs="+", default=list(CRITIC_CONFIGS.keys()),
                        help="Which critic configs to test")
    args = parser.parse_args()

    # Setup
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load notes
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    print(f"Loaded notes for {len(notes_df)} patients")

    # Load sample data
    samples = load_sample_data(n_per_fold=10, seed=42)
    n_correct = sum(s["binary_correct"] == 1 for s in samples)
    n_incorrect = sum(s["binary_correct"] == 0 for s in samples)
    print(f"Loaded {len(samples)} samples: {n_correct} correct, {n_incorrect} incorrect (per GPT-4o binary)")

    # =========================================================================
    # PHASE 0A: Extract GPT-4o criteria
    # =========================================================================
    criteria_file = OUTPUT_DIR / "gpt4o_criteria.txt"

    if not args.alignment_only:
        print("\n" + "=" * 60)
        print("PHASE 0A: Extracting GPT-4o judging criteria")
        print("=" * 60)

        criteria_text = extract_gpt4o_criteria(client, notes_df, samples)
        with open(criteria_file, "w") as f:
            f.write(criteria_text)
        print(f"Criteria saved to {criteria_file}")
        print(f"\n--- GPT-4o Criteria ---\n{criteria_text}\n---")

    if args.criteria_only:
        return

    # Load criteria for reasoning prompt
    criteria_text = ""
    if criteria_file.exists():
        criteria_text = criteria_file.read_text().strip()
        print(f"Loaded criteria from {criteria_file} ({len(criteria_text)} chars)")
    else:
        print("WARNING: No criteria file found. Reasoning prompts will have empty criteria.")

    # =========================================================================
    # PHASE 0B: Alignment test
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 0B: Critic Alignment Test (50 samples)")
    print("=" * 60)

    # Check vLLM for Qwen3 configs
    qwen3_configs = [c for c in args.configs if CRITIC_CONFIGS[c]["model"] == "qwen3"]
    if qwen3_configs:
        try:
            resp = requests.get(f"http://localhost:{args.port}/v1/models", timeout=5)
            model_name = resp.json()["data"][0]["id"]
            print(f"Qwen3 vLLM: {model_name} on port {args.port}")
        except Exception:
            print(f"WARNING: vLLM not available on port {args.port}. Skipping Qwen3 configs.")
            qwen3_configs = []

    # Run each config
    all_results = {}

    for config_name in args.configs:
        config = CRITIC_CONFIGS[config_name]

        if config["model"] == "qwen3" and config_name not in qwen3_configs:
            print(f"\n  Skipping {config_name} (no vLLM)")
            continue

        print(f"\n  --- {config_name} ---")
        predictions = []
        raw_outputs = []

        for i, sample in enumerate(samples):
            note = get_note_for_patient(sample["patient_id"], notes_df)

            if config["model"] == "qwen3":
                verdict, raw = evaluate_qwen3(
                    args.port, note, sample["question"],
                    sample["ground_truth"], sample["model_answer"],
                    thinking=config["thinking"],
                    prompt_type=config["prompt_type"],
                    criteria_text=criteria_text,
                )
            elif config["model"] == "gpt4o":
                verdict, raw = evaluate_gpt4o_critic(
                    client, note, sample["question"],
                    sample["ground_truth"], sample["model_answer"],
                )
                time.sleep(0.3)
            else:
                raise ValueError(f"Unknown model: {config['model']}")

            predictions.append(verdict)
            raw_outputs.append(raw)

            gt = sample["binary_correct"]
            marker = "✓" if verdict == gt else ("✗" if verdict is not None else "?")
            if (i + 1) % 10 == 0 or verdict != gt:
                print(f"    [{i+1}/{len(samples)}] GT={gt} Pred={verdict} {marker}")

        # Compute metrics
        valid_idx = [i for i, p in enumerate(predictions) if p is not None]
        gt_labels = [samples[i]["binary_correct"] for i in valid_idx]
        pred_labels = [predictions[i] for i in valid_idx]

        n = len(valid_idx)
        agree = sum(g == p for g, p in zip(gt_labels, pred_labels))
        pct = agree / n * 100 if n > 0 else 0
        kappa = cohen_kappa_score(gt_labels, pred_labels) if n > 1 else 0

        # Confusion matrix: FN = GT=correct, Pred=incorrect; FP = GT=incorrect, Pred=correct
        fn = sum(g == 1 and p == 0 for g, p in zip(gt_labels, pred_labels))
        fp = sum(g == 0 and p == 1 for g, p in zip(gt_labels, pred_labels))
        null_count = len(predictions) - len(valid_idx)

        result = {
            "config": config_name,
            "n_valid": n,
            "n_null": null_count,
            "agreement_pct": round(pct, 1),
            "kappa": round(kappa, 3),
            "fn": fn,
            "fp": fp,
        }
        all_results[config_name] = result
        print(f"    → Agr={pct:.1f}%, κ={kappa:.3f}, FN={fn}, FP={fp}, Null={null_count}")

        # Save per-config details
        details = []
        for i, sample in enumerate(samples):
            details.append({
                **sample,
                "prediction": predictions[i],
                "raw_output": raw_outputs[i][:500] if raw_outputs[i] else "",
            })
        details_df = pd.DataFrame(details)
        details_df.to_csv(OUTPUT_DIR / f"{config_name}_details.csv", index=False)

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 0B RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'N':>4} {'Agr%':>6} {'κ':>7} {'FN':>4} {'FP':>4} {'Null':>5}")
    print("-" * 60)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["agreement_pct"]):
        print(f"{name:<25} {r['n_valid']:>4} {r['agreement_pct']:>5.1f}% {r['kappa']:>7.3f} "
              f"{r['fn']:>4} {r['fp']:>4} {r['n_null']:>5}")

    # Save summary
    summary_file = OUTPUT_DIR / "alignment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # Recommendation
    if all_results:
        best = max(all_results.items(), key=lambda x: x[1]["kappa"])
        print(f"\n>>> RECOMMENDED CRITIC: {best[0]} (κ={best[1]['kappa']:.3f}, Agr={best[1]['agreement_pct']:.1f}%)")


if __name__ == "__main__":
    main()
