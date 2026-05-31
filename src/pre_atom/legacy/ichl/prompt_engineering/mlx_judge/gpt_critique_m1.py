"""GPT-4o critique of the M1 judge prompt.

Loads:
  - The M1 prompt template (full text including rules + format)
  - 10 errors from Magistral_M1_dev300.jsonl: items where M1 disagreed with gold

Asks GPT-4o:
  "Here is a judge prompt + 10 cases where this judge got it wrong.
   What's wrong with the prompt? What rule(s) would you add / remove / change
   to fix these errors? Be concrete — quote the rule number being changed."

Saves the critique to a file for human review and for M4 design.
"""
from __future__ import annotations
import json
import os
import random
from collections import Counter
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[4]
M1_RESULTS = ROOT / "output" / "ichl" / "mlx_judge" / "vllm_candidates" / "Magistral_M1_dev300.jsonl"
DEV_JSONL = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
CRITIQUE_OUT = ROOT / "output" / "ichl" / "mlx_judge" / "gpt_critique_m1.json"

# M1 prompt template (the prompt we asked Magistral to follow)
M1_RULES = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 if all specific claims align with the ground truth and the answer addresses the question — paraphrases, synonyms, different orderings, or different units that convey the same clinical fact all count as alignment.",
    "Output 0 if the answer includes additional, incorrect information not present in the ground truth.",
    "Output 1 if the answer provides correct additional context that does not contradict the ground truth.",
    "Be charitable on form, strict on content: do not penalize an answer for using different wording, abbreviations, generic vs. brand names, or restated phrasing — penalize only when the underlying clinical fact differs from the ground truth.",
]
M1_PROMPT_TEMPLATE = """SYSTEM: You are a medical expert evaluating an AI model's answer to a clinical question.

USER:
When judging, apply the following rules:

{rules}

DISCHARGE SUMMARY:
{{note}}

QUESTION:
{{question}}

CORRECT ANSWER (Ground Truth):
{{ground_truth}}

MODEL'S ANSWER:
{{model_answer}}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect""".format(rules="\n".join(f"{i}. {r}" for i, r in enumerate(M1_RULES, 1)))


def load_errors(n: int = 10, seed: int = 42):
    """Pick n M1 errors stratified by error direction."""
    rows = [json.loads(l) for l in M1_RESULTS.open() if l.strip()]
    errors = [r for r in rows if r.get("mlx_label") is not None and r["mlx_label"] != r["gold_label"]]
    fp = [r for r in errors if r["gold_label"] == 0 and r["mlx_label"] == 1]  # false positive (judge said 1, gold 0)
    fn = [r for r in errors if r["gold_label"] == 1 and r["mlx_label"] == 0]  # false negative (judge said 0, gold 1)
    print(f"  M1 errors total: {len(errors)} (FP={len(fp)}, FN={len(fn)})")
    half = n // 2
    rng = random.Random(seed)
    rng.shuffle(fp); rng.shuffle(fn)
    return fp[:half] + fn[:n - half]


def load_dev_lookup():
    return {(r["target"], r["patient_id"], r["fold_id"]): r
            for r in (json.loads(l) for l in DEV_JSONL.open() if l.strip())}


def main():
    print("Loading M1 errors + dev lookup…")
    errs = load_errors(10)
    dev_lookup = load_dev_lookup()
    cases = []
    for i, r in enumerate(errs, 1):
        key = (r["target"], r["patient_id"], r["fold_id"])
        dev_item = dev_lookup.get(key)
        if not dev_item:
            print(f"  WARN: missing dev item for {key}")
            continue
        cases.append({
            "idx": i,
            "question": dev_item["question"],
            "ground_truth": dev_item["ground_truth"],
            "model_answer": dev_item["model_answer"],
            "gold_label": r["gold_label"],
            "judge_verdict": r["mlx_label"],
            "error_type": "FALSE_POSITIVE" if (r["gold_label"] == 0 and r["mlx_label"] == 1) else "FALSE_NEGATIVE",
        })
    cases_block = "\n\n".join(
        f"--- Case {c['idx']}  [GOLD={c['gold_label']} | JUDGE={c['judge_verdict']} | {c['error_type']}] ---\n"
        f"QUESTION: {c['question']}\n"
        f"GROUND TRUTH: {c['ground_truth']}\n"
        f"MODEL'S ANSWER: {c['model_answer']}\n"
        f"(Note: discharge summary omitted to keep prompt small; the judge had access to it.)"
        for c in cases
    )

    critique_prompt = f"""You are a senior clinical evaluator helping debug an LLM judge.

Below is a judge prompt currently used to evaluate AI-model answers to clinical questions on EHR data, and 10 cases where this judge got the verdict wrong (4 false-positives where it said CORRECT but the answer is wrong, 6 false-negatives where it said INCORRECT but the answer is right).

Your task: identify what is wrong with the prompt that causes these errors, and suggest CONCRETE changes — add a rule, remove a rule, reword a rule, or reorder. Quote the rule number you'd change. Be specific and actionable. Limit your reply to ~400 words.

============== CURRENT JUDGE PROMPT ==============

{M1_PROMPT_TEMPLATE}

============== 10 ERROR CASES ==============

{cases_block}

============== YOUR CRITIQUE ==============

(Structure your reply as:)

1. Pattern across these errors (1-2 sentences)
2. Specific prompt changes (numbered list of concrete edits, each citing the rule number affected)
3. A revised rule list (so I can use it directly)"""

    # Pull API key from .env (per MEMORY.md convention)
    env_path = ROOT / ".env"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set and not in .env")

    client = OpenAI(api_key=api_key)
    print(f"\nSending {len(cases)} cases to GPT-4o (≈ {len(critique_prompt)//4} tokens)…")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior clinical evaluator helping debug an LLM judge prompt."},
            {"role": "user", "content": critique_prompt},
        ],
        temperature=0.0, max_tokens=1500,
    )
    critique = resp.choices[0].message.content
    usage = resp.usage
    print(f"\n=== GPT-4o critique ===\n{critique}\n=======================")
    print(f"\nTokens: prompt={usage.prompt_tokens}  completion={usage.completion_tokens}  cost≈${(usage.prompt_tokens*5e-6 + usage.completion_tokens*1.5e-5):.4f}")

    out = {
        "critique": critique,
        "n_cases": len(cases),
        "cases": cases,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "model": "gpt-4o",
    }
    CRITIQUE_OUT.parent.mkdir(parents=True, exist_ok=True)
    CRITIQUE_OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {CRITIQUE_OUT}")


if __name__ == "__main__":
    main()
