"""Run a parametrizable judge prompt on a vLLM-served candidate model.

Adds --prompt-variant flag (C3E baseline / M1 charitable / M2 ...) to the v1
runner. Captures truncation_report per Claude: Principle: Truncation Detection.

Usage:
    PYTHONPATH=src .venv/bin/python -m ichl.prompt_engineering.mlx_judge.run_vllm_judge_v2 \
        --model Magistral-Small-2509-AWQ \
        --prompt-variant M1_charitable \
        --output Magistral_M1_dev300.jsonl \
        --n 300 --max-tokens 256 --workers 4
"""
from __future__ import annotations
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from ichl.prompt_engineering.correction.truncation_detector import detect_truncation

ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
SPLIT_DIR = ROOT / "output" / "ichl" / "mlx_judge" / "splits"
OUT_DIR = ROOT / "output" / "ichl" / "mlx_judge" / "vllm_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL = "http://localhost:8003/v1"
SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

# Baseline rules (used by Qwen3 phase C, frozen)
C3E_RULES = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 only if all specific claims align with the ground truth and the answer directly addresses the question.",
    "Output 0 if the answer includes additional, incorrect information not present in the ground truth.",
    "Output 1 if the answer provides correct additional context that does not contradict the ground truth.",
]

# M1: target rejective bias — add explicit charitable-paraphrase guidance.
# Rule 5 reworded to be less strict; new rule 8 names common paraphrase patterns.
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

# M2: rejective bias + explicit calibration. Add a one-shot example of a
# paraphrased correct answer marked 1, alongside M1's charitable language.
M2_RULES = M1_RULES + [
    "Example: if ground truth is \"metformin 500 mg twice daily\" and the answer is \"500 mg metformin BID\", output 1 (same fact, different abbreviation/ordering).",
    "Example: if ground truth is \"acute kidney injury\" and the answer is \"AKI\", output 1 (synonym).",
]

# M4: rule set as suggested by GPT-4o critique on M1 + 10 errors (2026-04-25).
# Direction of GPT's suggestion: more permissive — adds "clinically insignificant"
# qualifier to contradiction rules, merges 5+7 into a single positive rule.
M4_RULES = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth, unless the discrepancy is clinically insignificant.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 if all specific claims align with the ground truth and the answer addresses the question — paraphrases, synonyms, different orderings, or different units that convey the same clinical fact all count as alignment. Correct additional context or detail that aligns with the clinical scenario should be considered positively.",
    "Output 0 if the answer includes additional, incorrect information that introduces clinically significant errors or contradictions not present in the ground truth.",
    "Be charitable on form, strict on content: do not penalize an answer for using different wording, abbreviations, generic vs. brand names, or restated phrasing — penalize only when the underlying clinical fact differs from the ground truth. Correctness in clinical interpretation takes precedence over verbatim matching.",
]

VARIANTS = {
    "C3E": C3E_RULES,
    "M1_charitable": M1_RULES,
    "M2_charitable_oneshot": M2_RULES,
    "M3_fewshot": M1_RULES,            # rules same as M1; few-shot block added in build_user
    "M4_gpt_critique": M4_RULES,       # GPT-4o-suggested revision of M1
    "M5_cot": M1_RULES,                # M1 rules + CoT instruction (different output format)
}

# 4 train_pool full-case examples (2 gold=0, 2 gold=1; mixed targets; no notes to keep prompt small)
M3_FEWSHOT_BLOCK = """\
Here are four worked examples — your task on the actual case will follow.

[Example 1 — VERDICT: 0]
QUESTION: What is the main variation in the patient's discharged condition between the two visits, recorded on 2148-01-02 and on 2148-11-05?
CORRECT ANSWER (Ground Truth): E: The first time, the patient was diagnosed with cervical stenosis, and on the second visit, he was diagnosed with back and right leg pain.
MODEL'S ANSWER: The main variation in the patient's discharged condition between the two visits is that on the first visit (2148-01-02), the patient was discharged with a condition of "B" (which is not specified), whereas on the second visit (2148-11-05), the patient was discharged with a clear and coherent mental status, alert and interactive level of consciousness, and ambulatory - independent activity status.
REASON: The question asks about diagnoses across two visits; the model answers about mental-status and ambulatory level, which is a different aspect (rule 3).

[Example 2 — VERDICT: 0]
QUESTION: What was the patient's final diagnosis upon her initial discharge from the hospital, and what treatment was given?
CORRECT ANSWER (Ground Truth): B. The patient was diagnosed with acute on chronic cholecystitis and was treated with IV antibiotics.
MODEL'S ANSWER: The patient's final diagnosis upon her initial discharge from the hospital was acute on chronic cholecystitis. The treatment given included a 2-week course of antibiotics: 500 mg of Ciprofloxacin (Cipro) twice daily and 500 mg of Metronidazole (Flagyl) three times daily.
REASON: The diagnosis matches but the treatment route contradicts: ground truth says IV antibiotics; the model gives an oral regimen (Cipro PO BID, Flagyl PO TID). Specific-fact contradiction on route (rule 1).

[Example 3 — VERDICT: 1]
QUESTION: What was the main postop complication from the infraclavicular first rib resection procedure?
CORRECT ANSWER (Ground Truth): D: The patient experienced difficulties with voiding.
MODEL'S ANSWER: The main postop complication from the infraclavicular first rib resection procedure was the patient's inability to void after the Foley catheter was discharged at midnight on POD0, resulting in a retained urine volume of 700cc. This was resolved after the patient was straight cathed in the morning of POD1.
REASON: "Inability to void" with retained urine is the same clinical fact as "difficulties with voiding"; extra detail (700cc, straight cath) is correct supporting context (rule 7).

[Example 4 — VERDICT: 1]
QUESTION: What was the patient's final diagnosis upon their last admission and what treatment was pursued for them?
CORRECT ANSWER (Ground Truth): B: The patient was diagnosed with two basal ganglia masses and underwent stereotactic brain biopsy.
MODEL'S ANSWER: Based on the discharge summary, the patient's final diagnosis upon their last admission was "Basal ganglia masses". The treatment pursued for them was a stereotactic right-sided brain biopsy, which was an elective procedure. The patient underwent the surgery on the same day as admission and was discharged home in stable condition. The final pathology was still pending at the time of discharge.
REASON: Diagnosis matches; treatment matches; the omission of "two" masses is minor and the additional details (right-sided, elective, pathology pending) do not contradict (rules 5 + 7).

Apply the same reasoning to the case below."""


def build_user(item, note, rules, *, fewshot_block: str = "", cot: bool = False):
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    fewshot = (f"\n{fewshot_block}\n\n" if fewshot_block else "")
    if cot:
        # CoT format: brief reasoning then explicit FINAL: digit on a new line
        instruction = (
            "Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
            "First write 1-3 short sentences explaining your reasoning (cite the specific rule that fires).\n"
            "Then on a NEW LINE, write exactly:\n"
            "FINAL: 1   (if correct)\n"
            "FINAL: 0   (if incorrect)"
        )
    else:
        instruction = (
            "Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
            "Respond with ONLY a single digit:\n"
            "1 = Correct\n"
            "0 = Incorrect"
        )
    return f"""When judging, apply the following rules:

{rules_block}
{fewshot}
DISCHARGE SUMMARY:
{note}

QUESTION:
{item["question"]}

CORRECT ANSWER (Ground Truth):
{item["ground_truth"]}

MODEL'S ANSWER:
{item["model_answer"]}

{instruction}"""


def parse_01(text):
    """Default parser: first [01] in text. Used for digit-only variants."""
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    m = re.search(r"[01]", s)
    return int(m.group(0)) if m else None


def parse_cot(text):
    """CoT parser: look for 'FINAL: X'; fallback to LAST [01] in text."""
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    m = re.search(r"FINAL\s*[:=]\s*([01])", s)
    if m:
        return int(m.group(1))
    # Fallback: last [01] in text (avoids picking up digits in reasoning prose)
    matches = re.findall(r"[01]", s)
    return int(matches[-1]) if matches else None


def build_note_lookup():
    import pandas as pd
    df = pd.read_json(ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    lookup = {}
    for _, r in df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in r.index:
                v = r.get(col)
                if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                    parts.append(f"[Note {i}]\n{v}")
        lookup[pid] = "\n\n".join(parts)
    return lookup


def call(client, model, system, user, max_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=max_tokens,
            )
            return resp
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 + attempt * 3)
            else:
                print(f"  ERROR: {e}")
                return None
    return None


def judge_one(args):
    client, model, item, note, rules, max_tokens, fewshot_block, cot = args
    user = build_user(item, note, rules, fewshot_block=fewshot_block, cot=cot)
    t0 = time.monotonic()
    resp = call(client, model, SYSTEM, user, max_tokens)
    lat = time.monotonic() - t0
    if resp is None:
        return {**{k: item.get(k) for k in ["target", "patient_id", "fold_id", "binary_correct"]},
                "gold_label": int(item["binary_correct"]), "mlx_label": None,
                "content": "", "finish_reason": "ERROR",
                "completion_tokens": None, "prompt_tokens": None,
                "latency_s": round(lat, 2), "truncation_report": None}
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    fin = resp.choices[0].finish_reason
    usage = resp.usage
    label = parse_cot(content) if cot else parse_01(content)
    raw_for_detector = (reasoning_content + "\n" + content) if reasoning_content else content
    report = detect_truncation(
        raw_response=raw_for_detector, text_clean=content,
        finish_reason=fin,
        usage={"completion_tokens": usage.completion_tokens if usage else None,
               "prompt_tokens": usage.prompt_tokens if usage else None},
        max_tokens=max_tokens, target=model, sub_variant="vllm",
    )
    return {
        "target": item["target"], "patient_id": item["patient_id"], "fold_id": item["fold_id"],
        "gold_label": int(item["binary_correct"]), "mlx_label": label,
        "content": content,
        "finish_reason": fin,
        "completion_tokens": usage.completion_tokens if usage else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "latency_s": round(lat, 2),
        "truncation_report": report.as_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-url", default=VLLM_URL)
    ap.add_argument("--split", default="dev",
                    help="Which split to evaluate on (default dev). Reads splits/<split>.jsonl.")
    ap.add_argument("--offset", type=int, default=0,
                    help="Per-class offset for stratified sampling (skip first N g0 + first N g1). "
                         "Used in Stage 2 to avoid overlap with Stage 1's items.")
    args = ap.parse_args()

    rules = VARIANTS[args.prompt_variant]
    fewshot_block = M3_FEWSHOT_BLOCK if args.prompt_variant == "M3_fewshot" else ""
    cot = (args.prompt_variant == "M5_cot")
    print(f"Loading dev + notes…")
    split_path = SPLIT_DIR / f"{args.split}.jsonl"
    print(f"  split: {split_path.name}")
    dev = [json.loads(l) for l in split_path.open() if l.strip()]
    notes = build_note_lookup()
    g0 = [d for d in dev if d["binary_correct"] == 0]
    g1 = [d for d in dev if d["binary_correct"] == 1]
    if args.n >= len(dev):
        # Full split, no stratified resampling — preserve natural distribution
        sample = list(dev)
        print(f"  variant={args.prompt_variant}  n={len(sample)}  (FULL split, natural balance: g0={len(g0)}, g1={len(g1)})")
    else:
        half = args.n // 2
        off = args.offset
        sample = g0[off:off + half] + g1[off:off + (args.n - half)]
        print(f"  variant={args.prompt_variant}  n={len(sample)}  (stratified: g0={half}, g1={args.n - half}, offset={off})")
        if len(sample) < args.n:
            print(f"  WARNING: sample shorter than requested ({len(sample)} < {args.n}); split may be too small at this offset")
    print(f"  rules ({len(rules)}):")
    for i, r in enumerate(rules, 1):
        print(f"    {i}. {r[:100]}{'…' if len(r) > 100 else ''}")

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    out_path = OUT_DIR / args.output
    tasks = [(client, args.model, item, notes.get(str(item["patient_id"]), ""), rules, args.max_tokens,
              fewshot_block, cot)
             for item in sample]

    t0 = time.monotonic()
    results = []
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(judge_one, tasks), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            results.append(r)
            if i % 25 == 0:
                dt = time.monotonic() - t0
                print(f"  {i}/{len(sample)}  elapsed={dt:.0f}s  eta={dt*(len(sample)-i)/i:.0f}s")
    elapsed = time.monotonic() - t0

    n = len(results)
    correct = sum(1 for r in results if r["mlx_label"] == r["gold_label"])
    none_cnt = sum(1 for r in results if r["mlx_label"] is None)
    truncated_cert = sum(1 for r in results if (r.get("truncation_report") or {}).get("is_truncated_certain"))
    parsed = n - none_cnt
    correct_parsed = sum(1 for r in results if r["mlx_label"] is not None and r["mlx_label"] == r["gold_label"])
    print(f"\n=== {args.model}  variant={args.prompt_variant}  n={n}  wall={elapsed:.0f}s ===")
    print(f"  agreement (None=wrong): {correct}/{n} = {100*correct/n:.1f}%")
    print(f"  agreement excl-None: {100*correct_parsed/parsed if parsed else 0:.1f}%  (parsed={parsed})")
    print(f"  None: {none_cnt}  truncated_certain: {truncated_cert}")
    try:
        from sklearn.metrics import cohen_kappa_score
        pp = [(r["gold_label"], r["mlx_label"]) for r in results if r["mlx_label"] is not None]
        if pp:
            k = cohen_kappa_score([p[0] for p in pp], [p[1] for p in pp])
            print(f"  Cohen's κ (excl-None): {k:.3f}")
    except Exception as e:
        print(f"  κ failed: {e}")
    from collections import Counter
    confs = Counter()
    for r in results:
        confs[(r["gold_label"], r["mlx_label"])] += 1
    print(f"  confusion (gold→mlx): {dict(confs)}")
    # Bias diagnostics
    gold1 = [r for r in results if r["gold_label"] == 1 and r["mlx_label"] is not None]
    gold0 = [r for r in results if r["gold_label"] == 0 and r["mlx_label"] is not None]
    tpr = sum(1 for r in gold1 if r["mlx_label"] == 1) / len(gold1) if gold1 else 0
    tnr = sum(1 for r in gold0 if r["mlx_label"] == 0) / len(gold0) if gold0 else 0
    print(f"  TPR (recall on gold=1): {tpr*100:.1f}%   TNR (specificity on gold=0): {tnr*100:.1f}%")
    print(f"  mean latency: {sum(r['latency_s'] for r in results)/n:.2f}s")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
