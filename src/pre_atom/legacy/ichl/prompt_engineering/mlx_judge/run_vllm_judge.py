"""Run the C3e judge prompt on a vLLM-served candidate model.

Usage:
    PYTHONPATH=src .venv/bin/python -m ichl.prompt_engineering.mlx_judge.run_vllm_judge \
        --model Magistral-Small-2509-AWQ \
        --output Magistral_Small_2509_dev50.jsonl \
        --n 50 \
        --max-tokens 4096 \
        [--no-thinking-prefix]

Applies truncation_detector per Claude: Principle: Truncation Detection on Every LLM Output.
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
OUT_DIR = ROOT / "output" / "ichl" / "mlx_judge" / "vllm_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL = "http://localhost:8003/v1"
LM_STUDIO_URL = "http://192.168.68.107:1234/v1"

C3E_RULES = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 only if all specific claims align with the ground truth and the answer directly addresses the question.",
    "Output 0 if the answer includes additional, incorrect information not present in the ground truth.",
    "Output 1 if the answer provides correct additional context that does not contradict the ground truth.",
]
SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."


def build_user(item, note):
    rules = "\n".join(f"{i}. {r}" for i, r in enumerate(C3E_RULES, 1))
    return f"""When judging, apply the following rules:

{rules}

DISCHARGE SUMMARY:
{note}

QUESTION:
{item["question"]}

CORRECT ANSWER (Ground Truth):
{item["ground_truth"]}

MODEL'S ANSWER:
{item["model_answer"]}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect"""


def parse_01(text):
    if not text: return None
    s = text.strip()
    if not s: return None
    m = re.search(r"[01]", s)
    return int(m.group(0)) if m else None


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
    client, model, item, note, max_tokens = args
    user = build_user(item, note)
    t0 = time.monotonic()
    resp = call(client, model, SYSTEM, user, max_tokens)
    lat = time.monotonic() - t0
    if resp is None:
        return {**{k: item.get(k) for k in ["target", "patient_id", "fold_id", "binary_correct"]},
                "gold_label": int(item["binary_correct"]), "mlx_label": None,
                "content": "", "reasoning_content": "", "finish_reason": "ERROR",
                "completion_tokens": None, "prompt_tokens": None,
                "latency_s": round(lat, 2), "truncation_report": None}
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    fin = resp.choices[0].finish_reason
    usage = resp.usage
    label = parse_01(content)
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
        "content": content, "reasoning_content_tail": reasoning_content[-300:] if reasoning_content else "",
        "finish_reason": fin,
        "completion_tokens": usage.completion_tokens if usage else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "latency_s": round(lat, 2),
        "truncation_report": report.as_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-url", default=VLLM_URL, help="API base URL (default vLLM 8003; use LM_STUDIO_URL for Mac)")
    args = ap.parse_args()

    print(f"Loading dev + notes…")
    dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
    notes = build_note_lookup()
    # Stratified n: balance gold 0/1
    g0 = [d for d in dev if d["binary_correct"] == 0]
    g1 = [d for d in dev if d["binary_correct"] == 1]
    half = args.n // 2
    sample = g0[:half] + g1[:args.n - half]
    print(f"  sampled n={len(sample)}  (g0={half}, g1={args.n - half})")

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    print(f"Model: {args.model}  max_tokens={args.max_tokens}  workers={args.workers}")

    out_path = OUT_DIR / args.output
    tasks = [(client, args.model, item, notes.get(str(item["patient_id"]), ""), args.max_tokens)
             for item in sample]

    t0 = time.monotonic()
    results = []
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(judge_one, tasks), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            results.append(r)
            if i % 10 == 0:
                dt = time.monotonic() - t0
                print(f"  {i}/{len(sample)}  elapsed={dt:.0f}s  eta={dt*(len(sample)-i)/i:.0f}s")
    elapsed = time.monotonic() - t0

    # Metrics
    n = len(results)
    correct = sum(1 for r in results if r["mlx_label"] == r["gold_label"])
    none_cnt = sum(1 for r in results if r["mlx_label"] is None)
    truncated_cert = sum(1 for r in results if (r.get("truncation_report") or {}).get("is_truncated_certain"))
    parsed = n - none_cnt
    correct_parsed = sum(1 for r in results if r["mlx_label"] is not None and r["mlx_label"] == r["gold_label"])
    print(f"\n=== {args.model}  n={n}  wall={elapsed:.0f}s ===")
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
    print(f"  mean latency: {sum(r['latency_s'] for r in results)/n:.1f}s")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
