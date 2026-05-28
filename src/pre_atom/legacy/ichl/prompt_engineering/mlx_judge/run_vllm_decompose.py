"""M6 atomic decomposition: 3 orthogonal sub-judgments per case, then aggregate.

Each item triggers 3 separate API calls, each asking ONE focused yes/no:
  S1 factual    — Are all specific facts consistent with ground truth?
  S2 addresses  — Does the answer directly address the question?
  S3 no_spurious — Does the answer avoid spurious/incorrect additions?

For each item we record (s1, s2, s3) and report 3 aggregations:
  STRICT     — final=1 iff s1==s2==s3==1   (mirrors C3E's "1 only if everything aligns")
  MAJORITY   — final=1 iff sum >= 2         (noise reduction)
  LENIENT    — final=1 iff sum >= 1         (rare; mostly diagnostic)

Truncation detector applied per Claude: Principle: Truncation Detection on Every LLM Output.
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
SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

SUB_PROMPTS = {
    "factual": (
        "Sub-question: Is EVERY specific clinical fact in MODEL'S ANSWER (medication, dose, "
        "diagnosis, timing, procedure, lab value) consistent with the CORRECT ANSWER?\n"
        "Paraphrases, synonyms, abbreviations, generic-vs-brand names, and equivalent units all "
        "count as consistent. Penalize ONLY when the underlying clinical fact differs.\n"
        "Respond with ONLY a single digit:\n"
        "1 = all specific facts consistent\n"
        "0 = at least one specific fact contradicts the ground truth"
    ),
    "addresses": (
        "Sub-question: Does MODEL'S ANSWER directly address what the QUESTION asks?\n"
        "Penalize if the answer addresses a different aspect than the question, hedges with "
        "multiple uncommitted options, or omits the specific fact the question asks about.\n"
        "Respond with ONLY a single digit:\n"
        "1 = directly addresses the question\n"
        "0 = different aspect / hedges / omits requested fact"
    ),
    "no_spurious": (
        "Sub-question: Does MODEL'S ANSWER avoid introducing additional INCORRECT information "
        "not present in the CORRECT ANSWER?\n"
        "Correct supporting context that does not contradict the ground truth is fine. Penalize "
        "ONLY when the addition introduces a clinically significant error or contradiction.\n"
        "Respond with ONLY a single digit:\n"
        "1 = no spurious incorrect additions\n"
        "0 = answer adds clinically incorrect information"
    ),
}


def build_user(item, note, sub_key):
    return f"""DISCHARGE SUMMARY:
{note}

QUESTION:
{item["question"]}

CORRECT ANSWER (Ground Truth):
{item["ground_truth"]}

MODEL'S ANSWER:
{item["model_answer"]}

{SUB_PROMPTS[sub_key]}"""


def parse_01(text):
    if not text:
        return None
    m = re.search(r"[01]", text.strip())
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


def call_one(args):
    client, model, item, note, sub_key, max_tokens = args
    user = build_user(item, note, sub_key)
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=max_tokens,
        )
    except Exception as e:
        return {"sub_key": sub_key, "patient_id": item["patient_id"],
                "fold_id": item["fold_id"], "target": item["target"],
                "label": None, "content": "", "error": str(e)[:200],
                "latency_s": round(time.monotonic() - t0, 2)}
    lat = time.monotonic() - t0
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    fin = resp.choices[0].finish_reason
    usage = resp.usage
    label = parse_01(content)
    report = detect_truncation(
        raw_response=content, text_clean=content,
        finish_reason=fin,
        usage={"completion_tokens": usage.completion_tokens if usage else None,
               "prompt_tokens": usage.prompt_tokens if usage else None},
        max_tokens=max_tokens, target=model, sub_variant=f"decompose_{sub_key}",
    )
    return {
        "sub_key": sub_key, "target": item["target"], "patient_id": item["patient_id"],
        "fold_id": item["fold_id"], "label": label, "content": content,
        "finish_reason": fin,
        "completion_tokens": usage.completion_tokens if usage else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "latency_s": round(lat, 2),
        "truncation_certain": report.is_truncated_certain,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-url", default=VLLM_URL)
    args = ap.parse_args()

    print("Loading dev + notes…")
    dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
    notes = build_note_lookup()
    g0 = [d for d in dev if d["binary_correct"] == 0]
    g1 = [d for d in dev if d["binary_correct"] == 1]
    half = args.n // 2
    sample = g0[:half] + g1[:args.n - half]
    print(f"  n={len(sample)}  (g0={half}, g1={args.n - half})")

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    out_path = OUT_DIR / args.output

    # Build 3× tasks (one per sub-question per item)
    tasks = []
    for it in sample:
        note = notes.get(str(it["patient_id"]), "")
        for sk in ("factual", "addresses", "no_spurious"):
            tasks.append((client, args.model, it, note, sk, args.max_tokens))
    print(f"  total sub-calls: {len(tasks)}  (3 per item)")

    t0 = time.monotonic()
    raw = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(call_one, tasks), 1):
            raw.append(r)
            if i % 100 == 0:
                dt = time.monotonic() - t0
                eta = dt * (len(tasks) - i) / i
                print(f"  {i}/{len(tasks)}  elapsed={dt:.0f}s  eta={eta:.0f}s")
    elapsed = time.monotonic() - t0
    print(f"DONE in {elapsed:.0f}s")

    # Aggregate per item
    by_item = {}
    for r in raw:
        key = (r["target"], r["patient_id"], r["fold_id"])
        by_item.setdefault(key, {})[r["sub_key"]] = r

    rows = []
    for it in sample:
        key = (it["target"], it["patient_id"], it["fold_id"])
        subs = by_item.get(key, {})
        s1 = subs.get("factual", {}).get("label")
        s2 = subs.get("addresses", {}).get("label")
        s3 = subs.get("no_spurious", {}).get("label")
        any_none = any(s is None for s in (s1, s2, s3))
        if any_none:
            agg_strict = agg_majority = agg_lenient = None
        else:
            agg_strict = 1 if (s1 == 1 and s2 == 1 and s3 == 1) else 0
            agg_majority = 1 if (s1 + s2 + s3) >= 2 else 0
            agg_lenient = 1 if (s1 + s2 + s3) >= 1 else 0
        rows.append({
            "target": it["target"], "patient_id": it["patient_id"], "fold_id": it["fold_id"],
            "gold_label": int(it["binary_correct"]),
            "s1_factual": s1, "s2_addresses": s2, "s3_no_spurious": s3,
            "agg_strict": agg_strict, "agg_majority": agg_majority, "agg_lenient": agg_lenient,
            "raw_subs": subs,
        })

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "raw_subs"}, default=str) + "\n")
    raw_path = OUT_DIR / args.output.replace(".jsonl", "_raw.jsonl")
    with raw_path.open("w") as f:
        for r in raw:
            f.write(json.dumps(r, default=str) + "\n")

    # Metrics for each aggregation
    from sklearn.metrics import cohen_kappa_score
    from collections import Counter
    n = len(rows)
    print(f"\n=== {args.model}  variant=M6_decompose  n={n}  wall={elapsed:.0f}s ===")
    for agg_name in ("agg_strict", "agg_majority", "agg_lenient"):
        labels = [r[agg_name] for r in rows]
        none_cnt = sum(1 for x in labels if x is None)
        correct = sum(1 for r in rows if r[agg_name] == r["gold_label"])
        parsed_pp = [(r["gold_label"], r[agg_name]) for r in rows if r[agg_name] is not None]
        kappa = cohen_kappa_score([p[0] for p in parsed_pp], [p[1] for p in parsed_pp]) if parsed_pp else None
        confs = Counter()
        for r in rows:
            confs[(r["gold_label"], r[agg_name])] += 1
        gold1 = [r for r in rows if r["gold_label"] == 1 and r[agg_name] is not None]
        gold0 = [r for r in rows if r["gold_label"] == 0 and r[agg_name] is not None]
        tpr = sum(1 for r in gold1 if r[agg_name] == 1) / len(gold1) if gold1 else 0
        tnr = sum(1 for r in gold0 if r[agg_name] == 0) / len(gold0) if gold0 else 0
        print(f"\n  [{agg_name}]")
        print(f"    agreement: {correct}/{n} = {100*correct/n:.1f}%  (None={none_cnt})")
        print(f"    Cohen's \u03ba (excl-None): {kappa:.3f}" if kappa is not None else "    \u03ba: n/a")
        print(f"    TPR={tpr*100:.1f}%   TNR={tnr*100:.1f}%")
        print(f"    confusion: {dict(confs)}")
    # Sub-score breakdown
    print(f"\n  Per-sub-question hit rates (vs gold):")
    for sk in ("s1_factual", "s2_addresses", "s3_no_spurious"):
        gold1 = [r for r in rows if r["gold_label"] == 1 and r[sk] is not None]
        gold0 = [r for r in rows if r["gold_label"] == 0 and r[sk] is not None]
        # For gold=1, expected sub=1 (consistent / addresses / no spurious)
        # For gold=0, expected sub varies — we compare overall sub-1 rate by gold
        if gold1:
            r1 = sum(1 for r in gold1 if r[sk] == 1) / len(gold1)
        else:
            r1 = 0
        if gold0:
            r0 = sum(1 for r in gold0 if r[sk] == 1) / len(gold0)
        else:
            r0 = 0
        print(f"    {sk}: P(sub=1 | gold=1)={r1*100:.1f}%   P(sub=1 | gold=0)={r0*100:.1f}%   gap={(r1-r0)*100:+.1f}pp")
    print(f"\nSaved aggregated: {out_path}")
    print(f"Saved raw subs:   {raw_path}")


if __name__ == "__main__":
    main()
