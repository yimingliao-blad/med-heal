#!/usr/bin/env python3
"""A/B: restrictive QA (#3) vs natural CoT answer prompt — both over the SAME retrieved spans.

User point: prompt #3 is over-restrictive ("Do not use outside knowledge; if the excerpts do not
contain it, say what they do show"), which may suppress the answer. Test a natural CoT-style answer
prompt (how the model normally answers) fed the SPANS instead of the whole note, on identical spans.

Usage: python scripts/expX_natural_qa.py --k 10 --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import topk_spans, get_embedder  # noqa
from llm_audit import set_ledger  # noqa

# A — restrictive #3
RESTR_SYS = "You answer a clinical question using ONLY the provided note excerpts. Do not use outside knowledge; if the excerpts do not contain it, say what they do show."
RESTR_USER = "Note excerpts:\n{spans}\n\nQuestion:\n{question}\n\nAnswer the question using only these excerpts. Be specific and quote values exactly."

# B — natural CoT answer over the spans
NAT_SYS = "You are a clinical expert answering a question about a patient based on information from their discharge note."
NAT_USER = "Information from the patient's discharge note:\n{spans}\n\nQuestion:\n{question}\n\nReason through the relevant information, then give a clear and complete answer."


def process_one(row, looked, k, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:14]
    spans = topk_spans(row["note"], [q] + items, k=k)
    sp = "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans))
    a = P2.vllm_chat(RESTR_SYS, RESTR_USER.format(spans=sp, question=q), port, 350, 0.0, tag="restr")
    b = P2.vllm_chat(NAT_SYS, NAT_USER.format(spans=sp, question=q), port, 500, 0.0, tag="nat")
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "restr_correct": P2.judge(row, a).get("label"),
            "nat_correct": P2.judge(row, b).get("label")}


def report(recs, key):
    W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
    fixes = sum(1 for r in W if r[key] == 1); breaks = sum(1 for r in C if r[key] == 0)
    print(f"  {key:14} FIX {fixes}/{len(W)}={fixes/max(1,len(W))*100:.0f}%   BREAK {breaks}/{len(C)}={breaks/max(1,len(C))*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expO", default=str(PROJECT_ROOT / "runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))
    args = ap.parse_args()
    looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open(args.expO))}
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    keys = list(looked)
    get_embedder()
    out_dir = PROJECT_ROOT / "runs" / "expX_natural_qa" / f"k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expX_natural_qa", served=P2.served_model_id(args.port))
    print(f"restrictive vs natural-CoT QA (same spans) on {len(keys)} cases, k={args.k}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[key], looked[key], args.k, args.port) for key in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    print("\n=== same spans, two answer prompts ===")
    report(recs, "restr_correct")
    report(recs, "nat_correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
