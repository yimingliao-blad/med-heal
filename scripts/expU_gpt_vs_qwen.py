#!/usr/bin/env python3
"""Oracle-model comparison: GPT vs Qwen doing focused-QA on the SAME retrieved spans.

The cerclage break was a MODEL failure (the fact was retrieved at rank 2 but Qwen omitted it), not
retrieval. So: feed the SAME top-k spans to GPT-4o and to Qwen, both answer, judge both vs gold.
If GPT >> Qwen, the bottleneck is the small model using the focused context, not retrieval.

Usage: python scripts/expU_gpt_vs_qwen.py --k 10 --concurrency 4
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

QA_SYS = "You answer a clinical question using ONLY the provided note excerpts. Do not use outside knowledge."
QA_USER = """Note excerpts (retrieved from the patient's discharge note):
{spans}

Question:
{question}

Answer the question using only these excerpts. Be specific and quote values exactly."""


def process_one(row, looked, k, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:12]
    spans = topk_spans(row["note"], [q] + items, k=k)
    sp = "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans))
    user = QA_USER.format(spans=sp, question=q)
    qwen = P2.vllm_chat(QA_SYS, user, port, 350, 0.0, tag="qa.qwen")
    gpt = P2.gpt("gpt-4o", QA_SYS, user, 350, 0.0, False, "qa.gpt")
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "qwen_correct": P2.judge(row, qwen).get("label"),
            "gpt_correct": P2.judge(row, gpt).get("label")}


def report(recs, model):
    W = [r for r in recs if r["stored_label"] == 0]
    C = [r for r in recs if r["stored_label"] == 1]
    key = f"{model}_correct"
    fixes = sum(1 for r in W if r[key] == 1)
    breaks = sum(1 for r in C if r[key] == 0)
    print(f"  {model:5} FIX {fixes}/{len(W)}={fixes/max(1,len(W))*100:.0f}%   BREAK {breaks}/{len(C)}={breaks/max(1,len(C))*100:.0f}%")


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
    out_dir = PROJECT_ROOT / "runs" / "expU_gpt_vs_qwen" / f"k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expU_gpt_vs_qwen", served=P2.served_model_id(args.port))
    print(f"GPT vs Qwen focused-QA (SAME spans) on {len(keys)} cases, k={args.k}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[key], looked[key], args.k, args.port) for key in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    print("\n=== focused-QA on the SAME retrieved spans (k={}) ===".format(args.k))
    report(recs, "qwen")
    report(recs, "gpt")
    print("\nGPT >> Qwen => bottleneck is the small model using the focused context, not retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
