#!/usr/bin/env python3
"""Focused QA: embedding-retrieved PARTIAL notes -> Qwen answers. No whole note.

User direction: do NOT retrieve from full notes. Embedding-retrieve the few relevant spans and let
Qwen answer from ONLY those. Long text -> hallucination; the fix is to make Qwen focus on few but
relevant content. So Qwen never sees the 24k note, only the top-k retrieved spans.

Per case:
  1. retrieve top-k GTR spans (queries = question + decompose items)   [partial, focused]
  2. Qwen answers the question using ONLY those spans                  [grounded -> no hallucination]
  3. judge the answer vs gold (project judge): fix-rate on wrong, break-rate on correct
  4. grounding check (gpt-4o-mini): is the answer supported by the spans? (hallucination guard)

Usage: python scripts/expT_focused_qa.py --k 10 --concurrency 4
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

QA_SYS = "You answer a clinical question using ONLY the provided note excerpts. Do not use outside knowledge; if the excerpts do not contain it, say what they do show."
QA_USER = """Note excerpts (retrieved from the patient's discharge note):
{spans}

Question:
{question}

Answer the question using only these excerpts. Be specific and quote values exactly."""

GROUND_SYS = "You check whether an answer is supported by provided note excerpts."
GROUND_USER = """Note excerpts:
{spans}

Answer:
{ans}

Is every claim in the answer supported by the excerpts (no invented facts)? Reply one word: GROUNDED or HALLUCINATED."""


def process_one(row, looked, k, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:12]
    spans = topk_spans(row["note"], [q] + items, k=k)
    sp = "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans))
    ans = P2.vllm_chat(QA_SYS, QA_USER.format(spans=sp, question=q), port, 350, 0.0, tag="fqa")
    label = P2.judge(row, ans).get("label")
    ground = P2.gpt("gpt-4o-mini", GROUND_SYS, GROUND_USER.format(spans=sp[:3500], ans=ans[:800]), 6, 0.0, False, "ground").upper()
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "qa_correct": label, "grounded": ("HALL" not in ground)}


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
    out_dir = PROJECT_ROOT / "runs" / "expT_focused_qa" / f"k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expT_focused_qa", served=P2.served_model_id(args.port))
    print(f"focused QA (partial notes, no whole note) on {len(keys)} cases, k={args.k}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[key], looked[key], args.k, args.port) for key in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    W = [r for r in recs if r["stored_label"] == 0]
    C = [r for r in recs if r["stored_label"] == 1]
    fixes = sum(1 for r in W if r["qa_correct"] == 1)
    breaks = sum(1 for r in C if r["qa_correct"] == 0)
    grounded = sum(1 for r in recs if r["grounded"])
    print(f"\n=== focused QA from embedding-retrieved partial notes (k={args.k}) ===")
    print(f"FIX-RATE   (wrong -> correct):  {fixes}/{len(W)} = {fixes/max(1,len(W))*100:.0f}%   (ZS=0 by definition; oracle-hint ~60%)")
    print(f"BREAK-RATE (correct -> wrong):  {breaks}/{len(C)} = {breaks/max(1,len(C))*100:.0f}%")
    print(f"GROUNDED   (no hallucination):  {grounded}/{len(recs)} = {grounded/len(recs)*100:.0f}%")
    print(f"\nnet over sample (unweighted): +{fixes} fixed, -{breaks} broken")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
