#!/usr/bin/env python3
"""Retrieval-strength sweep: does MORE / LONGER chunks raise the ceiling? GPT as the extractor.

GPT (better extractor) isolates retrieval completeness from model skill. If GPT's fix-rate rises
and break-rate falls as we retrieve more (higher k) or longer (sentence + neighbors) chunks, then
retrieval was the limit. Compares: k=10 (baseline), k=25, k=10-with-context (each span expanded to
its neighbors), on the same wrong+correct cases.

Usage: python scripts/expV_retrieval_sweep.py --concurrency 4
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
from note_span_index import topk_spans, get_embedder, split_sentences  # noqa
from llm_audit import set_ledger  # noqa

QA_SYS = "You answer a clinical question using ONLY the provided note excerpts. Do not use outside knowledge."
QA_USER = "Note excerpts:\n{spans}\n\nQuestion:\n{question}\n\nAnswer using only these excerpts. Be specific and quote values exactly."


def expand(note, spans):
    """Lengthen each retrieved sentence to include its immediate neighbors in the note."""
    sents = split_sentences(note)
    idx = {s: i for i, s in enumerate(sents)}
    out = []
    for sp in spans:
        i = idx.get(sp["sentence"])
        if i is None:
            out.append(sp["sentence"]); continue
        out.append(" ".join(sents[max(0, i - 1):i + 2]))
    return out


def ask(spans_txt, q):
    return P2.gpt("gpt-4o", QA_SYS, QA_USER.format(spans="\n".join(f"[{i+1}] {s}" for i, s in enumerate(spans_txt)), question=q), 350, 0.0, False, "sweep")


def process_one(row, looked, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:12]
    queries = [q] + items
    s10 = topk_spans(row["note"], queries, k=10)
    s25 = topk_spans(row["note"], queries, k=25)
    out = {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"]}
    out["k10"] = P2.judge(row, ask([s["sentence"] for s in s10], q)).get("label")
    out["k25"] = P2.judge(row, ask([s["sentence"] for s in s25], q)).get("label")
    out["k10ctx"] = P2.judge(row, ask(expand(row["note"], s10), q)).get("label")
    return out


def report(recs, key):
    W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
    fixes = sum(1 for r in W if r[key] == 1); breaks = sum(1 for r in C if r[key] == 0)
    print(f"  {key:8} FIX {fixes}/{len(W)}={fixes/max(1,len(W))*100:.0f}%   BREAK {breaks}/{len(C)}={breaks/max(1,len(C))*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expO", default=str(PROJECT_ROOT / "runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))
    args = ap.parse_args()
    looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open(args.expO))}
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    keys = list(looked)
    get_embedder()
    out_dir = PROJECT_ROOT / "runs" / "expV_retrieval_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expV_retrieval_sweep", served=P2.served_model_id(args.port))
    print(f"retrieval sweep (GPT extractor) on {len(keys)} cases", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[k], looked[k], args.port) for k in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    print("\n=== retrieval strength (GPT extractor; more vs longer chunks) ===")
    for key in ["k10", "k25", "k10ctx"]:
        report(recs, key)
    print("\nrising FIX / falling BREAK with more (k25) or longer (k10ctx) => retrieval was the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
