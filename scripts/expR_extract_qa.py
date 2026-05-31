#!/usr/bin/env python3
"""V1 extract (item-based, no question) -> QA -> prove vs gold.

User correction: the extract builds the fact list FROM the located items (which already carry the
direction) + the retrieved real spans. It does NOT refer to the question. The question returns only
at QA. Steps:
  1. items = decompose look-ups (reused from expO)
  2. retrieve real spans (GTR pooled, k)            [proven ~95% of ceiling]
  3. EXTRACT (items + spans, NO question): per item, what the note says, exact values, "not stated"
  4. QA (question + extracted facts, NO ZS): re-answer
  5. judge QA vs gold -> fix-rate on wrong, break-rate on correct

Judge-free-ish: correctness is P2.judge vs the gold answer (the project's standard correctness judge).
Usage: python scripts/expR_extract_qa.py --k 10 --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import topk_spans, get_embedder  # noqa
from llm_audit import set_ledger  # noqa

# EXTRACT — comprehensive normalization of the facts IN the spans (NOT an omit stage, NO question)
EXTRACT_SYS = "You turn raw discharge-note sentences into a clean, complete list of the facts they state."
EXTRACT_USER = """These note sentences were retrieved as relevant to these topics:
{items}

Note sentences (real, from the discharge note):
{spans}

Write a clean, normalized list of ALL the distinct facts stated in these sentences. Each fact = one clear statement, quoting values (numbers, dates, names, doses) EXACTLY. KEEP every piece of information — do NOT omit, do NOT summarize away, do NOT judge. Where a sentence clarifies an abbreviation or detail, include it. Output a plain numbered list."""

# QA — question + extracted facts, NO ZS answer
QA_SYS = "You answer a question about a patient using only the provided facts."
QA_USER = """Question:
{question}

Facts from the note:
{facts}

Answer the question using only these facts. Be specific and complete."""


def render(spans):
    return "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans)) if spans else "(none)"


def process_one(row, lookups_raw, k, port):
    out = {kk: row[kk] for kk in ["fold", "idx", "stored_label"]}
    items = [ln.strip(" -*0123456789.").strip() for ln in lookups_raw.splitlines() if ln.strip()]
    items = [x for x in items if len(x) > 4][:15]
    spans = topk_spans(row["note"], items, k=k)
    facts = P2.vllm_chat(EXTRACT_SYS, EXTRACT_USER.format(items="\n".join(f"- {x}" for x in items), spans=render(spans)), port, 800, 0.0, tag="extract")
    qa = P2.vllm_chat(QA_SYS, QA_USER.format(question=row["question"], facts=facts[:3500]), port, 500, 0.0, tag="qa")
    out["facts"], out["qa"] = facts, qa
    out["qa_correct"] = P2.judge(row, qa).get("label")
    out["zs_correct"] = 1 if row["stored_label"] == 1 else 0
    return out


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
    get_embedder()  # pre-load GTR (avoid concurrent lazy-init race)
    out_dir = PROJECT_ROOT / "runs" / "expR_extract_qa" / f"qwen25_k{args.k}_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expR_extract_qa", served=P2.served_model_id(args.port))
    print(f"extract->QA on {len(keys)} cases (40 wrong + 20 correct), k={args.k}", flush=True)
    recs = []
    f = (out_dir / "records.jsonl").open("w")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[key], looked[key], args.k, args.port) for key in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); recs.append(r)
            f.write(json.dumps(r, default=str) + "\n"); f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)
    f.close()
    W = [r for r in recs if r["stored_label"] == 0]
    C = [r for r in recs if r["stored_label"] == 1]
    fixes = sum(1 for r in W if r["qa_correct"] == 1)
    breaks = sum(1 for r in C if r["qa_correct"] == 0)
    print(f"\n=== extract(item-based)->QA, k={args.k} ===")
    print(f"FIX-RATE  (wrong -> QA correct):   {fixes}/{len(W)} = {fixes/max(1,len(W))*100:.0f}%   <- oracle-hint correction ~60%")
    print(f"BREAK-RATE (correct -> QA wrong):  {breaks}/{len(C)} = {breaks/max(1,len(C))*100:.0f}%")
    print(f"\nnet over this sample (unweighted): +{fixes} fixed, -{breaks} broken")
    print(f"saved: {out_dir/'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
