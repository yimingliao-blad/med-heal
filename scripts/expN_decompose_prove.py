#!/usr/bin/env python3
"""STAGE 1 ONLY — decompose ZS into a checklist, and PROVE it covers the real error.

Method (user): build the detection pipeline stage by stage and prove each before the next.
Stage 1 = DECOMPOSE: from the question + ZS answer, produce a checklist of specific, independently
checkable items — both the facts the answer asserts AND the facts the question requires (so an
OMISSION becomes an explicit item). No judging yet, no note lookup yet (that is stage 2).

PROOF for stage 1: on the WRONG cases, would verifying this checklist CATCH the error? i.e. is
there an item whose check would reveal the answer is wrong/incomplete? GPT-4o (with the gold
answer) judges coverage. If coverage is high, the checklist surfaces the mistake as a checkable
item — the ceiling for the new architecture's recall. (The monolithic CoT only 'found' 32%.)
Also run on CORRECT cases to watch for over-decomposition (manufacturing false must-checks).

Usage: python scripts/expN_decompose_prove.py --n-wrong 40 --n-correct 20
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa
from llm_audit import set_ledger  # noqa

DECOMP_SYS = "You turn a question and a proposed answer into a checklist of specific, independently verifiable factual items to look up in a clinical note."
DECOMP_USER = """Question:
{question}

A proposed answer:
{answer}

Break this into a CHECKLIST of the specific factual items that must each be TRUE and that together make the answer COMPLETE for the question. Include BOTH:
 (1) every distinct fact the answer asserts — each value, date, name, dose, count, finding;
 (2) every fact the QUESTION requires a complete answer to cover — even if the proposed answer did NOT mention it (these guard against omissions).
Write each item as a short, checkable statement of what must be confirmed in the note. Number them. Do NOT judge correctness — just list what to check."""

COVER_SYS = "You judge whether a verification checklist would catch a known error in an answer."
COVER_USER = """Question:
{question}

Proposed answer (this answer is WRONG):
{answer}

The TRUE correct answer:
{gold}

The error is the difference between the proposed answer and the true answer (a wrong value, or a missing required fact).

Checklist generated to verify the proposed answer:
{checklist}

If each checklist item were looked up in the note, would that CATCH this error — is there an item whose check would reveal the proposed answer is wrong or incomplete?
Reply JSON only: {{"covered": true|false, "which_item": "<the item number/text or NONE>", "why": "one sentence"}}"""


def process_one(row, port):
    out = {k: row[k] for k in ["fold", "idx", "stored_label"]}
    checklist = P2.vllm_chat(DECOMP_SYS, DECOMP_USER.format(question=row["question"], answer=row["original_answer"][:1500]), port, 700, 0.0, tag="decompose")
    out["checklist"] = checklist
    out["n_items"] = sum(1 for ln in checklist.splitlines() if ln.strip() and ln.strip()[0].isdigit())
    if row["stored_label"] == 0:  # wrong -> prove coverage of the real error
        aj = P2.gpt("gpt-4o-mini", COVER_SYS, COVER_USER.format(question=row["question"], answer=row["original_answer"][:1500], gold=row["ground_truth"][:800], checklist=checklist[:3000]), 200, 0.0, True, "cover")
        try:
            a = json.loads(aj)
        except Exception:
            a = {"covered": None, "which_item": "?", "why": aj[:80]}
        out["covered"] = a.get("covered")
        out["which_item"] = a.get("which_item", "")
        out["cover_why"] = a.get("why", "")
    return out


def summarize(recs):
    W = [r for r in recs if r["stored_label"] == 0]
    C = [r for r in recs if r["stored_label"] == 1]
    cov = sum(1 for r in W if r.get("covered") is True)
    nocov = sum(1 for r in W if r.get("covered") is False)
    avg_items_w = sum(r["n_items"] for r in W) / max(1, len(W))
    avg_items_c = sum(r["n_items"] for r in C) / max(1, len(C))
    print(f"\n=== STAGE 1 decomposition — coverage of the real error (wrong cases) ===")
    print(f"COVERED {cov}/{len(W)} = {cov/max(1,len(W))*100:.0f}%   (not covered {nocov})")
    print(f"avg checklist items: wrong {avg_items_w:.1f}, correct {avg_items_c:.1f}")
    print("\nThis is the recall CEILING for the new architecture: if the checklist doesn't contain")
    print("an item for the real error, no downstream blind-verify can catch it. (Monolithic CoT found 32%.)")
    print("\n-- misses (error NOT covered by any checklist item): --")
    for r in W:
        if r.get("covered") is False:
            print(f"   why: {r.get('cover_why','')[:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wrong", type=int, default=40)
    ap.add_argument("--n-correct", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    if any(not (r.get("note") or "").strip() for r in sample):
        raise RuntimeError("ABORT empty notes")
    out_dir = PROJECT_ROOT / "runs" / "expN_decompose" / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expN_decompose", served=P2.served_model_id(args.port))
    print(f"NOTE GUARD OK: {len(sample)} notes", flush=True)
    recs = []
    f = (out_dir / "records.jsonl").open("w")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); recs.append(r)
            f.write(json.dumps(r, default=str) + "\n"); f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)
    f.close()
    summarize(recs)
    print(f"\nsaved: {out_dir/'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
