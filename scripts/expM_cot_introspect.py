#!/usr/bin/env python3
"""Monolithic-CoT detection + step-by-step failure attribution.

Idea (user): ZS answer = the model's CONCLUSION; a CoT = its PROVING stage. We treat the CoT
as a HYPOTHESIS and audit it step-by-step to find WHICH stage the model gets wrong — so we know
which stage to design differently, instead of blindly sweeping the grid.

Per case:
  1. Run ONE CoT that does all detection functions in sequence (decompose question -> claims ->
     evidence -> compare -> conclude), emitting a labeled trace and a final FLAG.
  2. Parse the flag with the GPT-4o-mini semantic judge.
  3. ATTRIBUTE failure: GPT-4o reads the trace + the GOLD answer and, if the conclusion is wrong,
     names the FIRST broken step (question / claims / evidence / compare / conclude).

Output: the failure-stage distribution (which stage is the weak link) + example traces.

Usage: python scripts/expM_cot_introspect.py --n-wrong 40 --n-correct 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa
import expK_cascade_collect as K  # noqa
from llm_audit import set_ledger  # noqa

MONO_SYS = "You are a careful clinician checking whether an answer to a question about a discharge note is correct. Reason step by step and show every step."
MONO_USER = """Discharge note:
{note}

Question:
{question}

Answer given:
{answer}

Check whether the answer is correct, in these EXPLICIT labeled steps:
STEP1_QUESTION: what exactly does the question ask for, and what facts are required to answer it?
STEP2_CLAIMS: what does the given answer actually claim, point by point?
STEP3_EVIDENCE: for each required fact and each claim, what does the note actually say? Quote it. If the note is silent, say "SILENT".
STEP4_COMPARE: classify each claim SUPPORTED / CONTRADICTED / SILENT. A required fact the note states differently = CONTRADICTED; a fact simply not mentioned = SILENT (NOT an error).
STEP5_PROBLEM: if wrong, state exactly what is wrong or missing and the note-supported answer; if correct, say so.

Then output one final line:
FLAG: YES (real error or missing required fact) or FLAG: NO (answer is fine)."""

ATTRIB_SYS = "You audit a model's step-by-step reasoning that checked a clinical answer, comparing to the ground truth, to find WHERE the reasoning first went wrong."
ATTRIB_USER = """Question:
{question}

Answer the model was checking:
{answer}

GROUND-TRUTH correct answer:
{gold}

The answer was actually {truth} for the question, so a correct check should conclude FLAG: {expected}.

The model's step-by-step check:
<<<TRACE
{trace}
TRACE>>>

The model concluded FLAG: {model_flag}.

If the conclusion is WRONG, name the FIRST step where its reasoning broke:
- STEP1_QUESTION: misread what the question requires
- STEP2_CLAIMS: misread what the answer claims
- STEP3_EVIDENCE: missed or misquoted the note evidence
- STEP4_COMPARE: had the evidence but classified wrong (called SILENT what is CONTRADICTED, or vice versa)
- STEP5_CONCLUDE: steps were right but the final FLAG contradicts its own analysis
- NONE: the conclusion was actually correct
Reply JSON only: {{"conclusion_correct": true|false, "failure_step": "STEP1_QUESTION|STEP2_CLAIMS|STEP3_EVIDENCE|STEP4_COMPARE|STEP5_CONCLUDE|NONE", "why": "one sentence"}}"""


def process_one(row, port):
    out = {k: row[k] for k in ["fold", "idx", "stored_label"]}
    trace = P2.vllm_chat(MONO_SYS, MONO_USER.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500]), port, 1100, 0.0, tag="mono.cot")
    out["trace"] = trace
    flagged = K.llm_flag(trace, K.parse_flag(trace), "mono")
    out["flagged"] = flagged
    truth_wrong = (row["stored_label"] == 0)
    expected = "YES" if truth_wrong else "NO"
    out["correct_conclusion"] = (flagged == truth_wrong)
    # attribution (always run; NONE if conclusion correct)
    aj = P2.gpt("gpt-4o-mini", ATTRIB_SYS, ATTRIB_USER.format(
        question=row["question"], answer=row["original_answer"][:1500], gold=row["ground_truth"][:800],
        truth=("WRONG" if truth_wrong else "CORRECT"), expected=expected,
        trace=trace[:6000], model_flag=("YES" if flagged else "NO")), 200, 0.0, True, "attrib")
    try:
        a = json.loads(aj)
    except Exception:
        a = {"failure_step": "PARSE_ERR", "why": aj[:100]}
    out["failure_step"] = a.get("failure_step", "?")
    out["why"] = a.get("why", "")
    return out


def summarize(recs):
    W = [r for r in recs if r["stored_label"] == 0]
    C = [r for r in recs if r["stored_label"] == 1]
    rec_w = sum(1 for r in W if r["flagged"]) / max(1, len(W))
    of_c = sum(1 for r in C if r["flagged"]) / max(1, len(C))
    print(f"\n=== monolithic CoT detection: {len(W)} wrong + {len(C)} correct ===")
    print(f"recall on wrong: {sum(1 for r in W if r['flagged'])}/{len(W)} = {rec_w*100:.0f}%")
    print(f"over-flag on correct: {sum(1 for r in C if r['flagged'])}/{len(C)} = {of_c*100:.0f}%")
    print()
    print("FAILURE-STAGE ATTRIBUTION (where the CoT first went wrong):")
    print("  -- on WRONG cases the CoT MISSED (false negatives = lost recall):")
    miss = Counter(r["failure_step"] for r in W if not r["flagged"])
    for s, c in miss.most_common():
        print(f"     {s:16} {c}")
    print("  -- on CORRECT cases the CoT OVER-FLAGGED (false positives):")
    over = Counter(r["failure_step"] for r in C if r["flagged"])
    for s, c in over.most_common():
        print(f"     {s:16} {c}")
    print("\nThis says which STEP to redesign: dominant failure_step on misses = the recall bottleneck.")


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
    out_dir = PROJECT_ROOT / "runs" / "expM_cot_introspect" / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expM_cot_introspect", served=P2.served_model_id(args.port))
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
    print(f"\ntraces saved: {out_dir/'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
