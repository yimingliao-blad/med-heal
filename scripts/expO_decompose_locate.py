#!/usr/bin/env python3
"""Decompose -> blind-locate -> compare detection, proven on recall vs ground truth.

The new architecture (proven stage by stage):
  1. DECOMPOSE (MERGED-ONE): Q + ZS answer -> look-up list (OPEN question-driven items +
     SPECIFIC answer-claim items). One prompt; open item leads, specifics follow.
  2. BLIND LOCATE: note + Q + look-up list, WITHOUT the ZS answer -> what the note actually
     says per item (or "not stated"). Blind, so the model can't rationalize its own answer.
  3. COMPARE: located evidence vs the ZS answer -> wrong value / omission = the error. FLAG.

Metric (judge-free headline): recall = flagged / wrong; over-flag = flagged / correct; and the
idea-(1) GATE signal: of FLAG=NO cases, how many are actually correct (clean-verdict precision).
Flag is parsed by the GPT-4o-mini semantic judge (consistent with the rest of the pipeline);
recall/over-flag are vs ground-truth stored_label, so no judge in the measurement itself.

Usage: python scripts/expO_decompose_locate.py --n-wrong 40 --n-correct 20
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
import expK_cascade_collect as K  # noqa  (llm_flag, parse_flag, FLAG_TAIL)
from llm_audit import set_ledger  # noqa

# 1. DECOMPOSE (merged-one: open + specific look-ups)
DECOMP_SYS = "You list what to look up in a discharge note to fully check an answer to a question."
DECOMP_USER = """Question:
{question}

Answer that was given:
{answer}

Produce ONE numbered list of things to look up in the note, of two kinds together:
- OPEN look-ups for what the QUESTION itself requires, so nothing required is missed (for example "all treatments given during the stay", "all infections diagnosed");
- SPECIFIC look-ups for each distinct fact the ANSWER states (each value, date, name, dose, finding).
Output only the numbered list, one short look-up phrase per line. Do not judge whether the answer is right."""

# 2. BLIND LOCATE (note + lookups, NO ZS answer)
LOCATE_SYS = "You read a discharge note and locate specific information, quoting the note exactly."
LOCATE_USER = """Discharge note:
{note}

Question being investigated:
{question}

For each item below, find what the note actually says and quote the relevant sentence(s). If the note does not state it, say "not stated". Keep each answer short.
Items to look up:
{lookup}"""

# 3. COMPARE located evidence vs the ZS answer
COMPARE_SYS = "You decide whether a proposed answer is consistent with what was actually found in the note."
COMPARE_USER = """Question:
{question}

Proposed answer:
{answer}

What was actually found in the note (by looking up the relevant items, without seeing the proposed answer):
{located}

Compare the proposed answer against what was found in the note. Does the proposed answer contain a WRONG value, or leave out something the note shows the question requires (an OMISSION)?
If there is a real discrepancy, state what is wrong or missing and what the note shows instead.""" + K.FLAG_TAIL


def process_one(row, port):
    out = {k: row[k] for k in ["fold", "idx", "stored_label", "question", "ground_truth"]}
    q, a, note = row["question"], row["original_answer"][:1500], row["note"][:24000]
    lookup = P2.vllm_chat(DECOMP_SYS, DECOMP_USER.format(question=q, answer=a), port, 700, 0.0, tag="o.decompose")
    located = P2.vllm_chat(LOCATE_SYS, LOCATE_USER.format(note=note, question=q, lookup=lookup[:2500]), port, 1100, 0.0, tag="o.locate")
    comp = P2.vllm_chat(COMPARE_SYS, COMPARE_USER.format(question=q, answer=a, located=located[:3500]), port, 700, 0.0, tag="o.compare")
    out["lookup"], out["located"], out["compare"] = lookup, located, comp
    out["flagged"] = K.llm_flag(comp, K.parse_flag(comp), "o_compare")
    return out


def summarize(recs):
    W = [r for r in recs if r["stored_label"] == 0]
    C = [r for r in recs if r["stored_label"] == 1]
    fw = sum(1 for r in W if r["flagged"])
    fc = sum(1 for r in C if r["flagged"])
    rec = fw / max(1, len(W))
    of = fc / max(1, len(C))
    prec = fw / max(1, fw + fc)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    # idea-(1) gate: of FLAG=NO, how many are actually correct (clean-verdict precision)
    clean = [r for r in recs if not r["flagged"]]
    clean_correct = sum(1 for r in clean if r["stored_label"] == 1)
    print(f"\n=== decompose->blind-locate->compare: {len(W)} wrong + {len(C)} correct ===")
    print(f"RECALL (flag a wrong):   {fw}/{len(W)} = {rec*100:.0f}%   <- vs blind 22-32%, union 92%")
    print(f"OVER-FLAG (flag correct): {fc}/{len(C)} = {of*100:.0f}%   <- vs blind 8-10%, union 74%")
    print(f"precision {prec*100:.0f}%   F1 {f1:.2f}")
    print(f"\nGATE signal (idea 1): of {len(clean)} cases it called CLEAN (FLAG=NO), {clean_correct} are actually correct "
          f"= {clean_correct/max(1,len(clean))*100:.0f}% clean-verdict precision")
    print("(high recall + high clean-precision = detection can double as the pre-correction gate)")


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
    out_dir = PROJECT_ROOT / "runs" / "expO_decompose_locate" / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expO_decompose_locate", served=P2.served_model_id(args.port))
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
