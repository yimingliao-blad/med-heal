#!/usr/bin/env python3
"""Pilot: does feeding UNION's compared CONTENT to the diagnoser beat blind re-detection?

blind_plain re-derives detection from the note, ignoring what union found -> recall collapses
to ~27% (union's 96% is discarded). This tests the user's idea: have union output the content
it compared (it already does — the P3_COMPARE memo), and feed that to a VERIFY diagnoser that
confirms/rejects union's specific concern instead of detecting blind. Expected: recall inherits
union's ceiling, verify prunes the over-flag.

Three detectors compared on the SAME cases:
  union_raw    : union flag as-is (OR of 3 compares)            -> recall ceiling, big over-flag
  blind_plain  : current two-round blind diagnoser (baseline)
  union_verify : union-flagged cases handed to a verify step (union's memo as the concern)

Metrics (on N wrong + N correct): recall (flag a wrong), over-flag (flag a correct), and
LOCALIZATION (gpt_localize vs gold = is the flagged error the REAL one) — the true fix predictor.
A flag only helps if it is on a wrong case AND localized correctly -> 'useful'.

Usage: python scripts/expL_union_grounded.py --n-wrong 40 --n-correct 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa
import expA_detection_feedback as EA  # noqa
import expK_cascade_collect as K  # noqa
from llm_audit import set_ledger  # noqa

VERIFY_SYS = "You verify a flagged concern about a clinical answer against the discharge note, and decide if it is a real, note-supported error."
VERIFY_USER = """Discharge note:
{note}

Question:
{question}

Answer:
{answer}

An automated check compared this answer to the note and raised the following concern(s):
{concern}

Verify against the note. Is this a REAL, note-supported error in the answer — something the note actually contradicts, or a required fact that is clearly missing? Do NOT flag a fact the note is simply silent on, and do NOT flag a misreading of the concern.
If it is a real error, write:
WRONG: the wrong or missing claim.
CORRECT: the note-supported correct fact.
EVIDENCE: the exact note sentence(s).
If it is not a real error, write: WRONG: none""" + K.FLAG_TAIL


def union_verify(row, port, det):
    """Hand union's INCORRECT memo(s) to a verify step. Only runs if union flagged."""
    if not det["union_flag"]:
        return {"flagged": False, "error": "", "raw": "(union did not flag)"}
    concern = "\n\n".join(m for m, v in zip(det["memos"], det["verdicts"]) if v == "INCORRECT")[:3500]
    raw = P2.vllm_chat(VERIFY_SYS, VERIFY_USER.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500], concern=concern), port, 900, 0.0, tag="diag.union_verify")
    flagged = K.llm_flag(raw, K.parse_flag(raw), "union_verify")
    m = re.search(r"(WRONG\s*:[\s\S]*?)(?:\nFLAG\s*:|$)", raw or "", re.I)
    return {"flagged": flagged, "error": (m.group(1) if m else raw)[:1800], "raw": raw}


def process_one(row, port, parser):
    out = {k: row[k] for k in ["fold", "idx", "stored_label"]}
    det = EA.detect_k3_union(row, port, parser)
    out["union_raw"] = {"flagged": det["union_flag"]}
    # baseline blind_plain
    claims = K.round1(row, port)
    bp = K.diagnose("blind_plain", row, claims, port)
    out["blind_plain"] = {"flagged": bp["flagged"], "localized": K.gpt_localize(row, bp["error"]) if bp["flagged"] else False}
    # union-grounded verify
    uv = union_verify(row, port, det)
    out["union_verify"] = {"flagged": uv["flagged"], "localized": K.gpt_localize(row, uv["error"]) if uv["flagged"] else False}
    return out


def summarize(recs):
    nw = sum(1 for r in recs if r["stored_label"] == 0)
    ncorr = sum(1 for r in recs if r["stored_label"] == 1)
    print(f"\n=== pilot: {nw} wrong + {ncorr} correct ===")
    print(f"{'detector':14} {'recall(wrong)':>14} {'over-flag(corr)':>16} {'loc-rate':>9} {'USEFUL(w&loc)':>14}")
    for det in ["union_raw", "blind_plain", "union_verify"]:
        fw = sum(1 for r in recs if r["stored_label"] == 0 and r[det]["flagged"])
        fc = sum(1 for r in recs if r["stored_label"] == 1 and r[det]["flagged"])
        if det == "union_raw":
            loc = useful = "-"
            locs = "    -"
            useful_s = "       -"
        else:
            flagged = [r for r in recs if r[det]["flagged"]]
            nloc = sum(1 for r in flagged if r[det]["localized"])
            useful = sum(1 for r in recs if r["stored_label"] == 0 and r[det]["flagged"] and r[det]["localized"])
            locs = f"{nloc}/{len(flagged)}={nloc/max(1,len(flagged))*100:.0f}%"
            useful_s = f"{useful}/{nw}={useful/max(1,nw)*100:.0f}%"
        print(f"{det:14} {fw}/{nw}={fw/max(1,nw)*100:>3.0f}%{'':5} {fc}/{ncorr}={fc/max(1,ncorr)*100:>3.0f}%{'':7} {locs:>9} {useful_s:>14}")
    print("\nUSEFUL = flagged on a wrong case AND localized correctly (the cases correction can actually fix).")
    print("If union_verify USEFUL > blind_plain USEFUL, using union's content beats blind re-detection.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wrong", type=int, default=40)
    ap.add_argument("--n-correct", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--parser", default="helper-v2")
    args = ap.parse_args()
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    empty = [(r["fold"], r["idx"]) for r in sample if not (r.get("note") or "").strip()]
    if empty:
        raise RuntimeError(f"ABORT empty notes: {empty[:3]}")
    out_dir = PROJECT_ROOT / "runs" / "expL_union_grounded" / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expL_union_grounded", served=P2.served_model_id(args.port))
    print(f"NOTE GUARD OK: {len(sample)} notes, mean {sum(len(r['note']) for r in sample)//len(sample)} chars", flush=True)
    recs = []
    rec_path = out_dir / "records.jsonl"
    f = rec_path.open("w")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port, args.parser) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            recs.append(r)
            f.write(json.dumps(r, default=str) + "\n"); f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)
    f.close()
    summarize(recs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
