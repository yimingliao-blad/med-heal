#!/usr/bin/env python3
"""Experiment D2 — does the OLD verdict prompt beat our C3? All vs zeroshot.

Reuses expC's 139 flagged+corrected cases (labeled fix/break/neutral). Runs the two
proven OLD verdict prompts and compares to our C3 pairwise, every result reported as
LIFT OVER ZEROSHOT (= keep all originals, net 0):

  zeroshot                  : accept nothing (keep original) -> net 0 by definition.
  C3_pairwise               : our plain "which is better" pairwise (from expD).
  false_correction_sensitive: locked-default pairwise + explicit reject criteria.
  count_compare             : step9_v2 v1f — count contradicting claims, pick fewer.

Gate metrics: fix_keep, break_catch, accepted_fix/break, net (= accuracy lift over zeroshot).

Output: runs/expD2_verdict_old/qwen25/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expD2_verdict_old"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
EXPC = PROJECT_ROOT / "runs" / "expC_two_stage_verdict" / "qwen25_nw-1_nc50_seed42" / "judged_outputs.jsonl"

VARIANTS = ["C3_strict", "C3_natural", "C3_cot", "false_correction_sensitive", "count_compare"]

# ---------- old verdict prompts ----------

FCS_SYS = "You are a strict medical expert comparing two answers against the discharge note and question."
FCS_TMPL = """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Compare both answers against the discharge summary and the exact question. A correction should be accepted only if it is clearly more faithful to the note and better answers the question. Reject a correction if it adds unsupported facts, changes a supported answer unnecessarily, drops a required detail, or answers a different focus.

Choose the safer answer for final reporting.

Reply on the FIRST line with exactly one letter: A or B."""

CC_SYS = "You are a strict medical expert."
CC_TMPL = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count how many factual claims in each answer contradict the discharge notes.
Different wording for the same fact is NOT a contradiction.

A_ERRORS: <number>
B_ERRORS: <number>"""

# STRICT: single-token, no reasoning.
C3_STRICT_SYS = "You decide which of two answers is better for a clinical question. Reply with only A or B."
C3_STRICT_TMPL = """Discharge note:
{note}

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better and more correct for the question? Reply only A or B."""

# NATURAL: reason in plain prose first, then commit on the last line.
C3_NAT_SYS = "You are a thoughtful clinician deciding which of two answers to report for a question."
C3_NAT_TMPL = """Discharge note:
{note}

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Think it through in plain language: which answer is more faithful to the note and better answers the exact question, and why. Consider whether either adds unsupported facts or drops something the question needs.
On the very last line, state your choice as a single letter: A or B."""

# COT: explicit step-by-step reasoning, then commit.
C3_COT_SYS = "You compare two clinical answers carefully, step by step, before deciding."
C3_COT_TMPL = """Discharge note:
{note}

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Work through this step by step:
1. What exactly does the question require (the answer slot)?
2. What does Answer A claim, and is each claim supported by the note?
3. What does Answer B claim, and is each claim supported by the note?
4. Does either add unsupported facts, drop a required detail, or answer the wrong focus?
5. Which answer is more correct and complete for the exact question?
On the very last line, state your choice as a single letter: A or B."""


def _ab(row):
    rng = random.Random(42 + (row["fold"] << 16) + row["idx"])
    orig_is_a = rng.random() > 0.5
    return orig_is_a


def run_variant(variant: str, row, note, original, corrected, port) -> bool:
    orig_is_a = _ab(row)
    a, b = (original, corrected) if orig_is_a else (corrected, original)
    corrected_slot = "B" if orig_is_a else "A"
    if variant == "count_compare":
        raw = P2.vllm_chat(CC_SYS, CC_TMPL.format(note=note[:24000], question=row["question"], answer_a=a[:1500], answer_b=b[:1500]), port, 64, 0.0, tag="v.count_compare")
        ma = re.search(r"A_ERRORS:\s*(\d+)", raw or "", re.I)
        mb = re.search(r"B_ERRORS:\s*(\d+)", raw or "", re.I)
        ae = int(ma.group(1)) if ma else 99
        be = int(mb.group(1)) if mb else 99
        # pick the answer with fewer contradictions; tie -> keep original (A-slot of original)
        if ae == be:
            pick = "A" if orig_is_a else "B"  # the original's slot -> keep original
        else:
            pick = "A" if ae < be else "B"
    elif variant == "false_correction_sensitive":
        raw = P2.vllm_chat(FCS_SYS, FCS_TMPL.format(note=note[:24000], question=row["question"], answer_a=a[:1500], answer_b=b[:1500]), port, 16, 0.0, tag="v.fcs")
        m = re.search(r"\b([AB])\b", (raw or "").upper())
        pick = m.group(1) if m else ("A" if orig_is_a else "B")
    elif variant == "C3_strict":
        raw = P2.vllm_chat(C3_STRICT_SYS, C3_STRICT_TMPL.format(note=note[:24000], question=row["question"], answer_a=a[:1500], answer_b=b[:1500]), port, 8, 0.0, tag="v.c3_strict")
        m = re.search(r"\b([AB])\b", (raw or "").upper())
        pick = m.group(1) if m else ("A" if orig_is_a else "B")
    elif variant == "C3_natural":
        raw = P2.vllm_chat(C3_NAT_SYS, C3_NAT_TMPL.format(note=note[:24000], question=row["question"], answer_a=a[:1500], answer_b=b[:1500]), port, 400, 0.0, tag="v.c3_natural")
        pick = _last_ab(raw, orig_is_a)
    else:  # C3_cot
        raw = P2.vllm_chat(C3_COT_SYS, C3_COT_TMPL.format(note=note[:24000], question=row["question"], answer_a=a[:1500], answer_b=b[:1500]), port, 600, 0.0, tag="v.c3_cot")
        pick = _last_ab(raw, orig_is_a)
    return pick == corrected_slot  # True = accept correction


def _last_ab(raw: str, orig_is_a: bool) -> str:
    """Extract the A/B choice from the LAST line of a reasoned answer."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"\b([AB])\b", ln.upper())
        if m:
            return m.group(1)
    return "A" if orig_is_a else "B"


def label(r):
    jo = (r.get("judge_original") or {}).get("label")
    jc = (r.get("judge_corrected") or {}).get("label")
    return "fix" if (jo == 0 and jc == 1) else "break" if (jo == 1 and jc == 0) else "neutral"


def process_one(r, notes, port):
    out = {k: r.get(k) for k in ["fold", "idx", "patient_id", "question", "original_answer", "corrected"]}
    out["_label"] = label(r)
    try:
        note = notes.get(str(r["patient_id"]), "")
        corrected = r.get("corrected") or ""
        out["accept"] = {v: run_variant(v, r, note, r["original_answer"], corrected, port) for v in VARIANTS}
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows):
    fixes = [r for r in rows if r["_label"] == "fix"]
    breaks = [r for r in rows if r["_label"] == "break"]
    nf, nb = len(fixes), len(breaks)
    by = {"zeroshot": {"fix_keep": 0.0, "break_catch": 1.0, "acc_fix": 0, "acc_brk": 0, "net_lift_over_zeroshot": 0}}
    for v in VARIANTS:
        af = sum(1 for r in fixes if (r.get("accept") or {}).get(v))
        ab = sum(1 for r in breaks if (r.get("accept") or {}).get(v))
        by[v] = {"fix_keep": round(af / max(1, nf), 3), "break_catch": round(1 - ab / max(1, nb), 3),
                 "acc_fix": af, "acc_brk": ab, "net_lift_over_zeroshot": af - ab}
    return {"n_flagged": len(rows), "n_fix": nf, "n_break": nb,
            "note": "net_lift_over_zeroshot = accepted_fixes - accepted_breaks; zeroshot keeps all originals (net 0)",
            "by_verdict": dict(sorted(by.items(), key=lambda kv: -kv[1]["net_lift_over_zeroshot"])),
            "errors": sum(1 for r in rows if r.get("error"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    src = [json.loads(l) for l in EXPC.open()]
    flagged = [r for r in src if r.get("union_flag") and r.get("corrected") and (r.get("judge_corrected") or {}).get("label") is not None]
    notes = P2.load_notes()
    out_dir = OUT_ROOT / "qwen25"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expD2_verdict_old", served=served)
    print(f"flagged={len(flagged)} variants={VARIANTS} c={args.concurrency} out={out_dir}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, notes, args.port) for r in flagged]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 10 == 0 or i == len(futs):
                print(f"processed {i}/{len(futs)}", flush=True)
    with (out_dir / "judged_outputs.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
