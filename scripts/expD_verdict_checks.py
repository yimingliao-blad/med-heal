#!/usr/bin/env python3
"""Experiment D — decomposed verdict: three independent checks.

User design (2026-05-30): the verdict can do up to three checks, each anchored to a
different reference. Not all are needed — find the minimal effective set.

  C1 correctness : is the CORRECTED answer correct & note-supported for the question?
                   (reference = question + notes; catches corrections that added a wrong fact)
  C2 resolution  : did the corrected answer actually FIX the detected error?
                   (reference = the detection error; catches no-op / wrong-target corrections)
  C3 improvement : is the corrected answer BETTER than the original?
                   (reference = original answer; catches corrections that made it worse)

Reuses expC's 139 flagged+corrected cases (each labeled fix / break / neutral by GPT-4o),
so no detection/correction is re-run — only the three Qwen2.5 checks. Then every check and
combination (AND/OR) is evaluated as a gate: break-catch, fix-keep, net-after-gate, and a
full-scale projection at the 89%-correct base rate.

Output: runs/expD_verdict_checks/qwen25/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expD_verdict_checks"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
EXPC = PROJECT_ROOT / "runs" / "expC_two_stage_verdict" / "qwen25_nw-1_nc50_seed42" / "judged_outputs.jsonl"

CHECK_SYS = "You evaluate a clinical answer carefully and reply with only the requested single token."


def yn(raw: str) -> str:
    m = re.search(r"\b(YES|NO)\b", (raw or "").upper())
    return m.group(1) if m else "NO"


def check_C1(row, note, corrected, port) -> str:
    user = (f"Discharge note:\n{note[:24000]}\n\nQuestion:\n{row['question']}\n\n"
            f"Proposed answer:\n{corrected[:1500]}\n\n"
            "Is the proposed answer correct and fully supported by the note for this question? Reply only YES or NO.")
    return yn(P2.vllm_chat(CHECK_SYS, user, port, 8, 0.0, tag="C1.correct"))


def check_C2(row, error_stmt, corrected, port) -> str:
    user = (f"Question:\n{row['question']}\n\nA problem was identified in an earlier answer:\n{error_stmt[:2500]}\n\n"
            f"New answer:\n{corrected[:1500]}\n\n"
            "Does the new answer correctly resolve that identified problem? Reply only YES or NO.")
    return yn(P2.vllm_chat(CHECK_SYS, user, port, 8, 0.0, tag="C2.resolve"))


def check_C3(row, note, original, corrected, port) -> str:
    rng = random.Random(42 + (row["fold"] << 16) + row["idx"])
    orig_is_a = rng.random() > 0.5
    a, b = (original, corrected) if orig_is_a else (corrected, original)
    corrected_slot = "B" if orig_is_a else "A"
    user = (f"Discharge note:\n{note[:24000]}\n\nQuestion:\n{row['question']}\n\n"
            f"Answer A:\n{a[:1500]}\n\nAnswer B:\n{b[:1500]}\n\n"
            "Which answer is better and more correct for the question? Reply only A or B.")
    raw = P2.vllm_chat(CHECK_SYS, user, port, 8, 0.0, tag="C3.better")
    m = re.search(r"\b([AB])\b", (raw or "").upper())
    pick = m.group(1) if m else "A"
    return "YES" if pick == corrected_slot else "NO"  # YES = accept corrected


def label(r) -> str:
    jo = (r.get("judge_original") or {}).get("label")
    jc = (r.get("judge_corrected") or {}).get("label")
    if jo == 0 and jc == 1:
        return "fix"
    if jo == 1 and jc == 0:
        return "break"
    return "neutral"


def process_one(row, notes, port) -> dict[str, Any]:
    out = {k: row.get(k) for k in ["fold", "idx", "patient_id", "question", "original_answer", "corrected", "error_stmt"]}
    out["_label"] = label(row)
    try:
        note = notes.get(str(row["patient_id"]), "")
        corrected = row.get("corrected") or ""
        out["C1"] = check_C1(row, note, corrected, port)
        out["C2"] = check_C2(row, row.get("error_stmt") or "", corrected, port)
        out["C3"] = check_C3(row, note, row["original_answer"], corrected, port)
    except Exception as e:
        out["error"] = str(e)
    return out


def gate_accepts(r: dict[str, Any], rule: tuple[str, frozenset[str]]) -> bool:
    mode, checks = rule  # mode in {AND, OR}
    vals = [r.get(c) == "YES" for c in checks]
    return all(vals) if mode == "AND" else any(vals)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixes = [r for r in rows if r["_label"] == "fix"]
    breaks = [r for r in rows if r["_label"] == "break"]
    n_fix, n_brk = len(fixes), len(breaks)
    # full-scale base rate (Qwen2.5): 109 wrong, 853 correct. flagged-correction rates from expC.
    W, C = 109, 853
    # pre-gate fix-rate and break-rate from this labeled set:
    # fixes are wrong-origin, breaks are correct-origin. Use the rates observed.
    rules = []
    singles = ["C1", "C2", "C3"]
    for c in singles:
        rules.append(("AND", frozenset([c])))
    for combo in list(combinations(singles, 2)) + [tuple(singles)]:
        rules.append(("AND", frozenset(combo)))
        rules.append(("OR", frozenset(combo)))
    out: dict[str, Any] = {}
    for mode, checks in rules:
        name = ("&" if mode == "AND" else "|").join(sorted(checks))
        acc_fix = sum(1 for r in fixes if gate_accepts(r, (mode, checks)))
        acc_brk = sum(1 for r in breaks if gate_accepts(r, (mode, checks)))
        fix_keep = round(acc_fix / max(1, n_fix), 3)
        brk_catch = round(1 - acc_brk / max(1, n_brk), 3)
        out[name] = {"fix_keep": fix_keep, "break_catch": brk_catch, "acc_fix": acc_fix, "acc_brk": acc_brk,
                     "net_balanced": acc_fix - acc_brk}
    return {"n_flagged": len(rows), "n_fix": n_fix, "n_break": n_brk,
            "errors": sum(1 for r in rows if r.get("error")),
            "gates": dict(sorted(out.items(), key=lambda kv: -kv[1]["net_balanced"]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    if not EXPC.exists():
        raise RuntimeError(f"need expC outputs: {EXPC}")
    src = [json.loads(l) for l in EXPC.open()]
    flagged = [r for r in src if r.get("union_flag") and r.get("corrected") and (r.get("judge_corrected") or {}).get("label") is not None]
    notes = P2.load_notes()
    out_dir = OUT_ROOT / "qwen25"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expD_verdict_checks", served=served)
    print(f"flagged corrected cases={len(flagged)} c={args.concurrency} out={out_dir}", flush=True)
    rows: list[dict[str, Any]] = []
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
