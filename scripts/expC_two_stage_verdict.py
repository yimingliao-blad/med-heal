#!/usr/bin/env python3
"""Experiment C — model-does-all two-stage pipeline + 3-way verdict bakeoff.

Everything is Qwen2.5 (no oracle, no external model). Built on the A/B1/B1b/B2 findings:
correction needs only a correct, evidence-backed critical-error statement (B2: the note
is redundant given the error). So:

  Stage 1 DETECTION (error-focused output):
    k=3 natural union (recall 0.96) -> synthesize ONE focused error statement:
    what is wrong, the correct fact, and the note evidence. This is the output for correction.
  Stage 2 CORRECTION:
    apply the error statement directly (B2's error-led correction).

Then judge original + corrected (GPT-4o) to LABEL each correction fix / break / neutral.

VERDICT bakeoff — the Qwen2.5 gate decides keep-original vs accept-correction (A/B,
position-randomized). Three variants differ only in what the verdict sees:
    V_summary  : question + a question-relevant summary
    V_error    : question + the detected error statement
    V_fullnote : question + the full discharge note

Gate quality per variant: break-catch (reject corrections that are breaks), fix-keep
(accept corrections that are fixes), and net-after-gate. Answers: which information makes
the verdict best.

Wrong + correct cases. Real notes (24k), c=8, ledger, blocking.
Output: runs/expC_two_stage_verdict/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
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
import expA_detection_feedback as EA  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expC_two_stage_verdict"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VERDICTS = ["V_summary", "V_error", "V_fullnote"]

# ---------- Stage 1: error-focused detection output ----------

ERROR_FOCUS_SYS = "You state the single clearest error in a clinical answer, grounded in the note."
ERROR_FOCUS = """Question:
{question}

Answer that was given:
{answer}

Note facts that actually answer the question:
{question_facts}

Several quick checks raised these concerns:
{memos}

State the single clearest error in the given answer for THIS question, as a focused correction instruction with three parts:
WRONG: what in the answer is wrong or missing.
CORRECT: the note-supported correct fact for the question.
EVIDENCE: the exact note sentence(s) that establish the correct fact.
Keep it tight and specific."""


def error_focus(row: dict[str, Any], det: dict[str, Any], port: int) -> str:
    memos_block = "\n\n".join(f"Concern {i+1}: {m[:1200]}" for i, m in enumerate(det["memos"]))
    return P2.vllm_chat(ERROR_FOCUS_SYS, ERROR_FOCUS.format(question=row["question"], answer=row["original_answer"][:1500], question_facts=det["question_facts"][:3000], memos=memos_block[:4500]), port, 500, 0.0, tag="error_focus")


# ---------- Stage 2: correction (apply the error statement) ----------

CORR_SYS = ("You are a careful clinical QA assistant. Apply the correction instruction to the previous answer, "
            "grounded in the evidence it cites. Do not add facts beyond that evidence. Return only the final answer.")


def run_correction(row: dict[str, Any], error_stmt: str, port: int) -> str:
    user = f"""Question:
{row['question']}

Previous answer:
{row['original_answer']}

Correction instruction:
{error_stmt[:3000]}

Apply the correction instruction to fix the previous answer, grounded in the evidence it cites. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="correction")


# ---------- summary (for V_summary) ----------

SUMMARY_SYS = "You summarize a discharge note, keeping only what is relevant to a specific question."
SUMMARY_USER = """Discharge note:
{note}

Question:
{question}

Write a concise summary of the note keeping only facts relevant to answering this question — dates, values, medications, procedures, findings. Quote exact values. Leave out unrelated content."""


def make_summary(row: dict[str, Any], port: int) -> str:
    return P2.vllm_chat(SUMMARY_SYS, SUMMARY_USER.format(note=row["note"][:24000], question=row["question"]), port, 700, 0.0, tag="summary")


# ---------- Verdict (Qwen2.5 A/B; 3 input forms) ----------

VERDICT_SYS = "You decide which of two answers is better for a clinical question. Reply with only A or B."


def verdict_pick(variant: str, row: dict[str, Any], ans_a: str, ans_b: str, error_stmt: str, summary: str, port: int) -> str:
    if variant == "V_summary":
        ctx = f"Question-relevant note summary:\n{summary[:6000]}\n\n"
    elif variant == "V_error":
        ctx = f"An issue was flagged in one answer:\n{error_stmt[:2500]}\n\n"
    else:  # V_fullnote
        ctx = f"Discharge note:\n{row['note'][:24000]}\n\n"
    user = (f"{ctx}Question:\n{row['question']}\n\n"
            f"Answer A:\n{ans_a[:1500]}\n\nAnswer B:\n{ans_b[:1500]}\n\n"
            "Which answer is better supported and more correct for the question? Reply with only A or B.")
    raw = P2.vllm_chat(VERDICT_SYS, user, port, 8, 0.0, tag=f"verdict.{variant}")
    m = re.search(r"\b([AB])\b", (raw or "").upper())
    return m.group(1) if m else "A"


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int, parser: str) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        det = EA.detect_k3_union(row, port, parser)
        out["union_flag"] = det["union_flag"]
        if not det["union_flag"]:
            out["action"] = "kept_no_flag"
            out["final_answer"] = row["original_answer"]
            out["verdicts"] = {v: {"pick": "keep", "accepted": False} for v in VERDICTS}
            return out
        error_stmt = error_focus(row, det, port)
        corrected = run_correction(row, error_stmt, port)
        out["error_stmt"] = error_stmt
        out["corrected"] = corrected
        out["judge_corrected"] = P2.judge(row, corrected)
        summary = make_summary(row, port)
        # position-randomize original vs corrected per case
        rng = random.Random(42 + (row["fold"] << 16) + row["idx"])
        orig_is_a = rng.random() > 0.5
        ans_a, ans_b = (row["original_answer"], corrected) if orig_is_a else (corrected, row["original_answer"])
        corrected_slot = "B" if orig_is_a else "A"
        verds: dict[str, Any] = {}
        for v in VERDICTS:
            pick = verdict_pick(v, row, ans_a, ans_b, error_stmt, summary, port)
            accepted = (pick == corrected_slot)
            verds[v] = {"pick": pick, "corrected_slot": corrected_slot, "accepted": accepted}
        out["verdicts"] = verds
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [r for r in rows if r.get("union_flag") and "judge_corrected" in r]
    # label each flagged correction: fix (orig wrong->corr right), break (orig right->corr wrong), neutral
    def label(r):
        jo = (r.get("judge_original") or {}).get("label")
        jc = (r.get("judge_corrected") or {}).get("label")
        if jo == 0 and jc == 1:
            return "fix"
        if jo == 1 and jc == 0:
            return "break"
        return "neutral"
    for r in flagged:
        r["_corr_label"] = label(r)
    fixes = [r for r in flagged if r["_corr_label"] == "fix"]
    breaks = [r for r in flagged if r["_corr_label"] == "break"]
    by: dict[str, Any] = {}
    for v in VERDICTS:
        fix_kept = sum(1 for r in fixes if (r.get("verdicts") or {}).get(v, {}).get("accepted"))
        break_caught = sum(1 for r in breaks if not (r.get("verdicts") or {}).get(v, {}).get("accepted"))
        # net after gate = accepted fixes - accepted breaks (over all flagged)
        acc_fix = sum(1 for r in fixes if (r.get("verdicts") or {}).get(v, {}).get("accepted"))
        acc_break = sum(1 for r in breaks if (r.get("verdicts") or {}).get(v, {}).get("accepted"))
        by[v] = {
            "fix_keep_rate": round(fix_kept / max(1, len(fixes)), 3),
            "break_catch_rate": round(break_caught / max(1, len(breaks)), 3),
            "accepted_fixes": acc_fix, "accepted_breaks": acc_break,
            "net_after_gate": acc_fix - acc_break,
        }
    return {
        "n_cases": len(rows),
        "n_flagged": len(flagged),
        "correction_labels": {"fix": len(fixes), "break": len(breaks), "neutral": len(flagged) - len(fixes) - len(breaks)},
        "pre_gate_net": len(fixes) - len(breaks),
        "by_verdict": by,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--n-correct", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parser", choices=["gpt4o-mini", "helper-v2", "qwen35"], default="helper-v2")
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expC_two_stage_verdict", served=served, args=vars(args))
    print(f"sample={len(sample)} verdicts={VERDICTS} c={args.concurrency} out={out_dir}", flush=True)
    if sample:
        P2.topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port, args.parser) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 5 == 0 or i == len(futs):
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
