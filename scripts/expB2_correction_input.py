#!/usr/bin/env python3
"""Experiment B2 — does correction need the notes, a summary, or just the error?

Holds the ERROR constant (reuses B1's oracle #1 critical error, which already carries
its note evidence) and varies only the correction INPUT form:

  full_note   : full discharge note + retrieved spans + the critical error.  [B1's 67% setup]
  summary     : a question-relevant summary of the note + the critical error.
  error_only  : ONLY the critical error (which includes its note-evidence quote); NO note, NO spans.

Read:
  error_only ~ full_note  -> correction needs only the precise error+evidence, not the note.
  summary ~ full_note >> error_only -> the note context matters; a summary can stand in.
  full_note >> summary, error_only -> correction needs the full note.

Reuses B1 oracle errors from runs/expB1_oracle_ranked/. Wrong cases only. Real notes (24k),
c=8, ledger, blocking. Reference: B1 critical_only fix-rate = 0.667 (full note).

Output: runs/expB2_correction_input/qwen25_nw{NW}/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expB2_correction_input"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
B1_OUT = PROJECT_ROOT / "runs" / "expB1_oracle_ranked" / "qwen25_nw-1_seed42" / "judged_outputs.jsonl"

ARMS = ["full_note", "summary", "error_only"]


def load_b1_critical() -> dict[tuple[int, int], dict[str, Any]]:
    """(fold,idx) -> the #1 critical oracle error, from B1's run."""
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for line in B1_OUT.open():
        r = json.loads(line)
        errs = r.get("oracle_errors") or []
        if errs:
            out[(r["fold"], r["idx"])] = errs[0]
    return out


# ---------- summary (question-relevant) ----------

SUMMARY_SYS = "You summarize a discharge note, keeping only what is relevant to a specific question."
SUMMARY_USER = """Discharge note:
{note}

Question:
{question}

Write a concise summary of the note that keeps only the facts relevant to answering this question — dates, values, medications, procedures, findings that bear on the question. Quote exact values where they matter. Leave out unrelated content."""


def make_summary(row: dict[str, Any], port: int) -> str:
    return P2.vllm_chat(SUMMARY_SYS, SUMMARY_USER.format(note=row["note"][:24000], question=row["question"]), port, 700, 0.0, tag="summary")


# ---------- correction (tight; only the INPUT block differs) ----------

CORR_SYS = ("You are a careful clinical QA assistant. Use the named error and the provided evidence to fix the "
            "previous answer. Do not add facts not supported by the evidence. Fix only what the named error points to.")


def err_text(e: dict[str, Any]) -> str:
    return f"- {e.get('error','')} (note evidence: {e.get('evidence','')})"


def correct_full_note(row, err, spans, port):
    user = f"""Discharge note:
{row['note'][:24000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{P2.render_spans(spans)}

The error to fix:
{err_text(err)}

Fix the previous answer by correcting the named error, grounded in the note evidence. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="corr.full_note")


def correct_summary(row, err, summary, port):
    user = f"""Question-relevant note summary:
{summary[:6000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

The error to fix:
{err_text(err)}

Fix the previous answer by correcting the named error, grounded in the summary and the error evidence. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="corr.summary")


def correct_error_only(row, err, port):
    user = f"""Question:
{row['question']}

Previous answer:
{row['original_answer']}

The error to fix:
{err_text(err)}

Fix the previous answer by correcting the named error, using the evidence given in the error. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="corr.error_only")


# ---------- orchestration ----------

def process_one(row: dict[str, Any], crit: dict[tuple[int, int], dict[str, Any]], port: int) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        err = crit.get((row["fold"], row["idx"]))
        out["judge_original"] = P2.judge(row, row["original_answer"])
        if not err:
            out["no_oracle_error"] = True
            out["arms"] = {a: {"applicable": False} for a in ARMS}
            return out
        out["critical_error"] = err
        spans = P2.retrieve_spans(row, k=5)
        summary = make_summary(row, port)
        arms: dict[str, Any] = {}
        for a in ARMS:
            if a == "full_note":
                corrected = correct_full_note(row, err, spans, port)
            elif a == "summary":
                corrected = correct_summary(row, err, summary, port)
            else:
                corrected = correct_error_only(row, err, port)
            arms[a] = {"applicable": True, "corrected": corrected, "judge_final": P2.judge(row, corrected)}
        out["arms"] = arms
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    by: dict[str, Any] = {}
    for a in ARMS:
        app = [r for r in wrong if (r.get("arms") or {}).get(a, {}).get("applicable")]
        fix = sum(1 for r in app if (r.get("arms") or {}).get(a, {}).get("judge_final", {}).get("label") == 1)
        by[a] = {"n_applicable": len(app), "fix": fix, "fix_rate": round(fix / max(1, len(app)), 3)}
    return {"n_cases": len(rows), "n_wrong": len(wrong),
            "reference": {"B1_critical_only_full_note": 0.667},
            "by_arm": by, "errors": sum(1 for r in rows if r.get("error"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    if not B1_OUT.exists():
        raise RuntimeError(f"need B1 output for oracle errors: {B1_OUT}")
    crit = load_b1_critical()
    sample = P2.load_rows(args.n_wrong, 0, args.seed)  # wrong only
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expB2_correction_input", served=served, args=vars(args))
    print(f"sample={len(sample)} crit_errors={len(crit)} arms={ARMS} c={args.concurrency} out={out_dir}", flush=True)
    if sample:
        P2.topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, crit, args.port) for r in sample]
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
