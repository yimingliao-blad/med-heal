#!/usr/bin/env python3
"""Experiment F — Correction F1 / reliability axis.

Question (user): given the EXACT instruction, can Qwen2.5 reliably follow it to flip its
zero-shot answer? And how reliable is it? This decomposes the 67% oracle ceiling (B1):
is the unflipped 33% an instruction-FOLLOWING failure, or genuinely-hard cases?

Escalating instruction explicitness, on WRONG cases (should flip) and CORRECT cases
(should NOT flip):

  oracle_error : B1's WRONG/CORRECT/EVIDENCE statement (a precise diagnosis).
  giveaway     : hand the model the ground-truth answer and tell it to make the answer match.
                 This is the ceiling of instruction explicitness — if correction still can't
                 flip with the answer handed to it, the failure is intrinsic (hard case),
                 not instruction quality.

Reports per instruction:
  flip_rate   on wrong = correction RECALL (followed the instruction to fix)
  break_rate  on correct = correction (1 - precision) (wrongly flipped a correct answer)
  correction F1 = harmonic mean of flip_rate and (1 - break_rate)

Reuses B1 (wrong) and B1c (correct) oracle errors. Real notes (24k), c=8, ledger, blocking.
Output: runs/expF_correction_reliability/qwen25/{judged_outputs.jsonl, summary.json}
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

OUT_ROOT = PROJECT_ROOT / "runs" / "expF_correction_reliability"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
B1_OUT = PROJECT_ROOT / "runs" / "expB1_oracle_ranked" / "qwen25_nw-1_seed42" / "judged_outputs.jsonl"
B1C_OUT = PROJECT_ROOT / "runs" / "expB1c_oracle_break" / "qwen25_nc100_seed42" / "judged_outputs.jsonl"

ARMS = ["oracle_error", "giveaway"]

CORR_SYS = ("You are a careful clinical QA assistant. Apply the instruction to the previous answer, "
            "grounded in the note. Return only the final answer.")


def err_block(errors: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {e.get('error','')} (note evidence: {e.get('evidence','')})" for e in errors) if errors else "(none)"


def correct_oracle(row, errors, spans, port):
    user = f"""Discharge note:
{row['note'][:24000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{P2.render_spans(spans)}

Known error(s) to fix:
{err_block(errors)}

Fix the previous answer by correcting the known error(s), grounded in the note evidence. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="corr.oracle_error")


def correct_giveaway(row, port):
    user = f"""Question:
{row['question']}

Previous answer:
{row['original_answer']}

The correct answer to this question is:
{row['ground_truth']}

Rewrite the previous answer so that it states the correct answer above. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="corr.giveaway")


def process_one(row, errors, port):
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        spans = P2.retrieve_spans(row, k=5)
        arms = {}
        # oracle_error arm only applies if the oracle produced an error for this case
        if errors:
            c = correct_oracle(row, errors, spans, port)
            arms["oracle_error"] = {"applicable": True, "corrected": c, "judge_final": P2.judge(row, c)}
        else:
            arms["oracle_error"] = {"applicable": False}
        g = correct_giveaway(row, port)
        arms["giveaway"] = {"applicable": True, "corrected": g, "judge_final": P2.judge(row, g)}
        out["arms"] = arms
        out["had_oracle_error"] = bool(errors)
    except Exception as e:
        out["error"] = str(e)
    return out


def f1(flip, brk):
    rec = flip
    prec = 1 - brk
    return round(2 * rec * prec / max(1e-9, rec + prec), 3)


def summarize(rows):
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    out = {}
    for a in ARMS:
        w_app = [r for r in wrong if (r.get("arms") or {}).get(a, {}).get("applicable")]
        c_app = [r for r in correct if (r.get("arms") or {}).get(a, {}).get("applicable")]
        flips = sum(1 for r in w_app if (r.get("arms") or {}).get(a, {}).get("judge_final", {}).get("label") == 1)
        breaks = sum(1 for r in c_app if (r.get("arms") or {}).get(a, {}).get("judge_final", {}).get("label") == 0)
        fr = round(flips / max(1, len(w_app)), 3)
        br = round(breaks / max(1, len(c_app)), 3)
        out[a] = {"n_wrong": len(w_app), "flip_rate": fr, "n_correct": len(c_app), "break_rate": br, "correction_F1": f1(fr, br)}
    return {"n_wrong": len(wrong), "n_correct": len(correct),
            "note": "flip_rate=correction recall (followed instruction to fix); break_rate=wrongly flipped correct; F1 of (flip, 1-break)",
            "by_instruction": out, "errors": sum(1 for r in rows if r.get("error"))}


def load_oracle_errors():
    crit = {}
    for path in (B1_OUT, B1C_OUT):
        if path.exists():
            for line in path.open():
                r = json.loads(line)
                errs = r.get("oracle_errors") or []
                crit[(r["fold"], r["idx"])] = errs
    return crit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--n-correct", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    crit = load_oracle_errors()
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / "qwen25"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expF_correction_reliability", served=served, args=vars(args))
    print(f"sample={len(sample)} oracle_errors_loaded={len(crit)} c={args.concurrency} out={out_dir}", flush=True)
    if sample:
        P2.topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, crit.get((r["fold"], r["idx"]), []), args.port) for r in sample]
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
