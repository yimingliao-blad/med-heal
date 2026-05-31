#!/usr/bin/env python3
"""Experiment I — confirm-gate + CoT-planner rescue.

Combines two findings:
  - expG: Qwen2.5 is GOOD at confirming what is CORRECT (0.90-0.98), BAD at finding errors.
  - expH: asking "what are you NOT confident about" over-flags (0.80); but CoT reasoning
          roughly DOUBLES error localization (0.15 -> 0.30).

Design:
  GATE (confirm framing): Qwen confirms which parts of its answer the note SUPPORTS. Flag only
       if some material part is UNCONFIRMED (FULLY SUPPORTED -> no flag). Hope: using the
       confirm strength gives better PRECISION than the doubt framing.
  RESCUE (CoT planner): for flagged cases, a CoT planner reasons step-by-step about the
       unconfirmed part to produce a precise WRONG/CORRECT/EVIDENCE diagnosis, then correction
       applies it. This is where the fix is "rescued".

Metrics: gate recall/over-flag/F1; downstream fix/break/net vs zeroshot.
Wrong + correct cases, real notes (24k), c=8, ledger, blocking.
Output: runs/expI_confirm_gate_rescue/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expI_confirm_gate_rescue"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- GATE: confirm what the note supports ----------

CONFIRM_SYS = "You verify a clinical answer against the discharge note, confirming what is supported."
CONFIRM_GATE = """Discharge note:
{note}

Question:
{question}

Answer to verify:
{answer}

Go through the answer claim by claim. For each, check whether the discharge note clearly SUPPORTS it.
End with exactly two labeled sections:
SUPPORTED: the claims the note clearly supports.
UNCONFIRMED: the claims the note does NOT clearly support, or that may be wrong or off-topic (write "none" if every claim is clearly supported and the answer fully and correctly answers the question)."""


def confirm_gate(row, port) -> dict[str, Any]:
    raw = P2.vllm_chat(CONFIRM_SYS, CONFIRM_GATE.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500]), port, 700, 0.0, tag="gate.confirm")
    m = re.search(r"UNCONFIRMED\s*:\s*(.+)$", raw or "", re.I | re.S)
    unconf = (m.group(1).strip() if m else "").strip()
    flagged = bool(unconf) and not re.match(r"^\(?none\)?\.?$", unconf.strip(), re.I)
    return {"raw": raw, "unconfirmed": unconf[:1500], "flagged": flagged}


# ---------- RESCUE: CoT planner diagnoses the unconfirmed part ----------

COT_PLANNER_SYS = "You reason step by step to pin down a clinical answer's error from the note, then write a precise fix."
COT_PLANNER = """Discharge note:
{note}

Question:
{question}

Answer:
{answer}

This part of the answer is NOT clearly supported by the note:
{unconfirmed}

Reason step by step:
1. What exactly does the question ask for?
2. What does the note actually say about the unconfirmed part?
3. Is the answer wrong on this point, or just incomplete?
4. What is the note-supported correct fact?

Then write the precise fix as three lines:
WRONG: what in the answer is wrong or missing.
CORRECT: the note-supported correct fact.
EVIDENCE: the exact note sentence(s) that establish it."""


def cot_planner(row, unconf, port) -> str:
    return P2.vllm_chat(COT_PLANNER_SYS, COT_PLANNER.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500], unconfirmed=unconf[:1500]), port, 700, 0.0, tag="rescue.cot_planner")


# ---------- correction ----------

CORR_SYS = ("You are a careful clinical QA assistant. Apply the fix to the previous answer, grounded in the "
            "evidence it cites. Do not add facts beyond that evidence. Return only the final answer.")


def run_correction(row, fix, port) -> str:
    user = f"""Question:
{row['question']}

Previous answer:
{row['original_answer']}

Fix to apply:
{fix[:3000]}

Apply the fix to the previous answer, grounded in the evidence it cites. If the fix does not show a real, note-supported error, return the previous answer unchanged. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="correction")


# ---------- orchestration ----------

def process_one(row, port) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        gate = confirm_gate(row, port)
        out["flagged"] = gate["flagged"]
        out["unconfirmed"] = gate["unconfirmed"]
        if gate["flagged"]:
            fix = cot_planner(row, gate["unconfirmed"], port)
            corrected = run_correction(row, fix, port)
            out["fix_stmt"] = fix
            out["corrected"] = corrected
            out["judge_corrected"] = P2.judge(row, corrected)
            out["final_answer"] = corrected
        else:
            out["final_answer"] = row["original_answer"]
            out["judge_corrected"] = None
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows):
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    nw, nc = len(wrong), len(correct)
    fw = sum(1 for r in wrong if r.get("flagged"))
    fc = sum(1 for r in correct if r.get("flagged"))
    rec = fw / max(1, nw)
    prec = fw / max(1, fw + fc)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    # downstream (pre-verdict; correction applied to all flagged)
    fixes = sum(1 for r in wrong if (r.get("judge_corrected") or {}).get("label") == 1)
    breaks = sum(1 for r in correct if (r.get("judge_corrected") or {}).get("label") == 0)
    # full-scale projection at 89% base rate
    W, C = 109, 853
    proj = {"fixes": round(W * (fixes / max(1, nw))), "breaks": round(C * (breaks / max(1, nc)))}
    proj["net_pre_verdict"] = proj["fixes"] - proj["breaks"]
    return {
        "n_wrong": nw, "n_correct": nc,
        "GATE": {"recall_on_wrong": round(rec, 3), "over_flag_on_correct": round(fc / max(1, nc), 3),
                 "precision": round(prec, 3), "f1": round(f1, 3),
                 "compare": {"union": "rec0.95/prec0.64/of0.79", "doubt_flipside": "rec0.93/prec0.54/of0.80"}},
        "DOWNSTREAM_pre_verdict": {"fix": fixes, "break": breaks, "fix_rate_on_wrong": round(fixes / max(1, nw), 3),
                                   "break_rate_on_correct": round(breaks / max(1, nc), 3),
                                   "net_balanced": fixes - breaks},
        "full_scale_projection_pre_verdict": proj,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=40)
    ap.add_argument("--n-correct", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expI_confirm_gate_rescue", served=served, args=vars(args))
    print(f"sample={len(sample)} c={args.concurrency} out={out_dir}", flush=True)
    if sample:
        P2.topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port) for r in sample]
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
