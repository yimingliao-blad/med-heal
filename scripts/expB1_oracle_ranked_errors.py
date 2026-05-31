#!/usr/bin/env python3
"""Experiment B1 — does correction need the CRITICAL flaw, any flaw, or all flaws?

Exp A showed the feedback FORM does not change fix-rate (~25%). The remaining lever
is diagnosis CORRECTNESS. B1 isolates it with an oracle: GPT-4o ranks each wrong
answer's errors by criticality FOR ANSWERING THE QUESTION (rank 1 = most critical),
each with note evidence. Then Qwen2.5 correction runs three ways and we compare fix-rate:

  critical_only : correction is given only the #1 (most critical) error.
  lesser_only   : correction is given only the lowest-ranked (least critical) error.
  all_errors    : correction is given the full ranked list.

Read:
  critical_only ~ all_errors >> lesser_only  -> needs THE CRITICAL flaw (detection must prioritize).
  critical_only ~ lesser_only ~ all          -> ANY one real flaw suffices (detection needs recall only).
  all >> critical_only                       -> needs COMPLETENESS.

Wrong cases only (the oracle is given ground truth to enumerate errors). Same correction
prompt as Exp A (note + spans + the selected error(s)). Real notes (24k), c=8, ledger.

Output: runs/expB1_oracle_ranked/qwen25_nw{NW}/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expB1_oracle_ranked"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ARMS = ["critical_only", "lesser_only", "all_errors"]

# ---------- oracle: rank the errors ----------

RANK_SYS = "You are a medical expert auditing a wrong answer against the chart and the correct answer."

RANK_USER = """Discharge note:
{note}

Question:
{question}

Answer that was given (it is known to be wrong or incomplete for this question):
{answer}

Correct answer (ground truth):
{ground_truth}

List the distinct errors in the given answer, RANKED by how critical each is for correctly answering THIS question. Rank 1 = the single most critical error (the one that most makes the answer wrong for the question); higher numbers = less critical. For each error, state the error plainly and quote the note evidence that establishes the correct fact.

Return JSON only:
{{"errors": [{{"rank": 1, "error": "...", "evidence": "..."}}, {{"rank": 2, "error": "...", "evidence": "..."}}]}}
If the answer is actually correct, return {{"errors": []}}."""


def rank_errors(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = P2.gpt("gpt-4o", RANK_SYS,
                 RANK_USER.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500], ground_truth=row["ground_truth"][:800]),
                 max_tokens=900, temperature=0.0, json_mode=True, tag="oracle.rank")
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return []
    try:
        errs = json.loads(m.group()).get("errors", [])
    except Exception:
        return []
    out = []
    for e in errs:
        if isinstance(e, dict) and (e.get("error") or "").strip():
            out.append({"rank": int(e.get("rank", len(out) + 1)), "error": str(e.get("error", "")), "evidence": str(e.get("evidence", ""))})
    out.sort(key=lambda x: x["rank"])
    return out


def errors_for_arm(arm: str, errors: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not errors:
        return []  # oracle found no error -> keep original for all arms
    if arm == "critical_only":
        return [errors[0]]
    if arm == "all_errors":
        return errors
    if arm == "lesser_only":
        if len(errors) < 2:
            return None  # no distinct lesser error -> arm not applicable for this case
        return [errors[-1]]
    raise ValueError(arm)


def error_block(errors: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {e['error']} (note evidence: {e['evidence']})" for e in errors) if errors else "(no specific error)"


# ---------- correction (same shape as Exp A) ----------

CORRECTION_SYS = ("You are a careful clinical QA assistant. Use the known error(s) and the note evidence to fix the "
                  "previous answer. Do not add facts not supported by the note. Fix only what the known error(s) point to.")


def run_correction(row: dict[str, Any], errors: list[dict[str, Any]], spans: list[dict[str, Any]], port: int, tag: str) -> str:
    user = f"""Discharge note:
{row['note'][:24000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{P2.render_spans(spans)}

Known error(s) to fix:
{error_block(errors)}

Fix the previous answer by correcting the known error(s), grounded in the note evidence. Return only the final answer."""
    return P2.vllm_chat(CORRECTION_SYS, user, port, 700, 0.0, tag=tag)


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        spans = P2.retrieve_spans(row, k=5)
        out["judge_original"] = P2.judge(row, row["original_answer"])
        errors = rank_errors(row)
        out["oracle_errors"] = errors
        out["n_errors"] = len(errors)
        arms: dict[str, Any] = {}
        for arm in ARMS:
            sel = errors_for_arm(arm, errors)
            if sel is None:
                arms[arm] = {"applicable": False}
                continue
            corrected = run_correction(row, sel, spans, port, tag=f"correction.{arm}")
            arms[arm] = {"applicable": True, "n_errors_used": len(sel), "corrected": corrected, "judge_final": P2.judge(row, corrected)}
        out["arms"] = arms
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    by: dict[str, Any] = {}
    for arm in ARMS:
        applicable = [r for r in wrong if (r.get("arms") or {}).get(arm, {}).get("applicable")]
        fix = sum(1 for r in applicable if (r.get("arms") or {}).get(arm, {}).get("judge_final", {}).get("label") == 1)
        by[arm] = {"n_applicable": len(applicable), "fix": fix, "fix_rate": round(fix / max(1, len(applicable)), 3)}
    nerr = Counter(r.get("n_errors") for r in wrong)
    return {"n_cases": len(rows), "n_wrong": len(wrong),
            "n_errors_distribution": {str(k): v for k, v in sorted(nerr.items())},
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
    sample = P2.load_rows(args.n_wrong, 0, args.seed)  # wrong cases only
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expB1_oracle_ranked", served=served, args=vars(args))
    print(f"sample={len(sample)} arms={ARMS} c={args.concurrency} out={out_dir}", flush=True)
    if sample:
        P2.topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows: list[dict[str, Any]] = []
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
