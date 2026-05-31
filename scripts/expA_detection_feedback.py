#!/usr/bin/env python3
"""Experiment A — which detection feedback builder helps downstream correction most?

Base detector (shared across arms): k=3 natural compare at T=0.7, UNION flag.
  natural's job = "is it likely an error?" (high recall).

For each flagged case, build feedback THREE ways and run the SAME correction on each,
then judge. Only the feedback content differs, so this isolates feedback quality.

  cot             : reflect/reconsider the k=3 findings step by step, keep what holds.
  planner_verify  : checklist + per-item confirm/contradict/silent (careful verifier).
  planner_context : NEW. Takes the flag as given (does NOT re-judge). Plans HOW TO GET
                    THE SOURCE — identifies the exact answer slot the question needs and
                    gathers the note evidence a corrector needs. Context, not a verdict.

Sample: wrong cases (fix signal) + correct cases (break signal). Real notes, c=8, ledger.

Output: runs/expA_detection_feedback/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the phase2b infrastructure (prompts, vllm/gpt wrappers, loaders, judge, spans).
import phase2b_extract_compare_detection as P2  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expA_detection_feedback"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BUILDERS = ["cot", "planner_verify", "planner_context"]

# ---------- feedback builder prompts ----------

COT_REFLECT_SYS = "You reconsider several quick checks of a clinical answer and keep only what truly holds against the note."
COT_REFLECT = """Question:
{question}

Answer that was given:
{answer}

Note facts the answer talks about:
{answer_facts}

Note facts that actually answer the question:
{question_facts}

Several independent quick checks raised these concerns:
{memos}

Reconsider step by step. For each concern, check it against the note facts and decide whether it really holds. Then write the verified finding: what is actually wrong or missing in the answer, what the note-supported answer should be, and the note evidence — keeping only concerns that hold up."""

# planner_verify reuses phase2b's P0_PLANNER + P3_COMPARE_PLANNED (already a careful verifier).

PLANNER_CONTEXT_SYS = "You prepare the source material a corrector needs. You do not judge whether the answer is right."
PLANNER_CONTEXT = """The previous answer to this question has already been flagged as likely containing an error. That decision is final — do not re-judge it.

Question:
{question}

Previous answer:
{answer}

Note facts the answer talks about:
{answer_facts}

Note facts that actually answer the question:
{question_facts}

Your job is to prepare what a corrector needs to fix the answer. Do two things:
1. Identify the exact answer slot the question requires (e.g. a date, a value, a medication, a list, a yes/no, a procedure).
2. Gather the note evidence relevant to producing the correct answer for that slot — quote the source sentences from the note.

Give the corrector the required slot and the source evidence, grounded only in the note. Do not state a verdict."""


# ---------- shared detection: k=3 natural union ----------

def detect_k3_union(row: dict[str, Any], port: int, parser: str) -> dict[str, Any]:
    note = row["note"][:24000]
    a = row["original_answer"]
    answer_facts = P2.vllm_chat(P2.EXTRACT_SYS, P2.P1_ANSWER_EXTRACT.format(note=note, answer=a[:1500]), port, 600, 0.0, tag="extract.answer")
    question_facts = P2.vllm_chat(P2.EXTRACT_SYS, P2.P2_QUESTION_EXTRACT.format(note=note, question=row["question"]), port, 600, 0.0, tag="extract.question")
    memos, verdicts = [], []
    for i in range(3):
        mm = P2.vllm_chat(P2.P3_COMPARE_SYS, P2.P3_COMPARE.format(question=row["question"], answer=a[:1500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000]), port, 700, 0.7, tag=f"compare.k{i+1}")
        memos.append(mm)
        verdicts.append(P2.parse_memo(row, answer_facts, question_facts, mm, parser)["verdict"])
    union_flag = any(v == "INCORRECT" for v in verdicts)
    return {"answer_facts": answer_facts, "question_facts": question_facts, "memos": memos, "verdicts": verdicts, "union_flag": union_flag}


# ---------- feedback builders ----------

def build_feedback(builder: str, row: dict[str, Any], det: dict[str, Any], port: int) -> str:
    a = row["original_answer"]
    af, qf = det["answer_facts"], det["question_facts"]
    memos_block = "\n\n".join(f"Check {i+1}: {m[:1500]}" for i, m in enumerate(det["memos"]))
    if builder == "cot":
        return P2.vllm_chat(COT_REFLECT_SYS, COT_REFLECT.format(question=row["question"], answer=a[:1500], answer_facts=af[:3000], question_facts=qf[:3000], memos=memos_block[:5000]), port, 800, 0.0, tag="fb.cot")
    if builder == "planner_verify":
        plan = P2.vllm_chat(P2.P0_PLANNER_SYS, P2.P0_PLANNER.format(question=row["question"], answer=a[:1500]), port, 500, 0.0, tag="fb.planner_verify.plan")
        return P2.vllm_chat(P2.P3_COMPARE_SYS, P2.P3_COMPARE_PLANNED.format(question=row["question"], answer=a[:1500], plan=plan[:2500], answer_facts=af[:3000], question_facts=qf[:3000]), port, 800, 0.0, tag="fb.planner_verify")
    if builder == "planner_context":
        return P2.vllm_chat(PLANNER_CONTEXT_SYS, PLANNER_CONTEXT.format(question=row["question"], answer=a[:1500], answer_facts=af[:3000], question_facts=qf[:3000]), port, 800, 0.0, tag="fb.planner_context")
    raise ValueError(builder)


# ---------- correction (same prompt; only the feedback block differs) ----------

CORRECTION_SYS = ("You are a careful clinical QA assistant. Use the feedback and the note evidence to fix the "
                  "previous answer. Do not add facts not supported by the note. If the feedback does not show a "
                  "real, note-supported problem, keep the previous answer.")


def run_correction(row: dict[str, Any], feedback: str, spans: list[dict[str, Any]], port: int) -> str:
    user = f"""Discharge note:
{row['note'][:24000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{P2.render_spans(spans)}

Feedback to guide the fix:
{feedback[:4000]}

Fix the previous answer using the feedback and the note evidence. Return only the final answer."""
    return P2.vllm_chat(CORRECTION_SYS, user, port, 700, 0.0, tag="correction")


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int, parser: str) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        spans = P2.retrieve_spans(row, k=5)
        out["judge_original"] = P2.judge(row, row["original_answer"])
        det = detect_k3_union(row, port, parser)
        out["union_flag"] = det["union_flag"]
        out["k3_verdicts"] = det["verdicts"]
        if not det["union_flag"]:
            out["arms"] = {b: {"flagged": False, "judge_final": {"label": (out["judge_original"] or {}).get("label"), "raw": "kept_original"}} for b in BUILDERS}
            return out
        arms: dict[str, Any] = {}
        for b in BUILDERS:
            fb = build_feedback(b, row, det, port)
            corrected = run_correction(row, fb, spans, port)
            arms[b] = {"flagged": True, "feedback_chars": len(fb), "corrected": corrected, "judge_final": P2.judge(row, corrected)}
        out["arms"] = arms
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    union_recall = round(sum(1 for r in wrong if r.get("union_flag")) / max(1, len(wrong)), 3)
    union_overflag = round(sum(1 for r in correct if r.get("union_flag")) / max(1, len(correct)), 3)
    by: dict[str, Any] = {}
    for b in BUILDERS:
        fix = sum(1 for r in wrong if (r.get("arms") or {}).get(b, {}).get("judge_final", {}).get("label") == 1)
        brk = sum(1 for r in correct if (r.get("arms") or {}).get(b, {}).get("judge_final", {}).get("label") == 0)
        by[b] = {"fix": fix, "break": brk, "net": fix - brk, "fix_rate_on_wrong": round(fix / max(1, len(wrong)), 3)}
    return {"n_cases": len(rows), "n_wrong": len(wrong), "n_correct": len(correct),
            "union_recall_on_wrong": union_recall, "union_overflag_on_correct": union_overflag,
            "by_builder": by, "errors": sum(1 for r in rows if r.get("error"))}


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
    set_ledger(out_dir / "llm_calls.jsonl", script="expA_detection_feedback", served=served, args=vars(args))
    print(f"sample={len(sample)} builders={BUILDERS} c={args.concurrency} out={out_dir}", flush=True)
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
