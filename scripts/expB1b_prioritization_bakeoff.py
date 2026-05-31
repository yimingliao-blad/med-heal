#!/usr/bin/env python3
"""Experiment B1b — find a live prioritization prompt that picks THE critical flaw.

B1 (oracle) proved correction needs the single most critical flaw: critical-only fixes
67%, lesser-only 41%, all 71%. The live pipeline sits at 25% because k=3 union finds
*a* flaw (recall 0.93) but does not PRIORITIZE to the critical one.

This bakeoff tests prioritization prompts that, WITHOUT ground truth, pick the most
critical flaw from the k=3 union candidate concerns, then correct only that flaw.
Goal: lift fix-rate from 25% toward the 67% oracle-critical ceiling, and identify the
winning prompt so we can move on.

Shared base: k=3 natural union (recall engine). Variants differ only in the prioritize
step (or absence of it):

  baseline_all  : no prioritization — feed all concerns to correction (reproduces ~25%).
  prio_direct   : pick the single most critical concern for answering the question.
  prio_slot     : first name the question's required answer slot, then pick the concern
                  that most affects that slot.
  prio_impact   : pick the concern whose fix would most change the answer toward correct.

Wrong cases (fix) + correct cases (break). Real notes (24k), c=8, ledger, blocking run.

Output: runs/expB1b_prioritization/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
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
import expA_detection_feedback as EA  # noqa: E402  (reuse detect_k3_union, run_correction)
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expB1b_prioritization"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["baseline_all", "prio_direct", "prio_slot", "prio_impact"]
TIGHT = False  # set from --tight-correction in main()

# ---------- prioritization prompts ----------

PRIO_SYS = "You pick the single most important problem to fix in a clinical answer, using only the note."

PRIO_DIRECT = """Question:
{question}

Answer that was given:
{answer}

Note facts that actually answer the question:
{question_facts}

Several quick checks raised these concerns about the answer:
{memos}

Among these concerns, pick the SINGLE most critical one for correctly answering THIS exact question — the one that most makes the answer wrong for what the question asks. State that one critical problem plainly and quote the note evidence for the correct fact. Ignore the less important concerns."""

PRIO_SLOT = """Question:
{question}

Answer that was given:
{answer}

Note facts that actually answer the question:
{question_facts}

Several quick checks raised these concerns about the answer:
{memos}

First, name the exact answer slot this question requires (e.g. a date, a value, a medication, a list, a yes/no, a procedure). Then, among the concerns, pick the ONE that most directly affects that required slot — the problem that, if left unfixed, leaves the slot wrong. State that one critical problem plainly and quote the note evidence for the correct value of the slot. Ignore concerns that do not affect the required slot."""

PRIO_IMPACT = """Question:
{question}

Answer that was given:
{answer}

Note facts that actually answer the question:
{question_facts}

Several quick checks raised these concerns about the answer:
{memos}

Think about which single concern, if corrected, would most change the answer toward the correct answer for this question. That is the critical one. State it plainly and quote the note evidence for the correct fact. Do not list the others."""

PRIO_PROMPTS = {"prio_direct": PRIO_DIRECT, "prio_slot": PRIO_SLOT, "prio_impact": PRIO_IMPACT}


def prioritize(variant: str, row: dict[str, Any], det: dict[str, Any], port: int) -> str:
    memos_block = "\n\n".join(f"Concern {i+1}: {m[:1200]}" for i, m in enumerate(det["memos"]))
    tmpl = PRIO_PROMPTS[variant]
    return P2.vllm_chat(PRIO_SYS, tmpl.format(question=row["question"], answer=row["original_answer"][:1500], question_facts=det["question_facts"][:3000], memos=memos_block[:4500]), port, 500, 0.0, tag=f"prio.{variant}")


def feedback_for(variant: str, row: dict[str, Any], det: dict[str, Any], port: int) -> str:
    if variant == "baseline_all":
        return "\n\n".join(f"Concern {i+1}: {m[:1500]}" for i, m in enumerate(det["memos"]))[:4000]
    return prioritize(variant, row, det, port)


# Tight correction matching B1 (the 67% oracle run): "fix only what the named error points to".
# Isolates prioritization quality by removing the looser-prompt confound.
TIGHT_CORRECTION_SYS = ("You are a careful clinical QA assistant. Use the named error and the note evidence to fix the "
                        "previous answer. Do not add facts not supported by the note. Fix only what the named error points to.")


def run_correction_tight(row: dict[str, Any], feedback: str, spans: list[dict[str, Any]], port: int) -> str:
    user = f"""Discharge note:
{row['note'][:24000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{P2.render_spans(spans)}

The error to fix:
{feedback[:4000]}

Fix the previous answer by correcting the named error, grounded in the note evidence. Return only the final answer."""
    return P2.vllm_chat(TIGHT_CORRECTION_SYS, user, port, 700, 0.0, tag="correction.tight")


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int, parser: str) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        spans = P2.retrieve_spans(row, k=5)
        out["judge_original"] = P2.judge(row, row["original_answer"])
        det = EA.detect_k3_union(row, port, parser)
        out["union_flag"] = det["union_flag"]
        if not det["union_flag"]:
            out["arms"] = {v: {"flagged": False, "judge_final": {"label": (out["judge_original"] or {}).get("label"), "raw": "kept_original"}} for v in VARIANTS}
            return out
        arms: dict[str, Any] = {}
        corr_fn = run_correction_tight if TIGHT else EA.run_correction
        for v in VARIANTS:
            fb = feedback_for(v, row, det, port)
            corrected = corr_fn(row, fb, spans, port)
            arms[v] = {"flagged": True, "feedback_chars": len(fb), "corrected": corrected, "judge_final": P2.judge(row, corrected)}
        out["arms"] = arms
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    by: dict[str, Any] = {}
    for v in VARIANTS:
        fix = sum(1 for r in wrong if (r.get("arms") or {}).get(v, {}).get("judge_final", {}).get("label") == 1)
        brk = sum(1 for r in correct if (r.get("arms") or {}).get(v, {}).get("judge_final", {}).get("label") == 0)
        by[v] = {"fix": fix, "break": brk, "net": fix - brk, "fix_rate_on_wrong": round(fix / max(1, len(wrong)), 3)}
    return {"n_cases": len(rows), "n_wrong": len(wrong), "n_correct": len(correct),
            "union_recall_on_wrong": round(sum(1 for r in wrong if r.get("union_flag")) / max(1, len(wrong)), 3),
            "oracle_ceilings": {"critical_only": 0.667, "lesser_only": 0.409, "all": 0.710, "live_baseline_prior": 0.25},
            "by_variant": by, "errors": sum(1 for r in rows if r.get("error"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--n-correct", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parser", choices=["gpt4o-mini", "helper-v2", "qwen35"], default="helper-v2")
    ap.add_argument("--tight-correction", action="store_true", help="use B1's tight 'fix only the named error' correction to isolate prioritization")
    args = ap.parse_args()
    global TIGHT
    TIGHT = args.tight_correction
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}{'_tight' if TIGHT else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expB1b_prioritization", served=served, args=vars(args))
    print(f"sample={len(sample)} variants={VARIANTS} c={args.concurrency} out={out_dir}", flush=True)
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
