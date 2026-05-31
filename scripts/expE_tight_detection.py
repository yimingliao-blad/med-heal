#!/usr/bin/env python3
"""Experiment E — tighter detection: union (recall) -> material-error filter (precision).

Detection over-flag (0.78) is the error source (expC). The union (k=3 natural any-flag)
maximizes recall but flags nitpicks on correct answers. This cascades a material-error
FILTER after the union: keep a flag only if the concern is a MATERIAL error that makes the
answer wrong for the question — reject nitpicks / already-correct answers. The planner's
conservatism (weak as a standalone detector) becomes useful as a precision gate, because
the union already supplied recall.

Filter variants:
  material_default : MATERIAL (wrong/incomplete for the question) vs ACCEPTABLE (nitpick/correct).
  material_strict  : MATERIAL only if the note directly contradicts the answer or a required
                     fact is clearly missing; otherwise ACCEPTABLE.

Measures recall / precision / F1 / over-flag for: union-alone, union->default, union->strict,
and majority(>=2) as a reference threshold. Goal: cut over-flag while holding recall.

Wrong + correct cases. Real notes (24k), c=8, ledger, blocking.
Output: runs/expE_tight_detection/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
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
import expA_detection_feedback as EA  # noqa: E402
import expC_two_stage_verdict as EC  # noqa: E402  (reuse error_focus)
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expE_tight_detection"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FILTERS = ["material_default", "material_strict"]

FILTER_SYS = "You decide whether a flagged concern is a real, material error or just a nitpick. Reply with only one word."

FILTER_DEFAULT = """Question:
{question}

Answer given:
{answer}

Note facts relevant to the question:
{question_facts}

A check flagged this concern about the answer:
{error}

Decide: is this a MATERIAL error that makes the answer wrong or incomplete for the exact question? Or is the answer ACCEPTABLE (the concern is a minor nitpick, a matter of style or extra detail, or the answer is already correct for what was asked)?
Reply only MATERIAL or ACCEPTABLE."""

FILTER_STRICT = """Question:
{question}

Answer given:
{answer}

Note facts relevant to the question:
{question_facts}

A check flagged this concern about the answer:
{error}

Reply MATERIAL only if the note DIRECTLY CONTRADICTS the answer, or a fact REQUIRED to answer the question is clearly MISSING. If the concern is a nitpick, extra detail, phrasing, or the answer is already acceptable for the exact question, reply ACCEPTABLE.
Reply only MATERIAL or ACCEPTABLE."""

FILTER_PROMPTS = {"material_default": FILTER_DEFAULT, "material_strict": FILTER_STRICT}


def material_filter(variant: str, row: dict[str, Any], det: dict[str, Any], error: str, port: int) -> bool:
    tmpl = FILTER_PROMPTS[variant]
    raw = P2.vllm_chat(FILTER_SYS, tmpl.format(question=row["question"], answer=row["original_answer"][:1500], question_facts=det["question_facts"][:3000], error=error[:2500]), port, 8, 0.0, tag=f"filter.{variant}")
    return "MATERIAL" in (raw or "").upper()


def process_one(row: dict[str, Any], port: int, parser: str) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        det = EA.detect_k3_union(row, port, parser)
        out["union_flag"] = det["union_flag"]
        n_incorrect = sum(1 for v in det["verdicts"] if v == "INCORRECT")
        out["n_incorrect"] = n_incorrect
        out["majority_flag"] = n_incorrect >= 2
        if det["union_flag"]:
            error = EC.error_focus(row, det, port)
            out["error_stmt"] = error
            out["filters"] = {v: material_filter(v, row, det, error, port) for v in FILTERS}
        else:
            out["filters"] = {v: False for v in FILTERS}
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def pr(flagged_wrong: int, flagged_correct: int, n_wrong: int, n_correct: int) -> dict[str, float]:
    rec = flagged_wrong / max(1, n_wrong)
    prec = flagged_wrong / max(1, flagged_wrong + flagged_correct)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"recall": round(rec, 3), "precision": round(prec, 3), "f1": round(f1, 3), "over_flag": round(flagged_correct / max(1, n_correct), 3)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    nw, nc = len(wrong), len(correct)
    out: dict[str, Any] = {}
    # union alone
    out["union"] = pr(sum(1 for r in wrong if r.get("union_flag")), sum(1 for r in correct if r.get("union_flag")), nw, nc)
    # majority >=2
    out["majority>=2"] = pr(sum(1 for r in wrong if r.get("majority_flag")), sum(1 for r in correct if r.get("majority_flag")), nw, nc)
    # union -> material filter variants
    for v in FILTERS:
        fw = sum(1 for r in wrong if (r.get("filters") or {}).get(v))
        fc = sum(1 for r in correct if (r.get("filters") or {}).get(v))
        out[f"union->{v}"] = pr(fw, fc, nw, nc)
    return {"n_wrong": nw, "n_correct": nc, "by_config": out, "errors": sum(1 for r in rows if r.get("error"))}


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
    set_ledger(out_dir / "llm_calls.jsonl", script="expE_tight_detection", served=served, args=vars(args))
    print(f"sample={len(sample)} filters={FILTERS} c={args.concurrency} out={out_dir}", flush=True)
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
