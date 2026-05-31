#!/usr/bin/env python3
"""Experiment B1c — the oracle correction's BREAK rate on CORRECT cases (the ceiling test).

B1/B2 measured oracle correction fix-rate (67%) but only ran on WRONG cases, so the oracle
pipeline's break rate was never measured — my full-scale projection wrongly borrowed the
LIVE break rate. This measures it directly:

For each CORRECT answer, run the same oracle error-ranking (GPT-4o, given the answer + ground
truth) and ask for its errors. Expectation: the oracle returns errors=[] (no over-flag) ->
no correction -> 0 breaks. If instead it hallucinates errors, correction runs and may break.

Reports: oracle over-flag rate (correct cases the oracle flags) and oracle break rate
(flagged-correct cases that correction turns wrong). Establishes the true pipeline CEILING:
oracle fix-rate (67%) paired with the real oracle break-rate.

Correct cases only. Real notes (24k), c=8, ledger, blocking.
Output: runs/expB1c_oracle_break/qwen25_nc{NC}/{judged_outputs.jsonl, summary.json}
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
import expB1_oracle_ranked_errors as B1  # noqa: E402  (reuse rank_errors, run_correction)
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expB1c_oracle_break"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def process_one(row: dict[str, Any], port: int) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        # oracle ranks errors in this (correct) answer, given ground truth
        errors = B1.rank_errors(row)
        out["oracle_errors"] = errors
        out["oracle_flagged"] = len(errors) > 0
        if errors:
            spans = P2.retrieve_spans(row, k=5)
            corrected = B1.run_correction(row, errors, spans, port, tag="correction.oracle_on_correct")
            out["corrected"] = corrected
            out["judge_corrected"] = P2.judge(row, corrected)
        else:
            out["judge_corrected"] = None
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # restrict to cases the fresh judge agrees are originally CORRECT
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    flagged = [r for r in correct if r.get("oracle_flagged")]
    broke = [r for r in flagged if (r.get("judge_corrected") or {}).get("label") == 0]
    return {
        "n_cases": len(rows),
        "n_correct_by_judge": len(correct),
        "oracle_over_flag": {"flagged": len(flagged), "rate": round(len(flagged) / max(1, len(correct)), 3)},
        "oracle_break": {"breaks": len(broke), "rate_of_correct": round(len(broke) / max(1, len(correct)), 3),
                         "rate_of_flagged": round(len(broke) / max(1, len(flagged)), 3)},
        "note": "ceiling pipeline = oracle fix-rate 0.67 (B1) paired with this oracle break-rate",
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-correct", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = P2.load_rows(0, args.n_correct, args.seed)  # correct cases only
    out_dir = OUT_ROOT / f"qwen25_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expB1c_oracle_break", served=served, args=vars(args))
    print(f"sample={len(sample)} (correct cases) c={args.concurrency} out={out_dir}", flush=True)
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
