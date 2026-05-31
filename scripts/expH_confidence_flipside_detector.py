#!/usr/bin/env python3
"""Experiment H — confidence flipside as the detector.

expG showed Qwen2.5 is GOOD at confirming what is correct (0.90) but BAD at finding what
is wrong (suspicion precision 0.62, main-error recall 0.40). So flip it: ask Qwen, with
CoT reasoning, what it is CONFIDENT is right — and use the FLIPSIDE (what it is NOT confident
about) as the error detector.

Per case (wrong + correct):
  1. Qwen reviews its ZS answer with step-by-step reasoning, outputs CONFIDENT vs NOT CONFIDENT.
  2. flag = the NOT CONFIDENT section is non-empty (it doubts something material).
  3. GPT-4o judges whether the NOT CONFIDENT doubt correctly points at the real error.

Detector metrics vs the union detector (recall 0.95 / precision 0.64 / error-correctness 0.15):
  recall_on_wrong   : fraction of wrong answers the flipside flags (doubt raised)
  over_flag_on_correct: fraction of correct answers wrongly flagged (false doubt)
  precision, F1
  localization      : of flagged-wrong cases, fraction where the doubt matches the real error
                      (= the flipside's error-correctness)

Real notes (24k). c=8, ledger, blocking.
Output: runs/expH_confidence_flipside/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
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

OUT_ROOT = PROJECT_ROOT / "runs" / "expH_confidence_flipside"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CONF_SYS = "You honestly review your own clinical answer, reasoning step by step about what you are sure of and what you are not."
CONF_REVIEW = """Discharge note:
{note}

Question:
{question}

Your previous answer:
{answer}

Review your answer carefully, step by step. For each claim or part of the answer, reason about whether the discharge note actually supports it. Be honest about uncertainty — do not claim confidence you do not have.

Then end with exactly two labeled sections:
CONFIDENT: the parts you are sure are correct and well supported by the note.
NOT CONFIDENT: the parts you are unsure about or that may be wrong (write "none" if you are fully confident in the whole answer)."""


def confidence_review(row, port) -> dict[str, Any]:
    raw = P2.vllm_chat(CONF_SYS, CONF_REVIEW.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500]), port, 800, 0.0, tag="qwen.confidence")
    # extract NOT CONFIDENT section
    m = re.search(r"NOT\s*CONFIDENT\s*:\s*(.+)$", raw or "", re.I | re.S)
    not_conf = (m.group(1).strip() if m else "").strip()
    # flagged if the NOT CONFIDENT section is non-empty and not "none"
    flagged = bool(not_conf) and not re.match(r"^\(?none\)?\.?$", not_conf.strip(), re.I)
    return {"raw": raw, "not_confident": not_conf[:1500], "flagged": flagged}


def gpt_localize(row, not_conf) -> dict[str, Any]:
    user = (f"Discharge note:\n{row['note'][:18000]}\n\nQuestion:\n{row['question']}\n\n"
            f"Answer given:\n{row['original_answer'][:1500]}\n\nCorrect answer (gold):\n{row['ground_truth'][:800]}\n\n"
            f"The model said it was NOT CONFIDENT about:\n{not_conf[:1500]}\n\n"
            "Is the answer actually wrong? And does the model's stated uncertainty correctly point at the real error in the answer?\n"
            'Return JSON: {"answer_is_wrong":true/false, "doubt_matches_error":true/false}')
    raw = P2.gpt("gpt-4o", "You judge whether a model's expressed doubt matches the real error. Return JSON only.", user, max_tokens=120, temperature=0.0, json_mode=True, tag="gpt.localize")
    m = re.search(r"\{[\s\S]*\}", raw or "")
    try:
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


def process_one(row, port) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        cr = confidence_review(row, port)
        out["flagged"] = cr["flagged"]
        out["not_confident"] = cr["not_confident"]
        if cr["flagged"]:
            out["localize"] = gpt_localize(row, cr["not_confident"])
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
    flagged_wrong = [r for r in wrong if r.get("flagged")]
    loc = sum(1 for r in flagged_wrong if (r.get("localize") or {}).get("doubt_matches_error"))
    return {
        "n_wrong": nw, "n_correct": nc,
        "recall_on_wrong": round(rec, 3),
        "over_flag_on_correct": round(fc / max(1, nc), 3),
        "precision": round(prec, 3), "f1": round(f1, 3),
        "localization_of_flagged_wrong": round(loc / max(1, len(flagged_wrong)), 3),
        "localization_n": f"{loc}/{len(flagged_wrong)}",
        "compare_union": {"recall": 0.95, "precision": 0.64, "f1": 0.76, "error_correctness": "~0.15"},
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
    set_ledger(out_dir / "llm_calls.jsonl", script="expH_confidence_flipside", served=served, args=vars(args))
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
