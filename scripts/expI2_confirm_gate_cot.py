#!/usr/bin/env python3
"""Experiment I2 — does CoT make the confirm-gate more conservative (better precision)?

Gate-only test (no correction). Two variants of the confirm-gate, measured for
recall / precision / over-flag / F1 on wrong + correct cases:

  confirm_plain : claim-by-claim SUPPORTED / UNCONFIRMED (expI gate).
  confirm_cot   : reason step by step per claim BEFORE deciding, with an explicit
                  conservative instruction (only UNCONFIRMED if the note clearly fails to
                  support or contradicts).

Use as a PRECISION GATE candidate (to pair with the two-round-blind diagnoser, expJ).
Goal: does CoT push over-flag below the plain gate's 0.60 without killing recall?

Wrong + correct, real notes (24k), c=8, ledger, blocking.
Output: runs/expI2_confirm_gate_cot/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
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

OUT_ROOT = PROJECT_ROOT / "runs" / "expI2_confirm_gate_cot"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["confirm_plain", "confirm_cot"]

GATE_SYS = "You verify a clinical answer against the discharge note, confirming what is clearly supported."

PLAIN = """Discharge note:
{note}

Question:
{question}

Answer to verify:
{answer}

Go through the answer claim by claim. For each, check whether the discharge note clearly SUPPORTS it.
End with exactly two sections:
SUPPORTED: the claims the note clearly supports.
UNCONFIRMED: the claims the note does NOT clearly support, or that may be wrong or off-topic (write "none" if every claim is clearly supported and the answer correctly answers the question)."""

COT = """Discharge note:
{note}

Question:
{question}

Answer to verify:
{answer}

Go through the answer claim by claim. For EACH claim, reason step by step: what does the note actually say about this claim? Does it clearly support it, contradict it, or stay silent? Only after reasoning, decide.

Be conservative: mark a claim UNCONFIRMED only if the note clearly CONTRADICTS it or a fact REQUIRED to answer the question is clearly MISSING. Do not mark a claim unconfirmed merely because the note does not restate it or for extra detail.

End with exactly two sections:
SUPPORTED: the claims that are clearly supported.
UNCONFIRMED: the claims clearly contradicted or with a required fact missing (write "none" if the answer correctly answers the question)."""

PROMPTS = {"confirm_plain": PLAIN, "confirm_cot": COT}


def gate(variant, row, port) -> bool:
    raw = P2.vllm_chat(GATE_SYS, PROMPTS[variant].format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500]), port, 800 if variant == "confirm_cot" else 700, 0.0, tag=f"gate.{variant}")
    m = re.search(r"UNCONFIRMED\s*:\s*(.+)$", raw or "", re.I | re.S)
    unconf = (m.group(1).strip() if m else "").strip()
    return bool(unconf) and not re.match(r"^\(?none\)?\.?\s*$", unconf.strip(), re.I)


def process_one(row, port) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        out["flag"] = {v: gate(v, row, port) for v in VARIANTS}
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows):
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    nw, nc = len(wrong), len(correct)
    by = {}
    for v in VARIANTS:
        fw = sum(1 for r in wrong if (r.get("flag") or {}).get(v))
        fc = sum(1 for r in correct if (r.get("flag") or {}).get(v))
        rec = fw / max(1, nw)
        prec = fw / max(1, fw + fc)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        by[v] = {"recall": round(rec, 3), "over_flag": round(fc / max(1, nc), 3), "precision": round(prec, 3), "f1": round(f1, 3)}
    return {"n_wrong": nw, "n_correct": nc, "by_variant": by,
            "ref": {"expI_confirm_plain": "rec0.68/of0.60", "union": "rec0.95/of0.79"},
            "errors": sum(1 for r in rows if r.get("error"))}


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
    set_ledger(out_dir / "llm_calls.jsonl", script="expI2_confirm_gate_cot", served=served, args=vars(args))
    print(f"sample={len(sample)} c={args.concurrency} out={out_dir}", flush=True)
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
