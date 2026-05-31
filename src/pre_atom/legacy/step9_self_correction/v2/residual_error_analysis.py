#!/usr/bin/env python3
"""
Residual error analysis for the multi-model pilot.

For each pilot audit log, find the items where the answer is STILL WRONG
after the pipeline (judge_orig=0 AND outcome.delta != +1) and ask GPT-4o
to classify the residual error type. Aggregate to look for patterns —
e.g. "the pipeline never fixes lab-value misreads on Llama".

This is offline analysis only — uses GPT-4o as a research aid, not as a
runtime decision-maker. ~10-30 calls per model × $0.005 ≈ $0.05/model.

Categories the GPT-4o classifier picks from:
  MISREAD_FACT       — model named the wrong specific fact (drug, dose, date)
  FABRICATED         — model added a claim with no support in the notes
  OMITTED_KEY_FACT   — model dropped a fact essential to the answer
  WRONG_VISIT        — model answered about the wrong visit/admission
  WRONG_TIME_PERIOD  — model answered about the wrong time period (before/after)
  WRONG_BODY_PART    — model focused on the wrong anatomical site
  PARTIAL_COVERAGE   — multi-part question, model covered only some parts
  WRONG_INTERPRETATION — model misinterpreted the question's intent
  REASONING_LEAP     — model made an unsupported logical jump
  PARAPHRASE_ONLY    — corrected answer is essentially the same as the wrong original
  OTHER              — none of the above

Usage:
    python residual_error_analysis.py --models llama3,qwen3 --audit-name regen_p1.jsonl
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
sys.path.insert(0, str(Path(__file__).parent))
from judge import client
from multi_model_pilot import MODELS

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"


CATEGORIES = [
    "MISREAD_FACT",
    "FABRICATED",
    "OMITTED_KEY_FACT",
    "WRONG_VISIT",
    "WRONG_TIME_PERIOD",
    "WRONG_BODY_PART",
    "PARTIAL_COVERAGE",
    "WRONG_INTERPRETATION",
    "REASONING_LEAP",
    "PARAPHRASE_ONLY",
    "OTHER",
]


CLASSIFY_SYS = (
    "You are a medical informatician analyzing why an AI model gave a wrong "
    "answer to a clinical question. Classify the residual error into one of "
    "the listed categories."
)

CLASSIFY_USER_TMPL = """QUESTION:
{question}

GROUND TRUTH (the correct answer):
{ground_truth}

ORIGINAL WRONG ANSWER (the AI's first attempt):
{original_answer}

CORRECTED ANSWER (the AI's second attempt after self-critique; this is also wrong):
{proposed_answer}

The corrected answer is still wrong. Classify the type of residual error
in the corrected answer (the second attempt) using ONE of these categories:

  MISREAD_FACT       — named the wrong specific fact (drug, dose, date)
  FABRICATED         — added a claim with no support in the notes
  OMITTED_KEY_FACT   — dropped a fact essential to the answer
  WRONG_VISIT        — answered about the wrong visit / admission
  WRONG_TIME_PERIOD  — answered about the wrong time period (before/after)
  WRONG_BODY_PART    — focused on the wrong anatomical site
  PARTIAL_COVERAGE   — multi-part question, covered only some parts
  WRONG_INTERPRETATION — misinterpreted the question's intent
  REASONING_LEAP     — made an unsupported logical jump
  PARAPHRASE_ONLY    — essentially the same wrong content as the original
  OTHER              — none of the above

Reply in this exact format:

CATEGORY: <one of the categories above>
EXPLANATION: <one short sentence explaining the classification>"""


def classify(question: str, ground_truth: str, original_answer: str,
             proposed_answer: str) -> dict:
    user = CLASSIFY_USER_TMPL.format(
        question=question[:600],
        ground_truth=ground_truth[:400],
        original_answer=original_answer[:600],
        proposed_answer=proposed_answer[:600],
    )
    for attempt in range(3):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": CLASSIFY_SYS},
                          {"role": "user", "content": user}],
                max_tokens=200,
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            cat = None
            expl = ""
            for line in text.splitlines():
                if line.upper().startswith("CATEGORY:"):
                    val = line.split(":", 1)[1].strip().upper()
                    for c in CATEGORIES:
                        if c in val:
                            cat = c
                            break
                elif line.upper().startswith("EXPLANATION:"):
                    expl = line.split(":", 1)[1].strip()
            return {"category": cat or "OTHER", "explanation": expl, "raw": text}
        except Exception as e:
            print(f"  classify retry {attempt+1}/3: {e}", flush=True)
            time.sleep(5)
    return {"category": None, "explanation": "", "raw": ""}


def analyze_model(model_alias: str, audit_name: str) -> dict:
    cfg = MODELS[model_alias]
    log_path = OUT_DIR / cfg["step8_dir"] / audit_name
    if not log_path.exists():
        print(f"  {model_alias}: no log at {log_path}")
        return {"model": model_alias, "n": 0, "categories": {}}
    recs = [json.loads(l) for l in open(log_path)]

    # Find items where the FINAL answer (after pipeline) is still wrong
    # and the original was wrong (judge_orig=0).
    # This includes: kept_original on wrong items + corrected→still wrong (delta=0)
    residual = []
    for r in recs:
        j_orig = (r.get("judge_orig") or {}).get("label")
        j_cor = (r.get("judge_corrected") or {}).get("label")
        outcome = r.get("outcome") or {}
        # Original was wrong
        if j_orig != 0:
            continue
        # Final answer is wrong (no fix or corrected→still wrong)
        if outcome.get("delta") == 1:
            continue
        # Pull the proposed answer if a correction was generated; else use the original
        proposed = (r.get("correction") or {}).get("proposed", "")
        if not proposed:
            proposed = r["item"]["original_answer"]
        residual.append({
            "fold": r["fold"],
            "idx": r["idx"],
            "question": r["item"]["question"],
            "ground_truth": r["item"]["ground_truth"],
            "original_answer": r["item"]["original_answer"],
            "proposed_answer": proposed,
            "outcome_action": outcome.get("action"),
        })

    print(f"\n{model_alias}: {len(residual)} residual-wrong items to classify")
    classified = []
    for i, item in enumerate(residual, 1):
        result = classify(item["question"], item["ground_truth"],
                          item["original_answer"], item["proposed_answer"])
        item["classification"] = result
        classified.append(item)
        if i % 5 == 0:
            print(f"  {i}/{len(residual)}", flush=True)
        time.sleep(0.5)

    counts = Counter(it["classification"]["category"] for it in classified)
    return {"model": model_alias, "n": len(classified),
            "categories": dict(counts), "items": classified}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="llama3,qwen3",
                   help="comma-separated subset of " + ",".join(MODELS.keys()))
    p.add_argument("--audit-name", default="regen_p1.jsonl",
                   help="audit log filename inside each model dir")
    args = p.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    summaries = []
    for alias in selected:
        s = analyze_model(alias, args.audit_name)
        summaries.append(s)

    print()
    print("=" * 70)
    print("RESIDUAL ERROR ANALYSIS")
    print("=" * 70)
    print(f"{'Model':<14} {'N':>4}  ", end="")
    print("  ".join(f"{c[:12]:>12}" for c in CATEGORIES))
    for s in summaries:
        print(f"{s['model']:<14} {s['n']:>4}  ", end="")
        print("  ".join(f"{s['categories'].get(c, 0):>12}" for c in CATEGORIES))

    out_path = OUT_DIR / "residual_errors.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
