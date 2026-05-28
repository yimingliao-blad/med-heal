#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", ROOT.parent / "llm-ehr-hallucination"))
OUT_ROOT = ROOT / "runs" / "regen_verdict"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from pre_atom.llm import gpt4o_stage1_judge, gpt4omini_extract_json, served_model_id, vllm_chat


MODEL_STEP8_DIR = {
    "biomistral": "biomistral-7b",
    "biomistral-7b": "biomistral-7b",
    "qwen2.5": "qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "qwen3": "qwen3-8b",
    "qwen3-8b": "qwen3-8b",
    "deepseek": "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8b",
    "llama3": "llama-3.1-8b-instruct",
    "llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
}

REGEN_SYSTEM = "You are a medical expert answering questions about discharge summaries."
REGEN_USER = """Discharge note:
{note}

Question:
{question}

Answer the question using only information from the discharge note. Be specific and complete. If the question asks about multiple visits, conditions, or events, cover all of them."""

VERDICT_SYSTEM = "You are a strict medical expert comparing two clinical answers against discharge notes."
VERDICT_USER = """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

For each answer, count three things separately:

1. CONTRADICTIONS: factual claims that contradict the discharge notes.
2. UNADDRESSED_PARTS: parts of the question that the answer does not address.
3. UNSUPPORTED_CLAIMS: claims in the answer that have no supporting sentence in the discharge notes.

Choose the answer with the lower total. If both answers have the same total, choose A.

State your counts and final choice in natural text."""

EXTRACT_SYSTEM = (
    "You extract fields from a medical model's verdict text. Do not decide which "
    "answer is better yourself. Extract only what the model stated."
)
EXTRACT_USER = """Model verdict text:
{raw}

Return JSON only:
{{
  "a_contradictions": 0,
  "a_unaddressed": 0,
  "a_unsupported": 0,
  "b_contradictions": 0,
  "b_unaddressed": 0,
  "b_unsupported": 0,
  "winner": "A|B|UNCLEAR",
  "reason": "string",
  "parse_valid": true
}}

If a count or winner is not stated, use null for that count and UNCLEAR for winner."""


def load_notes() -> dict[str, str]:
    df = pd.read_json(SOURCE_REPO / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    out = {}
    for _, r in df.iterrows():
        parts = []
        for i in (1, 2, 3):
            t = r.get(f"note_{i}")
            if pd.notna(t) and str(t).strip() and str(t).strip().lower() != "nan":
                parts.append(f"[Note {i}]\n{str(t).strip()}")
        out[str(int(r["patient_id"]))] = "\n\n".join(parts)
    return out


def row_label(r: pd.Series) -> int:
    if "binary_correct" in r:
        return int(r["binary_correct"])
    if "label" in r:
        return int(r["label"])
    raise KeyError("step8 row needs binary_correct or label")


def load_rows(model: str, n_wrong: int, n_correct: int, seed: int) -> list[dict[str, Any]]:
    step8_dir = MODEL_STEP8_DIR[model]
    notes = load_notes()
    rows = []
    for fold in range(5):
        p = SOURCE_REPO / "output" / "step8" / step8_dir / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            pid = int(r["patient_id"])
            rows.append(
                {
                    "fold": fold,
                    "idx": int(r["idx"]),
                    "patient_id": pid,
                    "question": str(r["question"]),
                    "ground_truth": str(r["ground_truth"]),
                    "answer": str(r["model_answer"]),
                    "orig_label": row_label(r),
                    "note": notes[str(pid)],
                }
            )
    wrong = [r for r in rows if r["orig_label"] == 0]
    correct = [r for r in rows if r["orig_label"] == 1]
    rng = random.Random(seed)
    rng.shuffle(wrong)
    rng.shuffle(correct)
    if n_wrong < 0:
        n_wrong = len(wrong)
    if n_correct < 0:
        n_correct = len(correct)
    sample = wrong[: min(n_wrong, len(wrong))] + correct[: min(n_correct, len(correct))]
    rng.shuffle(sample)
    return sample


def parse_verdict(raw: str) -> dict[str, Any]:
    obj = gpt4omini_extract_json(EXTRACT_SYSTEM, EXTRACT_USER.format(raw=(raw or "")[:5000]), max_tokens=400)
    winner = str(obj.get("winner", "UNCLEAR")).upper()
    if winner not in {"A", "B", "UNCLEAR"}:
        winner = "UNCLEAR"
    out = {"winner": winner, "reason": str(obj.get("reason", "")), "parser_raw": obj, "parse_path": "gpt-4o-mini"}
    for key in ["a_contradictions", "a_unaddressed", "a_unsupported", "b_contradictions", "b_unaddressed", "b_unsupported"]:
        try:
            out[key] = None if obj.get(key) is None else int(obj.get(key))
        except Exception:
            out[key] = None
    out["parse_valid"] = "error" not in obj and winner in {"A", "B"}
    return out


def process_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "answer", "orig_label"]}
    try:
        regen = vllm_chat(
            REGEN_SYSTEM,
            REGEN_USER.format(note=row["note"][:18000], question=row["question"]),
            port=args.port,
            max_tokens=args.max_tokens,
            temperature=args.regen_temperature,
        )
        rng = random.Random(args.seed + (row["fold"] << 16) + row["idx"])
        orig_in_a = rng.random() > 0.5
        answer_a = row["answer"] if orig_in_a else regen
        answer_b = regen if orig_in_a else row["answer"]
        verdict_raw = vllm_chat(
            VERDICT_SYSTEM,
            VERDICT_USER.format(note=row["note"][:18000], question=row["question"], answer_a=answer_a[:1600], answer_b=answer_b[:1600]),
            port=args.port,
            max_tokens=700,
            temperature=args.verdict_temperature,
        )
        parsed = parse_verdict(verdict_raw)
        corrected_slot = "B" if orig_in_a else "A"
        accept = parsed["winner"] == corrected_slot
        final_answer = regen if accept else row["answer"]
        out.update(
            {
                "regen_answer": regen,
                "verdict": {
                    "raw": verdict_raw,
                    "parsed": parsed,
                    "orig_in_slot_A": orig_in_a,
                    "corrected_slot": corrected_slot,
                    "accept_correction": accept,
                    "decision_owner": "tested_model",
                    "parser_role": "gpt-4o-mini extraction only",
                },
                "action": "accepted_regen" if accept else "kept_original",
                "final_answer": final_answer,
            }
        )
    except Exception as e:
        out.update({"error": str(e), "action": "error_keep_original", "final_answer": row["answer"]})
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [r for r in rows if (r.get("judge_final") or {}).get("label") is not None]
    fixes = sum(1 for r in judged if r["orig_label"] == 0 and r["judge_final"]["label"] == 1)
    breaks = sum(1 for r in judged if r["orig_label"] == 1 and r["judge_final"]["label"] == 0)
    return {
        "n": len(rows),
        "n_judged": len(judged),
        "actions": dict(Counter(r.get("action") for r in rows)),
        "accepted": sum(1 for r in rows if r.get("action") == "accepted_regen"),
        "fixes": fixes,
        "breaks": breaks,
        "net": fixes - breaks,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_STEP8_DIR), default="qwen2.5")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=2)
    ap.add_argument("--n-correct", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regen-temperature", type=float, default=0.7)
    ap.add_argument("--verdict-temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    served = served_model_id(args.port)
    sample = load_rows(args.model, args.n_wrong, args.n_correct, args.seed)
    run_id = f"{MODEL_STEP8_DIR[args.model]}_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}_regen_verdict_mini"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"served_model={served} model={args.model} sample={len(sample)} c={args.concurrency}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, row, args) for row in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            write_jsonl(out_dir / "pipeline_outputs.jsonl", rows)
            print(f"pipeline {i}/{len(futs)}", flush=True)

    if args.judge:
        note_by_key = {(r["fold"], r["idx"]): r["note"] for r in sample}
        for i, r in enumerate(rows, 1):
            note = note_by_key[(r["fold"], r["idx"])]
            r["judge_final"] = gpt4o_stage1_judge(note, r["question"], r["ground_truth"], r["final_answer"])
            if i % 10 == 0 or i == len(rows):
                print(f"judged {i}/{len(rows)}", flush=True)
        write_jsonl(out_dir / "judged_outputs.jsonl", rows)

    summary = {
        "task": "regen_plus_verdict_test_model_decides_gpt4omini_extracts",
        "served_model": served,
        "settings": vars(args),
        "summary": summarize(rows),
        "outputs": {
            "pipeline": str(out_dir / "pipeline_outputs.jsonl"),
            "judged": str(out_dir / "judged_outputs.jsonl") if args.judge else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
