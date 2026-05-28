#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = PROJECT_ROOT / "refactor" / "pre_atom_pipeline" / "output" / "retrieval_correction"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

NOTE_SPAN_SRC = PROJECT_ROOT / "src" / "step9_self_correction" / "v2"
sys.path.insert(0, str(NOTE_SPAN_SRC))
from note_span_index import topk_spans  # noqa: E402

JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."


def build_judge_user(note: str, question: str, ground_truth: str, model_answer: str) -> str:
    return (
        f"DISCHARGE SUMMARY:\n{note}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
        f"MODEL'S ANSWER:\n{model_answer}\n\n"
        f"Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        f"Respond with ONLY a single digit:\n"
        f"1 = Correct\n"
        f"0 = Incorrect"
    )


def parse_binary(text: str | None) -> int | None:
    if text is None:
        return None
    if "1" in text and "0" not in text:
        return 1
    if "0" in text:
        return 0
    return None


def load_api_key() -> str:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("OPENAI_API_KEY not found")


_openai_client: OpenAI | None = None


def openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=load_api_key())
    return _openai_client


def gpt_judge(note: str, question: str, ground_truth: str, model_answer: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_judge_user(note, question, ground_truth, model_answer)},
    ]
    for attempt in range(5):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=10,
                temperature=0.1,
            )
            raw = (r.choices[0].message.content or "").strip()
            return {"label": parse_binary(raw), "raw": raw, "model": "gpt-4o", "temperature": 0.1}
        except Exception as e:
            if attempt == 4:
                return {"label": None, "raw": "", "error": str(e), "model": "gpt-4o", "temperature": 0.1}
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def served_model_id(port: int) -> str:
    r = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    if not data:
        raise RuntimeError(f"No model served on port {port}")
    return data[0]["id"]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I).strip()
    if "</think>" in text.lower():
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL | re.I).strip()
    return text


def vllm_chat(system: str, user: str, port: int, max_tokens: int = 512, temperature: float = 0.0) -> str:
    model = served_model_id(port)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
    body = r.json()
    if "choices" not in body:
        payload["messages"] = [{"role": "user", "content": f"{system}\n\n{user}"}]
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
        body = r.json()
    if "choices" not in body:
        raise RuntimeError(str(body))
    return strip_think((body["choices"][0]["message"]["content"] or "").strip())


def load_notes_lookup() -> dict[str, str]:
    df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    out = {}
    for _, r in df.iterrows():
        parts = []
        for i in (1, 2, 3):
            t = r.get(f"note_{i}")
            if pd.notna(t) and str(t).strip() and str(t).strip().lower() != "nan":
                parts.append(f"[Note {i}]\n{str(t).strip()}")
        out[str(int(r["patient_id"]))] = "\n\n".join(parts)
    return out


def load_qwen25_rows() -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        p = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows.append({
                "fold": fold,
                "idx": int(r["idx"]),
                "patient_id": int(r["patient_id"]),
                "question": str(r["question"]),
                "ground_truth": str(r["ground_truth"]),
                "original_answer": str(r["model_answer"]),
                "orig_label": int(r["binary_correct"]),
            })
    return rows


def load_taxonomy() -> dict[tuple[int, int], dict[str, Any]]:
    p = PROJECT_ROOT / "src" / "step9_self_correction" / "error_taxonomy" / "phase1_wrong_gpt4o.json"
    items = json.loads(p.read_text())
    return {(int(r["fold"]), int(r["idx"])): r for r in items}


def make_sample(n_wrong: int, n_correct: int, seed: int) -> list[dict[str, Any]]:
    rows = load_qwen25_rows()
    tax = load_taxonomy()
    notes = load_notes_lookup()
    wrong = []
    correct = []
    for r in rows:
        r["note"] = notes[str(r["patient_id"])]
        t = tax.get((r["fold"], r["idx"]))
        if t:
            r["taxonomy"] = {
                "primary_error": t.get("PRIMARY_ERROR", ""),
                "error_description": t.get("ERROR_DESCRIPTION", ""),
                "question_focus": t.get("QUESTION_FOCUS", ""),
                "model_claims": t.get("MODEL_CLAIMS", ""),
            }
        else:
            r["taxonomy"] = None
        if r["orig_label"] == 0:
            wrong.append(r)
        else:
            correct.append(r)
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


def spans_block(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return "(no spans retrieved)"
    return "\n".join(f"[{i + 1}] {s['sentence']}" for i, s in enumerate(spans))


def retrieve_spans(row: dict[str, Any], mode: str, k: int) -> list[dict[str, Any]]:
    tax = row.get("taxonomy") or {}
    base_queries = [row["question"], row["original_answer"][:800]]
    if mode == "gtr_question":
        queries = [row["question"]]
        scoring = "max"
    elif mode == "gtr_q_answer":
        queries = base_queries
        scoring = "agreement"
    elif mode == "gtr_oracle_error":
        queries = base_queries + [
            tax.get("error_description", ""),
            tax.get("question_focus", ""),
            tax.get("model_claims", ""),
        ]
        scoring = "agreement"
    else:
        raise ValueError(f"unknown retrieval mode: {mode}")
    return topk_spans(row["note"], queries, k=k, scoring=scoring)


CORRECTION_SYSTEM = (
    "You are a careful clinical QA assistant. Revise an answer only when the "
    "provided note evidence supports the revision. Do not add facts that are "
    "not supported by the evidence or discharge note."
)


def build_correction_user(row: dict[str, Any], spans: list[dict[str, Any]], arm: str) -> str:
    tax = row.get("taxonomy") or {}
    primary = tax.get("primary_error") or "UNKNOWN"
    if arm == "evidence_only":
        instruction = (
            "Use the evidence spans to check the previous answer. Return the best final answer. "
            "If the previous answer is already correct, keep it."
        )
    elif arm == "taxonomy_evidence":
        if primary in {"MISREADING", "FABRICATION"}:
            action = "Look for contradicted or unsupported claims, then replace or remove them."
        elif primary == "OMISSION":
            action = "Look for missing answer components and include only evidence-supported missing facts."
        elif primary == "QUESTION_MISALIGNMENT":
            action = "First identify the exact visit, date, or aspect asked by the question, then answer that target."
        else:
            action = "Check whether the prior answer directly answers the question and is supported by the spans."
        instruction = (
            f"The suspected error type is {primary}. {action} "
            "Return the best final answer in 1-3 sentences."
        )
    elif arm == "oracle_error_description":
        instruction = (
            f"The suspected error type is {primary}. Prior error analysis: "
            f"{tax.get('error_description', '')} "
            "Use this only as a hint; the final answer must be supported by the note evidence."
        )
    else:
        raise ValueError(f"unknown arm: {arm}")
    return f"""Discharge note:
{row['note'][:18000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Retrieved note evidence:
{spans_block(spans)}

Instruction:
{instruction}

Final answer:"""


def run_one(row: dict[str, Any], arm: str, retrieval_mode: str, k: int, port: int, spans: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out = {
        "fold": row["fold"],
        "idx": row["idx"],
        "patient_id": row["patient_id"],
        "orig_label": row["orig_label"],
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "original_answer": row["original_answer"],
        "taxonomy": row.get("taxonomy"),
        "arm": arm,
        "retrieval_mode": retrieval_mode,
        "temperature": 0.0,
    }
    try:
        if spans is None:
            spans = retrieve_spans(row, retrieval_mode, k)
        user = build_correction_user(row, spans, arm)
        ans = vllm_chat(CORRECTION_SYSTEM, user, port=port, max_tokens=512, temperature=0.0)
        out["spans"] = spans
        out["corrected_answer"] = ans
        out["error"] = None
    except Exception as e:
        out["spans"] = []
        out["corrected_answer"] = ""
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted({r["arm"] for r in rows}):
        subset = [r for r in rows if r["arm"] == arm]
        judged = [r for r in subset if (r.get("judge") or {}).get("label") is not None]
        fixes = sum(1 for r in judged if r["orig_label"] == 0 and r["judge"]["label"] == 1)
        breaks = sum(1 for r in judged if r["orig_label"] == 1 and r["judge"]["label"] == 0)
        kept_wrong = sum(1 for r in judged if r["orig_label"] == 0 and r["judge"]["label"] == 0)
        kept_correct = sum(1 for r in judged if r["orig_label"] == 1 and r["judge"]["label"] == 1)
        by_type = defaultdict(Counter)
        for r in judged:
            typ = ((r.get("taxonomy") or {}).get("primary_error") or "CORRECT_OR_UNKNOWN")
            if r["orig_label"] == 0 and r["judge"]["label"] == 1:
                by_type[typ]["fix"] += 1
            elif r["orig_label"] == 0:
                by_type[typ]["still_wrong"] += 1
            elif r["orig_label"] == 1 and r["judge"]["label"] == 0:
                by_type[typ]["break"] += 1
            else:
                by_type[typ]["still_correct"] += 1
        by_arm[arm] = {
            "n": len(subset),
            "n_judged": len(judged),
            "fixes": fixes,
            "breaks": breaks,
            "net": fixes - breaks,
            "kept_wrong": kept_wrong,
            "kept_correct": kept_correct,
            "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
            "errors": sum(1 for r in subset if r.get("error")),
        }
    return {"arms": by_arm}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retrieval-workers", type=int, default=4)
    ap.add_argument("--n-wrong", type=int, default=-1, help="-1 means all Qwen2.5 wrong cases")
    ap.add_argument("--n-correct", type=int, default=109)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=["evidence_only", "taxonomy_evidence"])
    ap.add_argument("--retrieval-mode", default="gtr_q_answer", choices=["gtr_question", "gtr_q_answer", "gtr_oracle_error"])
    ap.add_argument("--judge", action="store_true", help="Run sequential GPT-4o judge after vLLM generation")
    args = ap.parse_args()

    served = served_model_id(args.port)
    if "qwen2.5" not in served.lower() and "qwen2" not in served.lower():
        raise RuntimeError(f"Expected Qwen2.5 on port {args.port}, found {served}")

    sample = make_sample(args.n_wrong, args.n_correct, args.seed)
    run_id = f"qwen25_{args.retrieval_mode}_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"served_model={served}")
    print(f"sample={len(sample)} arms={args.arms} concurrency={args.concurrency} retrieval_workers={args.retrieval_workers} judge={args.judge}", flush=True)

    # Warm the lazy SentenceTransformer singleton before worker threads start.
    # Concurrent first-loads can race inside torch/safetensors and fail with
    # meta-tensor copy errors.
    if sample:
        _ = retrieve_spans(sample[0], args.retrieval_mode, min(args.k, 1))

    span_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    def retrieve_for_cache(row: dict[str, Any]) -> tuple[tuple[int, int], list[dict[str, Any]]]:
        key = (row["fold"], row["idx"])
        return key, retrieve_spans(row, args.retrieval_mode, args.k)

    with ThreadPoolExecutor(max_workers=max(1, args.retrieval_workers)) as ex:
        futs = [ex.submit(retrieve_for_cache, row) for row in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            key, spans = fut.result()
            span_cache[key] = spans
            if i % 10 == 0 or i == len(futs):
                print(f"retrieved {i}/{len(futs)}", flush=True)

    generated: list[dict[str, Any]] = []
    tasks = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for row in sample:
            spans = span_cache[(row["fold"], row["idx"])]
            for arm in args.arms:
                tasks.append(ex.submit(run_one, row, arm, args.retrieval_mode, args.k, args.port, spans))
        for fut in as_completed(tasks):
            generated.append(fut.result())
            if len(generated) % 10 == 0 or len(generated) == len(tasks):
                print(f"generated {len(generated)}/{len(tasks)}", flush=True)

    raw_path = out_dir / "generated.jsonl"
    write_jsonl(raw_path, generated)

    if args.judge:
        for i, r in enumerate(generated, 1):
            if r.get("error") or not r.get("corrected_answer"):
                r["judge"] = {"label": None, "raw": "", "skipped": True}
            else:
                note = next(row["note"] for row in sample if row["fold"] == r["fold"] and row["idx"] == r["idx"])
                r["judge"] = gpt_judge(note, r["question"], r["ground_truth"], r["corrected_answer"])
            if i % 10 == 0 or i == len(generated):
                print(f"judged {i}/{len(generated)}", flush=True)
        judged_path = out_dir / "judged.jsonl"
        write_jsonl(judged_path, generated)
    else:
        judged_path = None

    summary = {
        "task": "qwen25_retrieval_correction_quicktest",
        "served_model": served,
        "settings": {
            "port": args.port,
            "concurrency": args.concurrency,
            "retrieval_workers": args.retrieval_workers,
            "n_wrong": args.n_wrong,
            "n_correct": args.n_correct,
            "seed": args.seed,
            "k": args.k,
            "arms": args.arms,
            "retrieval_mode": args.retrieval_mode,
            "generation_temperature": 0.0,
            "judge": "gpt-4o old stage1 prompt temperature 0.1 sequential" if args.judge else None,
        },
        "sample_counts": dict(Counter(r["orig_label"] for r in sample)),
        "outputs": {"generated_jsonl": str(raw_path), "judged_jsonl": str(judged_path) if judged_path else None},
    }
    if args.judge:
        summary.update(summarize(generated))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = [
        "# Qwen2.5 Retrieval Correction Quick Test",
        "",
        f"- served model: `{served}`",
        f"- retrieval mode: `{args.retrieval_mode}`",
        f"- concurrency: `{args.concurrency}`",
        f"- generated rows: `{len(generated)}`",
        f"- judge: `{summary['settings']['judge']}`",
        "",
    ]
    if args.judge:
        md.append("| Arm | N judged | Fix | Break | Net |")
        md.append("|---|---:|---:|---:|---:|")
        for arm, s in summary["arms"].items():
            md.append(f"| `{arm}` | {s['n_judged']} | {s['fixes']} | {s['breaks']} | {s['net']} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
