#!/usr/bin/env python3
"""Phase 1: what information helps the model correct its error?

Oracle ablation on Qwen2.5 wrong cases. Each arm adds ONE information source to a
fixed baseline (question + wrong answer + same-patient spans). All oracle inputs
come from the offline GPT-4o error taxonomy (phase1_wrong_gpt4o.json) so we measure
the VALUE of each information source, separate from whether live detection can
produce it (that is Phase 2).

Note context is held constant (first18k) for Phase 1 — context-window tuning is a
separate later sweep.

Arms:
  1 baseline                 Q + wrong answer + spans
  2 +question_type           GPT-4o-mini one-line question-type (date/number/lab/medication/treatment/list/other)
  3 +question_focus          taxonomy QUESTION_FOCUS
  4 +error_type              taxonomy PRIMARY_ERROR
  5 +error_location          taxonomy MODEL_CLAIMS (where the wrong claim sits)
  6 +contradiction_quote     taxonomy ERROR_DESCRIPTION (oracle description of the contradiction)
  7 +analogy_wrong           similar past WRONG case from bm_contrast_pool (T2-matched by mapped op)
  8 +analogy_correct         similar past CORRECT answer pattern (pool ground_truth)
  9 +all                     union of 2..8

For arms 7 and 8 a per-case analogy-quality rating is recorded (GPT-4o-mini) so we
see retrieval quality alongside the correction effect.

Pre-flight: Qwen2.5-7B-Instruct on vLLM port 8003.

Output:
  runs/phase1_info_ablation/qwen25_nw{N}_seed{SEED}/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", PROJECT_ROOT.parent / "llm-ehr-hallucination"))
OUT_ROOT = PROJECT_ROOT / "runs" / "phase1_info_ablation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

POOL_DIR = SOURCE_REPO / "workspace" / "self_critique" / "data" / "bm_contrast_pool"
TAXONOMY = SOURCE_REPO / "src" / "step9_self_correction" / "error_taxonomy" / "phase1_wrong_gpt4o.json"
NOTE_SPAN_SRC = SOURCE_REPO / "src" / "step9_self_correction" / "v2"
sys.path.insert(0, str(NOTE_SPAN_SRC))
from note_span_index import topk_spans  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from llm_audit import set_ledger, log_call  # noqa: E402

ARMS = [
    "baseline",
    "question_type",
    "question_focus",
    "error_type",
    "error_location",
    "contradiction_quote",
    "analogy_wrong",
    "analogy_correct",
    "all",
]


# ---------- OpenAI ----------

def load_api_key() -> str:
    for env in (PROJECT_ROOT / ".env", SOURCE_REPO / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("OPENAI_API_KEY not found")


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=load_api_key())
    return _client


def gpt(model: str, system: str, user: str, max_tokens: int, temperature: float = 0.0, tag: str = "gpt") -> str:
    for attempt in range(4):
        try:
            r = client().chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature,
            )
            out = (r.choices[0].message.content or "").strip()
            log_call(tag, model, system, user, out, temperature=temperature, max_tokens=max_tokens)
            return out
        except Exception:
            if attempt == 3:
                log_call(tag + ".fail", model, system, user, "", temperature=temperature, max_tokens=max_tokens)
                return ""
            time.sleep(1 + attempt)
    return ""


# ---------- vLLM ----------

def served_model_id(port: int) -> str:
    r = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I).strip()
    if "</think>" in text.lower():
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL | re.I).strip()
    return text


def vllm_chat(system: str, user: str, port: int, max_tokens: int = 700, temperature: float = 0.0) -> str:
    model = served_model_id(port)
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
    body = r.json()
    if "choices" not in body:
        raise RuntimeError(str(body))
    out = strip_think((body["choices"][0]["message"]["content"] or "").strip())
    log_call("vllm", model, system, user, out, temperature=temperature, max_tokens=max_tokens)
    return out


# ---------- data ----------

def load_notes() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (SOURCE_REPO / "output" / "EHRNoteQA_processed.jsonl").open():
        d = json.loads(line)
        parts = []
        for i in (1, 2, 3):
            t = d.get(f"note_{i}")
            if t and str(t).strip() and str(t).strip().lower() != "nan":
                parts.append(f"[Note {i}]\n{str(t).strip()}")
        out[str(d["patient_id"])] = "\n\n".join(parts)
    return out


def load_taxonomy() -> dict[tuple[int, int], dict[str, Any]]:
    items = json.loads(TAXONOMY.read_text())
    return {(int(r["fold"]), int(r["idx"])): r for r in items}


def load_wrong_rows(n_wrong: int, seed: int) -> list[dict[str, Any]]:
    notes = load_notes()
    tax = load_taxonomy()
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        df = pd.read_csv(SOURCE_REPO / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv")
        for _, r in df.iterrows():
            if int(r["binary_correct"]) != 0:
                continue
            t = tax.get((fold, int(r["idx"]))) or {}
            rows.append({
                "fold": fold, "idx": int(r["idx"]), "patient_id": int(r["patient_id"]),
                "question": str(r["question"]), "ground_truth": str(r["ground_truth"]),
                "original_answer": str(r["model_answer"]), "note": notes[str(r["patient_id"])],
                "tax_question_focus": t.get("QUESTION_FOCUS", ""),
                "tax_primary_error": t.get("PRIMARY_ERROR", ""),
                "tax_model_claims": t.get("MODEL_CLAIMS", ""),
                "tax_error_description": t.get("ERROR_DESCRIPTION", ""),
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(n_wrong, len(rows))] if n_wrong > 0 else rows


# ---------- pool + analogy ----------

_pool_cache: dict[int, dict[str, Any]] = {}

T1_TO_T2 = {
    "MISREADING": {"REPLACE_VALUE", "REMOVE_UNSUPPORTED_CLAIM"},
    "FABRICATION": {"REMOVE_UNSUPPORTED_CLAIM", "REPLACE_VALUE"},
    "OMISSION": {"ADD_MISSING_SLOT"},
    "QUESTION_MISALIGNMENT": {"REFOCUS_TIME_OR_VISIT"},
    "HEDGING": set(),
}


def load_pool(fold: int) -> dict[str, Any]:
    if fold in _pool_cache:
        return _pool_cache[fold]
    pool = json.loads((POOL_DIR / f"fold_{fold}_pool.json").read_text())
    emb = np.load(POOL_DIR / f"fold_{fold}_question_embeddings.npy")
    tags = json.loads((POOL_DIR / f"fold_{fold}_t2_tags.json").read_text())
    t2 = {int(t["pool_index"]): t.get("operation", "UNCLEAR") for t in tags}
    _pool_cache[fold] = {"pool": pool, "emb": emb, "t2": t2}
    return _pool_cache[fold]


_gtr = None


def gtr():
    global _gtr
    if _gtr is None:
        from sentence_transformers import SentenceTransformer
        _gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
    return _gtr


def gtr_top1(query: str, emb: np.ndarray, cand: list[int]) -> int:
    if not cand:
        return -1
    q = gtr().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sub = emb[cand]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    sims = sub @ q
    return cand[int(np.argmax(sims))]


def retrieve_analogy(row: dict[str, Any]) -> dict[str, Any] | None:
    fd = load_pool(row["fold"])
    ops = T1_TO_T2.get(row["tax_primary_error"], set())
    query = " ".join([row["question"], row["tax_question_focus"], row["tax_model_claims"]])
    cand = [i for i in range(len(fd["pool"])) if fd["t2"].get(i) in ops] if ops else list(range(len(fd["pool"])))
    idx = gtr_top1(query, fd["emb"], cand)
    if idx < 0:
        return None
    return fd["pool"][idx]


# ---------- question type ----------

_qtype_cache: dict[tuple[int, int], str] = {}


def question_type(row: dict[str, Any]) -> str:
    key = (row["fold"], row["idx"])
    if key in _qtype_cache:
        return _qtype_cache[key]
    raw = gpt(
        "gpt-4o-mini",
        "Classify the clinical question's required answer type. One label only.",
        f"Question: {row['question']}\n\nReply with ONE label: DATE, NUMBER, LAB_TEST, MEDICATION, TREATMENT, PROCEDURE, DIAGNOSIS, LIST, YES_NO, or OTHER.",
        max_tokens=8,
    )
    label = (raw or "OTHER").strip().upper().split()[0] if raw else "OTHER"
    _qtype_cache[key] = label
    return label


# ---------- spans ----------

def retrieve_spans(row: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    queries = [row["question"], row["original_answer"][:800], row["tax_question_focus"]]
    queries = [q for q in queries if q]
    return topk_spans(row["note"], queries, k=k, scoring="agreement")


def render_spans(spans: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans)) if spans else "(none)"


# ---------- analogy quality ----------

def rate_analogy_quality(row: dict[str, Any], analogy: dict[str, Any]) -> str:
    raw = gpt(
        "gpt-4o-mini",
        "You judge whether a retrieved reference case is a useful analogy for correcting another case.",
        f"Test question: {row['question'][:500]}\nTest wrong answer: {row['original_answer'][:400]}\n\n"
        f"Reference question: {analogy.get('question','')[:500]}\nReference what-was-wrong: {analogy.get('what_was_wrong','')[:300]}\nReference correct answer: {analogy.get('ground_truth','')[:300]}\n\n"
        "Is the reference a USEFUL analogy (similar slot/error/answer-shape)? Reply ONE word: USEFUL, WEAK, or IRRELEVANT.",
        max_tokens=6,
    )
    m = re.search(r"(USEFUL|WEAK|IRRELEVANT)", (raw or "").upper())
    return m.group(1) if m else "UNKNOWN"


# ---------- correction ----------

CORRECTION_SYSTEM = (
    "You are a careful clinical QA assistant. Revise the previous answer only when the "
    "discharge note and provided evidence support the revision. Do not add facts not "
    "supported by the note."
)


def build_blocks(row: dict[str, Any], arm: str, spans: list[dict[str, Any]], analogy: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Return the extra-info block text and a meta record of what was injected."""
    parts: list[str] = []
    meta: dict[str, Any] = {}
    want = set()
    if arm == "all":
        want = {"question_type", "question_focus", "error_type", "error_location", "contradiction_quote", "analogy_wrong", "analogy_correct"}
    else:
        want = {arm} if arm != "baseline" else set()

    if "question_type" in want:
        qt = question_type(row)
        meta["question_type"] = qt
        parts.append(f"Question type: {qt}")
    if "question_focus" in want:
        parts.append(f"Question focus: {row['tax_question_focus']}")
        meta["question_focus"] = bool(row["tax_question_focus"])
    if "error_type" in want:
        parts.append(f"Error type: {row['tax_primary_error']}")
        meta["error_type"] = row["tax_primary_error"]
    if "error_location" in want:
        parts.append(f"Where the answer went wrong: {row['tax_model_claims']}")
        meta["error_location"] = bool(row["tax_model_claims"])
    if "contradiction_quote" in want:
        parts.append(f"What is wrong and why: {row['tax_error_description']}")
        meta["contradiction_quote"] = bool(row["tax_error_description"])
    if "analogy_wrong" in want:
        if analogy:
            parts.append(
                "Similar past WRONG case (pattern only):\n"
                f"  Question: {analogy.get('question','')}\n"
                f"  Wrong answer: {analogy.get('wrong_answer','')}\n"
                f"  What was wrong: {analogy.get('what_was_wrong','')}"
            )
            meta["analogy_wrong"] = True
        else:
            meta["analogy_wrong"] = False
    if "analogy_correct" in want:
        if analogy:
            parts.append(
                "Similar past CORRECT answer (pattern only):\n"
                f"  Question: {analogy.get('question','')}\n"
                f"  Correct answer: {analogy.get('ground_truth','')}"
            )
            meta["analogy_correct"] = True
        else:
            meta["analogy_correct"] = False
    block = "\n".join(parts) if parts else "(no extra information)"
    return block, meta


def build_user(row: dict[str, Any], spans: list[dict[str, Any]], extra_block: str) -> str:
    return f"""Discharge note:
{row['note'][:18000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{render_spans(spans)}

Additional information:
{extra_block}

Use the evidence and any additional information to check the previous answer. If it is wrong or incomplete, return the corrected answer grounded in the note. If it is already correct, keep it. Return only the final answer."""


# ---------- judge ----------

def parse_binary(text: str | None) -> int | None:
    if text is None:
        return None
    if "1" in text and "0" not in text:
        return 1
    if "0" in text:
        return 0
    return None


def judge(row: dict[str, Any], answer: str) -> dict[str, Any]:
    user = (
        f"DISCHARGE SUMMARY:\n{row['note']}\n\nQUESTION:\n{row['question']}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{row['ground_truth']}\n\nMODEL'S ANSWER:\n{answer}\n\n"
        "Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        "Respond with ONLY a single digit:\n1 = Correct\n0 = Incorrect"
    )
    raw = gpt("gpt-4o", "You are a medical expert evaluating an AI model's answer to a clinical question.", user, max_tokens=10, temperature=0.1)
    return {"label": parse_binary(raw), "raw": raw}


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer"]}
    out["tax_primary_error"] = row["tax_primary_error"]
    try:
        spans = retrieve_spans(row, k=5)
        analogy = retrieve_analogy(row)
        out["analogy_quality"] = rate_analogy_quality(row, analogy) if analogy else "NONE"
        out["judge_original"] = judge(row, row["original_answer"])
        arms_out: dict[str, Any] = {}
        for arm in ARMS:
            block, meta = build_blocks(row, arm, spans, analogy)
            corrected = vllm_chat(CORRECTION_SYSTEM, build_user(row, spans, block), port, max_tokens=700, temperature=0.0)
            arms_out[arm] = {"corrected": corrected, "judge_final": judge(row, corrected), "injected": meta}
        out["arms"] = arms_out
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        fix = brk = same1 = same0 = err = 0
        for r in rows:
            jo = (r.get("judge_original") or {}).get("label")
            ar = (r.get("arms") or {}).get(arm)
            jf = (ar or {}).get("judge_final", {}).get("label") if ar else None
            if jo is None or jf is None:
                err += 1
                continue
            if jo == 0 and jf == 1:
                fix += 1
            elif jo == 1 and jf == 0:
                brk += 1
            elif jo == 1 and jf == 1:
                same1 += 1
            else:
                same0 += 1
        by_arm[arm] = {"fix": fix, "break": brk, "net": fix - brk, "same_correct": same1, "same_wrong": same0, "err": err}
    return {
        "n_cases": len(rows),
        "analogy_quality": dict(Counter(r.get("analogy_quality") for r in rows)),
        "by_arm": by_arm,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = load_wrong_rows(args.n_wrong, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="phase1_info_ablation", served=served, args=vars(args))
    print(f"sample={len(sample)} arms={len(ARMS)} out={out_dir}", flush=True)
    if sample:
        topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
        gtr().encode(["warmup"], normalize_embeddings=True)
        load_pool(sample[0]["fold"])
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 5 == 0 or i == len(futs):
                print(f"processed {i}/{len(futs)}", flush=True)
    write_jsonl(out_dir / "judged_outputs.jsonl", rows)
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
