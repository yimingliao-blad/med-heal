#!/usr/bin/env python3
"""Channel-B taxonomy-alignment 3-arm pilot.

For each Qwen2.5 wrong case:
  1. Run natural-pipeline detection (meta_plan_confirm_natural + helper-v2) to
     get T1 (error_type) and T2 (correction_operation).
  2. Retrieve top-1 cross-patient analogy via THREE modes:
       C-1 untyped:        lexical token overlap (current runner default).
       C-2 T2-typed:       restrict pool to entries tagged with same T2 as
                           detection's correction_operation, rank by GTR
                           cosine over question embeddings.
       C-3 T2-typed+fallback: C-2; if T2 subset empty, fall back to T1-matched
                           subset; if that empty, fall back to full pool lexical.
  3. Retrieve same-patient note spans via gtr_q_answer top-5.
  4. Run correction once per arm using a new operation+analogy prompt.
  5. Judge candidate vs ground truth with GPT-4o.

Required pre-step: bm_contrast_pool/fold_X_t2_tags.json sidecar files (produced
by scripts/tag_bm_contrast_pool_t2.py).

Pre-flight: Qwen2.5-7B-Instruct must be loaded on vLLM port 8003.

Output:
  runs/taxonomy_alignment/qwen25_nw{N}_seed{SEED}/<arm>/{generated.jsonl, summary.json}
  runs/taxonomy_alignment/qwen25_nw{N}_seed{SEED}/summary.json
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
OUT_ROOT = PROJECT_ROOT / "runs" / "taxonomy_alignment"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

POOL_DIR = SOURCE_REPO / "workspace" / "self_critique" / "data" / "bm_contrast_pool"
NOTE_SPAN_SRC = SOURCE_REPO / "src" / "step9_self_correction" / "v2"
sys.path.insert(0, str(NOTE_SPAN_SRC))
from note_span_index import topk_spans  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from llm_audit import set_ledger, log_call  # noqa: E402

# Reuse natural-pipeline detection from the main runner by importing it.
RUNNER = str(PROJECT_ROOT / "scripts")
sys.path.insert(0, RUNNER)
from run_selfdetect_raicl_verdict import (  # noqa: E402
    DET_META_PLAN_NATURAL,
    DET_META_CONFIRM_NATURAL,
    DET_SYSTEM,
    PARSE_DET_HELPER_V2_USER,
    valid_detection,
)


# ---------- OpenAI client ----------

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


_openai_client: OpenAI | None = None


def openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=load_api_key())
    return _openai_client


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


def vllm_chat(system: str, user: str, port: int, max_tokens: int = 700, temperature: float = 0.0, tag: str = "vllm") -> str:
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
        raise RuntimeError(str(body))
    out = strip_think((body["choices"][0]["message"]["content"] or "").strip())
    log_call(tag, model, system, user, out, temperature=temperature, max_tokens=max_tokens)
    return out


# ---------- data ----------

def load_notes_lookup() -> dict[str, str]:
    p = SOURCE_REPO / "output" / "EHRNoteQA_processed.jsonl"
    out: dict[str, str] = {}
    for line in p.open():
        d = json.loads(line)
        parts = []
        for i in (1, 2, 3):
            t = d.get(f"note_{i}")
            if t and str(t).strip() and str(t).strip().lower() != "nan":
                parts.append(f"[Note {i}]\n{str(t).strip()}")
        out[str(d["patient_id"])] = "\n\n".join(parts)
    return out


def load_qwen25_wrong_rows(n_wrong: int, seed: int) -> list[dict[str, Any]]:
    notes = load_notes_lookup()
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        df = pd.read_csv(SOURCE_REPO / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv")
        for _, r in df.iterrows():
            if int(r["binary_correct"]) != 0:
                continue
            rows.append({
                "fold": fold,
                "idx": int(r["idx"]),
                "patient_id": int(r["patient_id"]),
                "question": str(r["question"]),
                "ground_truth": str(r["ground_truth"]),
                "original_answer": str(r["model_answer"]),
                "note": notes[str(r["patient_id"])],
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(n_wrong, len(rows))]


_pool_cache: dict[int, dict[str, Any]] = {}


def load_pool(fold: int) -> dict[str, Any]:
    if fold in _pool_cache:
        return _pool_cache[fold]
    pool = json.loads((POOL_DIR / f"fold_{fold}_pool.json").read_text())
    emb = np.load(POOL_DIR / f"fold_{fold}_question_embeddings.npy")
    if len(pool) != emb.shape[0]:
        raise RuntimeError(f"pool/embedding mismatch fold {fold}")
    tag_path = POOL_DIR / f"fold_{fold}_t2_tags.json"
    if not tag_path.exists():
        raise RuntimeError(f"missing T2 tags for fold {fold}: run tag_bm_contrast_pool_t2.py first")
    tags = json.loads(tag_path.read_text())
    t2 = {int(t["pool_index"]): t.get("operation", "UNCLEAR") for t in tags}
    _pool_cache[fold] = {"pool": pool, "emb": emb, "t2": t2}
    return _pool_cache[fold]


# ---------- detection (live) ----------

def parse_detection_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def gpt_parser(question: str, answer: str, plan: str, raw: str) -> dict[str, Any]:
    user = PARSE_DET_HELPER_V2_USER.format(
        question=question, answer=answer[:2000], plan=plan[:3500], raw=(raw or "")[:7000],
    )
    for attempt in range(4):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract structured fields from a clinical self-audit. Return JSON only."},
                    {"role": "user", "content": user},
                ],
                max_tokens=650,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_p = (r.choices[0].message.content or "").strip()
            obj = parse_detection_json(raw_p)
            return obj
        except Exception as e:
            if attempt == 3:
                return {"error": str(e)}
            time.sleep(1 + attempt)
    return {}


def run_detection(row: dict[str, Any], port: int) -> dict[str, Any]:
    plan_raw = vllm_chat(
        DET_SYSTEM,
        DET_META_PLAN_NATURAL.format(question=row["question"], answer=row["original_answer"][:2000]),
        port=port, max_tokens=700, temperature=0.0,
    )
    note_first18k = row["note"][:18000]
    confirm_raw = vllm_chat(
        DET_SYSTEM,
        DET_META_CONFIRM_NATURAL.format(note=note_first18k, question=row["question"], answer=row["original_answer"][:2000], plan=plan_raw[:3500]),
        port=port, max_tokens=1200, temperature=0.0,
    )
    obj = gpt_parser(row["question"], row["original_answer"], plan_raw, confirm_raw)
    parsed = {
        "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
        "error_type": str(obj.get("error_type", "UNCLEAR")).upper(),
        "correction_operation": str(obj.get("correction_operation", "UNCLEAR")).upper(),
        "wrong_claim": str(obj.get("wrong_claim", "")),
        "correct_or_missing_info": str(obj.get("correct_or_missing_info", "")),
        "decisive_evidence": str(obj.get("decisive_evidence", "")),
        "do_not_change": str(obj.get("do_not_change", "")),
        "correction_hint": str(obj.get("correction_hint", "")),
        "question_focus": str(obj.get("question_focus", "")),
        "answer_focus": str(obj.get("answer_focus", "")),
        "evidence_needed": str(obj.get("evidence_needed", "")),
        "retrieval_queries": obj.get("retrieval_queries") or [],
    }
    parsed["valid"] = valid_detection(parsed)
    return {"plan_raw": plan_raw, "confirm_raw": confirm_raw, "parsed": parsed}


# ---------- retrieval ----------

def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", (s or "").lower()))


def lexical_top1(query: str, pool: list[dict[str, Any]]) -> int:
    qt = toks(query)
    best_i, best_s = -1, -1
    for i, ex in enumerate(pool):
        text = " ".join([ex.get("question", ""), ex.get("what_was_wrong", ""), ex.get("ground_truth", "")])
        s = len(qt & toks(text))
        if s > best_s:
            best_i, best_s = i, s
    return best_i


_gtr_model = None


def gtr_encoder():
    global _gtr_model
    if _gtr_model is None:
        from sentence_transformers import SentenceTransformer
        _gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
    return _gtr_model


def gtr_top1_within(query: str, pool: list[dict[str, Any]], emb: np.ndarray, candidate_indices: list[int]) -> int:
    if not candidate_indices:
        return -1
    q_emb = gtr_encoder().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sub = emb[candidate_indices]
    sub_norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    sims = sub_norm @ q_emb
    best_local = int(np.argmax(sims))
    return candidate_indices[best_local]


T1_TO_T2 = {
    "CONTRADICTION": {"REPLACE_VALUE", "REMOVE_UNSUPPORTED_CLAIM"},
    "OMISSION": {"ADD_MISSING_SLOT"},
    "QUESTION_MISALIGNMENT": {"REFOCUS_TIME_OR_VISIT"},
    "NONE": set(),
    "UNCLEAR": set(),
}


def retrieve_analogy(arm: str, row: dict[str, Any], det: dict[str, Any], fold_data: dict[str, Any]) -> dict[str, Any]:
    pool = fold_data["pool"]
    emb = fold_data["emb"]
    t2_map = fold_data["t2"]
    query = " ".join([
        row["question"],
        det.get("question_focus", ""),
        det.get("wrong_claim", ""),
        det.get("correct_or_missing_info", ""),
    ])
    target_op = det.get("correction_operation", "UNCLEAR")
    target_t1 = det.get("error_type", "UNCLEAR")

    chosen_idx = -1
    chosen_path = "none"

    if arm == "C-1":
        chosen_idx = lexical_top1(query, pool)
        chosen_path = "lexical_full"
    elif arm == "C-2":
        cand = [i for i in range(len(pool)) if t2_map.get(i) == target_op]
        if cand:
            chosen_idx = gtr_top1_within(query, pool, emb, cand)
            chosen_path = "t2_match"
    elif arm == "C-3":
        cand_t2 = [i for i in range(len(pool)) if t2_map.get(i) == target_op]
        if cand_t2:
            chosen_idx = gtr_top1_within(query, pool, emb, cand_t2)
            chosen_path = "t2_match"
        else:
            t1_ops = T1_TO_T2.get(target_t1, set())
            cand_t1 = [i for i in range(len(pool)) if t2_map.get(i) in t1_ops]
            if cand_t1:
                chosen_idx = gtr_top1_within(query, pool, emb, cand_t1)
                chosen_path = "t1_fallback"
            else:
                chosen_idx = lexical_top1(query, pool)
                chosen_path = "lexical_fallback"
    else:
        raise ValueError(f"unknown arm {arm}")

    analogy = pool[chosen_idx] if chosen_idx >= 0 else None
    return {
        "arm": arm,
        "pool_index": chosen_idx,
        "retrieval_path": chosen_path,
        "target_operation": target_op,
        "target_error_type": target_t1,
        "analogy_operation_tag": t2_map.get(chosen_idx) if chosen_idx >= 0 else None,
        "analogy": {
            "fold": analogy.get("fold") if analogy else None,
            "idx": analogy.get("idx") if analogy else None,
            "patient_id": analogy.get("patient_id") if analogy else None,
            "question": analogy.get("question") if analogy else "",
            "wrong_answer": analogy.get("wrong_answer") if analogy else "",
            "what_was_wrong": analogy.get("what_was_wrong") if analogy else "",
            "ground_truth": analogy.get("ground_truth") if analogy else "",
        },
    }


def retrieve_spans(row: dict[str, Any], det: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    queries = [
        row["question"],
        row["original_answer"][:800],
        det.get("question_focus", ""),
        det.get("wrong_claim", ""),
        det.get("correct_or_missing_info", ""),
        det.get("evidence_needed", ""),
    ] + list(det.get("retrieval_queries") or [])
    queries = [q for q in queries if q]
    return topk_spans(row["note"], queries, k=k, scoring="agreement")


# ---------- correction ----------

CORRECTION_SYSTEM = (
    "You are a careful clinical QA assistant. Revise an answer only when the "
    "provided same-patient evidence supports the revision. Do not add facts not "
    "supported by the discharge note."
)


def render_spans(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return "(none)"
    return "\n".join(f"[{i + 1}] {s['sentence']}" for i, s in enumerate(spans))


def render_analogy(analogy: dict[str, Any]) -> str:
    if not analogy or analogy.get("pool_index", -1) < 0:
        return "(none)"
    a = analogy["analogy"]
    return (
        f"Past case (operation: {analogy.get('analogy_operation_tag', 'UNKNOWN')}):\n"
        f"Question: {a.get('question', '')}\n"
        f"Wrong answer: {a.get('wrong_answer', '')}\n"
        f"What was wrong: {a.get('what_was_wrong', '')}\n"
        f"Correct answer: {a.get('ground_truth', '')}"
    )


def build_correction_user(row: dict[str, Any], det: dict[str, Any], spans: list[dict[str, Any]], analogy: dict[str, Any]) -> str:
    return f"""Discharge note:
{row['note'][:18000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Audit result:
- error type: {det.get('error_type', '')}
- correction operation: {det.get('correction_operation', '')}
- wrong or missing part: {det.get('wrong_claim', '')}
- target fact: {det.get('correct_or_missing_info', '')}
- decisive evidence: {det.get('decisive_evidence', '')}
- do not change: {det.get('do_not_change', '')}
- correction hint: {det.get('correction_hint', '')}

Same-patient retrieved evidence:
{render_spans(spans)}

Cross-patient analogy of the same operation, for pattern only:
{render_analogy(analogy)}

Apply ONLY the named correction operation. Preserve supported parts listed in DO NOT CHANGE. Use the analogy only as a pattern hint, not as a source of facts about this patient. Do not add facts beyond the decisive evidence and same-patient retrieved evidence. Return only the final answer."""


def run_correction(row: dict[str, Any], det: dict[str, Any], spans: list[dict[str, Any]], analogy: dict[str, Any], port: int) -> str:
    user = build_correction_user(row, det, spans, analogy)
    return vllm_chat(CORRECTION_SYSTEM, user, port=port, max_tokens=700, temperature=0.0)


# ---------- judge ----------

JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."


def build_judge_user(note: str, question: str, ground_truth: str, model_answer: str) -> str:
    return (
        f"DISCHARGE SUMMARY:\n{note}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
        f"MODEL'S ANSWER:\n{model_answer}\n\n"
        f"Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        f"Respond with ONLY a single digit:\n1 = Correct\n0 = Incorrect"
    )


def parse_binary(text: str | None) -> int | None:
    if text is None:
        return None
    if "1" in text and "0" not in text:
        return 1
    if "0" in text:
        return 0
    return None


def gpt_judge(row: dict[str, Any], answer: str) -> dict[str, Any]:
    for attempt in range(4):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_judge_user(row["note"], row["question"], row["ground_truth"], answer)},
                ],
                max_tokens=10,
                temperature=0.1,
            )
            raw = (r.choices[0].message.content or "").strip()
            return {"label": parse_binary(raw), "raw": raw}
        except Exception as e:
            if attempt == 3:
                return {"label": None, "raw": "", "error": str(e)}
            time.sleep(1 + attempt)
    return {"label": None}


# ---------- per-case orchestration ----------

ARMS = ["C-1", "C-2", "C-3"]


def process_one(row: dict[str, Any], port: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fold": row["fold"],
        "idx": row["idx"],
        "patient_id": row["patient_id"],
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "original_answer": row["original_answer"],
        "orig_label": 0,
    }
    try:
        det_full = run_detection(row, port)
        det = det_full["parsed"]
        out["detection"] = det
        fold_data = load_pool(row["fold"])
        spans = retrieve_spans(row, det, k=5)
        out["spans"] = spans
        per_arm: dict[str, Any] = {}
        for arm in ARMS:
            analogy = retrieve_analogy(arm, row, det, fold_data)
            corrected = run_correction(row, det, spans, analogy, port)
            judge = gpt_judge(row, corrected)
            per_arm[arm] = {
                "analogy": analogy,
                "corrected_answer": corrected,
                "judge_final": judge,
            }
        out["arms"] = per_arm
        out["judge_original"] = gpt_judge(row, row["original_answer"])
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        fixes = breaks = still_wrong = errors = 0
        path_counts: Counter[str] = Counter()
        op_counts: Counter[str] = Counter()
        for r in rows:
            ar = (r.get("arms") or {}).get(arm)
            if not ar:
                errors += 1
                continue
            label = (ar.get("judge_final") or {}).get("label")
            if label == 1:
                fixes += 1
            elif label == 0:
                still_wrong += 1
            else:
                errors += 1
            path_counts[ar.get("analogy", {}).get("retrieval_path", "?")] += 1
            op_counts[ar.get("analogy", {}).get("target_operation", "?")] += 1
        by_arm[arm] = {
            "fixes": fixes,
            "still_wrong": still_wrong,
            "errors": errors,
            "retrieval_paths": dict(path_counts),
            "target_operations": dict(op_counts),
        }
    return {"n_cases": len(rows), "by_arm": by_arm}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-wrong", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = load_qwen25_wrong_rows(args.n_wrong, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="channelB_taxonomy_alignment_pilot", served=served, args=vars(args))
    print(f"sample={len(sample)} out={out_dir}", flush=True)

    # Warm GTR on first item.
    if sample:
        topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
        _ = gtr_encoder().encode(["warmup"], normalize_embeddings=True)
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
