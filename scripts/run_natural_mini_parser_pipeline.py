#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", PROJECT_ROOT.parent / "llm-ehr-hallucination"))
OUT_ROOT = PROJECT_ROOT / "runs" / "natural_mini_parser_pipeline"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(SOURCE_REPO / "src" / "step9_self_correction" / "v2"))
from note_span_index import topk_spans  # noqa: E402


NATURAL_DET_SYSTEM = (
    "You are a careful clinical QA auditor. Decide whether a draft answer is "
    "clinically acceptable for the question using the discharge note."
)

NATURAL_DET_USER = """Discharge note:
{note}

Question:
{question}

Draft answer:
{answer}

Read naturally and be conservative. Do not mark the answer wrong just because it is shorter than the note, uses different wording, or omits details that the question did not ask for.

Mark it wrong only when the answer has a clinically meaningful contradiction, a critical omission required by the question, answers the wrong visit/date/aspect, or relies on unsupported information.

If the answer is acceptable, explain briefly why it should be kept.

If it is wrong, explain:
- the specific question focus
- the smallest wrong or missing claim
- what note evidence would be needed to fix it
- two or three short retrieval queries that would help find that evidence
- a concise correction hint
- whether it is ready for correction now, or should be kept because the error is uncertain

Write in natural prose. Do not use JSON or a rigid template."""

NATURAL_PROBE_DET_USER = """Discharge note:
{note}

Question:
{question}

Draft answer:
{answer}

Audit the draft answer in natural prose. The draft answer may be correct or may contain a material error.

First identify the exact clinical fact, date, visit, or aspect the question asks for. Then compare the draft answer against the discharge note for:
- a contradiction with the note
- an unsupported speculative cause or treatment
- a missing fact that is required to answer the question
- answering the wrong date, visit, procedure, diagnosis, or aspect

Do not require unnecessary detail. Do not mark harmless extra context wrong. But do mark the answer wrong when the unsupported or missing content would change the final answer.

If the answer is acceptable, explain why it should be kept.

If the answer is wrong, explain the smallest wrong or missing claim, the evidence needed to fix it, useful short retrieval queries, and the correction the downstream answer writer should make.

Write in natural prose. Do not use JSON or a rigid template."""

COT_DET_USER = """Discharge note:
{note}

Question:
{question}

Draft answer:
{answer}

Audit the draft answer with a visible stepwise clinical checklist. Keep each step concise.

Step 1 - Question target:
State the exact clinical fact, date, visit, procedure, diagnosis, treatment, or causal explanation the question asks for.

Step 2 - Note evidence:
List the most relevant note evidence in short quoted or paraphrased snippets. Include evidence that supports the draft answer and evidence that may contradict it.

Step 3 - Draft answer claim:
State the central claim made by the draft answer. Ignore harmless extra context unless it changes the answer.

Step 4 - Mismatch check:
Decide whether the draft answer has a material contradiction, unsupported speculation, critical omission, or wrong question focus. Do not mark minor missing detail wrong. Do not mark an answer wrong only because it is less detailed than the note.

Step 5 - Decision:
Say either "The draft answer is acceptable" or "The draft answer is incorrect".

If incorrect, also give:
- error type
- smallest wrong or missing claim
- correct or missing information
- evidence needed
- two or three short retrieval queries
- one concise correction hint
- correction readiness: say exactly either "Ready for correction" or "Not ready for correction"

Use natural prose. Do not use JSON."""

COT_ROUTE_DET_USER = """Discharge note:
{note}

Question:
{question}

Draft answer:
{answer}

Audit the draft answer with a visible stepwise clinical checklist. Keep each step concise.

Step 1 - Question target:
State the exact clinical fact, date, visit, procedure, diagnosis, treatment, or causal explanation the question asks for.

Step 2 - Note evidence:
List the most relevant note evidence in short quoted or paraphrased snippets. Include evidence that supports the draft answer and evidence that may contradict it.

Step 3 - Draft answer claim:
State the central claim made by the draft answer. Ignore harmless extra context unless it changes the answer.

Step 4 - Mismatch check:
Decide whether the draft answer has a material contradiction, unsupported speculation, critical omission, or wrong question focus. Do not mark minor missing detail wrong. Do not mark an answer wrong only because it is less detailed than the note.

Step 5 - Error decision:
Say either "The draft answer is acceptable" or "The draft answer is incorrect".

If incorrect, give:
- error type
- smallest wrong or missing claim
- correct or missing information
- evidence needed
- two or three short retrieval queries
- one concise correction hint

Step 6 - Routing decision:
Choose exactly one:
- "Route to correction" if you identified a concrete wrong/missing claim and note evidence can be retrieved to revise it.
- "Keep original" if the possible error is uncertain, only stylistic, only asks for extra detail, or cannot be localized to a concrete correction target.

Use natural prose. Do not use JSON."""

MINI_PARSE_SYSTEM = (
    "You parse clinical QA text into JSON. Do not re-judge the clinical case, "
    "do not validate medical correctness, and do not decide whether correction "
    "is safe. Extract only the target model's stated choices and text fields."
)

MINI_PARSE_DET_USER = """Target model self-audit:
{raw}

Question:
{question}

Draft answer:
{answer}

Return JSON only with this schema:
{{
  "verdict": "CORRECT|INCORRECT|UNCLEAR",
  "error_type": "CONTRADICTION|OMISSION|QUESTION_MISALIGNMENT|UNSUPPORTED|NONE|UNCLEAR",
  "question_focus": "string",
  "wrong_claim": "string",
  "correct_or_missing_info": "string",
  "evidence_needed": "string",
  "retrieval_queries": ["string"],
  "correction_hint": "string",
  "why": "string",
  "confidence": 0.0,
  "correction_ready": "READY|NOT_READY|UNCLEAR",
  "parse_valid": true
}}

Rules:
- Extract the target model's stated verdict. Do not override it from the note.
- Extract correction_ready only from the target model's stated readiness/routing. Map "Ready for correction" or "Route to correction" to READY; map "Not ready for correction" or "Keep original" to NOT_READY. If readiness/routing is not stated, return UNCLEAR.
- Do not decide whether correction is safe based on your own judgment.
- Retrieval queries must be extracted from the audit text when present, or copied as short phrases from the audit text. Do not invent new clinical content."""

COR_SYSTEM = (
    "You are a careful clinical QA assistant. Revise only when same-patient "
    "evidence clearly supports the revision."
)

COR_USER = """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Parsed self-audit feedback:
- error type: {error_type}
- question focus: {question_focus}
- wrong or missing claim: {wrong_claim}
- correction target: {correct_or_missing_info}
- evidence needed: {evidence_needed}
- correction hint: {correction_hint}

Same-patient retrieved evidence. This is the factual source:
{spans_block}

Retrieved correction example from another patient. Use as style/pattern only, not factual content:
{example_block}

Write the best final answer to the question in 1-3 sentences. If the previous answer is already acceptable or the evidence does not clearly support a change, keep it essentially unchanged. Use only facts supported by the discharge note and evidence spans."""

NATURAL_VERDICT_SYSTEM = (
    "You are a conservative medical expert comparing an original answer and a "
    "candidate corrected answer to the same clinical question."
)

NATURAL_VERDICT_USER = """Discharge note:
{note}

Question:
{question}

Original answer:
{original}

Candidate corrected answer:
{corrected}

Choose whether to keep the original answer or switch to the candidate corrected answer.

Prefer the original unless the candidate clearly fixes a real contradiction, critical omission, or question-focus error without introducing unsupported content.

Do not switch only because the candidate is longer, more detailed, or more polished. Do not switch if the original already answers the question acceptably.

Write a short natural explanation and state your choice."""

MINI_PARSE_VERDICT_USER = """Target model verdict:
{raw}

Original answer:
{original}

Candidate corrected answer:
{corrected}

Return JSON only:
{{
  "pick": "ORIGINAL|CORRECTED|UNCLEAR",
  "reason": "string",
  "is_clear": true
}}

Extract the target model's choice. If it is ambiguous, hedged, or says both are acceptable, return UNCLEAR."""

JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."


def judge_user(note: str, question: str, ground_truth: str, answer: str) -> str:
    return (
        f"DISCHARGE SUMMARY:\n{note}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
        f"MODEL'S ANSWER:\n{answer}\n\n"
        "Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        "Respond with ONLY a single digit:\n"
        "1 = Correct\n"
        "0 = Incorrect"
    )


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


_client: OpenAI | None = None


def openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=load_api_key())
    return _client


def served_model_id(port: int) -> str:
    r = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


_served_cache: dict[int, str] = {}


def cached_served_model_id(port: int) -> str:
    if port not in _served_cache:
        _served_cache[port] = served_model_id(port)
    return _served_cache[port]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "</think>" in text.lower():
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.S | re.I).strip()
    return text


def vllm_chat(system: str, user: str, port: int, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": cached_served_model_id(port),
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


def gpt_json(system: str, user: str, max_tokens: int = 500) -> dict[str, Any]:
    for attempt in range(5):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads((r.choices[0].message.content or "{}").strip())
        except Exception as e:
            if attempt == 4:
                return {"error": str(e)}
            time.sleep(2 * (attempt + 1))
    return {}


def parse_binary(text: str | None) -> int | None:
    if text is None:
        return None
    if "1" in text and "0" not in text:
        return 1
    if "0" in text:
        return 0
    return None


def gpt_judge(note: str, question: str, gt: str, answer: str) -> dict[str, Any]:
    for attempt in range(5):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_user(note, question, gt, answer)}],
                temperature=0.1,
                max_tokens=10,
            )
            raw = (r.choices[0].message.content or "").strip()
            return {"label": parse_binary(raw), "raw": raw, "temperature": 0.1, "model": "gpt-4o"}
        except Exception as e:
            if attempt == 4:
                return {"label": None, "raw": "", "error": str(e), "temperature": 0.1, "model": "gpt-4o"}
            time.sleep(2 * (attempt + 1))
    return {"label": None}


def norm_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def parse_detection_with_mini(raw: str, row: dict[str, Any]) -> dict[str, Any]:
    obj = gpt_json(
        MINI_PARSE_SYSTEM,
        MINI_PARSE_DET_USER.format(raw=(raw or "")[:5000], question=row["question"], answer=row["answer"][:1600]),
        max_tokens=700,
    )
    queries = obj.get("retrieval_queries", [])
    if not isinstance(queries, list):
        queries = []
    verdict = str(obj.get("verdict", "UNCLEAR")).upper()
    if verdict not in {"CORRECT", "INCORRECT", "UNCLEAR"}:
        verdict = "UNCLEAR"
    error_type = str(obj.get("error_type", "UNCLEAR")).upper()
    if error_type not in {"CONTRADICTION", "OMISSION", "QUESTION_MISALIGNMENT", "UNSUPPORTED", "NONE", "UNCLEAR"}:
        error_type = "UNCLEAR"
    correction_ready = str(obj.get("correction_ready", "UNCLEAR")).upper()
    if correction_ready not in {"READY", "NOT_READY", "UNCLEAR"}:
        correction_ready = "UNCLEAR"
    parse_valid = norm_bool(obj.get("parse_valid", True)) and "error" not in obj
    safe = verdict == "INCORRECT" and correction_ready == "READY"
    return {
        "verdict": verdict,
        "error_type": error_type,
        "question_focus": str(obj.get("question_focus", "")),
        "wrong_claim": str(obj.get("wrong_claim", "")),
        "correct_or_missing_info": str(obj.get("correct_or_missing_info", "")),
        "evidence_needed": str(obj.get("evidence_needed", "")),
        "retrieval_queries": [str(q) for q in queries if str(q).strip()][:4],
        "correction_hint": str(obj.get("correction_hint", "")),
        "why": str(obj.get("why", "")),
        "confidence": obj.get("confidence"),
        "correction_ready": correction_ready,
        "safe_to_correct": safe,
        "parse_valid": parse_valid,
        "parser_raw": obj,
        "parse_path": "gpt-4o-mini",
    }


def parse_verdict_with_mini(raw: str, original: str, corrected: str) -> dict[str, Any]:
    obj = gpt_json(
        MINI_PARSE_SYSTEM,
        MINI_PARSE_VERDICT_USER.format(raw=(raw or "")[:3000], original=original[:1600], corrected=corrected[:1600]),
        max_tokens=250,
    )
    pick = str(obj.get("pick", "UNCLEAR")).upper()
    if pick not in {"ORIGINAL", "CORRECTED", "UNCLEAR"}:
        pick = "UNCLEAR"
    return {
        "pick": pick,
        "reason": str(obj.get("reason", ""))[:500],
        "is_clear": norm_bool(obj.get("is_clear")) and pick in {"ORIGINAL", "CORRECTED"},
        "parser_raw": obj,
        "parse_path": "gpt-4o-mini",
    }


def load_notes() -> dict[str, str]:
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


def load_rows(n_wrong: int, n_correct: int, seed: int) -> list[dict[str, Any]]:
    notes = load_notes()
    rows = []
    for fold in range(5):
        df = pd.read_csv(PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv")
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
                    "orig_label": int(r["binary_correct"]),
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


_pool_cache: dict[int, list[dict[str, Any]]] = {}


def load_pool(fold: int) -> list[dict[str, Any]]:
    if fold not in _pool_cache:
        p = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_contrast_pool" / f"fold_{fold}_pool.json"
        _pool_cache[fold] = json.loads(p.read_text()) if p.exists() else []
    return _pool_cache[fold]


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", (s or "").lower()))


def retrieve_example(row: dict[str, Any], det: dict[str, Any]) -> dict[str, Any] | None:
    pool = load_pool(row["fold"])
    if not pool:
        return None
    query = " ".join(
        [
            row["question"],
            det.get("error_type", ""),
            det.get("question_focus", ""),
            det.get("wrong_claim", ""),
            det.get("correct_or_missing_info", ""),
            det.get("correction_hint", ""),
        ]
    )
    qt = toks(query)

    def score(ex: dict[str, Any]) -> int:
        text = " ".join([ex.get("question", ""), ex.get("what_was_wrong", ""), ex.get("ground_truth", "")])
        return len(qt & toks(text))

    return max(pool, key=score)


def retrieve_spans(row: dict[str, Any], det: dict[str, Any], k: int) -> list[dict[str, Any]]:
    queries = [
        row["question"],
        det.get("question_focus", ""),
        det.get("wrong_claim", ""),
        det.get("correct_or_missing_info", ""),
        det.get("evidence_needed", ""),
    ] + list(det.get("retrieval_queries") or [])
    return topk_spans(row["note"], queries, k=k, scoring="agreement")


def render_spans(spans: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i + 1}] {s['sentence']}" for i, s in enumerate(spans)) if spans else "(none)"


def render_example(ex: dict[str, Any] | None) -> str:
    if not ex:
        return "(none)"
    ev = "; ".join(ex.get("evidence_from_notes") or [])[:600]
    return (
        f"Question: {ex.get('question', '')}\n"
        f"Wrong answer: {ex.get('wrong_answer', '')}\n"
        f"What was wrong: {ex.get('what_was_wrong', '')}\n"
        f"Correct answer pattern: {ex.get('ground_truth', '')}\n"
        f"Evidence style: {ev}"
    )


def run_detection(row: dict[str, Any], port: int, temp: float, prompt_variant: str) -> dict[str, Any]:
    if prompt_variant == "probe":
        template = NATURAL_PROBE_DET_USER
        prompt_name = "natural_probe_self_audit"
    elif prompt_variant == "cot":
        template = COT_DET_USER
        prompt_name = "visible_stepwise_self_audit"
    elif prompt_variant == "cot_route":
        template = COT_ROUTE_DET_USER
        prompt_name = "visible_stepwise_route_self_audit"
    elif prompt_variant == "conservative":
        template = NATURAL_DET_USER
        prompt_name = "natural_conservative_self_audit"
    else:
        raise ValueError(f"unknown detection prompt variant: {prompt_variant}")
    raw = vllm_chat(
        NATURAL_DET_SYSTEM,
        template.format(note=row["note"][:18000], question=row["question"], answer=row["answer"][:2000]),
        port,
        900,
        temp,
    )
    parsed = parse_detection_with_mini(raw, row)
    return {"raw": raw, "parsed": parsed, "prompt": prompt_name, "temperature": temp}


def run_correction(row: dict[str, Any], det: dict[str, Any], spans: list[dict[str, Any]], example: dict[str, Any] | None, port: int, temp: float) -> dict[str, Any]:
    user = COR_USER.format(
        note=row["note"][:18000],
        question=row["question"],
        answer=row["answer"][:1800],
        error_type=det.get("error_type", ""),
        question_focus=det.get("question_focus", ""),
        wrong_claim=det.get("wrong_claim", ""),
        correct_or_missing_info=det.get("correct_or_missing_info", ""),
        evidence_needed=det.get("evidence_needed", ""),
        correction_hint=det.get("correction_hint", ""),
        spans_block=render_spans(spans),
        example_block=render_example(example),
    )
    ans = vllm_chat(COR_SYSTEM, user, port, 600, temp)
    return {"answer": ans, "temperature": temp, "raicl_example": example, "spans": spans}


def run_verdict(row: dict[str, Any], corr_answer: str, port: int, k: int, temp: float) -> dict[str, Any]:
    samples = []
    for _ in range(k):
        raw = vllm_chat(
            NATURAL_VERDICT_SYSTEM,
            NATURAL_VERDICT_USER.format(note=row["note"][:18000], question=row["question"], original=row["answer"][:1500], corrected=corr_answer[:1500]),
            port,
            260,
            temp,
        )
        parsed = parse_verdict_with_mini(raw, row["answer"], corr_answer)
        samples.append({"raw": raw, **parsed})
    counts = Counter(s["pick"] for s in samples if s.get("is_clear"))
    corrected_votes = counts.get("CORRECTED", 0)
    original_votes = counts.get("ORIGINAL", 0)
    accept = corrected_votes > k / 2 and corrected_votes > original_votes
    majority = "TIE_OR_UNCLEAR"
    if corrected_votes > original_votes:
        majority = "CORRECTED"
    elif original_votes > corrected_votes:
        majority = "ORIGINAL"
    return {
        "variant": "natural_conservative_gate_mini_parsed",
        "k": k,
        "temperature": temp,
        "votes": dict(Counter(s["pick"] for s in samples)),
        "clear_votes": dict(counts),
        "majority_pick": majority,
        "accept_correction": accept,
        "samples": samples,
    }


def process_one(row: dict[str, Any], port: int, args: argparse.Namespace) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "answer", "orig_label"]}
    try:
        det = run_detection(row, port, args.det_temperature, args.det_prompt)
        out["detection"] = det
        p = det["parsed"]
        if p.get("verdict") != "INCORRECT" or not p.get("safe_to_correct") or not p.get("parse_valid"):
            out["action"] = "kept_original_no_safe_detection"
            out["final_answer"] = row["answer"]
            return out
        spans = retrieve_spans(row, p, args.k_spans)
        ex = retrieve_example(row, p)
        corr = run_correction(row, p, spans, ex, port, args.correction_temperature)
        out["correction"] = corr
        verdict = run_verdict(row, corr["answer"], port, args.verdict_k, args.verdict_temperature)
        out["verdict"] = verdict
        if verdict["accept_correction"]:
            out["action"] = "accepted_correction"
            out["final_answer"] = corr["answer"]
        else:
            out["action"] = "rejected_by_verdict"
            out["final_answer"] = row["answer"]
        return out
    except Exception as e:
        out["error"] = str(e)
        out["action"] = "error_keep_original"
        out["final_answer"] = row["answer"]
        return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [r for r in rows if (r.get("judge_final") or {}).get("label") is not None]
    fixes = sum(1 for r in judged if r["orig_label"] == 0 and r["judge_final"]["label"] == 1)
    breaks = sum(1 for r in judged if r["orig_label"] == 1 and r["judge_final"]["label"] == 0)
    wrong_n = sum(1 for r in rows if r["orig_label"] == 0)
    correct_n = sum(1 for r in rows if r["orig_label"] == 1)
    fix_rate = fixes / wrong_n if wrong_n else 0.0
    break_rate = breaks / correct_n if correct_n else 0.0
    projected_net = (109 * fix_rate) - (853 * break_rate)
    return {
        "n": len(rows),
        "n_judged": len(judged),
        "wrong_n": wrong_n,
        "correct_n": correct_n,
        "actions": dict(Counter(r.get("action") for r in rows)),
        "detected_incorrect": sum(1 for r in rows if ((r.get("detection") or {}).get("parsed") or {}).get("verdict") == "INCORRECT"),
        "safe_to_correct": sum(1 for r in rows if ((r.get("detection") or {}).get("parsed") or {}).get("safe_to_correct")),
        "accepted": sum(1 for r in rows if r.get("action") == "accepted_correction"),
        "fixes": fixes,
        "breaks": breaks,
        "net": fixes - breaks,
        "fix_rate_on_wrong_sample": fix_rate,
        "break_rate_on_correct_sample": break_rate,
        "projected_net_on_qwen25_109_wrong_853_correct": projected_net,
        "projected_accuracy_delta_pp": projected_net / 962 * 100,
        "detection_verdicts": dict(Counter(((r.get("detection") or {}).get("parsed") or {}).get("verdict", "none") for r in rows)),
        "detection_error_types": dict(Counter(((r.get("detection") or {}).get("parsed") or {}).get("error_type", "none") for r in rows)),
        "verdict_picks": dict(Counter(((s or {}).get("pick", "none")) for r in rows for s in ((r.get("verdict") or {}).get("samples") or []))),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=2)
    ap.add_argument("--n-correct", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--det-temperature", type=float, default=0.7)
    ap.add_argument("--det-prompt", choices=["conservative", "probe", "cot", "cot_route"], default="conservative")
    ap.add_argument("--correction-temperature", type=float, default=0.0)
    ap.add_argument("--verdict-temperature", type=float, default=0.7)
    ap.add_argument("--verdict-k", type=int, default=3)
    ap.add_argument("--k-spans", type=int, default=5)
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    served = cached_served_model_id(args.port)
    if "qwen2.5" not in served.lower() and "qwen2" not in served.lower():
        raise RuntimeError(f"Expected Qwen2.5, found {served}")
    sample = load_rows(args.n_wrong, args.n_correct, args.seed)
    temp_tag = str(args.det_temperature).replace(".", "p")
    run_id = f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}_natural_{args.det_prompt}_detT{temp_tag}_testmodel_decides_mini_extracts"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"served_model={served} sample={len(sample)} c={args.concurrency}", flush=True)
    if sample:
        topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, row, args.port, args) for row in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            write_jsonl(out_dir / "pipeline_outputs.jsonl", rows)
            print(f"pipeline {i}/{len(futs)}", flush=True)

    if args.judge:
        note_by_key = {(r["fold"], r["idx"]): r["note"] for r in sample}
        for i, r in enumerate(rows, 1):
            note = note_by_key[(r["fold"], r["idx"])]
            r["judge_final"] = gpt_judge(note, r["question"], r["ground_truth"], r["final_answer"])
            if i % 10 == 0 or i == len(rows):
                print(f"judged {i}/{len(rows)}", flush=True)
        write_jsonl(out_dir / "judged_outputs.jsonl", rows)

    summary = {
        "task": "natural_self_detection_test_model_decides_gpt4omini_extracts_raicl_correction_natural_verdict",
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
