#!/usr/bin/env python3
"""Phase 2 — can LIVE detection produce the diagnosis worth +59, and does persona help?

Phase 1 showed: a precise note-grounded "what is wrong and why" fixes 60% of Qwen2.5
errors (oracle ceiling). A category label fixes ~5%. So detection's real job is to
produce a precise diagnosis, not a category. This study measures how close live
detection comes, and whether the detection-system PERSONA changes diagnosis quality
and over-flagging.

Per case x per detection persona:
  1. Run natural detection (plan + confirm) with the persona as the system prompt.
  2. Parse to verdict + wrong_claim + correct_or_missing_info + decisive_evidence.
  3. Assemble a LIVE diagnosis string from those fields.
  4. Rate the live diagnosis vs the offline oracle ERROR_DESCRIPTION (GPT-4o-mini):
     AGREE / PARTIAL / WRONG (wrong cases only — correct cases have no oracle).
  5. Feed the live diagnosis into the SAME correction step (neutral persona) and judge.

Sample includes wrong AND correct cases so we measure:
  - recall: fraction of wrong cases detection flags INCORRECT
  - over_flag: fraction of correct cases detection wrongly flags INCORRECT
  - diagnosis_agree: of flagged wrong cases, how often the live diagnosis matches oracle
  - fix-rate: of wrong cases, how often the live-diagnosis-driven correction fixes it
    (compare to the +59 / 60% oracle ceiling from Phase 1)
  - break-rate: of correct cases, how often correction breaks them

Detection personas (GPT-4o-informed, 2026-05-29): neutral, clinical_detective,
meticulous_auditor, balanced_skeptic, strict_judge (deliberate over-flag endpoint).

Pre-flight: Qwen2.5-7B-Instruct on vLLM port 8003.
Output: runs/phase2_detection/qwen25_nw{NW}_nc{NC}_seed{SEED}/{judged_outputs.jsonl, summary.json}
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

import pandas as pd
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", PROJECT_ROOT.parent / "llm-ehr-hallucination"))
OUT_ROOT = PROJECT_ROOT / "runs" / "phase2_detection"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TAXONOMY = SOURCE_REPO / "src" / "step9_self_correction" / "error_taxonomy" / "phase1_wrong_gpt4o.json"
NOTE_SPAN_SRC = SOURCE_REPO / "src" / "step9_self_correction" / "v2"
sys.path.insert(0, str(NOTE_SPAN_SRC))
from note_span_index import topk_spans  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_selfdetect_raicl_verdict import (  # noqa: E402
    DET_META_PLAN_NATURAL,
    DET_META_CONFIRM_NATURAL,
    PARSE_DET_HELPER_V2_USER,
    valid_detection,
)
from llm_audit import set_ledger, log_call  # noqa: E402


# Detection personas (system prompt for the plan + confirm calls).
DET_PERSONAS: dict[str, str] = {
    "neutral": "You are a careful clinical QA auditor. You must check whether an answer is supported by the discharge note.",
    "clinical_detective": (
        "You are a clinical detective. Scrutinize the answer against the discharge note with precision. "
        "Flag a deviation only when the note unequivocally shows the answer is wrong, and when you flag it, "
        "state the exact discrepancy and the note evidence."
    ),
    "meticulous_auditor": (
        "You are a meticulous auditor. Verify each claim in the answer against the discharge note. "
        "Identify only clear, evidence-backed discrepancies; do not assume an error where the note is silent."
    ),
    "balanced_skeptic": (
        "You are a balanced skeptic. Question the answer's accuracy only when there is strong note evidence "
        "of a mistake. When you find one, name the specific wrong claim and the supporting note text."
    ),
    "strict_judge": (
        # Deliberate over-flag endpoint — NOT a ship candidate. Maps the false-positive ceiling.
        "You are a strict clinical examiner. Hold the answer to the highest standard. Treat any imprecision, "
        "omission, or unsupported wording as an error to be flagged and corrected."
    ),
}


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


def gpt(model: str, system: str, user: str, max_tokens: int, temperature: float = 0.0, json_mode: bool = False, tag: str = "gpt") -> str:
    kw: dict[str, Any] = {}
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    for attempt in range(4):
        try:
            r = client().chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature, **kw,
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


def vllm_chat(system: str, user: str, port: int, max_tokens: int, temperature: float = 0.0, tag: str = "vllm") -> str:
    model = served_model_id(port)
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
    body = r.json()
    if "choices" not in body:
        raise RuntimeError(str(body))
    out = strip_think((body["choices"][0]["message"]["content"] or "").strip())
    log_call(tag, model, system, user, out, temperature=temperature, max_tokens=max_tokens)
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


def load_rows(n_wrong: int, n_correct: int, seed: int) -> list[dict[str, Any]]:
    notes = load_notes()
    tax = load_taxonomy()
    wrong: list[dict[str, Any]] = []
    correct: list[dict[str, Any]] = []
    for fold in range(5):
        df = pd.read_csv(SOURCE_REPO / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv")
        for _, r in df.iterrows():
            t = tax.get((fold, int(r["idx"]))) or {}
            row = {
                "fold": fold, "idx": int(r["idx"]), "patient_id": int(r["patient_id"]),
                "question": str(r["question"]), "ground_truth": str(r["ground_truth"]),
                "original_answer": str(r["model_answer"]), "note": notes[str(r["patient_id"])],
                "stored_label": int(r["binary_correct"]),
                "oracle_error_description": t.get("ERROR_DESCRIPTION", ""),
            }
            (wrong if row["stored_label"] == 0 else correct).append(row)
    rng = random.Random(seed)
    rng.shuffle(wrong)
    rng.shuffle(correct)
    nw = len(wrong) if n_wrong < 0 else min(n_wrong, len(wrong))
    nc = len(correct) if n_correct < 0 else min(n_correct, len(correct))
    sample = wrong[:nw] + correct[:nc]
    rng.shuffle(sample)
    return sample


# ---------- detection ----------

def parse_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def run_detection(row: dict[str, Any], persona_system: str, port: int) -> dict[str, Any]:
    plan = vllm_chat(persona_system, DET_META_PLAN_NATURAL.format(question=row["question"], answer=row["original_answer"][:2000]), port, 700, 0.0)
    note = row["note"][:18000]
    confirm = vllm_chat(persona_system, DET_META_CONFIRM_NATURAL.format(note=note, question=row["question"], answer=row["original_answer"][:2000], plan=plan[:3500]), port, 1200, 0.0)
    parse_raw = gpt("gpt-4o-mini", "Extract structured fields from a clinical self-audit. Return JSON only.",
                    PARSE_DET_HELPER_V2_USER.format(question=row["question"], answer=row["original_answer"][:2000], plan=plan[:3500], raw=confirm[:7000]),
                    max_tokens=650, json_mode=True)
    obj = parse_json(parse_raw)
    rq = obj.get("retrieval_queries", [])
    parsed = {
        "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
        "error_type": str(obj.get("error_type", "UNCLEAR")).upper(),
        "wrong_claim": str(obj.get("wrong_claim", "")),
        "correct_or_missing_info": str(obj.get("correct_or_missing_info", "")),
        "decisive_evidence": str(obj.get("decisive_evidence", "")),
        # fields valid_detection() also inspects:
        "question_focus": str(obj.get("question_focus", "")),
        "evidence_needed": str(obj.get("evidence_needed", "")),
        "retrieval_queries": rq if isinstance(rq, list) else [],
    }
    parsed["valid"] = valid_detection(parsed)
    return {"confirm": confirm, "parsed": parsed}


def assemble_live_diagnosis(p: dict[str, Any]) -> str:
    parts = []
    if p.get("wrong_claim"):
        parts.append(f"What is wrong: {p['wrong_claim']}")
    if p.get("correct_or_missing_info"):
        parts.append(f"What it should be: {p['correct_or_missing_info']}")
    if p.get("decisive_evidence"):
        parts.append(f"Evidence: {p['decisive_evidence']}")
    return " ".join(parts) if parts else "(no specific diagnosis)"


# ---------- diagnosis quality vs oracle ----------

def rate_diagnosis(row: dict[str, Any], live_diag: str) -> str:
    oracle = row.get("oracle_error_description", "")
    if not oracle:
        return "NO_ORACLE"
    raw = gpt("gpt-4o-mini", "You compare two descriptions of what is wrong with a clinical answer.",
              f"Oracle diagnosis (gold):\n{oracle[:800]}\n\nLive diagnosis:\n{live_diag[:800]}\n\n"
              "Does the live diagnosis identify the SAME underlying error as the oracle? Reply ONE word: AGREE, PARTIAL, or WRONG.",
              max_tokens=6)
    m = re.search(r"(AGREE|PARTIAL|WRONG)", (raw or "").upper())
    return m.group(1) if m else "UNKNOWN"


# ---------- correction (neutral persona, live diagnosis) ----------

CORRECTION_SYSTEM = (
    "You are a careful clinical QA assistant. Revise the previous answer only when the "
    "discharge note and provided evidence support the revision. Do not add facts not supported by the note."
)


def retrieve_spans(row: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    queries = [q for q in [row["question"], row["original_answer"][:800]] if q]
    return topk_spans(row["note"], queries, k=k, scoring="agreement")


def render_spans(spans: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans)) if spans else "(none)"


def run_correction(row: dict[str, Any], live_diag: str, spans: list[dict[str, Any]], port: int) -> str:
    user = f"""Discharge note:
{row['note'][:18000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{render_spans(spans)}

Diagnosis of what is wrong:
{live_diag}

Use the diagnosis and the evidence to fix the previous answer. If the diagnosis does not identify a real, note-supported error, keep the previous answer. Return only the final answer."""
    return vllm_chat(CORRECTION_SYSTEM, user, port, 700, 0.0)


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
    user = (f"DISCHARGE SUMMARY:\n{row['note']}\n\nQUESTION:\n{row['question']}\n\n"
            f"CORRECT ANSWER (Ground Truth):\n{row['ground_truth']}\n\nMODEL'S ANSWER:\n{answer}\n\n"
            "Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
            "Respond with ONLY a single digit:\n1 = Correct\n0 = Incorrect")
    raw = gpt("gpt-4o", "You are a medical expert evaluating an AI model's answer to a clinical question.", user, max_tokens=10, temperature=0.1)
    return {"label": parse_binary(raw), "raw": raw}


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int, personas: list[str]) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        spans = retrieve_spans(row, k=5)
        out["judge_original"] = judge(row, row["original_answer"])
        per: dict[str, Any] = {}
        for name in personas:
            det = run_detection(row, DET_PERSONAS[name], port)
            p = det["parsed"]
            flagged = (p.get("verdict") == "INCORRECT" and p.get("valid"))
            rec: dict[str, Any] = {"verdict": p.get("verdict"), "valid": p.get("valid"), "flagged": flagged}
            if flagged:
                live = assemble_live_diagnosis(p)
                rec["live_diagnosis"] = live
                rec["diagnosis_quality"] = rate_diagnosis(row, live)
                corrected = run_correction(row, live, spans, port)
                rec["corrected"] = corrected
                rec["judge_final"] = judge(row, corrected)
            else:
                rec["judge_final"] = {"label": (out["judge_original"] or {}).get("label"), "raw": "kept_original"}
            per[name] = rec
        out["personas"] = per
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]], personas: list[str]) -> dict[str, Any]:
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    by: dict[str, Any] = {}
    for name in personas:
        flagged_wrong = sum(1 for r in wrong if (r.get("personas") or {}).get(name, {}).get("flagged"))
        flagged_correct = sum(1 for r in correct if (r.get("personas") or {}).get(name, {}).get("flagged"))
        dq = Counter((r.get("personas") or {}).get(name, {}).get("diagnosis_quality") for r in wrong if (r.get("personas") or {}).get(name, {}).get("flagged"))
        fix = sum(1 for r in wrong if (r.get("personas") or {}).get(name, {}).get("judge_final", {}).get("label") == 1)
        brk = sum(1 for r in correct if (r.get("personas") or {}).get(name, {}).get("judge_final", {}).get("label") == 0)
        by[name] = {
            "recall_on_wrong": round(flagged_wrong / max(1, len(wrong)), 3),
            "overflag_on_correct": round(flagged_correct / max(1, len(correct)), 3),
            "flagged_wrong": flagged_wrong, "flagged_correct": flagged_correct,
            "diagnosis_quality": dict(dq),
            "fix": fix, "break": brk, "net": fix - brk,
            "fix_rate_on_wrong": round(fix / max(1, len(wrong)), 3),
        }
    return {"n_cases": len(rows), "n_wrong": len(wrong), "n_correct": len(correct),
            "oracle_ceiling_note": "Phase 1 contradiction_quote fixed 62/103 wrong (60%) on this model",
            "by_persona": by}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--n-correct", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--personas", nargs="+", default=list(DET_PERSONAS), choices=list(DET_PERSONAS))
    args = ap.parse_args()
    served = served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="phase2_detection_quality", served=served, args=vars(args))
    print(f"sample={len(sample)} personas={args.personas} out={out_dir}", flush=True)
    if sample:
        topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port, args.personas) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 5 == 0 or i == len(futs):
                print(f"processed {i}/{len(futs)}", flush=True)
    write_jsonl(out_dir / "judged_outputs.jsonl", rows)
    summary = summarize(rows, args.personas)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
