#!/usr/bin/env python3
"""Phase 2b — extract-and-compare detection, anchored on the zero-shot answer.

Hypothesis: the current single-pass detection asks Qwen2.5 to read note+question+
answer and decide what is wrong all at once. Decomposing it into three natural-tone
prompts may (a) ground the diagnosis in extracted quotes and (b) focus a long note
down to the relevant facts before comparing.

Three prompts (natural tone, minimal instruction; the model under test = Qwen2.5):
  P1 answer-side extraction:  pull note sentences the ZS answer makes claims about.
  P2 question-side extraction: pull note sentences that actually answer the question.
  P3 compare/judge:           hold the ZS answer against both extractions; say whether
                              it is wrong and what is contradicted or missing.

The P3 memo is then PARSED (gpt-4o-mini or qwen35 on mlx:8803) into a structured
diagnosis. We measure the diagnosis vs the offline oracle ERROR_DESCRIPTION (AGREE/
PARTIAL/WRONG), recall, over-flag, and downstream fix-rate (feed the live diagnosis
into the same neutral correction step). Compare to: oracle ceiling (~60% fix, Phase 1)
and the current plan->confirm detection (Phase 2 neutral: ~5% AGREE, 5-10% fix).

First goal: does the multi-prompt decomposition lift AGREE in principle? Later:
add a planner prompt or fold into CoT.

Pre-flight: Qwen2.5-7B-Instruct on vLLM :8003. Parser gpt-4o-mini (default) or
qwen35 (mlx http://192.168.68.107:8803).

Output: runs/phase2b_extract_compare/qwen25_nw{NW}_nc{NC}_{parser}/{judged_outputs.jsonl, summary.json}
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
OUT_ROOT = PROJECT_ROOT / "runs" / "phase2b_extract_compare"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TAXONOMY = SOURCE_REPO / "src" / "step9_self_correction" / "error_taxonomy" / "phase1_wrong_gpt4o.json"
NOTE_SPAN_SRC = SOURCE_REPO / "src" / "step9_self_correction" / "v2"
sys.path.insert(0, str(NOTE_SPAN_SRC))
from note_span_index import topk_spans  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from llm_audit import set_ledger, log_call  # noqa: E402
# Borrow yesterday's tuned natural-memo parser (helper-v2). It is specifically
# tuned to classify only correction-worthy issues as INCORRECT (anti over-flag),
# and fills the wrong_claim / correct_or_missing_info / decisive_evidence fields
# the correction stage relies on.
from run_selfdetect_raicl_verdict import PARSE_DET_HELPER_V2_USER  # noqa: E402

MLX_URL = "http://192.168.68.107:8803/v1/chat/completions"

# ---------- natural-tone detection prompts ----------

EXTRACT_SYS = "You read clinical discharge notes carefully and quote from them accurately."

P1_ANSWER_EXTRACT = """Here is a patient's discharge note:

{note}

Someone answered a question about this patient with:
{answer}

Read through the note and pull out the sentences that relate to what this answer is talking about — the facts in the note that bear on the claims this answer makes. Quote them directly from the note. If the note does not mention something the answer claims, say so."""

P2_QUESTION_EXTRACT = """Here is a patient's discharge note:

{note}

Here is a question about this patient:
{question}

Read through the note and pull out the sentences that actually answer this question. Quote them directly from the note. If the note does not contain the answer, say so."""

P3_COMPARE_SYS = "You are a thoughtful clinician comparing an answer against what the note actually says."

P3_COMPARE = """Question:
{question}

The answer that was given:
{answer}

Note facts that the given answer is talking about:
{answer_facts}

Note facts that actually answer the question:
{question_facts}

Think about whether the given answer matches what the note really says for this exact question. Consider whether the answer contradicts the note, or whether it leaves out something the question needs. Explain in plain clinical language whether the answer is right or wrong, and if it is wrong, say specifically what is contradicted or missing and what the note-supported answer should be."""

# Planner: read only the question + answer (no note) and list what to verify.
P0_PLANNER_SYS = "You plan how to check a clinical answer before reading the chart."
P0_PLANNER = """Question:
{question}

Answer given:
{answer}

Before looking at the note, think about what this question is really asking for and what would make this answer wrong. List the specific things to verify against the note — the key facts, values, dates, medications, or list items the answer depends on, and the exact slot the question needs filled."""

# Planner-guided compare: same as P3 but works through the plan.
P3_COMPARE_PLANNED = """Question:
{question}

The answer that was given:
{answer}

Things to verify (made before reading the note):
{plan}

Note facts that the given answer is talking about:
{answer_facts}

Note facts that actually answer the question:
{question_facts}

Go through the things-to-verify one by one against the note facts. For each, decide whether the note confirms the answer, contradicts it, or is silent. Then state in plain clinical language whether the answer is right or wrong, and if wrong, exactly what is contradicted or missing and what the note-supported answer should be."""

# Planner designs the fix by synthesizing several natural-compare findings.
P_DESIGN_FIX_SYS = "You design a precise, note-grounded correction from several quick audits."
P_DESIGN_FIX = """Question:
{question}

Answer that was given:
{answer}

Note facts the answer talks about:
{answer_facts}

Note facts that actually answer the question:
{question_facts}

Several independent quick checks of this answer raised these concerns:
{memos}

Now design the precise fix. Weigh all the concerns together against the note facts, decide which are genuinely supported, and specify exactly: what in the answer is wrong or missing, what the note-supported answer should be, the note evidence that proves it, and what must be kept unchanged. Be specific and grounded only in the note. If on reflection the answer is actually correct, say so."""

# CoT compare: explicit step-by-step reasoning over the two extractions.
P3_COMPARE_COT = """Question:
{question}

The answer that was given:
{answer}

Note facts that the given answer is talking about:
{answer_facts}

Note facts that actually answer the question:
{question_facts}

Work through this step by step:
1. What exactly does the question ask for (the required answer slot)?
2. What does the given answer claim?
3. What do the note facts actually say about that?
4. Does the answer match the note for this exact question — fully, partly, or not at all?
5. If it does not match, state specifically what is contradicted or missing, and what the note-supported answer should be.

After the steps, give your plain-language conclusion: is the answer right or wrong, and if wrong, what is the specific error."""

# ---------- parsing (memo -> structured) ----------

PARSE_SYS = "You convert a clinician's natural audit note into structured fields. Return JSON only."

PARSE_USER = """Question:
{question}

Answer that was audited:
{answer}

The clinician's audit note:
{memo}

From the audit note, extract:
- verdict: INCORRECT if the clinician judged the answer wrong/incomplete for the question, else CORRECT.
- wrong_claim: the specific claim in the answer that is contradicted, or NONE.
- correct_or_missing_info: the note-supported fact that should replace it or that is missing, or NONE.
- decisive_evidence: the note quote that proves it, or NONE.

Return JSON only:
{{"verdict":"CORRECT|INCORRECT","wrong_claim":"...","correct_or_missing_info":"...","decisive_evidence":"..."}}"""


# ---------- OpenAI / clients ----------

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
    kw: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
    for attempt in range(4):
        try:
            r = client().chat.completions.create(
                model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature, **kw,
            )
            out = (r.choices[0].message.content or "").strip()
            log_call(tag, model, system, user, out, temperature=temperature)
            return out
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(1 + attempt)
    return ""


def mlx_chat(system: str, user: str, max_tokens: int = 600, temperature: float = 0.0, tag: str = "mlx-parser") -> str:
    payload = {"model": "qwen35", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": temperature}
    for attempt in range(4):
        try:
            r = requests.post(MLX_URL, json=payload, timeout=180)
            body = r.json()
            out = (body["choices"][0]["message"].get("content") or "").strip()
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.S | re.I).strip()
            log_call(tag, "qwen35-mlx", system, user, out, temperature=temperature)
            return out
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(1 + attempt)
    return ""


# ---------- vLLM (Qwen2.5 under test) ----------

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
    return {(int(r["fold"]), int(r["idx"])): r for r in json.loads(TAXONOMY.read_text())}


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


# ---------- detection (3-prompt extract-compare) ----------

def parse_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def parse_memo(row: dict[str, Any], answer_facts: str, question_facts: str, memo: str, parser: str) -> dict[str, Any]:
    """Parse a compare memo into the full structured field set the corrector needs."""
    if parser == "helper-v2":
        plan_text = f"Note facts the answer talks about:\n{answer_facts[:1800]}\n\nNote facts that answer the question:\n{question_facts[:1800]}"
        hu = PARSE_DET_HELPER_V2_USER.format(question=row["question"], answer=row["original_answer"][:2000], plan=plan_text[:3500], raw=memo[:7000])
        raw = gpt("gpt-4o-mini", "Extract structured fields from a clinical self-audit. Return JSON only.", hu, max_tokens=650, json_mode=True, tag="parse.helper-v2")
    elif parser == "qwen35":
        raw = mlx_chat(PARSE_SYS, PARSE_USER.format(question=row["question"], answer=row["original_answer"][:1200], memo=memo[:5000]), max_tokens=400, tag="parse.qwen35")
    else:
        raw = gpt("gpt-4o-mini", PARSE_SYS, PARSE_USER.format(question=row["question"], answer=row["original_answer"][:1200], memo=memo[:5000]), max_tokens=400, json_mode=True, tag="parse.gpt4omini")
    obj = parse_json(raw)
    return {
        "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
        "wrong_claim": str(obj.get("wrong_claim", "")),
        "correct_or_missing_info": str(obj.get("correct_or_missing_info", "")),
        "decisive_evidence": str(obj.get("decisive_evidence", "")),
        "error_type": str(obj.get("error_type", "")),
        "correction_operation": str(obj.get("correction_operation", "")),
        "do_not_change": str(obj.get("do_not_change", "")),
    }


def _flagged(p: dict[str, Any]) -> bool:
    return p["verdict"] == "INCORRECT" and (p["wrong_claim"] not in ("", "NONE") or p["correct_or_missing_info"] not in ("", "NONE"))


def run_detection(row: dict[str, Any], port: int, parser: str, mode: str = "natural") -> dict[str, Any]:
    note = row["note"][:24000]
    a = row["original_answer"]
    answer_facts = vllm_chat(EXTRACT_SYS, P1_ANSWER_EXTRACT.format(note=note, answer=a[:1500]), port, 600, 0.0, tag="extract.answer")
    question_facts = vllm_chat(EXTRACT_SYS, P2_QUESTION_EXTRACT.format(note=note, question=row["question"]), port, 600, 0.0, tag="extract.question")

    if mode == "k3union":
        # Natural compare x3 at T=0.7; UNION of flags (any sample says wrong -> flagged,
        # maximizes recall); then the planner designs the fix by synthesizing the 3 memos.
        memos = []
        verdicts = []
        for i in range(3):
            mm = vllm_chat(P3_COMPARE_SYS, P3_COMPARE.format(question=row["question"], answer=a[:1500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000]), port, 700, 0.7, tag=f"compare.k{i+1}")
            memos.append(mm)
            verdicts.append(parse_memo(row, answer_facts, question_facts, mm, parser)["verdict"])
        union_flag = any(v == "INCORRECT" for v in verdicts)
        if not union_flag:
            return {"answer_facts": answer_facts, "question_facts": question_facts, "memo": memos[0], "k3_verdicts": verdicts, "parsed": {"verdict": "CORRECT", "flagged": False, "wrong_claim": "", "correct_or_missing_info": "", "decisive_evidence": "", "error_type": "", "correction_operation": "", "do_not_change": ""}, "stage": "k3union_clear"}
        memos_block = "\n\n".join(f"Check {i+1}: {m[:1500]}" for i, m in enumerate(memos))
        design = vllm_chat(P_DESIGN_FIX_SYS, P_DESIGN_FIX.format(question=row["question"], answer=a[:1500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000], memos=memos_block[:5000]), port, 800, 0.0, tag="design.fix")
        parsed = parse_memo(row, answer_facts, question_facts, design, parser)
        parsed["flagged"] = (parsed["wrong_claim"] not in ("", "NONE") or parsed["correct_or_missing_info"] not in ("", "NONE"))
        return {"answer_facts": answer_facts, "question_facts": question_facts, "k3_verdicts": verdicts, "memo": design, "parsed": parsed, "stage": "k3union_designed"}

    if mode == "hybrid":
        # 1) natural holistic compare decides WHETHER to flag (keeps high recall).
        nat_memo = vllm_chat(P3_COMPARE_SYS, P3_COMPARE.format(question=row["question"], answer=a[:1500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000]), port, 700, 0.0, tag="compare.judge")
        nat_parsed = parse_memo(row, answer_facts, question_facts, nat_memo, parser)
        if nat_parsed["verdict"] != "INCORRECT":
            nat_parsed["flagged"] = False
            return {"answer_facts": answer_facts, "question_facts": question_facts, "memo": nat_memo, "parsed": nat_parsed, "stage": "natural_flag_clear"}
        # 2) flagged -> planner item-by-item produces the PRECISE diagnosis (high AGREE).
        plan = vllm_chat(P0_PLANNER_SYS, P0_PLANNER.format(question=row["question"], answer=a[:1500]), port, 500, 0.0, tag="planner")
        memo = vllm_chat(P3_COMPARE_SYS, P3_COMPARE_PLANNED.format(question=row["question"], answer=a[:1500], plan=plan[:2500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000]), port, 800, 0.0, tag="compare.planned")
        parsed = parse_memo(row, answer_facts, question_facts, memo, parser)
        # natural already said wrong; keep flagged true and prefer the precise diagnosis fields.
        parsed["flagged"] = (parsed["wrong_claim"] not in ("", "NONE") or parsed["correct_or_missing_info"] not in ("", "NONE"))
        if not parsed["flagged"]:
            # planner could not pin it; fall back to natural's diagnosis so we don't lose the flag.
            parsed = nat_parsed
            parsed["flagged"] = True
        return {"answer_facts": answer_facts, "question_facts": question_facts, "nat_memo": nat_memo, "memo": memo, "parsed": parsed, "stage": "hybrid_diagnosed"}

    if mode == "planner":
        plan = vllm_chat(P0_PLANNER_SYS, P0_PLANNER.format(question=row["question"], answer=a[:1500]), port, 500, 0.0, tag="planner")
        memo = vllm_chat(P3_COMPARE_SYS, P3_COMPARE_PLANNED.format(question=row["question"], answer=a[:1500], plan=plan[:2500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000]), port, 800, 0.0, tag="compare.planned")
    elif mode == "cot":
        memo = vllm_chat(P3_COMPARE_SYS, P3_COMPARE_COT.format(question=row["question"], answer=a[:1500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000]), port, 900, 0.0, tag="compare.cot")
    else:
        memo = vllm_chat(P3_COMPARE_SYS, P3_COMPARE.format(question=row["question"], answer=a[:1500], answer_facts=answer_facts[:3000], question_facts=question_facts[:3000]), port, 700, 0.0, tag="compare.judge")
    parsed = parse_memo(row, answer_facts, question_facts, memo, parser)
    parsed["flagged"] = _flagged(parsed)
    return {"answer_facts": answer_facts, "question_facts": question_facts, "memo": memo, "parsed": parsed, "stage": mode}


def assemble_live_diagnosis(p: dict[str, Any]) -> str:
    parts = []
    if p.get("wrong_claim") and p["wrong_claim"] != "NONE":
        parts.append(f"What is wrong: {p['wrong_claim']}")
    if p.get("correct_or_missing_info") and p["correct_or_missing_info"] != "NONE":
        parts.append(f"What it should be: {p['correct_or_missing_info']}")
    if p.get("decisive_evidence") and p["decisive_evidence"] != "NONE":
        parts.append(f"Evidence: {p['decisive_evidence']}")
    return " ".join(parts) if parts else "(no specific diagnosis)"


def rate_diagnosis(row: dict[str, Any], live: str) -> str:
    oracle = row.get("oracle_error_description", "")
    if not oracle:
        return "NO_ORACLE"
    raw = gpt("gpt-4o-mini", "You compare two descriptions of what is wrong with a clinical answer.",
              f"Oracle diagnosis (gold):\n{oracle[:800]}\n\nLive diagnosis:\n{live[:800]}\n\n"
              "Does the live diagnosis identify the SAME underlying error as the oracle? Reply ONE word: AGREE, PARTIAL, or WRONG.",
              max_tokens=6, tag="rate.diagnosis")
    m = re.search(r"(AGREE|PARTIAL|WRONG)", (raw or "").upper())
    return m.group(1) if m else "UNKNOWN"


# ---------- correction + judge ----------

CORRECTION_SYS = ("You are a careful clinical QA assistant. Revise the previous answer only when the "
                  "discharge note and provided evidence support the revision. Do not add facts not supported by the note.")


def retrieve_spans(row: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    return topk_spans(row["note"], [q for q in [row["question"], row["original_answer"][:800]] if q], k=k, scoring="agreement")


def render_spans(spans: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans)) if spans else "(none)"


def _f(v: str) -> str:
    v = (v or "").strip()
    return v if v and v.upper() != "NONE" else "(none)"


def run_correction(row: dict[str, Any], p: dict[str, Any], spans: list[dict[str, Any]], port: int) -> str:
    """Operation-guided correction: hand the corrector the parsed structured fields
    in the exact shape it needs — the named operation, the wrong claim, the
    note-supported target, the decisive evidence, and what to preserve."""
    op = _f(p.get("correction_operation"))
    user = f"""Discharge note:
{row['note'][:24000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{render_spans(spans)}

Audit result (apply exactly this):
- correction operation: {op}
- wrong or missing part: {_f(p.get('wrong_claim'))}
- note-supported target (what it should be): {_f(p.get('correct_or_missing_info'))}
- decisive note evidence: {_f(p.get('decisive_evidence'))}
- preserve unchanged: {_f(p.get('do_not_change'))}

Perform only the named correction operation on the previous answer, grounded in the decisive evidence and the retrieved note evidence. Keep everything in 'preserve unchanged'. Do not add facts beyond the evidence. If the audit does not name a real, note-supported error, return the previous answer unchanged. Return only the final answer."""
    return vllm_chat(CORRECTION_SYS, user, port, 700, 0.0, tag="correction")


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
    raw = gpt("gpt-4o", "You are a medical expert evaluating an AI model's answer to a clinical question.", user, max_tokens=10, temperature=0.1, tag="judge")
    return {"label": parse_binary(raw), "raw": raw}


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int, parser: str, mode: str = "natural") -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        spans = retrieve_spans(row, k=5)
        out["judge_original"] = judge(row, row["original_answer"])
        det = run_detection(row, port, parser, mode)
        p = det["parsed"]
        out["detection"] = {"flagged": p["flagged"], "verdict": p["verdict"], "stage": det.get("stage"),
                            "k3_verdicts": det.get("k3_verdicts"),
                            "parsed_fields": {k: p.get(k) for k in ["error_type", "correction_operation", "wrong_claim", "correct_or_missing_info", "decisive_evidence", "do_not_change"]}}
        if p["flagged"]:
            live = assemble_live_diagnosis(p)
            out["live_diagnosis"] = live
            out["diagnosis_quality"] = rate_diagnosis(row, live)
            corrected = run_correction(row, p, spans, port)
            out["corrected"] = corrected
            out["judge_final"] = judge(row, corrected)
        else:
            out["diagnosis_quality"] = "NOT_FLAGGED"
            out["judge_final"] = {"label": (out["judge_original"] or {}).get("label"), "raw": "kept_original"}
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    flagged_w = sum(1 for r in wrong if (r.get("detection") or {}).get("flagged"))
    flagged_c = sum(1 for r in correct if (r.get("detection") or {}).get("flagged"))
    dq = Counter(r.get("diagnosis_quality") for r in wrong if (r.get("detection") or {}).get("flagged"))
    fix = sum(1 for r in wrong if (r.get("judge_final") or {}).get("label") == 1)
    brk = sum(1 for r in correct if (r.get("judge_final") or {}).get("label") == 0)
    return {
        "n_cases": len(rows), "n_wrong": len(wrong), "n_correct": len(correct),
        "recall_on_wrong": round(flagged_w / max(1, len(wrong)), 3),
        "overflag_on_correct": round(flagged_c / max(1, len(correct)), 3),
        "diagnosis_quality": dict(dq),
        "agree_rate_of_flagged": round(dq.get("AGREE", 0) / max(1, sum(dq.values())), 3),
        "fix": fix, "break": brk, "net": fix - brk,
        "fix_rate_on_wrong": round(fix / max(1, len(wrong)), 3),
        "compare_note": "oracle ceiling ~60% fix (Phase1); plan->confirm detection ~5% AGREE, 5-10% fix (Phase2)",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--n-correct", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parser", choices=["gpt4o-mini", "helper-v2", "qwen35"], default="helper-v2")
    ap.add_argument("--detect-mode", choices=["natural", "planner", "cot", "hybrid", "k3union"], default="natural")
    args = ap.parse_args()
    served = served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}_{args.parser}_{args.detect_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="phase2b_extract_compare", served=served, args=vars(args))
    print(f"sample={len(sample)} parser={args.parser} mode={args.detect_mode} c={args.concurrency} out={out_dir}", flush=True)
    if sample:
        topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port, args.parser, args.detect_mode) for r in sample]
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
