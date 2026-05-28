#!/usr/bin/env python3
"""
Module 1 — Detection-format bake-off.

Compares 4 detection variants on a 60-item labeled set:

  F1  free-form Qwen2.5 + Qwen3-32B JSON extractor          (current baseline)
  J1  direct JSON, single combined prompt                    (one prompt → JSON)
  J2  direct JSON, 3 sub-prompts (contra/qmis/omis)
  J3  direct JSON, single combined, with claim grounding    (model first lists
       3-5 atomic claims and quotes the supporting/contradicting span before
       emitting the verdict)

Primary metric (per user direction): semantic validity of the captured
`correct_statement` against the actual notes — judged by GPT-4o (separate
from the answer judge) as a binary "is this correct_statement supported by
the notes and a valid characterization of the error?".

Secondary metrics: TP / FP rates (vs the temp=0 GPT-4o judge labels from
Module 0), vote agreement at K=5 samples per item.

Outputs:
  output/step9_v2/detection_bakeoff_results.json
  output/step9_v2/detection_bakeoff_summary.md
"""
from __future__ import annotations

import os
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sampling import vote_call
from judge import _load_notes_lookup, client

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUT_DIR = RUN_ROOT / "output" / "step9_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

# ---------- vLLM helpers (mirror run_fullscale.py:174-182 verbatim) ----------

def build_chatml(system: str, user: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def vllm_gen(prompt: str, port: int, *, max_tokens: int = 1024,
             temperature: float = 0.7, stop: list[str] | None = None) -> str:
    model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
    payload = {
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": temperature,
        "stop": stop or ["<|im_end|>", "<|endoftext|>"],
    }
    r = requests.post(f"http://localhost:{port}/v1/completions", json=payload, timeout=180)
    return r.json()["choices"][0]["text"].strip()


_THINK_PAIRED_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# DeepSeek-R1 distill emits reasoning starting from the first token with NO
# opening <think> tag, just a closing </think> tag at the end. Strip
# everything up to and including the first </think>.
_THINK_HEAD_RE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Module-level default chat_template_kwargs. Pilots that want to disable
# Qwen3 thinking set this once at the start of the model run via
# set_default_chat_template_kwargs({"enable_thinking": False}).
_DEFAULT_CHAT_TEMPLATE_KWARGS: dict | None = None


def set_default_chat_template_kwargs(kwargs: dict | None) -> None:
    """Set the module-level default chat_template_kwargs that vllm_chat
    will inject when no per-call override is given. Used to enable Qwen3
    non-thinking mode for the duration of a pilot run."""
    global _DEFAULT_CHAT_TEMPLATE_KWARGS
    _DEFAULT_CHAT_TEMPLATE_KWARGS = kwargs


def vllm_chat(system: str, user: str, port: int, *,
              max_tokens: int = 1024,
              temperature: float = 0.7,
              strip_think: bool = True,
              chat_template_kwargs: dict | None = None) -> str:
    """Model-agnostic chat call via /v1/chat/completions.

    The vLLM server applies the loaded model's tokenizer chat template
    automatically (Qwen ChatML, Llama-3 special tokens, DeepSeek format,
    etc.), so the same code works across models. Used by the multi-model
    pilot — no hardcoded ChatML.

    Per-model quirks handled here:
      - BioMistral / Mistral chat templates don't support a separate system
        role. If the server returns 400 "Conversation roles must alternate",
        we retry with the system message merged into the user message.
      - DeepSeek-R1 / Qwen3 thinking blocks are stripped (via strip_think).
      - Qwen3 hard-disable thinking: pass
            chat_template_kwargs={"enable_thinking": False}
        which the server forwards to the model's Jinja chat template.
        Confirmed via the Qwen3 docs and vLLM 0.9+ docs (2025-04-08
        research). The /no_think soft directive is unreliable and the
        chat_template_kwargs form is the canonical way to disable.

    `strip_think=True` removes <think>...</think> blocks emitted by Qwen3 /
    DeepSeek-R1 reasoning models so downstream parsers see only the final
    answer.
    """
    served = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()
    model_id = served["data"][0]["id"]
    base = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Per-call override > module-level default > none
    effective_kwargs = chat_template_kwargs or _DEFAULT_CHAT_TEMPLATE_KWARGS
    if effective_kwargs:
        base["chat_template_kwargs"] = effective_kwargs

    def _post(messages):
        payload = dict(base, messages=messages)
        return requests.post(f"http://localhost:{port}/v1/chat/completions",
                             json=payload, timeout=240)

    # First try: separate system + user
    r = _post([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    body = r.json()
    if "choices" not in body:
        # Likely BioMistral / Mistral: no system role allowed → merge
        err = (body.get("error") or {}).get("message", "")
        if "alternate" in err.lower() or "system" in err.lower():
            r = _post([{"role": "user", "content": f"{system}\n\n{user}"}])
            body = r.json()
        if "choices" not in body:
            raise RuntimeError(f"vLLM chat call failed: {body}")
    text = body["choices"][0]["message"]["content"]
    if text is None:
        return ""
    text = text.strip()
    if strip_think:
        text = _THINK_PAIRED_RE.sub("", text)
        if "</think>" in text.lower():
            text = _THINK_HEAD_RE.sub("", text, count=1)
        text = text.strip()
    return text


def served_model_id(port: int) -> str:
    """Return the id of the model currently loaded by vLLM at `port`."""
    return requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]


def q32_extract(raw: str, schema_template: str) -> dict | None:
    """Use Qwen3-32B (Mac Studio) to extract JSON from a free-form output.
    Same scaffold as run_fullscale.py:184-197."""
    try:
        r = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [{"role": "system", "content": "Extract info. JSON only."},
                         {"role": "user", "content": schema_template.format(raw=raw[:2500])}],
            "max_tokens": 400, "temperature": 0.0,
        }, timeout=90)
        text = r.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"  q32 extract error: {e}", flush=True)
    return None


# ---------- Detection prompts ----------

# F1 — free-form, three sub-prompts (mirror run_fullscale.py 38-74 exactly)
DET_F1_CONTRA = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CONTRADICTION: Does the answer state any fact that DIRECTLY CONFLICTS with the discharge notes?

For each key claim (medication names, dosages, procedures, diagnoses, lab values, dates):
1. Find the matching information in the notes
2. Does the answer say something DIFFERENT?

Only flag OPPOSING information. If you find a contradiction, explain what the answer says vs what the notes say."""

DET_F1_QMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR QUESTION MISALIGNMENT: Does the answer address the WRONG thing?

Parse the question: which visit, what aspect, what time period?
Does the answer match? If misaligned, explain what the question asks vs what the answer discusses."""

DET_F1_OMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CRITICAL OMISSION: Is there information in the notes ESSENTIAL to answer the question but COMPLETELY ABSENT?

Only flag omissions that would CHANGE the conclusion. Do NOT flag minor details."""

EXTRACT_F1 = """/nothink
Read this self-critique. Extract as JSON. Only INCORRECT if critical errors found.

TEXT:
{raw}

{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the error as one sentence", "correct_statement": "what notes say"}}"""

# J1 — direct JSON, single combined prompt
DET_J1 = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

You are checking the answer for three error types:
1. CONTRADICTION   — answer states a fact that DIRECTLY CONFLICTS with the notes
2. QUESTION_MISALIGNMENT — answer addresses the wrong aspect of the question
3. OMISSION        — answer is missing information from the notes that is ESSENTIAL to the question

Output ONLY a JSON object with this exact schema. No other text:

{{"verdict": "CORRECT" or "INCORRECT",
 "error_type": "CONTRADICTION" or "QUESTION_MISALIGNMENT" or "OMISSION" or "NONE",
 "error_statement": "<one-sentence description of the wrong/missing/misaligned content in the answer, or empty if CORRECT>",
 "correct_statement": "<one-sentence statement of what the notes actually say about this point, or empty if CORRECT>"}}"""

# J2 — direct JSON, 3 sub-prompts (one JSON each)
DET_J2_CONTRA = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check ONLY for CONTRADICTION: does any factual claim in the answer DIRECTLY CONFLICT with the discharge notes?

Output ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT",
 "error_type": "CONTRADICTION" or "NONE",
 "error_statement": "<the contradicted claim from the answer, or empty>",
 "correct_statement": "<what the notes actually say about it, or empty>"}}"""

DET_J2_QMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check ONLY for QUESTION_MISALIGNMENT: does the answer address the wrong visit / aspect / time-period?

Output ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT",
 "error_type": "QUESTION_MISALIGNMENT" or "NONE",
 "error_statement": "<the misaligned aspect, or empty>",
 "correct_statement": "<what the question is actually asking, or empty>"}}"""

DET_J2_OMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check ONLY for critical OMISSION: is there information in the notes ESSENTIAL to the question that the answer omits? Only flag omissions that would change the conclusion.

Output ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT",
 "error_type": "OMISSION" or "NONE",
 "error_statement": "<a phrase saying what is missing, or empty>",
 "correct_statement": "<the relevant fact from the notes, or empty>"}}"""

# J3 — direct JSON, single combined, with claim grounding
DET_J3 = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Step 1: Extract 3-5 atomic claims from the answer (medications, diagnoses,
        procedures, dates, outcomes, etc.).

Step 2: For each claim, quote the supporting OR contradicting span from the
        notes verbatim (or "NO MATCH" if the notes say nothing).

Step 3: Decide whether the answer has any of these errors:
        - CONTRADICTION       (a claim contradicts the notes)
        - QUESTION_MISALIGNMENT (the answer addresses the wrong aspect)
        - OMISSION             (a fact in the notes essential to the question is missing)

Output a single JSON object containing both your reasoning and the verdict:

{{"claims": [{{"claim": "...", "note_span": "..."}}, ...],
 "verdict": "CORRECT" or "INCORRECT",
 "error_type": "CONTRADICTION" or "QUESTION_MISALIGNMENT" or "OMISSION" or "NONE",
 "error_statement": "<one-sentence error description, or empty>",
 "correct_statement": "<one-sentence statement of what the notes say, or empty>"}}"""


# ---------- Variant runners ----------

SYS_DET = "You are a strict medical expert checking clinical answers."


def _vote_key_for_det(parsed: dict | None) -> str:
    if not parsed:
        return "NONE"
    return str(parsed.get("verdict", "NONE")).upper()


def _parse_inline_json(raw: str) -> dict | None:
    """Parse a JSON object from a model output. Strip thinking, find first {…}."""
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _normalise_det(obj: dict | None) -> dict:
    if not obj:
        return {"verdict": "UNCLEAR", "error_type": "NONE",
                "error_statement": "", "correct_statement": ""}
    return {
        "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
        "error_type": str(obj.get("error_type", "NONE")).upper(),
        "error_statement": str(obj.get("error_statement", ""))[:300],
        "correct_statement": str(obj.get("correct_statement", ""))[:300],
    }


def _aggregate_subprompts(sub_results: dict[str, dict]) -> dict:
    """Aggregate 3 sub-prompt detection dicts into one final dict.
    Priority for picking the 'critical' error: qmis > contra > omis (matches
    run_fullscale.py)."""
    detected = {k: v for k, v in sub_results.items() if v["verdict"] == "INCORRECT"}
    if not detected:
        # Use any sub-prompt verdict (CORRECT) as the aggregate verdict
        return {"verdict": "CORRECT", "error_type": "NONE",
                "error_statement": "", "correct_statement": "",
                "detected_sub_keys": []}
    for sub_key in ("qmis", "contra", "omis"):
        if sub_key in detected:
            chosen = detected[sub_key]
            type_map = {"qmis": "QUESTION_MISALIGNMENT",
                        "contra": "CONTRADICTION", "omis": "OMISSION"}
            return {
                "verdict": "INCORRECT",
                "error_type": type_map[sub_key],
                "error_statement": chosen["error_statement"],
                "correct_statement": chosen["correct_statement"],
                "detected_sub_keys": list(detected.keys()),
            }
    # unreachable
    return {"verdict": "CORRECT", "error_type": "NONE",
            "error_statement": "", "correct_statement": "",
            "detected_sub_keys": []}


def run_F1(note: str, question: str, answer: str, port: int, k: int) -> dict:
    """Free-form 3 sub-prompts + Qwen3-32B extraction. Returns aggregated dict
    + per-sub-prompt vote results for traceability."""
    sub_votes = {}
    final_aggregates: list[dict] = []  # one aggregate per sample (for vote_call)
    raw_per_sample: list[dict] = []
    for sample_i in range(k):
        sub_results = {}
        sub_raws = {}
        for sub_key, prompt in [("contra", DET_F1_CONTRA),
                                ("qmis", DET_F1_QMIS),
                                ("omis", DET_F1_OMIS)]:
            user = prompt.format(note=note, question=question, answer=answer[:800])
            raw = vllm_gen(build_chatml(SYS_DET, user), port, temperature=0.7,
                           max_tokens=1024)
            obj = q32_extract(raw, EXTRACT_F1)
            sub_results[sub_key] = _normalise_det(obj)
            sub_raws[sub_key] = raw
        agg = _aggregate_subprompts(sub_results)
        final_aggregates.append(agg)
        raw_per_sample.append({"sub_results": sub_results, "sub_raws": sub_raws})
    return _vote_over_aggregates(final_aggregates, raw_per_sample)


def run_J1(note: str, question: str, answer: str, port: int, k: int) -> dict:
    """Single direct-JSON prompt; sample k times; vote on verdict."""
    final_aggregates: list[dict] = []
    raw_per_sample: list[dict] = []
    user = DET_J1.format(note=note, question=question, answer=answer[:800])
    prompt = build_chatml(SYS_DET, user)
    for _ in range(k):
        raw = vllm_gen(prompt, port, temperature=0.7, max_tokens=400)
        obj = _parse_inline_json(raw)
        agg = _normalise_det(obj)
        final_aggregates.append(agg)
        raw_per_sample.append({"raw": raw})
    return _vote_over_aggregates(final_aggregates, raw_per_sample)


def run_J2(note: str, question: str, answer: str, port: int, k: int) -> dict:
    final_aggregates: list[dict] = []
    raw_per_sample: list[dict] = []
    for _ in range(k):
        sub_results = {}
        sub_raws = {}
        for sub_key, prompt in [("contra", DET_J2_CONTRA),
                                ("qmis", DET_J2_QMIS),
                                ("omis", DET_J2_OMIS)]:
            user = prompt.format(note=note, question=question, answer=answer[:800])
            raw = vllm_gen(build_chatml(SYS_DET, user), port, temperature=0.7, max_tokens=300)
            sub_results[sub_key] = _normalise_det(_parse_inline_json(raw))
            sub_raws[sub_key] = raw
        agg = _aggregate_subprompts(sub_results)
        final_aggregates.append(agg)
        raw_per_sample.append({"sub_results": sub_results, "sub_raws": sub_raws})
    return _vote_over_aggregates(final_aggregates, raw_per_sample)


def run_J3(note: str, question: str, answer: str, port: int, k: int) -> dict:
    final_aggregates: list[dict] = []
    raw_per_sample: list[dict] = []
    user = DET_J3.format(note=note, question=question, answer=answer[:800])
    prompt = build_chatml(SYS_DET, user)
    for _ in range(k):
        raw = vllm_gen(prompt, port, temperature=0.7, max_tokens=900)
        obj = _parse_inline_json(raw)
        agg = _normalise_det(obj)
        # also persist the claim list if present
        if obj and "claims" in obj:
            agg["claims"] = obj["claims"]
        final_aggregates.append(agg)
        raw_per_sample.append({"raw": raw})
    return _vote_over_aggregates(final_aggregates, raw_per_sample)


def _vote_over_aggregates(aggregates: list[dict], raws: list[dict]) -> dict:
    """Take K aggregates and produce a vote on the verdict + the canonical
    output drawn from a sample whose verdict matches the majority."""
    keys = [a["verdict"] for a in aggregates]
    from collections import Counter
    counts = Counter(keys)
    if not counts:
        return {"final": _normalise_det(None), "samples": [], "raws": raws,
                "vote_distribution": {}, "unanimity": 0.0, "n_valid": 0}
    majority_key, _ = counts.most_common(1)[0]
    final = next(a for a in aggregates if a["verdict"] == majority_key)
    return {
        "final": final,
        "samples": aggregates,
        "raws": raws,
        "vote_distribution": dict(counts),
        "unanimity": counts[majority_key] / len(keys),
        "n_valid": len(keys),
    }


# ---------- Test set ----------

def build_test_set(n_wrong: int = 30, n_correct: int = 30, seed: int = 42) -> list[dict]:
    """Sample n items from the temp=0 re-judged step8 Qwen2.5 set.
    Falls back to legacy temp=0.1 binary_correct if T0 file missing."""
    t0_path = OUT_DIR / "zeroshot_evaluated_binary_T0.csv"
    if t0_path.exists():
        t0 = pd.read_csv(t0_path)
    else:
        t0 = None

    parts = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            parts.append(df)
    base = pd.concat(parts, ignore_index=True)

    if t0 is not None:
        base = base.merge(t0[["fold", "idx", "binary_correct_T0"]], on=["fold", "idx"],
                           how="left")
        # Use T0 label where present, else fall back to legacy
        base["label"] = base["binary_correct_T0"].fillna(base["binary_correct"]).astype(int)
    else:
        base["label"] = base["binary_correct"].astype(int)

    rng = random.Random(seed)
    wrong = base[base["label"] == 0].sample(n=n_wrong, random_state=rng.randint(0, 2**31)).to_dict("records")
    correct = base[base["label"] == 1].sample(n=n_correct, random_state=rng.randint(0, 2**31)).to_dict("records")
    items = []
    for r in wrong + correct:
        items.append({
            "fold": int(r["fold"]),
            "idx": int(r["idx"]),
            "patient_id": int(r["patient_id"]),
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "model_answer": str(r["model_answer"]),
            "label": int(r["label"]),  # 1=correct, 0=wrong
        })
    return items


# ---------- Semantic-validity GPT-4o judge ----------

VALIDITY_SYSTEM = (
    "You are a medical expert auditing whether a self-critique correctly characterised "
    "the error in an AI model's answer to a clinical question."
)

VALIDITY_USER_TMPL = """DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

MODEL'S ANSWER (the answer being critiqued):
{model_answer}

SELF-CRITIQUE OUTPUT:
- claimed error type:    {error_type}
- error_statement:       {error_statement}
- correct_statement:     {correct_statement}

Evaluate ONLY the `correct_statement` field on two criteria:

NOTE_SUPPORTED: yes|no    (does the discharge summary actually say what `correct_statement` says about this fact?)
ADDRESSES_ERROR: yes|no   (does `correct_statement` address the same fact as `error_statement` — i.e. is it actually fixing what was wrong, not pointing at something unrelated?)

Reply with exactly two lines in that format. No other text.
"""


def _parse_validity(text: str) -> dict:
    out = {"note_supported": None, "addresses_error": None, "raw": text}
    for line in text.splitlines():
        u = line.strip().upper()
        if u.startswith("NOTE_SUPPORTED:"):
            after = line.split(":", 1)[1].strip().lower()
            out["note_supported"] = "yes" if after.startswith("yes") else ("no" if after.startswith("no") else None)
        elif u.startswith("ADDRESSES_ERROR:"):
            after = line.split(":", 1)[1].strip().lower()
            out["addresses_error"] = "yes" if after.startswith("yes") else ("no" if after.startswith("no") else None)
    return out


def gpt4o_validity(note: str, question: str, model_answer: str, det_final: dict) -> dict:
    """Ask GPT-4o whether the captured `correct_statement` is semantically valid."""
    user = VALIDITY_USER_TMPL.format(
        note=note,
        question=question,
        model_answer=model_answer[:800],
        error_type=det_final.get("error_type", "NONE"),
        error_statement=det_final.get("error_statement", ""),
        correct_statement=det_final.get("correct_statement", ""),
    )
    for attempt in range(3):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": VALIDITY_SYSTEM},
                          {"role": "user", "content": user}],
                max_tokens=60,
                temperature=0.0,
            )
            return _parse_validity(r.choices[0].message.content.strip())
        except Exception as e:
            print(f"  validity retry {attempt+1}/3: {e}", flush=True)
            time.sleep(5)
    return {"note_supported": None, "addresses_error": None, "raw": ""}


# ---------- Main ----------

VARIANTS = {
    "F1": run_F1,
    "J1": run_J1,
    "J2": run_J2,
    "J3": run_J3,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--k", type=int, default=5, help="samples per item per variant")
    p.add_argument("--n-wrong", type=int, default=30)
    p.add_argument("--n-correct", type=int, default=30)
    p.add_argument("--variants", default="F1,J1,J2,J3")
    p.add_argument("--no-validity", action="store_true",
                   help="skip GPT-4o semantic-validity step (vLLM only)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap items for a smoke test")
    args = p.parse_args()

    variants = [v for v in args.variants.split(",") if v in VARIANTS]
    if not variants:
        print("!! no valid variants selected"); return 1

    items = build_test_set(args.n_wrong, args.n_correct)
    if args.limit:
        items = items[:args.limit]
    notes = _load_notes_lookup()

    print(f"Test set: {len(items)} items "
          f"({sum(1 for i in items if i['label']==0)} wrong, "
          f"{sum(1 for i in items if i['label']==1)} correct)", flush=True)
    print(f"Variants: {variants}, K={args.k}", flush=True)

    out_path = OUT_DIR / "detection_bakeoff_results.json"
    # Resume support
    results: dict[str, list] = {v: [] for v in variants}
    done: dict[str, set[tuple[int, int]]] = {v: set() for v in variants}
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        for v in variants:
            results[v] = prior.get(v, [])
            done[v] = {(r["fold"], r["idx"]) for r in results[v]}
        print(f"Resuming with prior: " + ", ".join(f"{v}={len(results[v])}" for v in variants),
              flush=True)

    for v in variants:
        runner = VARIANTS[v]
        print(f"\n=== Variant {v} ===", flush=True)
        for i, item in enumerate(items, 1):
            if (item["fold"], item["idx"]) in done[v]:
                continue
            note = notes.get(str(item["patient_id"]), "")
            if not note:
                continue
            try:
                det = runner(note, item["question"], item["model_answer"], args.port, args.k)
            except Exception as e:
                print(f"  ❌ {v} item ({item['fold']},{item['idx']}): {e}", flush=True)
                continue
            entry = {
                "variant": v,
                "fold": item["fold"], "idx": item["idx"],
                "label": item["label"],
                "patient_id": item["patient_id"],
                "det_final": det["final"],
                "vote_distribution": det["vote_distribution"],
                "unanimity": det["unanimity"],
                "samples": det["samples"],
                "raws": det["raws"],
            }
            # Semantic validity check (GPT-4o) only when detection fired
            if (not args.no_validity) and det["final"]["verdict"] == "INCORRECT":
                vresult = gpt4o_validity(note, item["question"], item["model_answer"], det["final"])
                entry["validity"] = vresult
                time.sleep(0.5)
            results[v].append(entry)
            if i % 5 == 0:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  {v}: {i}/{len(items)} done", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    summary_lines = ["# Detection Format Bake-off Results", "", f"K={args.k} samples per item, "
                     f"N_wrong={args.n_wrong}, N_correct={args.n_correct}", "",
                     "| Variant | TP rate | FP rate | TP-FP | Validity yes-rate | Mean unanimity |",
                     "|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        rs = results[v]
        if not rs:
            continue
        n_w = sum(1 for r in rs if r["label"] == 0)
        n_c = sum(1 for r in rs if r["label"] == 1)
        tp = sum(1 for r in rs if r["label"] == 0 and r["det_final"]["verdict"] == "INCORRECT")
        fp = sum(1 for r in rs if r["label"] == 1 and r["det_final"]["verdict"] == "INCORRECT")
        with_val = [r for r in rs if r.get("validity")]
        valid_yes = sum(1 for r in with_val
                        if r["validity"].get("note_supported") == "yes"
                        and r["validity"].get("addresses_error") == "yes")
        mean_u = float(np.mean([r["unanimity"] for r in rs])) if rs else 0.0
        tp_rate = 100*tp/n_w if n_w else 0
        fp_rate = 100*fp/n_c if n_c else 0
        valid_rate = 100*valid_yes/len(with_val) if with_val else 0
        summary_lines.append(
            f"| {v} | {tp_rate:.0f}% ({tp}/{n_w}) | {fp_rate:.0f}% ({fp}/{n_c}) "
            f"| {tp_rate-fp_rate:+.0f} | {valid_rate:.0f}% ({valid_yes}/{len(with_val)}) | {mean_u:.2f} |"
        )

    summary_lines.append("")
    summary_lines.append("Decision rule: highest validity yes-rate first, then break ties by (TP - FP).")
    (OUT_DIR / "detection_bakeoff_summary.md").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
