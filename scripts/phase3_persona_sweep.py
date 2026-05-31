#!/usr/bin/env python3
"""Phase 3 (persona axis): does WHO the model is told to be change correction behavior?

Holds information content and reasoning structure constant; varies only the system
persona. Persona is the willingness-to-edit lever, so the study MUST include both
wrong cases (to measure fixes) and originally-correct cases (to measure breaks /
over-editing). Phase 1 was wrong-only and therefore could not see this.

Information set is configurable via --info-set. Default is spans_only — a clean,
symmetric setting where every case gets the same kind of input (same-patient spans),
so the persona effect is isolated. To layer the Phase-1-winning information on top,
pass --info-set contradiction_quote (correct cases have no taxonomy entry, so they
naturally receive spans-only; wrong cases receive the oracle hint — this mirrors a
deployment where detection emits a hint only for flagged cases).

Personas (system prompt only; user body is identical across arms):
  neutral           current production wording
  student           medical student double-checking own work
  judge             strict examiner deciding stand-or-revise
  senior_clinician  senior attending reviewing a junior, decisive but not over-editing
  peer_reviewer     cautious reviewer, change only what evidence forces

Metrics per persona:
  fix         wrong->correct
  break       correct->wrong (over-editing)
  net         fix - break
  edit_rate   fraction of cases where the answer was materially changed (willingness)

Pre-flight: Qwen2.5-7B-Instruct on vLLM port 8003.

Output:
  runs/phase3_persona/qwen25_nw{NW}_nc{NC}_seed{SEED}_{info}/{judged_outputs.jsonl, summary.json}
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
OUT_ROOT = PROJECT_ROOT / "runs" / "phase3_persona"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TAXONOMY = SOURCE_REPO / "src" / "step9_self_correction" / "error_taxonomy" / "phase1_wrong_gpt4o.json"
NOTE_SPAN_SRC = SOURCE_REPO / "src" / "step9_self_correction" / "v2"
sys.path.insert(0, str(NOTE_SPAN_SRC))
from note_span_index import topk_spans  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from llm_audit import set_ledger, log_call  # noqa: E402


# ---------- personas (system prompts) ----------

# Correction personas. Informed by GPT-4o suggestions (2026-05-29), spanning the
# edit-willingness dial: a conservative cluster (GPT's recommended winners) plus a
# decisive middle and a deliberately aggressive endpoint to map where over-editing peaks.
PERSONAS: dict[str, str] = {
    "neutral": (
        "You are a careful clinical QA assistant. Revise the previous answer only when the "
        "discharge note and provided evidence support the revision. Do not add facts not "
        "supported by the note."
    ),
    "surgical_editor": (
        "You are a surgical editor. Make only the minimal necessary change to correct the specific "
        "error, preserving all correct information from the original answer. Do not rewrite, restyle, "
        "or expand anything the diagnosis did not flag."
    ),
    "precision_fixer": (
        "You are a precision fixer. Address the diagnosed error precisely and leave all other "
        "information unchanged. If the evidence does not prove a specific change, keep the original."
    ),
    "senior_attending": (
        "You are a senior attending physician reviewing a junior colleague's answer. You are decisive "
        "and willing to correct real clinical errors using the chart, but you do not rewrite sound "
        "answers or add unnecessary detail. Preserve what is correct; fix what is clinically wrong."
    ),
    "overzealous_improver": (
        # Deliberate backfire endpoint — NOT a ship candidate. Maps the upper bound of over-editing.
        "You are an eager clinical improver. Make the answer as complete, precise, and polished as "
        "possible. Add any clinically relevant detail from the note, sharpen vague statements, and "
        "rewrite anything that could be clearer or more thorough."
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
    for attempt in range(4):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical expert evaluating an AI model's answer to a clinical question."},
                    {"role": "user", "content": user},
                ],
                max_tokens=10, temperature=0.1,
            )
            raw = (r.choices[0].message.content or "").strip()
            log_call("judge", "gpt-4o", "judge", user, raw, fold=row.get("fold"), idx=row.get("idx"))
            return {"label": parse_binary(raw), "raw": raw}
        except Exception as e:
            if attempt == 3:
                return {"label": None, "raw": "", "error": str(e)}
            time.sleep(1 + attempt)
    return {"label": None}


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
                "tax_error_description": t.get("ERROR_DESCRIPTION", ""),
                "tax_question_focus": t.get("QUESTION_FOCUS", ""),
                "tax_model_claims": t.get("MODEL_CLAIMS", ""),
                "tax_primary_error": t.get("PRIMARY_ERROR", ""),
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


# ---------- spans + info ----------

def retrieve_spans(row: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    queries = [row["question"], row["original_answer"][:800], row.get("tax_question_focus", "")]
    queries = [q for q in queries if q]
    return topk_spans(row["note"], queries, k=k, scoring="agreement")


def render_spans(spans: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans)) if spans else "(none)"


def info_block(row: dict[str, Any], info_set: str) -> str:
    """Information held constant across personas. Correct cases have no taxonomy,
    so any oracle field is empty for them (spans-only), mirroring deployment."""
    if info_set == "spans_only":
        return "(no extra information)"
    if info_set == "contradiction_quote":
        desc = row.get("tax_error_description", "")
        return f"What is wrong and why: {desc}" if desc else "(no extra information)"
    if info_set == "error_type":
        et = row.get("tax_primary_error", "")
        return f"Error type: {et}" if et else "(no extra information)"
    raise ValueError(f"unknown info_set {info_set}")


def build_user(row: dict[str, Any], spans: list[dict[str, Any]], extra: str) -> str:
    return f"""Discharge note:
{row['note'][:18000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

Same-patient retrieved evidence:
{render_spans(spans)}

Additional information:
{extra}

Use the evidence and any additional information to check the previous answer. If it is wrong or incomplete, return the corrected answer grounded in the note. If it is already correct, keep it. Return only the final answer."""


# ---------- edit detection ----------

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)  # drop punctuation so trailing '.' is not an edit
    return re.sub(r"\s+", " ", s).strip()


def materially_changed(original: str, corrected: str) -> bool:
    o, c = normalize(original), normalize(corrected)
    if o == c:
        return False
    # token Jaccard; treat >0.92 overlap as "not materially changed"
    ot, ct = set(o.split()), set(c.split())
    if not ot or not ct:
        return o != c
    jac = len(ot & ct) / len(ot | ct)
    return jac < 0.92


# ---------- orchestration ----------

def process_one(row: dict[str, Any], port: int, personas: list[str], info_set: str) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        spans = retrieve_spans(row, k=5)
        extra = info_block(row, info_set)
        out["judge_original"] = judge(row, row["original_answer"])
        per_persona: dict[str, Any] = {}
        for name in personas:
            corrected = vllm_chat(PERSONAS[name], build_user(row, spans, extra), port, max_tokens=700, temperature=0.0)
            per_persona[name] = {
                "corrected": corrected,
                "judge_final": judge(row, corrected),
                "edited": materially_changed(row["original_answer"], corrected),
            }
        out["personas"] = per_persona
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows: list[dict[str, Any]], personas: list[str]) -> dict[str, Any]:
    by_persona: dict[str, Any] = {}
    n_wrong = sum(1 for r in rows if (r.get("judge_original") or {}).get("label") == 0)
    n_correct = sum(1 for r in rows if (r.get("judge_original") or {}).get("label") == 1)
    for name in personas:
        fix = brk = same1 = same0 = edits = err = 0
        edits_on_wrong = edits_on_correct = 0
        for r in rows:
            jo = (r.get("judge_original") or {}).get("label")
            pp = (r.get("personas") or {}).get(name)
            if pp is None or jo is None:
                err += 1
                continue
            jf = pp.get("judge_final", {}).get("label")
            edited = pp.get("edited", False)
            if edited:
                edits += 1
                if jo == 0:
                    edits_on_wrong += 1
                elif jo == 1:
                    edits_on_correct += 1
            if jf is None:
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
        by_persona[name] = {
            "fix": fix, "break": brk, "net": fix - brk,
            "same_correct": same1, "same_wrong": same0, "err": err,
            "edit_rate": round(edits / max(1, len(rows)), 3),
            "edit_rate_on_wrong": round(edits_on_wrong / max(1, n_wrong), 3),
            "edit_rate_on_correct": round(edits_on_correct / max(1, n_correct), 3),
        }
    return {
        "n_cases": len(rows),
        "n_wrong_by_judge": n_wrong,
        "n_correct_by_judge": n_correct,
        "info_set": rows and None,  # placeholder, set by caller
        "by_persona": by_persona,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-wrong", type=int, default=50)
    ap.add_argument("--n-correct", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--info-set", choices=["spans_only", "contradiction_quote", "error_type"], default="spans_only")
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS), choices=list(PERSONAS))
    args = ap.parse_args()
    served = served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}_{args.info_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="phase3_persona_sweep", served=served, args=vars(args))
    print(f"sample={len(sample)} personas={args.personas} info_set={args.info_set} out={out_dir}", flush=True)
    if sample:
        topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port, args.personas, args.info_set) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 5 == 0 or i == len(futs):
                print(f"processed {i}/{len(futs)}", flush=True)
    write_jsonl(out_dir / "judged_outputs.jsonl", rows)
    summary = summarize(rows, args.personas)
    summary["info_set"] = args.info_set
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
