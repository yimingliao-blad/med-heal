#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = PROJECT_ROOT / "refactor" / "pre_atom_pipeline" / "output" / "detection_bakeoff"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SYSTEM = "You are a careful clinical QA auditor. You must check whether an answer is supported by the discharge note."

PROMPTS: dict[str, str] = {
    "p1_three_axis_freeform": """Discharge note:\n{note}\n\nQuestion:\n{question}\n\nAnswer to audit:\n{answer}\n\nAudit the answer on three axes:\n1. CONTRADICTION: a factual claim conflicts with the note.\n2. OMISSION: critical information required by the question is absent.\n3. QUESTION_MISALIGNMENT: the answer addresses the wrong visit, date, time period, or clinical aspect.\n\nBe strict about critical errors, but do not flag minor missing details that do not change the answer.\n\nReturn exactly this template:\nVERDICT: CORRECT or INCORRECT\nERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE\nWRONG_CLAIM: one sentence, or NONE\nCORRECT_OR_MISSING_INFO: one sentence grounded in the note, or NONE\nQUESTION_FOCUS: the visit/date/aspect the question asks about\nWHY: one short explanation""",
    "p2_claim_quote_check": """Discharge note:\n{note}\n\nQuestion:\n{question}\n\nAnswer to audit:\n{answer}\n\nFirst list up to four important claims in the answer. For each claim, decide whether the discharge note supports it, contradicts it, or whether it is irrelevant to the question. Then decide if the final answer is acceptable.\n\nOnly mark INCORRECT for a critical contradiction, a missing required fact, or answering the wrong question focus.\n\nReturn exactly this template:\nCLAIMS_CHECKED:\n- claim: ... | status: supported/contradicted/irrelevant/missing_required | note evidence: ...\nVERDICT: CORRECT or INCORRECT\nERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE\nWRONG_CLAIM: one sentence, or NONE\nCORRECT_OR_MISSING_INFO: one sentence grounded in the note, or NONE\nWHY: one short explanation""",
    "p3_question_focus_first": """Discharge note:\n{note}\n\nQuestion:\n{question}\n\nAnswer to audit:\n{answer}\n\nStep 1: identify the exact question focus: patient event, visit, date, time period, treatment, medication change, complication, symptom, or outcome.\nStep 2: decide whether the answer addresses that exact focus.\nStep 3: decide whether the answer conflicts with the note or omits a critical required fact.\n\nDo not penalize harmless wording differences. Do penalize wrong admission/date/aspect.\n\nReturn exactly this template:\nQUESTION_FOCUS: one sentence\nANSWER_FOCUS: one sentence\nVERDICT: CORRECT or INCORRECT\nERROR_TYPE: QUESTION_MISALIGNMENT or CONTRADICTION or OMISSION or NONE\nWRONG_CLAIM: one sentence, or NONE\nCORRECT_OR_MISSING_INFO: one sentence grounded in the note, or NONE\nWHY: one short explanation""",
    "p4_fewshot_conservative": """Discharge note:\n{note}\n\nQuestion:\n{question}\n\nAnswer to audit:\n{answer}\n\nUse these examples only as auditing rules:\nExample A: If the answer says no complication but the note says urinary retention requiring catheterization, mark CONTRADICTION.\nExample B: If the question asks for medication changes and the answer lists only one of several clinically important changes, mark OMISSION.\nExample C: If the question asks about the second admission but the answer describes the first admission, mark QUESTION_MISALIGNMENT.\nExample D: If the answer is shorter than the ground-truth-style response but captures the clinically necessary fact, mark CORRECT.\n\nNow audit the answer. Avoid false positives: only mark INCORRECT when the error changes the answer to the question.\n\nReturn exactly this template:\nVERDICT: CORRECT or INCORRECT\nERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE\nWRONG_CLAIM: one sentence, or NONE\nCORRECT_OR_MISSING_INFO: one sentence grounded in the note, or NONE\nWHY: one short explanation""",
    "p5_retrieval_payload": """Discharge note:\n{note}\n\nQuestion:\n{question}\n\nAnswer to audit:\n{answer}\n\nYour job is not only to decide whether the answer is wrong. Your job is to create a correction payload that a downstream retrieval step can use.\n\nCheck in this order:\n1. QUESTION_FOCUS: What exact visit/date/aspect/fact does the question ask for?\n2. ANSWER_FOCUS: What does the answer actually focus on?\n3. WRONG_OR_MISSING_TARGET: If wrong, identify the smallest wrong claim or missing required fact.\n4. EVIDENCE_NEEDED: What note evidence would prove the correction?\n\nOnly mark INCORRECT if the issue changes the answer to the question. Do not flag minor extra details.\n\nReturn exactly this template:\nVERDICT: CORRECT or INCORRECT\nERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE\nQUESTION_FOCUS: one sentence\nANSWER_FOCUS: one sentence\nWRONG_CLAIM: the smallest wrong claim, or NONE\nCORRECT_OR_MISSING_INFO: the fact that should replace/add/refocus the answer, or NONE\nEVIDENCE_NEEDED: what kind of note span should be retrieved\nRETRIEVAL_QUERY_1: short query using the question focus\nRETRIEVAL_QUERY_2: short query using the wrong/missing target\nRETRIEVAL_QUERY_3: short query using key clinical entities\nCORRECTION_HINT: one sentence telling the downstream corrector what to change\nWHY: one short explanation""",
    "p6_claims_to_queries": """Discharge note:\n{note}\n\nQuestion:\n{question}\n\nAnswer to audit:\n{answer}\n\nCreate a compact claim audit for downstream correction.\n\nA. Extract up to 5 clinically important claims from the answer.\nB. For each claim, mark supported / contradicted / not-in-note / wrong-focus.\nC. Check whether the answer omits a critical fact required by the question.\nD. If incorrect, produce retrieval queries that would find the exact correcting evidence.\n\nBe conservative: if the answer is clinically sufficient, mark CORRECT even if it is not exhaustive.\n\nReturn exactly this template:\nCLAIM_AUDIT:\n- claim: ... | status: supported/contradicted/not-in-note/wrong-focus | evidence target: ...\nVERDICT: CORRECT or INCORRECT\nERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE\nQUESTION_FOCUS: one sentence\nWRONG_CLAIM: the claim to repair, or NONE\nCORRECT_OR_MISSING_INFO: the target fact to retrieve, or NONE\nEVIDENCE_NEEDED: exact evidence needed for correction, or NONE\nRETRIEVAL_QUERY_1: short query\nRETRIEVAL_QUERY_2: short query\nRETRIEVAL_QUERY_3: short query\nCORRECTION_HINT: one sentence\nWHY: one short explanation""",
    "p7_error_gate_payload": """Discharge note:\n{note}\n\nQuestion:\n{question}\n\nAnswer to audit:\n{answer}\n\nYou are a high-precision gate before an automatic correction system. A false alarm may break a correct answer, so only mark INCORRECT when there is a correction-worthy error.\n\nCorrection-worthy errors are:\n- CONTRADICTION: the answer makes a claim that the note contradicts.\n- OMISSION: the answer misses a critical fact required to answer the question.\n- QUESTION_MISALIGNMENT: the answer addresses the wrong visit/date/aspect.\n\nIf incorrect, provide the exact payload needed for retrieval. If you cannot name the wrong claim or missing/refocus target, mark CORRECT.\n\nReturn exactly this template:\nVERDICT: CORRECT or INCORRECT\nERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE\nQUESTION_FOCUS: one sentence\nWRONG_CLAIM: exact wrong/missing/misaligned part of the answer, or NONE\nCORRECT_OR_MISSING_INFO: exact correction target grounded in the note, or NONE\nEVIDENCE_NEEDED: what evidence span should be retrieved, or NONE\nRETRIEVAL_QUERY_1: short query, or NONE\nRETRIEVAL_QUERY_2: short query, or NONE\nRETRIEVAL_QUERY_3: short query, or NONE\nCORRECTION_HINT: one sentence, or NONE\nWHY: one short explanation""",
}

PARSE_SYSTEM = "You extract structured fields from a clinical self-audit. Return JSON only."
PARSE_USER = """Extract the auditor's decision from this text. Use only what the text says; do not re-judge the clinical case.\n\nTEXT:\n{raw}\n\nReturn JSON with exactly these keys:\n{{"verdict":"CORRECT|INCORRECT|UNCLEAR", "error_type":"CONTRADICTION|OMISSION|QUESTION_MISALIGNMENT|NONE|UNCLEAR", "wrong_claim":"string", "correct_or_missing_info":"string", "question_focus":"string", "evidence_needed":"string", "retrieval_queries":["string"], "correction_hint":"string", "why":"string"}}"""


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


def served_model_id(port: int) -> str:
    r = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I).strip()
    if "</think>" in text.lower():
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL | re.I).strip()
    return text


def vllm_chat(system: str, user: str, port: int, max_tokens: int, temperature: float) -> str:
    model = served_model_id(port)
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": temperature}
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


def load_rows(n_wrong: int, n_correct: int, seed: int) -> list[dict[str, Any]]:
    notes = load_notes_lookup()
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
                "answer": str(r["model_answer"]),
                "ground_truth": str(r["ground_truth"]),
                "label": int(r["binary_correct"]),
                "note": notes[str(int(r["patient_id"]))],
            })
    wrong = [r for r in rows if r["label"] == 0]
    correct = [r for r in rows if r["label"] == 1]
    rng = random.Random(seed)
    rng.shuffle(wrong); rng.shuffle(correct)
    if n_wrong < 0:
        n_wrong = len(wrong)
    if n_correct < 0:
        n_correct = len(correct)
    sample = wrong[:min(n_wrong, len(wrong))] + correct[:min(n_correct, len(correct))]
    rng.shuffle(sample)
    return sample


def generate_one(row: dict[str, Any], prompt_id: str, port: int, temperature: float) -> dict[str, Any]:
    user = PROMPTS[prompt_id].format(note=row["note"][:18000], question=row["question"], answer=row["answer"][:2000])
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "answer", "ground_truth", "label"]}
    out["prompt_id"] = prompt_id
    out["temperature"] = temperature
    try:
        out["raw"] = vllm_chat(SYSTEM, user, port=port, max_tokens=900, temperature=temperature)
        out["error"] = None
    except Exception as e:
        out["raw"] = ""
        out["error"] = str(e)
    return out


def parse_regex(raw: str) -> dict[str, Any]:
    text = raw or ""
    def field(name: str) -> str:
        m = re.search(rf"^\s*\*?\*?{re.escape(name)}\*?\*?\s*:\s*(.+)$", text, re.I | re.M)
        return m.group(1).strip() if m else ""
    verdict_s = field("VERDICT").upper()
    if "INCORRECT" in verdict_s:
        verdict = "INCORRECT"
    elif "CORRECT" in verdict_s:
        verdict = "CORRECT"
    else:
        low = text.lower()
        verdict = "INCORRECT" if re.search(r"\b(incorrect|not correct|wrong|unsupported|contradict)", low) else ("CORRECT" if re.search(r"\b(correct|supported)\b", low) else "UNCLEAR")
    et_s = field("ERROR_TYPE").upper()
    error_type = "UNCLEAR"
    for t in ["QUESTION_MISALIGNMENT", "CONTRADICTION", "OMISSION", "NONE"]:
        if t in et_s:
            error_type = t
            break
    if verdict == "CORRECT" and error_type == "UNCLEAR":
        error_type = "NONE"
    return {
        "verdict": verdict,
        "error_type": error_type,
        "wrong_claim": field("WRONG_CLAIM"),
        "correct_or_missing_info": field("CORRECT_OR_MISSING_INFO"),
        "question_focus": field("QUESTION_FOCUS"),
        "evidence_needed": field("EVIDENCE_NEEDED"),
        "retrieval_queries": [q for q in [field("RETRIEVAL_QUERY_1"), field("RETRIEVAL_QUERY_2"), field("RETRIEVAL_QUERY_3")] if q and q.upper() != "NONE"],
        "correction_hint": field("CORRECTION_HINT"),
        "why": field("WHY"),
    }


def parse_gpt_mini(raw: str) -> dict[str, Any]:
    for attempt in range(5):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": PARSE_SYSTEM}, {"role": "user", "content": PARSE_USER.format(raw=(raw or "")[:5000])}],
                max_tokens=220,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = (r.choices[0].message.content or "").strip()
            obj = json.loads(text)
            return {
                "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
                "error_type": str(obj.get("error_type", "UNCLEAR")).upper(),
                "wrong_claim": str(obj.get("wrong_claim", "")),
                "correct_or_missing_info": str(obj.get("correct_or_missing_info", "")),
                "question_focus": str(obj.get("question_focus", "")),
                "evidence_needed": str(obj.get("evidence_needed", "")),
                "retrieval_queries": obj.get("retrieval_queries", []) if isinstance(obj.get("retrieval_queries", []), list) else [],
                "correction_hint": str(obj.get("correction_hint", "")),
                "why": str(obj.get("why", "")),
                "raw_json": text,
            }
        except Exception as e:
            if attempt == 4:
                return {"verdict": "UNCLEAR", "error_type": "UNCLEAR", "wrong_claim": "", "correct_or_missing_info": "", "question_focus": "", "evidence_needed": "", "retrieval_queries": [], "correction_hint": "", "why": "", "error": str(e)}
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def score(rows: list[dict[str, Any]], parser_key: str) -> dict[str, Any]:
    by_prompt = {}
    for pid in sorted({r["prompt_id"] for r in rows}):
        sub = [r for r in rows if r["prompt_id"] == pid]
        tp = fp = tn = fn = unclear = 0
        type_counts = Counter()
        usable_feedback = 0
        retrieval_ready = 0
        for r in sub:
            parsed = r[parser_key]
            pred_bad = parsed.get("verdict") == "INCORRECT"
            if parsed.get("verdict") == "UNCLEAR":
                unclear += 1
            gold_bad = r["label"] == 0
            if pred_bad and gold_bad: tp += 1
            elif pred_bad and not gold_bad: fp += 1
            elif (not pred_bad) and gold_bad: fn += 1
            else: tn += 1
            if pred_bad:
                type_counts[parsed.get("error_type", "UNCLEAR")] += 1
                if parsed.get("wrong_claim") and parsed.get("correct_or_missing_info"):
                    usable_feedback += 1
                if (parsed.get("correct_or_missing_info") and (parsed.get("evidence_needed") or parsed.get("retrieval_queries"))):
                    retrieval_ready += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_prompt[pid] = {"n": len(sub), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "unclear": unclear, "precision": precision, "recall": recall, "f1": f1, "type_counts": dict(type_counts), "usable_feedback_on_detected": usable_feedback, "retrieval_ready_on_detected": retrieval_ready}
    return by_prompt


def parser_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for pid in sorted({r["prompt_id"] for r in rows}):
        sub = [r for r in rows if r["prompt_id"] == pid]
        same_verdict = sum(1 for r in sub if r["regex_parse"].get("verdict") == r["gpt4o_mini_parse"].get("verdict"))
        same_type = sum(1 for r in sub if r["regex_parse"].get("error_type") == r["gpt4o_mini_parse"].get("error_type"))
        out[pid] = {"n": len(sub), "same_verdict": same_verdict, "same_verdict_rate": same_verdict / len(sub), "same_error_type": same_type, "same_error_type_rate": same_type / len(sub)}
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=50)
    ap.add_argument("--n-correct", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    ap.add_argument("--skip-gpt-parse", action="store_true")
    args = ap.parse_args()

    served = served_model_id(args.port)
    if "qwen2.5" not in served.lower() and "qwen2" not in served.lower():
        raise RuntimeError(f"Expected Qwen2.5 on port {args.port}, found {served}")
    sample = load_rows(args.n_wrong, args.n_correct, args.seed)
    prompt_tag = "-".join(args.prompts)
    prompt_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_tag)[:120]
    run_id = f"qwen25_detection_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}_t{str(args.temperature).replace('.', 'p')}_{prompt_tag}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"served_model={served}")
    print(f"sample={len(sample)} prompts={args.prompts} concurrency={args.concurrency} temperature={args.temperature}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(generate_one, row, pid, args.port, args.temperature) for row in sample for pid in args.prompts]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 20 == 0 or i == len(futs):
                print(f"generated {i}/{len(futs)}", flush=True)
    write_jsonl(out_dir / "raw_outputs.jsonl", rows)

    for r in rows:
        r["regex_parse"] = parse_regex(r.get("raw", ""))
    if not args.skip_gpt_parse:
        for i, r in enumerate(rows, 1):
            r["gpt4o_mini_parse"] = parse_gpt_mini(r.get("raw", ""))
            if i % 20 == 0 or i == len(rows):
                print(f"gpt4o-mini parsed {i}/{len(rows)}", flush=True)
    else:
        for r in rows:
            r["gpt4o_mini_parse"] = {"verdict": "UNCLEAR", "error_type": "UNCLEAR"}

    write_jsonl(out_dir / "parsed_outputs.jsonl", rows)
    summary = {
        "task": "qwen25_self_detection_prompt_bakeoff",
        "served_model": served,
        "settings": vars(args),
        "sample_counts": dict(Counter(r["label"] for r in sample)),
        "regex_scores": score(rows, "regex_parse"),
        "gpt4o_mini_scores": score(rows, "gpt4o_mini_parse"),
        "parser_agreement": parser_agreement(rows),
        "outputs": {"raw": str(out_dir / "raw_outputs.jsonl"), "parsed": str(out_dir / "parsed_outputs.jsonl")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = ["# Qwen2.5 Self-Detection Prompt Bakeoff", "", f"Served model: `{served}`", f"Sample: {summary['sample_counts']}", f"Temperature: `{args.temperature}`", "", "## GPT-4o-Mini Parser Scores", "", "| Prompt | TP | FP | TN | FN | Precision | Recall | F1 | Usable feedback | Retrieval-ready |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pid, s in summary["gpt4o_mini_scores"].items():
        lines.append(f"| `{pid}` | {s['tp']} | {s['fp']} | {s['tn']} | {s['fn']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['usable_feedback_on_detected']} | {s.get('retrieval_ready_on_detected', 0)} |")
    lines += ["", "## Regex Parser Scores", "", "| Prompt | TP | FP | TN | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for pid, s in summary["regex_scores"].items():
        lines.append(f"| `{pid}` | {s['tp']} | {s['fp']} | {s['tn']} | {s['fn']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |")
    lines += ["", "## Parser Agreement", "", "| Prompt | Same verdict | Same error type |", "|---|---:|---:|"]
    for pid, s in summary["parser_agreement"].items():
        lines.append(f"| `{pid}` | {s['same_verdict']}/{s['n']} ({s['same_verdict_rate']:.1%}) | {s['same_error_type']}/{s['n']} ({s['same_error_type_rate']:.1%}) |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
