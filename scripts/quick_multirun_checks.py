#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = PROJECT_ROOT / "refactor" / "pre_atom_pipeline" / "output" / "quick_tests"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

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

def openai_client() -> OpenAI:
    return OpenAI(api_key=load_api_key())

def gpt_judge_one(row: dict[str, Any], temperature: float, model: str = "gpt-4o") -> dict[str, Any]:
    client = openai_client()
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_judge_user(row["note"], row["question"], row["ground_truth"], row["model_answer"])},
    ]
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=10,
                temperature=temperature,
            )
            raw = (r.choices[0].message.content or "").strip()
            out = dict(row)
            out.update({"judge_model": model, "temperature": temperature, "raw": raw, "label": parse_binary(raw)})
            return out
        except Exception as e:
            if attempt == 4:
                out = dict(row)
                out.update({"judge_model": model, "temperature": temperature, "raw": "", "label": None, "error": str(e)})
                return out
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")

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

def gt_from_step2(row: pd.Series) -> str:
    letter = str(row.get("answer", "")).strip()
    choice = row.get(f"choice_{letter}", "")
    return f"{letter}. {choice}" if pd.notna(choice) and str(choice).strip() else letter

def load_human_gold() -> pd.DataFrame:
    path = PROJECT_ROOT / "datasets" / "external" / "all_users_openended_BioMistral-7B_1775740232208.csv"
    df = pd.read_csv(path)
    df["human_binary"] = (df["Answer Quality"] == 5).astype(int)
    sara = df[df["User Name"] == "Sara Saif"].drop_duplicates("Patient ID").set_index("Patient ID")["human_binary"]
    jose = df[df["User Name"] == "Jose E. Lizarraga Mazab"].drop_duplicates("Patient ID").set_index("Patient ID")["human_binary"]
    common = sara.index.intersection(jose.index)
    gold = pd.DataFrame({"sara": sara.loc[common], "jose": jose.loc[common]})
    gold = gold[gold["sara"] == gold["jose"]].copy()
    gold["gold_label"] = gold["sara"]
    gold.index.name = "patient_id"
    return gold.reset_index()

def metric_summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    n = len(y_true)
    agree = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return {
        "n": n,
        "agreement": agree / n if n else None,
        "agree_count": agree,
        "kappa": cohen_kappa_score(y_true, y_pred) if n and len(set(y_true)) > 1 and len(set(y_pred)) > 1 else None,
        "truth_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
    }

def cmd_judge_biomistral(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    notes = load_notes_lookup()
    gold = load_human_gold()
    step2 = pd.read_csv(PROJECT_ROOT / "output" / "ours_biomistral-7b_EHRNoteQA_processed.csv")
    step2["patient_id"] = step2["patient_id"].astype(int)
    step2 = step2.drop_duplicates("patient_id").set_index("patient_id")
    candidates = []
    for _, g in gold.iterrows():
        pid = int(g["patient_id"])
        if pid not in step2.index or str(pid) not in notes:
            continue
        r = step2.loc[pid]
        candidates.append({
            "patient_id": pid,
            "question": str(r["question"]),
            "ground_truth": gt_from_step2(r),
            "model_answer": str(r["openended_answer"]),
            "note": notes[str(pid)],
            "human_gold": int(g["gold_label"]),
            "source_answer": "output/ours_biomistral-7b_EHRNoteQA_processed.csv",
        })
    sample = rng.sample(candidates, min(args.n, len(candidates)))
    out_dir = OUT_ROOT / "judge_biomistral"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for temp in args.temperatures:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(gpt_judge_one, row, temp, args.gpt_model) for row in sample]
            for fut in as_completed(futs):
                all_results.append(fut.result())
    jsonl = out_dir / f"judge_biomistral_n{len(sample)}_seed{args.seed}.jsonl"
    with jsonl.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "task": "judge_biomistral_50case_confirmation",
        "n_sampled": len(sample),
        "seed": args.seed,
        "concurrency": args.concurrency,
        "prompt": {
            "system": JUDGE_SYSTEM,
            "user_template": "DISCHARGE SUMMARY / QUESTION / CORRECT ANSWER (Ground Truth) / MODEL'S ANSWER / single digit 1 or 0",
            "max_tokens": 10,
        },
        "temperatures": {},
        "jsonl": str(jsonl.relative_to(PROJECT_ROOT)),
    }
    for temp in args.temperatures:
        rows = [r for r in all_results if float(r["temperature"]) == float(temp) and r.get("label") is not None]
        summary["temperatures"][str(temp)] = metric_summary([int(r["human_gold"]) for r in rows], [int(r["label"]) for r in rows])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

# Step 8 prompt replicas
BASE_SYSTEM = "You are a medical expert answering questions about discharge summaries."
BIOMISTRAL_SYSTEM = "You are a helpful, respectful and honest assistant."
USER_TASK = "Discharge Summary:\n{note}\n\nQuestion: {question}\n\nAnswer:"

def build_llama2(system: str, user: str) -> str:
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

def served_model_id(port: int) -> str:
    return requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]

def vllm_completion(prompt: str, port: int, max_tokens: int, temperature: float) -> str:
    model = served_model_id(port)
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()["choices"][0]["text"].strip()

def cmd_step8_biomistral(args: argparse.Namespace) -> None:
    served = served_model_id(args.port)
    if "biomistral" not in served.lower():
        raise RuntimeError(f"Expected BioMistral on port {args.port}, found {served}")
    rng = random.Random(args.seed)
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    existing_parts = []
    for fold in range(5):
        p = PROJECT_ROOT / "output" / "step8" / "biomistral-7b" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        df = pd.read_csv(p)
        df["fold"] = fold
        existing_parts.append(df)
    existing = pd.concat(existing_parts, ignore_index=True)
    sample = existing.sample(n=min(args.n, len(existing)), random_state=args.seed).to_dict("records")
    note_lookup = load_notes_lookup()
    out_dir = OUT_ROOT / "step8_biomistral"
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(row: dict[str, Any]) -> dict[str, Any]:
        note = note_lookup[str(int(row["patient_id"]))]
        user = USER_TASK.format(note=note, question=row["question"])
        prompt = build_llama2(BIOMISTRAL_SYSTEM, user)
        try:
            ans = vllm_completion(prompt, args.port, 512, 0.1)
        except Exception as e:
            ans = ""
            err = str(e)
        else:
            err = ""
        return {
            "fold": int(row["fold"]),
            "idx": int(row["idx"]),
            "patient_id": int(row["patient_id"]),
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "existing_model_answer": row["model_answer"],
            "fresh_model_answer": ans,
            "existing_binary_correct": int(row["binary_correct"]),
            "prompt_length": len(prompt),
            "error": err,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(one, sample))
    judged = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = []
        for r in rows:
            if not r["fresh_model_answer"]:
                continue
            jr = {"note": note_lookup[str(r["patient_id"])], "question": r["question"], "ground_truth": r["ground_truth"], "model_answer": r["fresh_model_answer"], "patient_id": r["patient_id"], "fold": r["fold"], "idx": r["idx"], "existing_binary_correct": r["existing_binary_correct"]}
            futs.append(ex.submit(gpt_judge_one, jr, 0.0, args.gpt_model))
        for fut in as_completed(futs):
            judged.append(fut.result())
    label_by_key = {(r["fold"], r["idx"]): r.get("label") for r in judged}
    for r in rows:
        r["fresh_binary_correct_T0"] = label_by_key.get((r["fold"], r["idx"]))
    jsonl = out_dir / f"step8_biomistral_n{len(rows)}_seed{args.seed}.jsonl"
    with jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    valid = [r for r in rows if r.get("fresh_binary_correct_T0") is not None]
    summary = {
        "task": "step8_biomistral_50case_generation_confirmation",
        "n_sampled": len(rows),
        "seed": args.seed,
        "served_model": served,
        "concurrency": args.concurrency,
        "generation_settings": {"temperature": 0.1, "max_tokens": 512, "system_prompt": BIOMISTRAL_SYSTEM, "prompt_format": "Llama2 [INST] with discharge summary/question/Answer:"},
        "judge_settings": {"model": args.gpt_model, "temperature": 0.0, "max_tokens": 10, "system_prompt": JUDGE_SYSTEM},
        "agreement_with_existing_step8_binary": metric_summary([int(r["existing_binary_correct"]) for r in valid], [int(r["fresh_binary_correct_T0"]) for r in valid]),
        "fresh_accuracy": sum(int(r["fresh_binary_correct_T0"]) for r in valid) / len(valid) if valid else None,
        "existing_accuracy_on_sample": sum(int(r["existing_binary_correct"]) for r in valid) / len(valid) if valid else None,
        "jsonl": str(jsonl.relative_to(PROJECT_ROOT)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

# F1 / regen / verdict replicas
SYS_DET = "You are a strict medical expert checking clinical answers."
DET_F1_CONTRA = """Discharge summary:\n{note}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nCHECK FOR CONTRADICTION: Does the answer state any fact that DIRECTLY CONFLICTS with the discharge notes?\n\nFor each key claim (medication names, dosages, procedures, diagnoses, lab values, dates):\n1. Find the matching information in the notes\n2. Does the answer say something DIFFERENT?\n\nOnly flag OPPOSING information. If you find a contradiction, explain what the answer says vs what the notes say."""
DET_F1_QMIS = """Discharge summary:\n{note}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nCHECK FOR QUESTION MISALIGNMENT: Does the answer address the WRONG thing?\n\nParse the question: which visit, what aspect, what time period?\nDoes the answer match? If misaligned, explain what the question asks vs what the answer discusses."""
DET_F1_OMIS = """Discharge summary:\n{note}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nCHECK FOR CRITICAL OMISSION: Is there information in the notes ESSENTIAL to answer the question but COMPLETELY ABSENT?\n\nOnly flag omissions that would CHANGE the conclusion. Do NOT flag minor details."""
EXTRACT_F1 = """/nothink\nRead this self-critique. Extract as JSON. Only INCORRECT if critical errors found.\n\nTEXT:\n{raw}\n\n{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the error as one sentence", "correct_statement": "what notes say"}}"""
VALIDITY_SYSTEM = "You are a medical expert auditing whether a self-critique correctly characterised the error in an AI model's answer to a clinical question."
VALIDITY_USER = """DISCHARGE SUMMARY:\n{note}\n\nQUESTION:\n{question}\n\nMODEL'S ANSWER (the answer being critiqued):\n{model_answer}\n\nSELF-CRITIQUE OUTPUT:\n- claimed error type:    {error_type}\n- error_statement:       {error_statement}\n- correct_statement:     {correct_statement}\n\nEvaluate ONLY the `correct_statement` field on two criteria:\n\nNOTE_SUPPORTED: yes|no    (does the discharge summary actually say what `correct_statement` says about this fact?)\nADDRESSES_ERROR: yes|no   (does `correct_statement` address the same fact as `error_statement` -- i.e. is it actually fixing what was wrong, not pointing at something unrelated?)\n\nReply with exactly two lines in that format. No other text."""
REGEN_SYS = "You are a medical expert."
REGEN_USER = """Discharge note:\n{note}\n\nQuestion: {question}\n\nAnswer the question using only information from the discharge note. Be specific\nand complete. If the question asks about multiple visits, conditions, or events,\ncover all of them."""
COUNT_COMPARE_SYS = "You are a strict medical expert."
COUNT_COMPARE = """Discharge summary:\n{note}\n\nQuestion: {question}\n\nANSWER A:\n{answer_a}\n\nANSWER B:\n{answer_b}\n\nCount how many factual claims in each answer contradict the discharge notes.\nDifferent wording for the same fact is NOT a contradiction.\n\nA_ERRORS: <number>\nB_ERRORS: <number>"""
QWEN_PARSE_SYS = "You interpret a medical expert's analysis and extract their decision about which of two answers is more reliable."
QWEN_PARSE = """A medical expert compared two clinical answers (A and B) against discharge notes. Here is their analysis:\n\n---\n{analysis}\n---\n\nBased on this analysis, which answer should we keep? The expert counted contradictions in each -- pick the answer with FEWER contradictions. If both have the same count, pick A.\n\nDECISION: A or B\nREASON: <one sentence>\n/no_think"""

def build_chatml(system: str, user: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I).strip()
    if "</think>" in text.lower():
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL | re.I).strip()
    return text

def vllm_prompt(prompt: str, port: int, max_tokens: int, temperature: float) -> str:
    model = served_model_id(port)
    r = requests.post(f"http://localhost:{port}/v1/completions", json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return strip_think(r.json()["choices"][0]["text"].strip())

def vllm_chat(system: str, user: str, port: int, max_tokens: int, temperature: float, chat_template_kwargs: dict | None = None) -> str:
    model = served_model_id(port)
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": temperature}
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
    body = r.json()
    if "choices" not in body:
        payload["messages"] = [{"role": "user", "content": f"{system}\n\n{user}"}]
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
        body = r.json()
    if "choices" not in body:
        raise RuntimeError(str(body))
    return strip_think((body["choices"][0]["message"]["content"] or "").strip())

def q32_extract(raw: str, schema_template: str) -> dict[str, Any] | None:
    r = requests.post(QWEN32B_URL, json={"model": "Qwen/Qwen3-32B-MLX-bf16", "messages": [{"role": "system", "content": "Extract info. JSON only."}, {"role": "user", "content": schema_template.format(raw=raw[:2500])}], "max_tokens": 400, "temperature": 0.0}, timeout=120)
    text = strip_think(r.json()["choices"][0]["message"]["content"].strip())
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else None

def normalise_det(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not obj:
        return {"verdict": "UNCLEAR", "error_type": "NONE", "error_statement": "", "correct_statement": ""}
    return {"verdict": str(obj.get("verdict", "UNCLEAR")).upper(), "error_type": str(obj.get("error_type", "NONE")).upper(), "error_statement": str(obj.get("error_statement", ""))[:300], "correct_statement": str(obj.get("correct_statement", ""))[:300]}

def aggregate_f1(parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bad = {k: v for k, v in parts.items() if v.get("verdict") == "INCORRECT"}
    if not bad:
        return {"verdict": "CORRECT", "error_type": "NONE", "error_statement": "", "correct_statement": "", "detected_sub_keys": []}
    for key, typ in [("qmis", "QUESTION_MISALIGNMENT"), ("contra", "CONTRADICTION"), ("omis", "OMISSION")]:
        if key in bad:
            v = bad[key]
            return {"verdict": "INCORRECT", "error_type": typ, "error_statement": v["error_statement"], "correct_statement": v["correct_statement"], "detected_sub_keys": list(bad)}
    return {"verdict": "CORRECT", "error_type": "NONE", "error_statement": "", "correct_statement": "", "detected_sub_keys": []}

def run_f1(note: str, question: str, answer: str, port: int) -> dict[str, Any]:
    raw = {}
    parts = {}
    for key, tmpl in [("contra", DET_F1_CONTRA), ("qmis", DET_F1_QMIS), ("omis", DET_F1_OMIS)]:
        text = vllm_prompt(build_chatml(SYS_DET, tmpl.format(note=note, question=question, answer=answer[:800])), port, 1024, 0.7)
        raw[key] = text
        parts[key] = normalise_det(q32_extract(text, EXTRACT_F1))
    return {"final": aggregate_f1(parts), "sub_results": parts, "raw": raw, "k": 1, "temperature": 0.7}

def gpt_validity(note: str, question: str, answer: str, det: dict[str, Any]) -> dict[str, Any]:
    if det.get("verdict") != "INCORRECT":
        return {"note_supported": None, "addresses_error": None, "raw": "not_applicable"}
    client = openai_client()
    user = VALIDITY_USER.format(note=note, question=question, model_answer=answer[:800], error_type=det.get("error_type", "NONE"), error_statement=det.get("error_statement", ""), correct_statement=det.get("correct_statement", ""))
    r = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": VALIDITY_SYSTEM}, {"role": "user", "content": user}], max_tokens=60, temperature=0.0)
    text = (r.choices[0].message.content or "").strip()
    out = {"note_supported": None, "addresses_error": None, "raw": text}
    for line in text.splitlines():
        up = line.upper()
        if up.startswith("NOTE_SUPPORTED:"):
            out["note_supported"] = "yes" if line.split(":", 1)[1].strip().lower().startswith("yes") else "no"
        if up.startswith("ADDRESSES_ERROR:"):
            out["addresses_error"] = "yes" if line.split(":", 1)[1].strip().lower().startswith("yes") else "no"
    return out

def q32_parse_decision(analysis: str) -> tuple[str, str]:
    r = requests.post(QWEN32B_URL, json={"model": "Qwen/Qwen3-32B-MLX-bf16", "messages": [{"role": "system", "content": QWEN_PARSE_SYS}, {"role": "user", "content": QWEN_PARSE.format(analysis=analysis[:1500])}], "max_tokens": 128, "temperature": 0.0}, timeout=120)
    text = strip_think(r.json()["choices"][0]["message"]["content"].strip())
    m = re.search(r"DECISION:\s*([AB])", text.upper())
    pick = m.group(1) if m else "A"
    rm = re.search(r"REASON:\s*(.+)$", text, re.I | re.M)
    return pick, (rm.group(1).strip()[:200] if rm else text[:200])

def sample_step8_items(model_dir: str, n: int, seed: int) -> list[dict[str, Any]]:
    parts = []
    for fold in range(5):
        p = PROJECT_ROOT / "output" / "step8" / model_dir / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        df = pd.read_csv(p)
        df["fold"] = fold
        parts.append(df)
    base = pd.concat(parts, ignore_index=True)
    wrong = base[base["binary_correct"] == 0]
    correct = base[base["binary_correct"] == 1]
    n_w = min(len(wrong), n // 2)
    n_c = min(len(correct), n - n_w)
    return pd.concat([wrong.sample(n=n_w, random_state=seed), correct.sample(n=n_c, random_state=seed + 1)], ignore_index=True).sample(frac=1, random_state=seed + 2).to_dict("records")

def run_f1_regen_verdict_one(row: dict[str, Any], note_lookup: dict[str, str], port: int, chat_kwargs: dict | None) -> dict[str, Any]:
    note = note_lookup[str(int(row["patient_id"]))]
    question = str(row["question"])
    original = str(row["model_answer"])
    gt = str(row["ground_truth"])
    out = {"fold": int(row["fold"]), "idx": int(row["idx"]), "patient_id": int(row["patient_id"]), "question": question, "ground_truth": gt, "original_answer": original, "orig_label": int(row["binary_correct"])}
    try:
        det = run_f1(note, question, original, port)
        val = gpt_validity(note, question, original, det["final"])
    except Exception as e:
        det = {"error": str(e)}
        val = {"error": str(e)}
    out["detection"] = det
    out["validity_gate"] = val
    try:
        regen = vllm_chat(REGEN_SYS, REGEN_USER.format(note=note[:18000], question=question), port, 1024, 0.0, chat_kwargs)
        rng = random.Random(42 + (out["fold"] << 16) + out["idx"])
        orig_in_a = rng.random() > 0.5
        ans_a = original if orig_in_a else regen
        ans_b = regen if orig_in_a else original
        cc = vllm_chat(COUNT_COMPARE_SYS, COUNT_COMPARE.format(note=note[:18000], question=question, answer_a=ans_a[:1500], answer_b=ans_b[:1500]), port, 1024, 0.0, chat_kwargs)
        pick, reason = q32_parse_decision(cc)
        accept = (pick == "B") if orig_in_a else (pick == "A")
        out["regen"] = {"method": "regen_zeroshot", "system_prompt": REGEN_SYS, "temperature": 0.0, "max_tokens": 1024, "answer": regen}
        out["verdict"] = {"variant": "count_compare_qwen3parse_v1f", "system_prompt": COUNT_COMPARE_SYS, "temperature": 0.0, "max_tokens": 1024, "orig_in_slot_A": orig_in_a, "raw": cc[:1500], "qwen3_pick": pick, "qwen3_reason": reason, "accept_correction": accept}
        if accept:
            jr = {"note": note, "question": question, "ground_truth": gt, "model_answer": regen, "patient_id": out["patient_id"]}
            j = gpt_judge_one(jr, 0.0, "gpt-4o")
            final_label = j.get("label") if j.get("label") is not None else out["orig_label"]
            delta = 1 if final_label == 1 and out["orig_label"] == 0 else (-1 if final_label == 0 and out["orig_label"] == 1 else 0)
            out["judge_corrected"] = {"label": final_label, "raw": j.get("raw")}
            out["outcome"] = {"action": "corrected", "delta": delta, "final_eval": final_label}
        else:
            out["judge_corrected"] = None
            out["outcome"] = {"action": "kept_original", "delta": 0, "final_eval": out["orig_label"]}
    except Exception as e:
        out["regen_error"] = str(e)
        out["outcome"] = {"action": "error", "delta": 0, "final_eval": out["orig_label"]}
    return out

def cmd_step9_component(args: argparse.Namespace) -> None:
    model_map = {"qwen3": "qwen3-8b", "qwen2.5": "qwen2.5-7b-instruct"}
    expected = {"qwen3": "qwen3", "qwen2.5": "qwen2.5"}[args.model]
    served = served_model_id(args.port)
    if expected not in served.lower():
        raise RuntimeError(f"Expected {args.model} on port {args.port}, found {served}")
    chat_kwargs = {"enable_thinking": False} if args.model == "qwen3" else None
    sample = sample_step8_items(model_map[args.model], args.n, args.seed)
    notes = load_notes_lookup()
    out_dir = OUT_ROOT / "step9_components" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_f1_regen_verdict_one, r, notes, args.port, chat_kwargs) for r in sample]
        for fut in as_completed(futs):
            rows.append(fut.result())
            print(f"done {len(rows)}/{len(sample)}", flush=True)
    jsonl = out_dir / f"f1_regen_verdict_n{len(rows)}_seed{args.seed}.jsonl"
    with jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    actions = Counter((r.get("outcome") or {}).get("action") for r in rows)
    fixes = sum(1 for r in rows if (r.get("outcome") or {}).get("delta") == 1)
    breaks = sum(1 for r in rows if (r.get("outcome") or {}).get("delta") == -1)
    det_incorrect = sum(1 for r in rows if (((r.get("detection") or {}).get("final") or {}).get("verdict") == "INCORRECT"))
    val_yes = sum(1 for r in rows if (r.get("validity_gate") or {}).get("note_supported") == "yes" and (r.get("validity_gate") or {}).get("addresses_error") == "yes")
    summary = {
        "task": "step9_f1_regen_verdict_component_confirmation",
        "model_alias": args.model,
        "served_model": served,
        "n_sampled": len(rows),
        "seed": args.seed,
        "concurrency": args.concurrency,
        "settings": {
            "detection": {"variant": "F1", "temperature": 0.7, "k": 1, "extractor": "Qwen3-32B", "validity_gate": "GPT-4o temperature 0"},
            "regen": {"variant": "regen_zeroshot", "system_prompt": REGEN_SYS, "temperature": 0.0, "max_tokens": 1024},
            "verdict": {"variant": "v1f count_compare_qwen3parse", "system_prompt": COUNT_COMPARE_SYS, "temperature": 0.0, "max_tokens": 1024, "ties": "keep original / pick A"},
            "qwen3_chat_template_kwargs": chat_kwargs,
        },
        "actions": dict(actions),
        "fixes": fixes,
        "breaks": breaks,
        "net": fixes - breaks,
        "f1_detections": det_incorrect,
        "valid_f1_detections": val_yes,
        "jsonl": str(jsonl.relative_to(PROJECT_ROOT)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge-biomistral")
    j.add_argument("--n", type=int, default=50)
    j.add_argument("--seed", type=int, default=42)
    j.add_argument("--concurrency", type=int, default=8)
    j.add_argument("--gpt-model", default="gpt-4o")
    j.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.1])
    j.set_defaults(func=cmd_judge_biomistral)
    s8 = sub.add_parser("step8-biomistral")
    s8.add_argument("--n", type=int, default=50)
    s8.add_argument("--seed", type=int, default=42)
    s8.add_argument("--concurrency", type=int, default=8)
    s8.add_argument("--port", type=int, default=8003)
    s8.add_argument("--gpt-model", default="gpt-4o")
    s8.set_defaults(func=cmd_step8_biomistral)
    s9 = sub.add_parser("step9-component")
    s9.add_argument("--model", choices=["qwen3", "qwen2.5"], required=True)
    s9.add_argument("--n", type=int, default=50)
    s9.add_argument("--seed", type=int, default=42)
    s9.add_argument("--concurrency", type=int, default=8)
    s9.add_argument("--port", type=int, default=8003)
    s9.set_defaults(func=cmd_step9_component)
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
