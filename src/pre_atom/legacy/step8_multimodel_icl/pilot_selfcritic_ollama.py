#!/usr/bin/env python3
"""Test self-critic on Mac Ollama and compare with vLLM results."""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent

PROMPT_BLIND = """You are an experienced medical professor reviewing a student's answer to a clinical question based on a patient's discharge summary.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

STUDENT'S ANSWER:
{model_answer}

Review the student's answer carefully against the discharge summary. Check for:
- Facts that contradict the discharge notes
- Important information from the notes that was omitted
- Information the student made up that isn't in the notes
- Vague answers when specific details are available in the notes

If the answer is INCORRECT, identify the error type:
- omission: missed critical information from the notes
- hallucination: included facts not in the discharge summary
- reasoning_failure: misinterpreted or drew wrong conclusions
- specificity: gave vague answer when specific details exist
- context_confusion: mixed up information between different parts of the notes
- temporal_error: got timing, sequence, or dates wrong

Respond in this exact format:
GRADE: CORRECT or INCORRECT
ERROR_TYPE: <type> (only if incorrect)
EXPLANATION: <brief explanation>"""

PROMPT_GROUNDTRUTH = """You are an experienced medical professor grading a student's exam answer. You have the answer key.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (from answer key):
{ground_truth}

STUDENT'S ANSWER:
{model_answer}

Compare the student's answer against the correct answer and the discharge summary.

If the student's answer is wrong, identify the specific error type:
- omission: missed critical information that the correct answer includes
- hallucination: included facts not supported by the discharge summary
- reasoning_failure: misinterpreted the information or drew wrong conclusions
- specificity: gave a vague answer when specific details were needed
- context_confusion: mixed up information from different parts of the notes
- temporal_error: got timing, sequence, or dates wrong

Respond in this exact format:
GRADE: CORRECT or INCORRECT
ERROR_TYPE: <type> (only if incorrect)
EXPLANATION: <brief explanation of what specifically is wrong>"""

PROMPT_REFLECT = """You are an experienced medical professor reviewing a student's answer to a clinical question based on a patient's discharge summary. You must be thorough and skeptical.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

STUDENT'S ANSWER:
{model_answer}

You MUST complete ALL three steps below before giving your grade:

STEP 1 — EXTRACT: Read the discharge summary carefully. List the key facts from the notes that are relevant to the question. Be specific — include dates, medications, dosages, lab values, diagnoses, procedures, and any other concrete details.

STEP 2 — COMPARE: For each claim in the student's answer, check whether it is supported by the facts you extracted in Step 1. Flag any:
- Claims NOT supported by the discharge summary (hallucination)
- Important facts from the notes that the student failed to mention (omission)
- Facts that the student got wrong — wrong dates, wrong values, wrong medications (contradiction)
- Vague statements where the notes contain specific information (specificity gap)

STEP 3 — VERDICT: Based on your comparison, give your final grade.

Error types (use exactly one if incorrect):
- omission: missed critical information that the discharge summary contains
- hallucination: stated facts not found anywhere in the discharge summary
- reasoning_failure: misinterpreted information or drew wrong conclusions from the notes
- specificity: gave a vague answer when the notes contain specific details (dates, values, names)
- context_confusion: mixed up information between different sections or notes
- temporal_error: got timing, sequence of events, or dates wrong

Respond in this exact format:
STEP_1_FACTS: <list the key relevant facts from the notes>
STEP_2_COMPARISON: <compare each student claim against the facts>
GRADE: CORRECT or INCORRECT
ERROR_TYPE: <type> (only if incorrect)
EXPLANATION: <brief explanation>"""

PROMPT_ADVERSARIAL = """You are a strict medical examiner whose job is to find errors in clinical answers. Assume the answer below contains an error until you can prove otherwise.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

STUDENT'S ANSWER:
{model_answer}

Your task: Try to find an error in the student's answer. Be thorough and skeptical.

Check each of these error types carefully:

1. HALLUCINATION — Does the answer state any fact that is NOT in the discharge summary? Check every medication name, every dosage, every diagnosis, every date. If the student mentions something not explicitly written in the notes, it is a hallucination.

2. OMISSION — Does the discharge summary contain important information relevant to the question that the student failed to mention? Compare what the notes say vs what the student included.

3. TEMPORAL ERROR — Did the student get any dates, timelines, or sequence of events wrong? Check admission dates, discharge dates, procedure dates, medication start/stop times.

4. REASONING FAILURE — Did the student misinterpret the clinical information? Did they draw a wrong conclusion from the evidence in the notes?

5. SPECIFICITY — Did the student give a vague or general answer when the notes contain specific details (exact values, specific medications, precise dates)?

6. CONTEXT CONFUSION — Did the student mix up information from different notes or different sections?

If after thorough checking you find NO errors, then and only then mark as CORRECT.

Respond in this exact format:
GRADE: CORRECT or INCORRECT
ERROR_TYPE: <type> (only if incorrect)
EXPLANATION: <what specifically is wrong, citing the relevant part of the notes>"""

PROMPTS = {"blind": PROMPT_BLIND, "reflect": PROMPT_REFLECT, "adversarial": PROMPT_ADVERSARIAL, "groundtruth": PROMPT_GROUNDTRUTH}

OLLAMA_MODELS = {
    "biomistral-7b": "cniongolo/biomistral:latest",
    "deepseek-r1-distill-llama-8b": "deepseek-r1:8b-llama-distill-q4_K_M",
    "llama-3.1-8b-instruct": "llama3.1:8b",
    "qwen2.5-7b-instruct": "qwen2.5:7b",
    "qwen3-8b": "qwen3:8b",
}


def ollama_generate(host, model, system_msg, user_msg, max_tokens=1024, temperature=0.1):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    resp = requests.post(f"http://{host}/api/chat", json=payload, timeout=300)
    if resp.status_code != 200:
        raise Exception(f"Ollama error: {resp.text}")
    return resp.json()["message"]["content"].strip()


def parse_critic_response(text):
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"</think>", "", clean).strip()
    parse_text = re.sub(r"\*\*", "", clean)

    grade = None
    grade_match = re.search(r"GRADE:\s*(CORRECT|INCORRECT)", parse_text, re.IGNORECASE)
    if grade_match:
        grade = 1 if grade_match.group(1).upper() == "CORRECT" else 0

    error_type = None
    error_match = re.search(r"ERROR_TYPE:\s*(\w+)", parse_text)
    if error_match:
        error_type = error_match.group(1).lower()
    if not error_type:
        error_match2 = re.search(r"Error\s*Type:\s*(\w+)", parse_text, re.IGNORECASE)
        if error_match2:
            error_type = error_match2.group(1).lower()

    explanation = ""
    expl_match = re.search(r"EXPLANATION:\s*(.+?)(?:\n|$)", parse_text)
    if expl_match:
        explanation = expl_match.group(1).strip()

    return grade, error_type, explanation, clean


def load_notes():
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    lookup = {}
    for _, r in notes_df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(f"[Note {i}]\n{t}")
        lookup[pid] = "\n\n".join(parts)
    return lookup


def gpt4o_classify_error(note, question, ground_truth, model_answer):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical expert classifying errors in AI answers."},
            {"role": "user", "content": (
                f"DISCHARGE SUMMARY:\n{note}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"CORRECT ANSWER:\n{ground_truth}\n\n"
                f"MODEL'S WRONG ANSWER:\n{model_answer}\n\n"
                f"Classify the error type as exactly ONE of:\n"
                f"omission, hallucination, reasoning_failure, specificity, context_confusion, temporal_error\n\n"
                f"Respond with ONLY the error type, nothing else."
            )},
        ],
        max_tokens=20, temperature=0.1,
    )
    et = resp.choices[0].message.content.strip().lower()
    valid = {"omission", "hallucination", "reasoning_failure", "specificity", "context_confusion", "temporal_error"}
    return et if et in valid else None


def run_model(host, model_key, pilot_df, notes_lookup, prompt_name, prompt_template, gpt4o_errors=None):
    ollama_model = OLLAMA_MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt_name} | Model: {model_key} | Backend: Ollama ({host})")
    print(f"{'='*70}")

    results = []
    for _, row in pilot_df.iterrows():
        pid = str(row["patient_id"])
        note = notes_lookup.get(pid, "")
        gt = str(row.get("ground_truth", ""))

        fmt_args = dict(note=note, question=row["question"], model_answer=row["model_answer"])
        if "{ground_truth}" in prompt_template:
            fmt_args["ground_truth"] = gt
        user_content = prompt_template.format(**fmt_args)

        tok_limit = 2048 if "reflect" in prompt_name or "adversarial" in prompt_name else 1024
        try:
            raw = ollama_generate(host, ollama_model, "", user_content, max_tokens=tok_limit, temperature=0.1)
            grade, error_type, explanation, clean = parse_critic_response(raw)
        except Exception as e:
            grade, error_type, explanation = None, None, str(e)[:80]

        gpt_label = int(row["binary_correct"])
        agree = grade == gpt_label
        gpt4o_et = (gpt4o_errors or {}).get(row["idx"], "-")

        results.append({
            "idx": row["idx"], "gpt_label": gpt_label, "critic_label": grade,
            "agree": agree, "critic_error_type": error_type, "gpt4o_error_type": gpt4o_et,
        })

        gpt_str = "correct" if gpt_label == 1 else "WRONG"
        critic_str = "correct" if grade == 1 else ("WRONG" if grade == 0 else "???")
        et_match = ""
        if gpt_label == 0 and error_type and gpt4o_et and gpt4o_et != "-":
            et_match = " ✓ET" if error_type == gpt4o_et else f" ✗ET({gpt4o_et})"
        symbol = "✓" if agree else "✗"
        print(f"  idx={row['idx']:3d} | GPT={gpt_str:7s} | Critic={critic_str:7s} | {symbol} | "
              f"err={error_type or '-':20s}{et_match}")

    # Summary
    tp = sum(1 for r in results if r["gpt_label"] == 0 and r["critic_label"] == 0)
    fn = sum(1 for r in results if r["gpt_label"] == 0 and r["critic_label"] == 1)
    tn = sum(1 for r in results if r["gpt_label"] == 1 and r["critic_label"] == 1)
    fp = sum(1 for r in results if r["gpt_label"] == 1 and r["critic_label"] == 0)
    n_parse = sum(1 for r in results if r["critic_label"] is None)
    n_agree = sum(1 for r in results if r["agree"])

    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    print(f"\n  Agreement: {n_agree}/{len(results)} = {n_agree/len(results):.0%}")
    print(f"  Detection: TP={tp} FN={fn} → recall={recall:.0%}")
    print(f"  False alarms: FP={fp} TN={tn} → FPR={fpr:.0%}")
    print(f"  Precision: {precision:.0%}")
    if n_parse:
        print(f"  Parse failures: {n_parse}")

    if gpt4o_errors:
        wrong_tp = [r for r in results if r["gpt_label"] == 0 and r["critic_label"] == 0]
        if wrong_tp:
            et_matches = sum(1 for r in wrong_tp if r["critic_error_type"] == r["gpt4o_error_type"])
            print(f"  Error type match (TP only): {et_matches}/{len(wrong_tp)} = "
                  f"{et_matches/len(wrong_tp):.0%}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.68.116:11434")
    parser.add_argument("--model-key", default="llama-3.1-8b-instruct")
    parser.add_argument("--pilot-csv", default="/tmp/llama_pilot_20.csv")
    parser.add_argument("--prompt", default="all", choices=list(PROMPTS.keys()) + ["all"])
    parser.add_argument("--with-gpt4o", action="store_true")
    parser.add_argument("--all-models", action="store_true", help="Test all available models")
    parser.add_argument("--exclude-models", nargs="*", default=[], help="Models to skip")
    args = parser.parse_args()

    notes_lookup = load_notes()
    print(f"Loaded notes for {len(notes_lookup)} patients")

    models_to_test = list(OLLAMA_MODELS.keys()) if args.all_models else [args.model_key]
    models_to_test = [m for m in models_to_test if m not in args.exclude_models]
    prompts_to_test = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]

    for model_key in models_to_test:
        # Load pilot CSV for this model
        pilot_csv = f"/tmp/{model_key}_pilot_20.csv" if args.all_models else args.pilot_csv
        try:
            pilot_df = pd.read_csv(pilot_csv)
        except Exception:
            print(f"\nSkipping {model_key}: no pilot CSV at {pilot_csv}")
            continue

        n_wrong = (pilot_df.binary_correct == 0).sum()
        n_correct = (pilot_df.binary_correct == 1).sum()
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_key} ({n_wrong} wrong, {n_correct} correct)")
        print(f"{'#'*70}")

        # GPT-4o error types
        gpt4o_errors = {}
        if args.with_gpt4o:
            print("Getting GPT-4o error types for wrong cases...")
            for _, row in pilot_df[pilot_df.binary_correct == 0].iterrows():
                pid = str(row["patient_id"])
                note = notes_lookup.get(pid, "")
                et = gpt4o_classify_error(note, row["question"], row["ground_truth"], row["model_answer"])
                gpt4o_errors[row["idx"]] = et
                print(f"  idx={row['idx']:3d} → {et}")

        for prompt_name in prompts_to_test:
            run_model(args.host, model_key, pilot_df, notes_lookup,
                      prompt_name, PROMPTS[prompt_name], gpt4o_errors)


if __name__ == "__main__":
    main()
