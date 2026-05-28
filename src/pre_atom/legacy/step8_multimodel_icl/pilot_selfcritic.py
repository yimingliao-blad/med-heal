#!/usr/bin/env python3
"""Pilot self-critic with teacher-student prompt.

Tests prompt framings for self-critic on a small sample per model.
Compares blind self-critic vs ground-truth-aware critic.
Measures agreement with GPT-4o labels.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ─── Prompt: Blind self-critic (teacher_v1 — best from initial test) ──

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


# ─── Prompt: Forced reflection — extract facts THEN compare ──────────

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


# ─── Prompt: Adversarial — assume error exists, prove otherwise ──────

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


# ─── Prompt: Ground-truth-aware critic (model sees correct answer) ────

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


# ─── Prompt: Ground-truth + forced reflection ─────────────────────────

PROMPT_GROUNDTRUTH_REFLECT = """You are an experienced medical professor grading a student's exam answer. You have the answer key. You must be thorough.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (from answer key):
{ground_truth}

STUDENT'S ANSWER:
{model_answer}

You MUST complete ALL three steps below before giving your grade:

STEP 1 — KEY ANSWER ANALYSIS: List the key facts and claims in the correct answer. What specific information does it contain?

STEP 2 — STUDENT ANSWER COMPARISON: Compare the student's answer against the correct answer point by point:
- What did the student get right?
- What did the student get wrong or say differently?
- What did the student omit that the correct answer includes?
- What did the student add that is NOT in the correct answer or discharge summary?

STEP 3 — VERDICT: Based on your comparison, is the student's answer correct or incorrect?

Error types (use exactly one if incorrect):
- omission: missed critical information that the correct answer includes
- hallucination: stated facts not found in the discharge summary
- reasoning_failure: misinterpreted or drew wrong conclusions
- specificity: gave vague answer when specific details were needed
- context_confusion: mixed up information from different sections
- temporal_error: got timing, dates, or sequence wrong

Respond in this exact format:
STEP_1_KEY_FACTS: <facts from correct answer>
STEP_2_COMPARISON: <point-by-point comparison>
GRADE: CORRECT or INCORRECT
ERROR_TYPE: <type> (only if incorrect)
EXPLANATION: <brief explanation>"""


PROMPTS = {
    "blind": PROMPT_BLIND,
    "reflect": PROMPT_REFLECT,
    "adversarial": PROMPT_ADVERSARIAL,
    "groundtruth": PROMPT_GROUNDTRUTH,
    "groundtruth_reflect": PROMPT_GROUNDTRUTH_REFLECT,
}


# ─── vLLM generation ──────────────────────────────────────────────

def vllm_generate(url, model, prompt, max_tokens=1024, temperature=0.1):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<|im_end|>", "<|eot_id|>", "</s>"],
    }
    resp = requests.post(f"{url}/completions", json=payload, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"vLLM error: {resp.text}")
    return resp.json()["choices"][0]["text"].strip()


def get_vllm_model(port, host="localhost"):
    resp = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
    return resp.json()["data"][0]["id"]


# ─── Chat template builders per model ─────────────────────────────

def build_chatml(system, user):
    """Qwen2.5, Qwen3 (chatml)"""
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def build_llama31(system, user):
    """Llama 3.1 Instruct"""
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

def build_mistral(system, user):
    """BioMistral (Mistral [INST] format)"""
    return f"[INST] {system}\n\n{user} [/INST]"

def build_deepseek(system, user):
    """DeepSeek R1 distill (uses Llama template)"""
    return build_llama31(system, user)

CHAT_BUILDERS = {
    "biomistral-7b": build_mistral,
    "deepseek-r1-distill-llama-8b": build_llama31,
    "llama-3.1-8b-instruct": build_llama31,
    "qwen2.5-7b-instruct": build_chatml,
    "qwen3-8b": build_chatml,
}


# ─── Parse critic response ────────────────────────────────────────

def parse_critic_response(text):
    """Parse GRADE and ERROR_TYPE from critic response.

    Handles markdown bold (**GRADE:**), extra whitespace, and <think> tags.
    """
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"</think>", "", clean).strip()
    # Strip markdown bold markers for parsing
    parse_text = re.sub(r"\*\*", "", clean)

    grade = None
    grade_match = re.search(r"GRADE:\s*(CORRECT|INCORRECT)", parse_text, re.IGNORECASE)
    if grade_match:
        grade = 1 if grade_match.group(1).upper() == "CORRECT" else 0

    error_type = None
    error_match = re.search(r"ERROR_TYPE:\s*(\w+)", parse_text)
    if error_match:
        error_type = error_match.group(1).lower()
    # Also catch "Error Type:" variant
    if not error_type:
        error_match2 = re.search(r"Error\s*Type:\s*(\w+)", parse_text, re.IGNORECASE)
        if error_match2:
            error_type = error_match2.group(1).lower()

    explanation = ""
    expl_match = re.search(r"EXPLANATION:\s*(.+?)(?:\n|$)", parse_text)
    if expl_match:
        explanation = expl_match.group(1).strip()

    return grade, error_type, explanation, clean


# ─── GPT-4o error type classification ─────────────────────────────

def gpt4o_classify_error(note, question, ground_truth, model_answer):
    """Use GPT-4o to classify error type for a known-wrong answer."""
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
        max_tokens=20,
        temperature=0.1,
    )
    et = resp.choices[0].message.content.strip().lower()
    valid = {"omission", "hallucination", "reasoning_failure", "specificity", "context_confusion", "temporal_error"}
    return et if et in valid else None


# ─── Load notes ───────────────────────────────────────────────────

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


# ─── Main ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--model-key", default="llama-3.1-8b-instruct")
    parser.add_argument("--prompt", default="all", choices=list(PROMPTS.keys()) + ["all"])
    parser.add_argument("--pilot-csv", default="/tmp/llama_pilot_20.csv")
    parser.add_argument("--with-gpt4o", action="store_true", help="Get GPT-4o error types for wrong cases")
    args = parser.parse_args()

    # Connect to vLLM
    vllm_model = get_vllm_model(args.port)
    url = f"http://localhost:{args.port}/v1"
    print(f"vLLM model: {vllm_model}")

    # Load data
    notes_lookup = load_notes()
    pilot_df = pd.read_csv(args.pilot_csv)
    n_wrong = (pilot_df.binary_correct == 0).sum()
    n_correct = (pilot_df.binary_correct == 1).sum()
    print(f"Pilot samples: {len(pilot_df)} ({n_wrong} wrong, {n_correct} correct per GPT-4o)")

    builder = CHAT_BUILDERS.get(args.model_key, build_chatml)

    # Get GPT-4o error types for wrong cases
    gpt4o_errors = {}
    if args.with_gpt4o:
        print("\nGetting GPT-4o error type classifications for wrong cases...")
        for _, row in pilot_df[pilot_df.binary_correct == 0].iterrows():
            pid = str(row["patient_id"])
            note = notes_lookup.get(pid, "")
            et = gpt4o_classify_error(note, row["question"], row["ground_truth"], row["model_answer"])
            gpt4o_errors[row["idx"]] = et
            print(f"  idx={row['idx']:3d} → GPT-4o error: {et}")
        print()

    # Test prompts
    prompts_to_test = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]

    for prompt_name in prompts_to_test:
        prompt_template = PROMPTS[prompt_name]
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt_name} | Model: {args.model_key}")
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
            system_msg = ""
            prompt = builder(system_msg, user_content)

            # Reflection prompts need more tokens for step-by-step output
            tok_limit = 2048 if "reflect" in prompt_name or "adversarial" in prompt_name else 1024
            try:
                raw = vllm_generate(url, vllm_model, prompt, max_tokens=tok_limit, temperature=0.1)
                grade, error_type, explanation, clean = parse_critic_response(raw)
            except Exception as e:
                grade, error_type, explanation = None, None, str(e)

            gpt_label = int(row["binary_correct"])
            agree = grade == gpt_label
            gpt4o_et = gpt4o_errors.get(row["idx"], "-")

            results.append({
                "idx": row["idx"],
                "gpt_label": gpt_label,
                "critic_label": grade,
                "agree": agree,
                "critic_error_type": error_type,
                "gpt4o_error_type": gpt4o_et,
                "explanation": explanation[:120],
            })

            gpt_str = "correct" if gpt_label == 1 else "WRONG"
            critic_str = "correct" if grade == 1 else ("WRONG" if grade == 0 else "???")
            et_match = ""
            if gpt_label == 0 and error_type and gpt4o_et and gpt4o_et != "-":
                et_match = " ✓ET" if error_type == gpt4o_et else f" ✗ET({gpt4o_et})"
            symbol = "✓" if agree else "✗"
            print(f"  idx={row['idx']:3d} | GPT={gpt_str:7s} | Critic={critic_str:7s} | {symbol} | "
                  f"err={error_type or '-':20s}{et_match}")

        # Summary stats
        n_agree = sum(1 for r in results if r["agree"])
        n_total = len(results)
        tp = sum(1 for r in results if r["gpt_label"] == 0 and r["critic_label"] == 0)
        fn = sum(1 for r in results if r["gpt_label"] == 0 and r["critic_label"] == 1)
        tn = sum(1 for r in results if r["gpt_label"] == 1 and r["critic_label"] == 1)
        fp = sum(1 for r in results if r["gpt_label"] == 1 and r["critic_label"] == 0)
        n_parse_fail = sum(1 for r in results if r["critic_label"] is None)

        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)

        print(f"\n  Agreement: {n_agree}/{n_total} = {n_agree/n_total:.0%}")
        print(f"  Detection: TP={tp} FN={fn} → recall={recall:.0%}")
        print(f"  False alarms: FP={fp} TN={tn} → FPR={fpr:.0%}")
        print(f"  Precision: {precision:.0%}")
        if n_parse_fail:
            print(f"  Parse failures: {n_parse_fail}")

        # Error type comparison (for wrong cases only)
        if gpt4o_errors:
            wrong_results = [r for r in results if r["gpt_label"] == 0 and r["critic_label"] == 0]
            if wrong_results:
                et_matches = sum(1 for r in wrong_results
                                 if r["critic_error_type"] == r["gpt4o_error_type"])
                print(f"  Error type match (TP only): {et_matches}/{len(wrong_results)} = "
                      f"{et_matches/len(wrong_results):.0%}")


if __name__ == "__main__":
    main()
