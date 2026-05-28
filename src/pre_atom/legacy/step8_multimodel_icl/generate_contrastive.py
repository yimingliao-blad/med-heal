#!/usr/bin/env python3
"""Phase 3-4: Contrastive RA-ICL generation pipeline.

Pipeline (original):
  1. Load existing zeroshot answers
  2. Qwen3-think blind critic predicts error type (or "correct")
  3. For flagged cases: retrieve error-typed negative + note-similar positive
  4. Re-generate with contrastive prompt (error-specific instruction)
  5. For non-flagged: keep original zeroshot answer

Pipeline (oracle mode, --oracle):
  1. Load existing zeroshot answers + GPT-4o binary labels
  2. Only process cases where GPT-4o says binary_correct=0 (eliminates FP)
  3. Qwen3-think diagnoses the specific error (knows answer is wrong)
  4. Retrieve error-typed negative + note-similar positive
  5. Edit-based regeneration: includes original answer, asks to revise
  6. Correct cases (binary_correct=1) keep original zeroshot answer

Conditions:
  - contrastive_random: random neg + random pos, generic instruction
  - contrastive_targeted: critic → error-typed retrieval → specific instruction
  - oracle_targeted: GPT-4o flags → Qwen3 diagnosis → error-typed retrieval → edit-based
  - oracle_random: GPT-4o flags → random examples → edit-based

Usage:
    # Original pilot
    python generate_contrastive.py --model biomistral-7b --gen-port 8003 --critic-port 8003 --pilot 10

    # Oracle pilot (recommended)
    python generate_contrastive.py --model biomistral-7b --gen-port 8003 --critic-port 8003 --pilot 10 \
        --oracle --conditions oracle_targeted oracle_random
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Lazy imports for retrieval (not needed for oracle_concise conditions)
SentenceTransformer = None
NoteRetriever = None

def _load_retrieval_deps():
    global SentenceTransformer, NoteRetriever
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    if NoteRetriever is None:
        sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_12_ra_icl"))
        from retrieval_strategies import NoteRetriever as NR
        NoteRetriever = NR

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "step8"
BIO_INDEX_DIR = PROJECT_ROOT / "output" / "fullscale_4_biomistral" / "indices"
QWEN_INDEX_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "indices"
ERROR_CLASSIFICATION_DIR = PROJECT_ROOT / "output" / "step8" / "error_classification"

BASE_SYSTEM = "You are a medical expert answering questions about discharge summaries."
BIOMISTRAL_SYSTEM = "You are a helpful, respectful and honest assistant."
USER_TASK = "Discharge Summary:\n{note}\n\nQuestion: {question}\n\nAnswer:"

ERROR_TYPES = [
    "omission", "hallucination", "reasoning_failure",
    "specificity", "context_confusion", "temporal_error",
]

ERROR_INSTRUCTIONS = {
    "omission": (
        "Ensure you include ALL relevant details from the discharge notes. "
        "Do not leave out diagnoses, medications, procedures, or test results."
    ),
    "hallucination": (
        "Do NOT fabricate any information. If something is not stated in the notes, "
        "say 'not specified'. Never invent dates, measurements, or procedures."
    ),
    "reasoning_failure": (
        "Consider the patient's full medical history when answering. Make reasonable "
        "clinical inferences from documented diagnoses, but distinguish inference from fact."
    ),
    "specificity": (
        "Provide specific names, values, and details from the notes. "
        "Avoid vague or general statements when specific information is available."
    ),
    "context_confusion": (
        "When multiple discharge summaries are provided, carefully distinguish which "
        "information comes from which admission. Do not conflate findings across notes."
    ),
    "temporal_error": (
        "Pay careful attention to the chronological sequence of events. "
        "Verify dates and timelines against the discharge notes before answering."
    ),
}

# V2: More actionable, specific instructions for edit-based oracle mode
ERROR_INSTRUCTIONS_V2 = {
    "omission": (
        "The previous answer missed important information from the discharge notes. "
        "Go through the notes and ensure every relevant diagnosis, medication, procedure, "
        "lab value, and clinical finding asked about is included. "
        "If the question asks about multiple items, list ALL of them."
    ),
    "hallucination": (
        "The previous answer included information NOT in the discharge notes. "
        "For each claim, verify it against the notes. NEVER assume a procedure was performed "
        "just because a condition was present — only state what is explicitly documented. "
        "If unsure, say 'not specified in the notes'."
    ),
    "reasoning_failure": (
        "The previous answer misinterpreted the clinical information. "
        "Distinguish carefully between: (a) initial/suspected diagnosis vs. confirmed/final diagnosis, "
        "(b) planned procedures vs. actually performed procedures, "
        "(c) the patient's prior history vs. the current admission's findings."
    ),
    "specificity": (
        "The previous answer was too vague when specific information is available. "
        "Use exact medication names with dosages, exact lab values with units, "
        "specific procedure names, and dates from the notes."
    ),
    "context_confusion": (
        "The previous answer confused information between different admissions or notes. "
        "When multiple notes are provided, track which events, procedures, and medications "
        "belong to which admission. Refer to each note's dates to keep them separate."
    ),
    "temporal_error": (
        "The previous answer got the timeline or sequence of events wrong. "
        "Note the dates in each discharge note and present events in correct chronological order. "
        "Do not attribute events from one admission to another."
    ),
}

GENERIC_INSTRUCTION = (
    "Learn from the example mistake shown below. Answer based only on what is "
    "explicitly stated in the discharge summary."
)

GENERIC_INSTRUCTION_V2 = (
    "The previous answer contained errors. Learn from the examples below, then revise "
    "the answer based strictly on what is stated in the discharge summary."
)

# ─── Model-specific self-critic prompts (best from pilot testing) ────────
# blind: teacher-student framing (works for Llama-3.1, Qwen2.5)
# reflect: forced step-by-step fact extraction (works for DeepSeek, Qwen3)
# adversarial: assume error exists, prove otherwise (works for BioMistral)

CRITIC_PROMPT_BLIND = """You are an experienced medical professor reviewing a student's answer to a clinical question based on a patient's discharge summary.

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

CRITIC_PROMPT_REFLECT = """You are an experienced medical professor reviewing a student's answer to a clinical question based on a patient's discharge summary. You must be thorough and skeptical.

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

CRITIC_PROMPT_ADVERSARIAL = """You are a strict medical examiner whose job is to find errors in clinical answers. Assume the answer below contains an error until you can prove otherwise.

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

# Map model → best prompt from pilot testing
MODEL_BEST_CRITIC_PROMPT = {
    "biomistral-7b": CRITIC_PROMPT_ADVERSARIAL,
    "deepseek-r1-distill-llama-8b": CRITIC_PROMPT_REFLECT,
    "llama-3.1-8b-instruct": CRITIC_PROMPT_BLIND,
    "qwen2.5-7b-instruct": CRITIC_PROMPT_BLIND,
    "qwen3-8b": CRITIC_PROMPT_REFLECT,
}


# Oracle mode: Qwen3 diagnoses error (knows answer is wrong, just identifies what's wrong)
ORACLE_DIAGNOSIS_PROMPT = """You are a medical expert analyzing an AI model's answer that has been VERIFIED AS INCORRECT by an expert reviewer.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

AI MODEL'S ANSWER (VERIFIED INCORRECT):
{model_answer}

Your task: identify the SPECIFIC error(s) in this answer by comparing it to the discharge summary.

For each error:
1. Quote the specific incorrect or missing part
2. State what the discharge summary actually says

Then classify the PRIMARY error and provide a concise diagnosis:

ERROR_TYPE: <one of: omission, hallucination, reasoning_failure, specificity, context_confusion, temporal_error>
DIAGNOSIS: <2-3 sentence explanation of exactly what is wrong and what the correct information should be, based on the notes>"""


# =============================================================================
# CHAT TEMPLATES (same as generate_step8.py)
# =============================================================================

def build_llama2(system, user):
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

def build_llama3(system, user):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def build_chatml(system, user):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

MODEL_CONFIGS = {
    "biomistral-7b": {
        "build_fn": build_llama2,
        "max_tokens": 512,
        "is_thinking": False,
        "index_source": "biomistral",
        "system_prompt": BIOMISTRAL_SYSTEM,
    },
    "deepseek-r1-distill-llama-8b": {
        "build_fn": build_llama3,
        "max_tokens": 2048,
        "is_thinking": True,
        "index_source": "qwen",
    },
    "qwen2.5-7b-instruct": {
        "build_fn": build_chatml,
        "max_tokens": 1024,
        "is_thinking": False,
        "index_source": "qwen",
    },
    "llama-3.1-8b-instruct": {
        "build_fn": build_llama3,
        "max_tokens": 1024,
        "is_thinking": False,
        "index_source": "qwen",
    },
    "qwen3-8b": {
        "build_fn": build_chatml,
        "max_tokens": 2048,
        "is_thinking": True,
        "index_source": "qwen",
    },
}


# =============================================================================
# VLLM CLIENT
# =============================================================================

def vllm_generate(base_url, model_name, prompt, max_tokens, temperature=0.1):
    """Generate text from vLLM server."""
    resp = requests.post(
        f"{base_url}/completions",
        json={"model": model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        timeout=300,
    )
    if resp.status_code != 200:
        raise Exception(f"vLLM error: {resp.text}")
    return resp.json()["choices"][0]["text"].strip()


# Ollama model name mapping
OLLAMA_MODELS = {
    "biomistral-7b": "cniongolo/biomistral:latest",
    "deepseek-r1-distill-llama-8b": "deepseek-r1:8b-llama-distill-q4_K_M",
    "llama-3.1-8b-instruct": "llama3.1:8b",
    "qwen2.5-7b-instruct": "qwen2.5:7b",
    "qwen3-8b": "qwen3:8b",
}


def ollama_generate(host, model_key, system_msg, user_msg, max_tokens, temperature=0.1, think=False):
    """Generate text via Ollama chat API."""
    payload = {
        "model": OLLAMA_MODELS.get(model_key, model_key),
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if model_key == "qwen3-8b":
        payload["think"] = think
    resp = requests.post(f"http://{host}/api/chat", json=payload, timeout=300)
    if resp.status_code != 200:
        raise Exception(f"Ollama error: {resp.text}")
    return resp.json()["message"]["content"].strip()


def get_vllm_model(port, host="localhost"):
    """Get model name from vLLM server."""
    resp = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
    return resp.json()["data"][0]["id"]


def extract_thinking_answer(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"</think>", "", cleaned).strip()
    return cleaned if cleaned else text


# =============================================================================
# CRITIC
# =============================================================================

def run_critic(critic_url, critic_model, note, question, model_answer,
               critic_backend="vllm", ollama_host=None, ollama_model_key=None,
               model_key=None):
    """Run blind self-critic. Uses model-specific best prompt and chat template.

    Returns (verdict, error_type, reasoning).
    verdict: 0 (incorrect) or 1 (correct) or None (parse failure).
    """
    # Pick model-specific prompt
    critic_prompt = MODEL_BEST_CRITIC_PROMPT.get(model_key, CRITIC_PROMPT_BLIND)
    user_content = critic_prompt.format(
        note=note, question=question, model_answer=model_answer,
    )
    system_msg = ""

    # Pick model-specific chat template for vLLM
    build_fn = MODEL_CONFIGS.get(model_key, {}).get("build_fn", build_chatml)
    is_thinking = MODEL_CONFIGS.get(model_key, {}).get("is_thinking", False)
    max_tokens = 2048 if is_thinking or "reflect" in (MODEL_BEST_CRITIC_PROMPT.get(model_key, "").__class__.__name__ or "") else 1024
    # Reflect/adversarial prompts need more tokens
    if critic_prompt in (CRITIC_PROMPT_REFLECT, CRITIC_PROMPT_ADVERSARIAL):
        max_tokens = 2048

    try:
        if critic_backend == "ollama" and ollama_host and ollama_model_key:
            raw = ollama_generate(ollama_host, ollama_model_key, system_msg, user_content,
                                  max_tokens=max_tokens, temperature=0.1, think=is_thinking)
        else:
            prompt = build_fn(system_msg, user_content)
            raw = vllm_generate(critic_url, critic_model, prompt, max_tokens=max_tokens, temperature=0.1)

        # Strip thinking tags and markdown bold
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"</think>", "", clean).strip()
        parse_text = re.sub(r"\*\*", "", clean)

        # Extract verdict — supports both GRADE and VERDICT formats
        verdict = None
        grade_match = re.search(r"GRADE:\s*(CORRECT|INCORRECT)", parse_text, re.IGNORECASE)
        if grade_match:
            verdict = 1 if grade_match.group(1).upper() == "CORRECT" else 0
        else:
            verdict_match = re.search(r"VERDICT:\s*([01])", parse_text)
            if verdict_match:
                verdict = int(verdict_match.group(1))

        # Extract error type
        error_type = None
        error_match = re.search(r"ERROR_TYPE:\s*(\w+)", parse_text)
        if error_match:
            et = error_match.group(1).lower()
            if et in ERROR_TYPES:
                error_type = et
            else:
                for valid_et in ERROR_TYPES:
                    if valid_et.startswith(et[:4]):
                        error_type = valid_et
                        break

        return verdict, error_type, clean[:500]

    except Exception as e:
        return None, None, str(e)[:200]


def run_oracle_diagnosis(critic_url, critic_model, note, question, model_answer,
                         critic_backend="vllm", ollama_host=None, ollama_model_key=None):
    """Run diagnosis on a KNOWN-WRONG answer. Returns (error_type, diagnosis)."""
    user_content = ORACLE_DIAGNOSIS_PROMPT.format(
        note=note, question=question, model_answer=model_answer,
    )
    system_msg = "You are a medical expert analyzing errors in AI-generated clinical answers."

    try:
        if critic_backend == "ollama" and ollama_host and ollama_model_key:
            raw = ollama_generate(ollama_host, ollama_model_key, system_msg, user_content,
                                  max_tokens=2048, temperature=0.1, think=False)
        else:
            prompt = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            raw = vllm_generate(critic_url, critic_model, prompt, max_tokens=2048, temperature=0.1)

        # Strip thinking tags
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"</think>", "", clean).strip()

        # Extract error type
        error_type = None
        error_match = re.search(r"ERROR_TYPE:\s*(\w+)", clean)
        if error_match:
            et = error_match.group(1).lower()
            if et in ERROR_TYPES:
                error_type = et
            else:
                for valid_et in ERROR_TYPES:
                    if valid_et.startswith(et[:4]):
                        error_type = valid_et
                        break
        if not error_type:
            error_type = "hallucination"  # default

        # Extract diagnosis
        diagnosis = ""
        diag_match = re.search(r"DIAGNOSIS:\s*(.+)", clean, re.DOTALL)
        if diag_match:
            diagnosis = diag_match.group(1).strip()[:500]
        else:
            # Use last few sentences as diagnosis
            sentences = clean.split(". ")
            diagnosis = ". ".join(sentences[-3:])[:500] if sentences else clean[:500]

        return error_type, diagnosis

    except Exception as e:
        return "hallucination", str(e)[:200]


# =============================================================================
# NOTE ASSEMBLY
# =============================================================================

def assemble_note(row, notes_df):
    pid = row.get("patient_id", "")
    if pid:
        nr = notes_df[notes_df["patient_id"] == pid]
        if len(nr) > 0:
            nr = nr.iloc[0]
            parts = []
            for i in [1, 2, 3]:
                col = f"note_{i}"
                if col in nr and pd.notna(nr[col]):
                    t = str(nr[col]).strip()
                    if t and t.lower() != "nan":
                        parts.append(f"[Note {i}]\n{t}")
            if parts:
                return "\n\n".join(parts)
    return ""


def get_ground_truth(row):
    gt = row.get("ground_truth", "")
    if gt and str(gt).strip() and str(gt).lower() != "nan":
        return str(gt)
    al = str(row.get("answer", row.get("ground_truth_letter", ""))).strip()
    ck = f"choice_{al}"
    if ck in row and pd.notna(row.get(ck)):
        return f"{al}: {row[ck]}"
    return al


# =============================================================================
# CONTRASTIVE PROMPT BUILDER
# =============================================================================

def build_contrastive_prompt(build_fn, sys_prompt, note, question,
                             neg_ex, pos_ex, error_type, error_instruction):
    """Build contrastive ICL prompt with negative + positive examples."""
    # Build error explanation from neg_ex's classification
    error_summary = neg_ex.get("error_summary", "")
    errors_list = neg_ex.get("errors", [])
    if errors_list:
        # Use first error's correction as explanation
        what_went_wrong = errors_list[0].get("correct", error_summary)
    else:
        what_went_wrong = error_summary or "The answer contained errors."

    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {error_instruction}\n\n"
        f"Here is an example where an AI made a \"{error_type}\" error:\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Wrong Answer]: {neg_ex['openended_answer']}\n"
        f"[What went wrong]: {what_went_wrong}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
        f"Here is an example of a correct answer for a similar case:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['openended_answer']}\n\n"
        f"Apply the same care and precision. Avoid the type of mistake shown above."
    )
    user = USER_TASK.format(note=note, question=question)
    return build_fn(system, user)


def build_random_contrastive_prompt(build_fn, sys_prompt, note, question, neg_ex, pos_ex):
    """Build contrastive ICL prompt with random examples and generic instruction."""
    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {GENERIC_INSTRUCTION}\n\n"
        f"Here is an example of a common mistake to avoid:\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Incorrect Answer]: {neg_ex['openended_answer']}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
        f"Here is an example of a correct answer:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['openended_answer']}\n\n"
        f"Apply the same precision and avoid similar mistakes."
    )
    user = USER_TASK.format(note=note, question=question)
    return build_fn(system, user)


# =============================================================================
# ORACLE (EDIT-BASED) PROMPT BUILDERS
# =============================================================================

def build_oracle_targeted_prompt(build_fn, sys_prompt, note, question,
                                  original_answer, diagnosis,
                                  neg_ex, pos_ex, error_type, error_instruction):
    """Edit-based contrastive prompt: includes original answer for revision.

    Key improvements over from-scratch:
    1. Model sees its original answer → preserves correct parts
    2. Specific diagnosis tells it exactly what to fix
    3. Error-typed retrieval shows the pattern to avoid
    4. V2 instructions are more actionable
    """
    error_summary = neg_ex.get("error_summary", "")
    errors_list = neg_ex.get("errors", [])
    if errors_list:
        what_went_wrong = errors_list[0].get("correct", error_summary)
    else:
        what_went_wrong = error_summary or "The answer contained errors."

    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {error_instruction}\n\n"
        f"Here is an example where an AI made a \"{error_type}\" error:\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Wrong Answer]: {neg_ex['openended_answer']}\n"
        f"[What went wrong]: {what_went_wrong}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
        f"Here is an example of a correct answer for a similar case:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['openended_answer']}\n\n"
        f"A previous attempt at answering the question below contained errors.\n"
        f"Specific issue identified: {diagnosis}\n\n"
        f"Revise the previous answer below. Keep the parts that are correct, "
        f"fix the identified errors. Base your revision STRICTLY on the discharge summary."
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains errors — revise this):\n{original_answer}\n\n"
        f"Revised Answer:"
    )
    return build_fn(system, user)


def build_oracle_random_prompt(build_fn, sys_prompt, note, question,
                                original_answer, diagnosis, neg_ex, pos_ex):
    """Edit-based contrastive prompt with random examples (no error-type targeting)."""
    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {GENERIC_INSTRUCTION_V2}\n\n"
        f"Here is an example of a common mistake to avoid:\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Incorrect Answer]: {neg_ex['openended_answer']}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
        f"Here is an example of a correct answer:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['openended_answer']}\n\n"
        f"A previous attempt at answering the question below contained errors.\n"
        f"Specific issue identified: {diagnosis}\n\n"
        f"Revise the previous answer below. Keep the parts that are correct, "
        f"fix the identified errors. Base your revision STRICTLY on the discharge summary."
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains errors — revise this):\n{original_answer}\n\n"
        f"Revised Answer:"
    )
    return build_fn(system, user)


# =============================================================================
# ORACLE CONCISE PROMPT BUILDERS (best performers from prompt engineering)
# =============================================================================

def build_oracle_concise_prompt(build_fn, sys_prompt, note, question, diagnosis):
    """Prompt E: Concise fix with diagnosis only. No ICL examples.
    Score: 6/7 in prompt engineering test."""
    system = sys_prompt
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"A previous answer had this error: {diagnosis}\n\n"
        f"Provide a corrected answer in 2-3 sentences, based strictly on the discharge summary."
    )
    return build_fn(system, user)


def build_oracle_concise_neg_prompt(build_fn, sys_prompt, note, question, diagnosis, neg_ex):
    """Prompt G: Concise fix with diagnosis + one negative ICL example.
    Score: 7/7 in prompt engineering test — BEST PERFORMER."""
    system = (
        f"{sys_prompt}\n\n"
        f"Here is an example of a common mistake:\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Wrong Answer]: {neg_ex['openended_answer'][:300]}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"A previous answer had this error: {diagnosis}\n\n"
        f"Provide a corrected answer in 2-3 sentences, based strictly on the discharge summary."
    )
    return build_fn(system, user)


def build_concise_pos_prompt(build_fn, sys_prompt, note, question, diagnosis, pos_ex):
    """Concise fix with diagnosis + 1 positive example (no negative)."""
    system = (
        f"{sys_prompt}\n\n"
        f"Here is an example of a CORRECT answer:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['ground_truth']}\n"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"A previous answer had this error: {diagnosis}\n\n"
        f"Provide a corrected answer in 2-3 sentences, based strictly on the discharge summary."
    )
    return build_fn(system, user)


def build_concise_posneg_prompt(build_fn, sys_prompt, note, question, diagnosis, neg_ex, pos_ex):
    """Concise fix with diagnosis + 1 negative example + 1 positive example."""
    system = (
        f"{sys_prompt}\n\n"
        f"Here is an example of a CORRECT answer:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['ground_truth']}\n\n"
        f"Here is an example of a WRONG answer (mistake to avoid):\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Wrong Answer]: {neg_ex['openended_answer'][:300]}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"A previous answer had this error: {diagnosis}\n\n"
        f"Provide a corrected answer in 2-3 sentences, based strictly on the discharge summary."
    )
    return build_fn(system, user)


def build_oracle_concise_neg_multi_prompt(build_fn, sys_prompt, note, question, diagnosis, neg_examples):
    """Concise fix with diagnosis + k negative ICL examples (k=1-5)."""
    neg_section = ""
    for i, neg_ex in enumerate(neg_examples, 1):
        neg_section += (
            f"\nMistake example {i}:\n"
            f"[Question]: {neg_ex['question']}\n"
            f"[Wrong Answer]: {neg_ex['openended_answer'][:200]}\n"
            f"[Correct Answer]: {neg_ex['ground_truth']}\n"
        )
    system = (
        f"{sys_prompt}\n\n"
        f"Here are examples of common mistakes to avoid:{neg_section}"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"A previous answer had this error: {diagnosis}\n\n"
        f"Provide a corrected answer in 2-3 sentences, based strictly on the discharge summary."
    )
    return build_fn(system, user)


# =============================================================================
# CRITIC REVISION PROMPT BUILDERS (edit-based, with error type + original answer)
# =============================================================================

def build_critic_revision_prompt(build_fn, sys_prompt, note, question,
                                  original_answer, diagnosis, error_type, error_instruction):
    """Edit-based revision: original answer + error type guidance + critic diagnosis.
    Mirrors oracle_targeted but without ICL examples."""
    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {error_instruction}"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains a {error_type} error — revise this):\n{original_answer}\n\n"
        f"Error identified: {diagnosis}\n\n"
        f"Revise the previous answer. Keep the parts that are correct, "
        f"fix the identified error. Base your revision STRICTLY on the discharge summary."
    )
    return build_fn(system, user)


def build_critic_revision_neg_prompt(build_fn, sys_prompt, note, question,
                                      original_answer, diagnosis, error_type, error_instruction, neg_ex):
    """Edit-based revision + 1 error-typed negative example."""
    error_summary = neg_ex.get("error_summary", "")
    errors_list = neg_ex.get("errors", [])
    what_went_wrong = errors_list[0].get("correct", error_summary) if errors_list else error_summary or "The answer contained errors."

    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {error_instruction}\n\n"
        f"Here is an example where an AI made a \"{error_type}\" error:\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Wrong Answer]: {neg_ex['openended_answer'][:300]}\n"
        f"[What went wrong]: {what_went_wrong}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains a {error_type} error — revise this):\n{original_answer}\n\n"
        f"Error identified: {diagnosis}\n\n"
        f"Revise the previous answer. Keep the parts that are correct, "
        f"fix the identified error. Base your revision STRICTLY on the discharge summary."
    )
    return build_fn(system, user)


def build_critic_revision_pos_prompt(build_fn, sys_prompt, note, question,
                                      original_answer, diagnosis, error_type, error_instruction, pos_ex):
    """Edit-based revision + 1 positive example for correct answer style."""
    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {error_instruction}\n\n"
        f"Here is an example of a correct answer for a similar case:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['ground_truth']}\n"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains a {error_type} error — revise this):\n{original_answer}\n\n"
        f"Error identified: {diagnosis}\n\n"
        f"Revise the previous answer. Keep the parts that are correct, "
        f"fix the identified error. Base your revision STRICTLY on the discharge summary."
    )
    return build_fn(system, user)


def build_critic_revision_posneg_prompt(build_fn, sys_prompt, note, question,
                                         original_answer, diagnosis, error_type, error_instruction,
                                         neg_ex, pos_ex):
    """Edit-based revision + 1 negative + 1 positive example."""
    error_summary = neg_ex.get("error_summary", "")
    errors_list = neg_ex.get("errors", [])
    what_went_wrong = errors_list[0].get("correct", error_summary) if errors_list else error_summary or "The answer contained errors."

    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {error_instruction}\n\n"
        f"Here is an example where an AI made a \"{error_type}\" error:\n"
        f"[Question]: {neg_ex['question']}\n"
        f"[Wrong Answer]: {neg_ex['openended_answer'][:300]}\n"
        f"[What went wrong]: {what_went_wrong}\n"
        f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
        f"Here is an example of a correct answer for a similar case:\n"
        f"[Question]: {pos_ex['question']}\n"
        f"[Correct Answer]: {pos_ex['ground_truth']}\n"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains a {error_type} error — revise this):\n{original_answer}\n\n"
        f"Error identified: {diagnosis}\n\n"
        f"Revise the previous answer. Keep the parts that are correct, "
        f"fix the identified error. Base your revision STRICTLY on the discharge summary."
    )
    return build_fn(system, user)


def build_critic_revision_neg_multi_prompt(build_fn, sys_prompt, note, question,
                                            original_answer, diagnosis, error_type, error_instruction,
                                            neg_examples):
    """Edit-based revision + k negative examples."""
    neg_section = ""
    for i, neg_ex in enumerate(neg_examples, 1):
        error_summary = neg_ex.get("error_summary", "")
        errors_list = neg_ex.get("errors", [])
        what_went_wrong = errors_list[0].get("correct", error_summary) if errors_list else error_summary or "The answer contained errors."
        neg_section += (
            f"\nMistake example {i}:\n"
            f"[Question]: {neg_ex['question']}\n"
            f"[Wrong Answer]: {neg_ex['openended_answer'][:200]}\n"
            f"[What went wrong]: {what_went_wrong}\n"
            f"[Correct Answer]: {neg_ex['ground_truth']}\n"
        )
    system = (
        f"{sys_prompt}\n\n"
        f"CAUTION — {error_instruction}\n\n"
        f"Here are examples of \"{error_type}\" errors to avoid:{neg_section}"
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains a {error_type} error — revise this):\n{original_answer}\n\n"
        f"Error identified: {diagnosis}\n\n"
        f"Revise the previous answer. Keep the parts that are correct, "
        f"fix the identified error. Base your revision STRICTLY on the discharge summary."
    )
    return build_fn(system, user)


# =============================================================================
# MAIN
# =============================================================================

ALL_CONDITIONS = [
    "contrastive_random", "contrastive_targeted",
    "oracle_concise", "oracle_concise_neg", "oracle_concise_neg_k1",
    "oracle_targeted", "oracle_random",
    "critic_concise", "critic_concise_neg", "critic_concise_neg_k1",
    "critic_concise_pos_k1", "critic_concise_posneg_k1",
    "critic_concise_neg_k2", "critic_concise_neg_k3",
    "critic_concise_neg_k4", "critic_concise_neg_k5",
    "oracle_concise_neg_k2", "oracle_concise_neg_k3",
    "oracle_concise_neg_k4", "oracle_concise_neg_k5",
]


def main():
    parser = argparse.ArgumentParser(description="Contrastive RA-ICL generation pipeline")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--gen-port", type=int, default=8003, help="vLLM port for target model generation")
    parser.add_argument("--gen-host", type=str, default="localhost", help="Host for target model generation")
    parser.add_argument("--gen-backend", type=str, default="vllm", choices=["vllm", "ollama"],
                        help="Backend for generation: vllm (OpenAI-compat) or ollama")
    parser.add_argument("--critic-port", type=int, default=8003, help="vLLM port for critic model")
    parser.add_argument("--critic-host", type=str, default="localhost", help="Host for critic model")
    parser.add_argument("--critic-backend", type=str, default="vllm", choices=["vllm", "ollama"],
                        help="Backend for critic: vllm (separate model) or ollama (self-critic using gen model)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pilot", type=int, default=0, help="If >0, only process N random samples per fold")
    parser.add_argument("--conditions", nargs="+",
                        default=["contrastive_random", "contrastive_targeted"],
                        help="Which conditions to run")
    parser.add_argument("--critic-only", action="store_true", help="Only run critic/diagnosis stage")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle mode: use GPT-4o binary labels to flag wrong cases, "
                             "Qwen3 only diagnoses error type (no false positives)")
    parser.add_argument("--stratified-half", action="store_true",
                        help="Only process half the test set: 50%% correct + 50%% incorrect per GPT-4o labels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Auto-detect oracle vs original vs critic conditions
    oracle_conditions = [c for c in args.conditions if c.startswith("oracle_")]
    critic_conditions = [c for c in args.conditions if c.startswith("critic_")]
    original_conditions = [c for c in args.conditions
                           if not c.startswith("oracle_") and not c.startswith("critic_")]

    if args.oracle and not oracle_conditions:
        args.conditions = ["oracle_concise", "oracle_concise_neg"]
        oracle_conditions = args.conditions
        original_conditions = []

    has_oracle = bool(oracle_conditions)
    has_original = bool(original_conditions)
    has_critic = bool(critic_conditions)

    cfg = MODEL_CONFIGS[args.model]
    sys_prompt = cfg.get("system_prompt", BASE_SYSTEM)
    build_fn = cfg["build_fn"]

    # Check critic backend
    critic_model = None
    critic_url = None
    critic_backend = getattr(args, 'critic_backend', 'vllm')
    need_critic = has_original or has_oracle or has_critic

    gen_model = None
    gen_url = f"http://{args.gen_host}:{args.gen_port}/v1"
    use_ollama = (args.gen_backend == "ollama")
    ollama_host = f"{args.gen_host}:{args.gen_port}" if use_ollama else None
    critic_ollama_host = None
    critic_ollama_model = None

    if need_critic:
        if critic_backend == "ollama":
            # Self-critic: use same Ollama host as generation, same model
            critic_ollama_host = ollama_host or f"{args.critic_host}:{args.critic_port}"
            critic_ollama_model = args.model
            print(f"Critic Ollama (self): {OLLAMA_MODELS.get(args.model, args.model)} on {critic_ollama_host}")
        else:
            try:
                critic_model = get_vllm_model(args.critic_port, args.critic_host)
                critic_url = f"http://{args.critic_host}:{args.critic_port}/v1"
                print(f"Critic vLLM: {critic_model} on {args.critic_host}:{args.critic_port}")
            except Exception:
                if has_original:
                    print(f"ERROR: Critic vLLM not available on {args.critic_host}:{args.critic_port}")
                    sys.exit(1)
                else:
                    print(f"WARNING: No critic vLLM — will skip oracle diagnosis if needed")

    if not args.critic_only:
        if use_ollama:
            gen_model = OLLAMA_MODELS.get(args.model, args.model)
            print(f"Generation Ollama: {gen_model} on {ollama_host}")
        elif args.gen_port == args.critic_port and args.gen_host == args.critic_host:
            gen_model = critic_model
            print(f"Generation vLLM: same as critic ({gen_model})")
        else:
            try:
                gen_model = get_vllm_model(args.gen_port, args.gen_host)
                print(f"Generation vLLM: {gen_model} on {args.gen_host}:{args.gen_port}")
            except Exception:
                print(f"ERROR: Generation vLLM not available on {args.gen_host}:{args.gen_port}")
                sys.exit(1)

    # Load notes
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    print(f"Loaded notes for {len(notes_df)} patients")

    # Load GTR model for retrieval (only for targeted conditions)
    needs_retrieval = any(c in args.conditions for c in
                          ["contrastive_targeted", "oracle_targeted",
                           "critic_concise_pos_k1", "critic_concise_posneg_k1"]) and not args.critic_only
    needs_neg_pool = any(
        "neg" in c
        or c in ["contrastive_random", "contrastive_targeted", "oracle_targeted", "oracle_random"]
        for c in args.conditions
    )
    gtr_model = None
    if needs_retrieval:
        _load_retrieval_deps()
        print("Loading GTR model on CPU...")
        gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    folds_dir = PROJECT_ROOT / "output" / "folds"
    total_folds = len(args.folds)
    global_start = time.time()

    for fold_num, fold_id in enumerate(args.folds):
        print(f"\n>>> Progress: fold {fold_num+1}/{total_folds} "
              f"(elapsed: {(time.time()-global_start)/60:.1f} min) <<<")

        # Load test data
        test_file = folds_dir / f"fold_{fold_id}" / "test.jsonl"
        test_data = []
        with open(test_file) as f:
            for line in f:
                test_data.append(json.loads(line))
        for i, s in enumerate(test_data):
            if "idx" not in s:
                s["idx"] = i

        if args.pilot > 0:
            rng_pilot = random.Random(args.seed + fold_id)
            test_data = rng_pilot.sample(test_data, min(args.pilot, len(test_data)))

        # Load zeroshot answers
        zs_file = OUTPUT_DIR / args.model / f"fold_{fold_id}" / "zeroshot_generated.csv"
        if not zs_file.exists():
            print(f"Fold {fold_id}: No zeroshot file at {zs_file}, skipping")
            continue
        zs_df = pd.read_csv(zs_file)
        zs_lookup = {int(row["idx"]): str(row["model_answer"]) for _, row in zs_df.iterrows()}

        # Random half-sampling: randomly pick 50% of test set
        if args.stratified_half and args.pilot == 0:
            rng_strat = random.Random(args.seed + fold_id)
            n_half = len(test_data) // 2
            test_data = rng_strat.sample(test_data, n_half)
            rng_strat.shuffle(test_data)
            print(f"  Random half: {n_half} samples (from {n_half * 2} total)")

        # Load GPT-4o binary labels for oracle mode
        oracle_wrong_idxs = set()
        if has_oracle:
            eval_file = OUTPUT_DIR / args.model / f"fold_{fold_id}" / "zeroshot_evaluated_binary.csv"
            if eval_file.exists():
                eval_df = pd.read_csv(eval_file)
                oracle_wrong_idxs = set(eval_df[eval_df["binary_correct"] == 0]["idx"].astype(int).tolist())
                print(f"  Oracle: {len(oracle_wrong_idxs)} wrong cases from GPT-4o labels")
            else:
                print(f"  WARNING: No evaluated binary file for oracle mode")

        # Load pools
        bio_fold_dir = BIO_INDEX_DIR / f"fold_{fold_id}"
        with open(bio_fold_dir / "correct_pool.json") as f:
            correct_pool = json.load(f)
        pos_embs = np.load(bio_fold_dir / "gtr_note_embeddings.npy")
        with open(bio_fold_dir / "incorrect_pool.json") as f:
            incorrect_pool = json.load(f)

        error_file = ERROR_CLASSIFICATION_DIR / f"fold_{fold_id}_errors.json"
        if error_file.exists():
            with open(error_file) as f:
                error_pool = json.load(f)
        else:
            error_pool = incorrect_pool

        # Build retrievers if needed
        error_sub_retrievers = {}
        sub_dir = bio_fold_dir / "error_subindex"
        if needs_retrieval and sub_dir.exists():
            _load_retrieval_deps()
            for etype in ERROR_TYPES:
                pool_file = sub_dir / f"{etype}_pool.json"
                emb_file = sub_dir / f"{etype}_embeddings.npy"
                if pool_file.exists() and emb_file.exists():
                    with open(pool_file) as f:
                        sub_pool = json.load(f)
                    sub_embs = np.load(emb_file)
                    error_sub_retrievers[etype] = NoteRetriever(sub_pool, sub_embs, model=gtr_model)

        pos_retriever = None
        if needs_retrieval:
            _load_retrieval_deps()
            pos_retriever = NoteRetriever(correct_pool, pos_embs, model=gtr_model)

        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}: {len(test_data)} samples | Model: {args.model}")
        print(f"  Correct pool: {len(correct_pool)}, Error pool: {len(error_pool)}")
        print(f"{'='*60}")

        # --- Stage 1A: Blind critic pass (for original and critic conditions) ---
        critic_results = {}
        if has_original or has_critic:
            critic_file = OUTPUT_DIR / args.model / f"fold_{fold_id}" / "critic_results.json"
            if critic_file.exists():
                with open(critic_file) as f:
                    critic_results = {int(r["idx"]): r for r in json.load(f)}
                print(f"  Loaded {len(critic_results)} existing critic results")

            to_critique = [s for s in test_data
                           if s["idx"] not in critic_results
                           or critic_results[s["idx"]].get("verdict") is None]
            if to_critique:
                print(f"  Running blind critic on {len(to_critique)} samples...")
                start_time = time.time()
                for i, sample in enumerate(to_critique):
                    idx = sample["idx"]
                    note = assemble_note(sample, notes_df)
                    zs_answer = zs_lookup.get(idx, "")
                    if not zs_answer or zs_answer == "nan":
                        critic_results[idx] = {
                            "idx": idx, "patient_id": sample.get("patient_id", ""),
                            "verdict": None, "error_type": None, "reasoning": "no zeroshot answer",
                        }
                        continue

                    verdict, error_type, reasoning = run_critic(
                        critic_url, critic_model, note, sample["question"], zs_answer,
                        critic_backend=critic_backend, ollama_host=critic_ollama_host,
                        ollama_model_key=critic_ollama_model,
                        model_key=args.model,
                    )
                    critic_results[idx] = {
                        "idx": idx, "patient_id": sample.get("patient_id", ""),
                        "verdict": verdict, "error_type": error_type, "reasoning": reasoning,
                    }

                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        remaining = (len(to_critique) - i - 1) / rate if rate > 0 else 0
                        flagged = sum(1 for r in critic_results.values() if r.get("verdict") == 0)
                        print(f"    [{i+1}/{len(to_critique)}] {rate:.1f}/s, ~{remaining/60:.1f}min left | "
                              f"flagged: {flagged}")
                        critic_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(critic_file, "w") as f:
                            json.dump(list(critic_results.values()), f)

                critic_file.parent.mkdir(parents=True, exist_ok=True)
                with open(critic_file, "w") as f:
                    json.dump(list(critic_results.values()), f, indent=2)

            flagged = sum(1 for r in critic_results.values() if r.get("verdict") == 0)
            print(f"  Blind critic: {flagged} flagged wrong")

        # --- Stage 1B: Oracle diagnosis (for oracle conditions) ---
        oracle_diagnoses = {}
        if has_oracle and oracle_wrong_idxs:
            diag_file = OUTPUT_DIR / args.model / f"fold_{fold_id}" / "oracle_diagnoses.json"
            if diag_file.exists():
                with open(diag_file) as f:
                    oracle_diagnoses = {int(r["idx"]): r for r in json.load(f)}
                print(f"  Loaded {len(oracle_diagnoses)} existing oracle diagnoses")

            # Reuse blind critic reasoning for cases where critic also flagged wrong
            reused = 0
            for s in test_data:
                idx = s["idx"]
                if idx in oracle_wrong_idxs and idx not in oracle_diagnoses:
                    cr = critic_results.get(idx, {})
                    if cr.get("verdict") == 0 and cr.get("reasoning"):
                        # Critic agrees it's wrong — reuse reasoning as diagnosis
                        raw_reasoning = cr["reasoning"]
                        diag = re.sub(r"VERDICT:\s*\d\s*", "", raw_reasoning)
                        diag = re.sub(r"ERROR_TYPE:\s*\w+\s*", "", diag).strip()
                        if len(diag) > 500:
                            diag = diag[:500]
                        oracle_diagnoses[idx] = {
                            "idx": idx,
                            "patient_id": s.get("patient_id", ""),
                            "error_type": cr.get("error_type", "hallucination"),
                            "diagnosis": diag,
                            "source": "critic_reuse",
                        }
                        reused += 1
            if reused:
                print(f"  Reused {reused} critic diagnoses for oracle")

            # Only run new diagnosis for cases critic missed (FN: critic=1, GPT-4o=0)
            pilot_wrong = [s for s in test_data
                           if s["idx"] in oracle_wrong_idxs
                           and s["idx"] not in oracle_diagnoses]
            if pilot_wrong and critic_url:
                print(f"  Running oracle diagnosis on {len(pilot_wrong)} wrong cases...")
                start_time = time.time()
                for i, sample in enumerate(pilot_wrong):
                    idx = sample["idx"]
                    note = assemble_note(sample, notes_df)
                    zs_answer = zs_lookup.get(idx, "")
                    if not zs_answer or zs_answer == "nan":
                        oracle_diagnoses[idx] = {
                            "idx": idx, "error_type": "hallucination", "diagnosis": "no answer",
                        }
                        continue

                    error_type, diagnosis = run_oracle_diagnosis(
                        critic_url, critic_model, note, sample["question"], zs_answer,
                        critic_backend=critic_backend, ollama_host=critic_ollama_host,
                        ollama_model_key=critic_ollama_model,
                    )
                    oracle_diagnoses[idx] = {
                        "idx": idx,
                        "patient_id": sample.get("patient_id", ""),
                        "error_type": error_type,
                        "diagnosis": diagnosis,
                    }

                    if (i + 1) % 5 == 0:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        remaining = (len(pilot_wrong) - i - 1) / rate if rate > 0 else 0
                        print(f"    [{i+1}/{len(pilot_wrong)}] {rate:.1f}/s, ~{remaining/60:.1f}min left")
                        diag_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(diag_file, "w") as f:
                            json.dump(list(oracle_diagnoses.values()), f)

                diag_file.parent.mkdir(parents=True, exist_ok=True)
                with open(diag_file, "w") as f:
                    json.dump(list(oracle_diagnoses.values()), f, indent=2)

            print(f"  Oracle: {len(oracle_diagnoses)} diagnoses for wrong cases")

        if args.critic_only:
            continue

        # --- Stage 2: Generation ---
        for condition in args.conditions:
            model_out = OUTPUT_DIR / args.model / f"fold_{fold_id}"
            model_out.mkdir(parents=True, exist_ok=True)
            output_file = model_out / f"{condition}_generated.csv"

            if output_file.exists():
                existing = pd.read_csv(output_file)
                if len(existing) >= len(test_data):
                    print(f"\n  {condition}: Already complete ({len(existing)} rows)")
                    continue

            print(f"\n  Generating: {condition}")

            results = []
            done_ids = set()
            if output_file.exists():
                existing = pd.read_csv(output_file)
                done_ids = set(existing["idx"].tolist())
                results = existing.to_dict("records")
                print(f"    Resuming from {len(results)}")

            rng = random.Random(args.seed + fold_id)
            start_time = time.time()
            is_oracle = condition.startswith("oracle_")

            for sample in test_data:
                idx = sample["idx"]
                if idx in done_ids:
                    continue

                note = assemble_note(sample, notes_df)
                question = sample["question"]
                gt = get_ground_truth(sample)
                zs_answer = zs_lookup.get(idx, "")

                # Determine if we regenerate
                is_critic_cond = condition.startswith("critic_")
                if is_oracle:
                    regenerate = (idx in oracle_wrong_idxs)
                    diag_r = oracle_diagnoses.get(idx, {})
                    error_type = diag_r.get("error_type", "hallucination")
                    diagnosis = diag_r.get("diagnosis", "")
                elif is_critic_cond:
                    # Critic conditions: use blind critic verdicts + full reasoning
                    critic_r = critic_results.get(idx, {})
                    regenerate = (critic_r.get("verdict") == 0)
                    error_type = critic_r.get("error_type")
                    # Keep the full critic reasoning — includes fact extraction,
                    # comparison, error type, and error location. Only clean
                    # redundant VERDICT/ERROR_TYPE lines.
                    raw_reasoning = critic_r.get("reasoning", "")
                    diagnosis = re.sub(r"VERDICT:\s*\d\s*", "", raw_reasoning)
                    diagnosis = re.sub(r"ERROR_TYPE:\s*\w+\s*", "", diagnosis).strip()
                    # Allow up to 1500 chars to preserve step-by-step reasoning
                    if len(diagnosis) > 1500:
                        diagnosis = diagnosis[:1500]
                else:
                    critic_r = critic_results.get(idx, {})
                    regenerate = (critic_r.get("verdict") == 0)
                    error_type = critic_r.get("error_type")
                    diagnosis = critic_r.get("reasoning", "")

                if not regenerate:
                    results.append({
                        "idx": idx, "patient_id": sample.get("patient_id", ""),
                        "fold_id": fold_id, "question": question,
                        "question_type": sample.get("question_type", ""),
                        "ground_truth": gt, "model_answer": zs_answer,
                        "model": args.model, "method": condition,
                        "regenerated": False, "critic_verdict": 1 if not is_oracle else (0 if idx in oracle_wrong_idxs else 1),
                        "critic_error_type": None, "retrieval_sim_score": 0.0,
                        "prompt_length": 0, "answer_length": len(str(zs_answer)),
                    })
                    continue

                # --- Build prompt based on condition ---
                # Use a capture wrapper so we can extract (system, user) for Ollama
                captured = {}
                def capture_fn(system, user):
                    captured['system'] = system
                    captured['user'] = user
                    return build_fn(system, user)

                sim_score = 0.0

                if condition == "oracle_concise":
                    prompt = build_oracle_concise_prompt(
                        capture_fn, sys_prompt, note, question, diagnosis,
                    )

                elif condition in ("oracle_concise_neg", "oracle_concise_neg_k1"):
                    neg_ex = rng.choice(error_pool)
                    prompt = build_oracle_concise_neg_prompt(
                        capture_fn, sys_prompt, note, question, diagnosis, neg_ex,
                    )

                elif condition == "contrastive_random":
                    neg_ex = rng.choice(error_pool)
                    pos_ex = rng.choice(correct_pool)
                    prompt = build_random_contrastive_prompt(
                        capture_fn, sys_prompt, note, question, neg_ex, pos_ex,
                    )

                elif condition == "contrastive_targeted":
                    used_error_type = error_type or "hallucination"
                    error_instruction = ERROR_INSTRUCTIONS.get(used_error_type, GENERIC_INSTRUCTION)
                    if used_error_type in error_sub_retrievers:
                        retrieved_neg = error_sub_retrievers[used_error_type].retrieve(note, k=1)
                        neg_ex, neg_sim = retrieved_neg[0]
                    else:
                        neg_ex = rng.choice(error_pool)
                        neg_sim = 0.0
                    retrieved_pos = pos_retriever.retrieve(note, k=1)
                    pos_ex, pos_sim = retrieved_pos[0]
                    sim_score = (neg_sim + pos_sim) / 2
                    prompt = build_contrastive_prompt(
                        capture_fn, sys_prompt, note, question,
                        neg_ex, pos_ex, used_error_type, error_instruction,
                    )

                elif condition == "oracle_targeted":
                    used_error_type = error_type or "hallucination"
                    error_instruction = ERROR_INSTRUCTIONS_V2.get(used_error_type, GENERIC_INSTRUCTION_V2)
                    if used_error_type in error_sub_retrievers:
                        retrieved_neg = error_sub_retrievers[used_error_type].retrieve(note, k=1)
                        neg_ex, neg_sim = retrieved_neg[0]
                    else:
                        neg_ex = rng.choice(error_pool)
                        neg_sim = 0.0
                    retrieved_pos = pos_retriever.retrieve(note, k=1)
                    pos_ex, pos_sim = retrieved_pos[0]
                    sim_score = (neg_sim + pos_sim) / 2
                    prompt = build_oracle_targeted_prompt(
                        capture_fn, sys_prompt, note, question, zs_answer, diagnosis,
                        neg_ex, pos_ex, used_error_type, error_instruction,
                    )

                elif condition == "oracle_random":
                    neg_ex = rng.choice(error_pool)
                    pos_ex = rng.choice(correct_pool)
                    prompt = build_oracle_random_prompt(
                        capture_fn, sys_prompt, note, question, zs_answer, diagnosis,
                        neg_ex, pos_ex,
                    )

                elif condition == "critic_concise":
                    # Edit-based: show original answer + error type guidance + critic diagnosis
                    used_error_type = error_type or "hallucination"
                    error_instruction = ERROR_INSTRUCTIONS_V2.get(used_error_type, GENERIC_INSTRUCTION_V2)
                    prompt = build_critic_revision_prompt(
                        capture_fn, sys_prompt, note, question,
                        zs_answer, diagnosis, used_error_type, error_instruction,
                    )

                elif condition in ("critic_concise_neg", "critic_concise_neg_k1"):
                    used_error_type = error_type or "hallucination"
                    error_instruction = ERROR_INSTRUCTIONS_V2.get(used_error_type, GENERIC_INSTRUCTION_V2)
                    if used_error_type in error_sub_retrievers:
                        retrieved_neg = error_sub_retrievers[used_error_type].retrieve(note, k=1)
                        neg_ex, neg_sim = retrieved_neg[0]
                    else:
                        neg_ex = rng.choice(error_pool)
                    prompt = build_critic_revision_neg_prompt(
                        capture_fn, sys_prompt, note, question,
                        zs_answer, diagnosis, used_error_type, error_instruction, neg_ex,
                    )

                elif condition == "critic_concise_pos_k1":
                    used_error_type = error_type or "hallucination"
                    error_instruction = ERROR_INSTRUCTIONS_V2.get(used_error_type, GENERIC_INSTRUCTION_V2)
                    retrieved_pos = pos_retriever.retrieve(note, k=1)
                    pos_ex, pos_sim = retrieved_pos[0]
                    sim_score = pos_sim
                    prompt = build_critic_revision_pos_prompt(
                        capture_fn, sys_prompt, note, question,
                        zs_answer, diagnosis, used_error_type, error_instruction, pos_ex,
                    )

                elif condition == "critic_concise_posneg_k1":
                    used_error_type = error_type or "hallucination"
                    error_instruction = ERROR_INSTRUCTIONS_V2.get(used_error_type, GENERIC_INSTRUCTION_V2)
                    if used_error_type in error_sub_retrievers:
                        retrieved_neg = error_sub_retrievers[used_error_type].retrieve(note, k=1)
                        neg_ex, neg_sim = retrieved_neg[0]
                    else:
                        neg_ex = rng.choice(error_pool)
                    retrieved_pos = pos_retriever.retrieve(note, k=1)
                    pos_ex, pos_sim = retrieved_pos[0]
                    prompt = build_critic_revision_posneg_prompt(
                        capture_fn, sys_prompt, note, question,
                        zs_answer, diagnosis, used_error_type, error_instruction, neg_ex, pos_ex,
                    )

                elif condition.startswith("critic_concise_neg_k") or condition.startswith("oracle_concise_neg_k"):
                    k = int(condition.split("_k")[-1])
                    used_error_type = error_type or "hallucination"
                    error_instruction = ERROR_INSTRUCTIONS_V2.get(used_error_type, GENERIC_INSTRUCTION_V2)
                    neg_examples = rng.sample(error_pool, min(k, len(error_pool)))
                    prompt = build_critic_revision_neg_multi_prompt(
                        capture_fn, sys_prompt, note, question,
                        zs_answer, diagnosis, used_error_type, error_instruction, neg_examples,
                    )

                else:
                    raise ValueError(f"Unknown condition: {condition}")

                # --- Generate answer via appropriate backend ---
                if use_ollama:
                    sys_msg = captured.get('system', sys_prompt)
                    usr_msg = captured.get('user', '')
                    think = False  # no thinking for generation
                    raw_answer = ollama_generate(
                        ollama_host, args.model, sys_msg, usr_msg,
                        cfg["max_tokens"], think=think,
                    )
                else:
                    raw_answer = vllm_generate(gen_url, gen_model, prompt, cfg["max_tokens"])
                if cfg["is_thinking"]:
                    clean_answer = extract_thinking_answer(raw_answer)
                else:
                    clean_answer = raw_answer

                results.append({
                    "idx": idx, "patient_id": sample.get("patient_id", ""),
                    "fold_id": fold_id, "question": question,
                    "question_type": sample.get("question_type", ""),
                    "ground_truth": gt, "model_answer": clean_answer,
                    "model": args.model, "method": condition,
                    "regenerated": True, "critic_verdict": 0,
                    "critic_error_type": error_type,
                    "retrieval_sim_score": round(float(sim_score), 4),
                    "prompt_length": len(prompt), "answer_length": len(clean_answer),
                })

                if len(results) % 10 == 0:
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    regen = sum(1 for r in results if r.get("regenerated"))
                    elapsed = time.time() - start_time
                    print(f"    [{len(results)}/{len(test_data)}] {regen} regenerated, {elapsed:.0f}s")

            pd.DataFrame(results).to_csv(output_file, index=False)
            regen = sum(1 for r in results if r.get("regenerated"))
            print(f"    Done: {len(results)} total, {regen} regenerated -> {output_file.name}")

    print(f"\nContrastive generation complete for {args.model}.")


if __name__ == "__main__":
    main()
