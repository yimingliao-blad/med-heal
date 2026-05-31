#!/usr/bin/env python3
"""Prompt engineering test: try different prompt formulations on known-wrong DeepSeek cases.

Tests edit-based vs from-scratch, different levels of specificity, etc.
Runs against vLLM on port 8003 (DeepSeek currently loaded).
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "step8"
BIO_INDEX_DIR = PROJECT_ROOT / "output" / "fullscale_4_biomistral" / "indices"
ERROR_CLASSIFICATION_DIR = PROJECT_ROOT / "output" / "step8" / "error_classification"

BASE_SYSTEM = "You are a medical expert answering questions about discharge summaries."
VLLM_URL = "http://localhost:8003/v1"


def vllm_generate(prompt, max_tokens=2048, temperature=0.1):
    resp = requests.get(f"{VLLM_URL}/models", timeout=5)
    model_name = resp.json()["data"][0]["id"]

    resp = requests.post(
        f"{VLLM_URL}/completions",
        json={"model": model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        timeout=300,
    )
    text = resp.json()["choices"][0]["text"].strip()
    # Strip thinking tags for DeepSeek
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"</think>", "", clean).strip()
    return clean


def build_llama3(system, user):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def assemble_note(fold_id, idx):
    """Get the discharge note for a case."""
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    test_file = PROJECT_ROOT / "output" / "folds" / f"fold_{fold_id}" / "test.jsonl"
    with open(test_file) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if "idx" not in d:
                d["idx"] = i
            if d["idx"] == idx:
                pid = d.get("patient_id", "")
                nr = notes_df[notes_df["patient_id"] == pid]
                if len(nr) > 0:
                    nr = nr.iloc[0]
                    parts = []
                    for j in [1, 2, 3]:
                        col = f"note_{j}"
                        if col in nr and pd.notna(nr[col]):
                            t = str(nr[col]).strip()
                            if t and t.lower() != "nan":
                                parts.append(f"[Note {j}]\n{t}")
                    return "\n\n".join(parts)
    return ""


def get_negative_example(fold_id, error_type="hallucination"):
    """Get a negative example from the error pool."""
    error_file = ERROR_CLASSIFICATION_DIR / f"fold_{fold_id}_errors.json"
    with open(error_file) as f:
        error_pool = json.load(f)
    # Find one matching the error type
    for ex in error_pool:
        if ex.get("primary_error") == error_type:
            return ex
    return error_pool[0]


def get_positive_example(fold_id):
    """Get a positive example from the correct pool."""
    with open(BIO_INDEX_DIR / f"fold_{fold_id}" / "correct_pool.json") as f:
        correct_pool = json.load(f)
    return correct_pool[0]


# =============================================================================
# PROMPT VARIANTS
# =============================================================================

def prompt_A_from_scratch(note, question, neg_ex, pos_ex, error_type, **_):
    """Original from-scratch contrastive (current approach)."""
    error_instructions = {
        "hallucination": (
            "Do NOT fabricate any information. If something is not stated in the notes, "
            "say 'not specified'. Never invent dates, measurements, or procedures."
        ),
        "omission": (
            "Ensure you include ALL relevant details from the discharge notes. "
            "Do not leave out diagnoses, medications, procedures, or test results."
        ),
    }
    instr = error_instructions.get(error_type, error_instructions["hallucination"])

    errors_list = neg_ex.get("errors", [])
    what_went_wrong = errors_list[0].get("correct", "") if errors_list else "The answer contained errors."

    system = (
        f"{BASE_SYSTEM}\n\n"
        f"CAUTION — {instr}\n\n"
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
    user = f"Discharge Summary:\n{note}\n\nQuestion: {question}\n\nAnswer:"
    return build_llama3(system, user)


def prompt_B_edit_with_diagnosis(note, question, original_answer, diagnosis, neg_ex, pos_ex, error_type, **_):
    """Edit-based: include original answer + Qwen3 diagnosis."""
    error_instructions_v2 = {
        "hallucination": (
            "The previous answer included information NOT in the discharge notes. "
            "For each claim, verify it against the notes. NEVER assume a procedure was performed "
            "just because a condition was present — only state what is explicitly documented."
        ),
        "omission": (
            "The previous answer missed important information from the discharge notes. "
            "Go through the notes and ensure every relevant detail asked about is included."
        ),
    }
    instr = error_instructions_v2.get(error_type, error_instructions_v2["hallucination"])

    errors_list = neg_ex.get("errors", [])
    what_went_wrong = errors_list[0].get("correct", "") if errors_list else "The answer contained errors."

    system = (
        f"{BASE_SYSTEM}\n\n"
        f"CAUTION — {instr}\n\n"
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
    return build_llama3(system, user)


def prompt_C_pinpoint(note, question, original_answer, diagnosis, **_):
    """Minimal: just pinpoint the error and ask to fix it, no ICL examples."""
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"A previous attempt at answering the question below was found to have errors.\n"
        f"Specific issue: {diagnosis}\n\n"
        f"Revise the previous answer. Fix ONLY the identified error(s). "
        f"Keep everything else the same. Base corrections strictly on the discharge summary."
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Previous Answer (contains errors):\n{original_answer}\n\n"
        f"Revised Answer:"
    )
    return build_llama3(system, user)


def prompt_D_verify_then_revise(note, question, original_answer, **_):
    """Step-by-step: ask model to verify each claim, then revise."""
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"The following answer may contain errors. Your task:\n"
        f"1. For each factual claim in the answer, check if it is supported by the discharge summary.\n"
        f"2. If any claim is NOT supported, note what the notes actually say.\n"
        f"3. Provide a corrected answer that fixes any errors found.\n\n"
        f"Be precise: only state facts explicitly documented in the notes."
    )
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"Answer to verify and correct:\n{original_answer}\n\n"
        f"Step 1 - Verify each claim:\n"
    )
    return build_llama3(system, user)


def prompt_E_concise_fix(note, question, original_answer, diagnosis, **_):
    """Ultra-concise: tell it exactly what's wrong, ask for short answer."""
    system = BASE_SYSTEM
    user = (
        f"Discharge Summary:\n{note}\n\n"
        f"Question: {question}\n\n"
        f"A previous answer had this error: {diagnosis}\n\n"
        f"Provide a corrected answer in 2-3 sentences, based strictly on the discharge summary."
    )
    return build_llama3(system, user)


def prompt_F_hybrid(note, question, original_answer, diagnosis, error_type="hallucination", neg_ex=None, pos_ex=None, **_):
    """Hybrid: edit-based for hallucination/reasoning, concise for omission.
    Always includes the diagnosis. Uses ICL example as reference."""
    if error_type == "omission":
        # For omission: don't edit, give fresh focused answer
        system = (
            f"{BASE_SYSTEM}\n\n"
            f"A previous AI answer to this question was incomplete. "
            f"Specifically: {diagnosis}\n\n"
            f"Answer the question completely, ensuring you include the missing information. "
            f"Be concise but thorough. Base your answer strictly on the discharge summary."
        )
        user = (
            f"Discharge Summary:\n{note}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    else:
        # For hallucination/reasoning/etc: edit-based with pinpoint
        system = (
            f"{BASE_SYSTEM}\n\n"
            f"A previous AI answer to this question contained an error.\n"
            f"Error identified: {diagnosis}\n\n"
            f"Revise the answer below. Fix the identified error while keeping correct parts. "
            f"Only state facts explicitly documented in the discharge summary."
        )
        user = (
            f"Discharge Summary:\n{note}\n\n"
            f"Question: {question}\n\n"
            f"Previous Answer (contains an error):\n{original_answer}\n\n"
            f"Revised Answer:"
        )
    return build_llama3(system, user)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Test cases: fold_id, idx, question, ground_truth, zeroshot_answer, diagnosis
    test_cases = [
        {
            "fold_id": 1, "idx": 36,
            "question": "What antibiotics was the patient given around the time of the Incision and Drainage procedure in his second hospital visit and for what duration?",
            "ground_truth": "B: The patient was given Augmentin for ten days.",
            "zeroshot_answer": "The patient was prescribed **Augmentin** (875 mg/125 mg tablets) once daily for 10 days during his second hospital visit.",
            "diagnosis": "The answer states Augmentin was prescribed 'once daily' but the discharge notes state 'One Tablet PO twice a day for 10 days'. The dosing frequency is wrong — it should be twice daily, not once daily.",
            "error_type": "hallucination",
        },
        {
            "fold_id": 1, "idx": 118,
            "question": "What was found in the level of neutrophils and platelets in the patient's blood on the day of discharge, and what might these levels suggest about his health status at the time of discharge?",
            "ground_truth": "B: The patient showed a increased level of neutrophils and decreased level of platelets, suggesting potential infection or injury.",
            "zeroshot_answer": "The patient's neutrophils and platelets levels on the day of discharge were:\n\n- **Neutrophils**: 3.9* (normal range is typically 1.5-11.0 x10^9/L)\n- **Platelets**: 64* (normal range is typically 150-450 x10^9/L)\n\nThese levels are within the normal range. Neutrophils are a type of white blood cell that help fight infections, and platelets are important for blood clotting and preventing bleeding. The normal levels suggest that the patient's immune system and blood clotting functions were functioning adequately at the time of discharge.",
            "diagnosis": "The answer incorrectly states platelet level of 64 is 'within normal range' — normal platelets are 150-450 x10^9/L, so 64 is significantly LOW (thrombocytopenia). The interpretation should note elevated neutrophils and decreased platelets, suggesting potential infection or bone marrow involvement.",
            "error_type": "hallucination",
        },
        {
            "fold_id": 4, "idx": 149,
            "question": "In the context of the alterations to the patient's medication list between the initial admission on 2188-12-29 and the subsequent admission on 2195-01-16, which of the following options accurately characterizes the modifications?",
            "ground_truth": "E: The patient started taking cimetidine.",
            "zeroshot_answer": "The patient's medication list underwent several modifications between the two admissions. Key changes include the addition of new medications such as **enoxaparin** for prophylaxis and **oxycodone** for pain management, as well as adjustments to dosages or discontinuations of certain drugs. Specifically, **metoprolol** was held during the second admission and was restarted upon discharge. Additionally, **traMADol** was added for pain management, and **acetaminophen** was continued for mild pain or fever. The **gastrostomy tube** was replaced during the second admission, which affected the route of medication administration. Overall, the modifications reflect updates to the patient's medical condition and treatment plan during the second hospital stay.",
            "diagnosis": "The answer fails to mention that cimetidine was added to the medication list in the second admission. The discharge notes for the second admission include cimetidine as a new medication that was not present in the first admission's medication list.",
            "error_type": "omission",
        },
    ]

    prompt_variants = {
        # "A_from_scratch": prompt_A_from_scratch,
        "B_edit_diagnosis": prompt_B_edit_with_diagnosis,
        # "C_pinpoint": prompt_C_pinpoint,
        # "D_verify_revise": prompt_D_verify_then_revise,
        "E_concise_fix": prompt_E_concise_fix,
        "F_hybrid": prompt_F_hybrid,
    }

    for case in test_cases:
        fold_id = case["fold_id"]
        idx = case["idx"]
        print(f"\n{'#'*80}")
        print(f"# CASE: fold={fold_id} idx={idx}")
        print(f"# Q: {case['question'][:80]}...")
        print(f"# GT: {case['ground_truth']}")
        print(f"# ZS answer (WRONG): {case['zeroshot_answer'][:100]}...")
        print(f"{'#'*80}")

        # Load note
        note = assemble_note(fold_id, idx)
        if not note:
            print("  ERROR: Could not load note")
            continue

        # Get ICL examples
        neg_ex = get_negative_example(fold_id, case["error_type"])
        pos_ex = get_positive_example(fold_id)

        for variant_name, prompt_fn in prompt_variants.items():
            print(f"\n  --- Variant {variant_name} ---")
            try:
                prompt = prompt_fn(
                    note=note,
                    question=case["question"],
                    original_answer=case["zeroshot_answer"],
                    diagnosis=case["diagnosis"],
                    neg_ex=neg_ex,
                    pos_ex=pos_ex,
                    error_type=case["error_type"],
                )
                print(f"  Prompt length: {len(prompt)} chars")
                answer = vllm_generate(prompt, max_tokens=2048)
                # Truncate for display
                display = answer[:400].replace('\n', '\n  ')
                print(f"  Answer: {display}")
                if len(answer) > 400:
                    print(f"  ... ({len(answer)} chars total)")
            except Exception as e:
                print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
