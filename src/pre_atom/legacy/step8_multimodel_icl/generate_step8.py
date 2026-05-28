#!/usr/bin/env python3
"""
Step 8: Full-scale Multi-Model ICL Experiment (5-Fold CV)

Experiment 1: 5 models × 8 conditions × 5 folds
Experiment 2: 4 models × neg k-sweep (k=2,3,4,5) × 5 folds

Usage:
    python generate_step8.py --model biomistral-7b --port 8003
    python generate_step8.py --model qwen2.5-7b-instruct --conditions cot_evidence cot_conclusion multiturn --port 8003
    python generate_step8.py --model qwen3-8b --conditions gtr_note_neg_k2 gtr_note_neg_k3 --folds 0 1 --port 8003
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_12_ra_icl"))
from retrieval_strategies import NoteRetriever

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[4]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUTPUT_DIR = RUN_ROOT / "output" / "step8"

# Index directories per model
QWEN_INDEX_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "indices"
BIO_INDEX_DIR = PROJECT_ROOT / "output" / "fullscale_4_biomistral" / "indices"

ALL_CONDITIONS = [
    "zeroshot",
    "gtr_note_pos_k1", "gtr_note_neg_k1", "gtr_note_posneg_k1",
    "cot_evidence", "cot_conclusion", "multiturn",
    "gtr_note_any_unlabeled_k1",
    # Negative k-sweep (Experiment 2)
    "gtr_note_neg_k2", "gtr_note_neg_k3", "gtr_note_neg_k4", "gtr_note_neg_k5",
    # Random baselines (no retrieval, fixed seed)
    "random_pos_k1", "random_neg_k1",
]

STEP1_CONDITIONS = ALL_CONDITIONS[:8]

NEEDS_POS = {"gtr_note_pos_k1", "gtr_note_posneg_k1", "multiturn", "random_pos_k1"}
NEEDS_NEG = {
    "gtr_note_neg_k1", "gtr_note_posneg_k1",
    "gtr_note_neg_k2", "gtr_note_neg_k3", "gtr_note_neg_k4", "gtr_note_neg_k5",
    "random_neg_k1",
}
NEEDS_MIXED = {"gtr_note_any_unlabeled_k1"}


# =============================================================================
# CHAT TEMPLATES
# =============================================================================

BASE_SYSTEM = "You are a medical expert answering questions about discharge summaries."
# BioMistral uses the original step2 prompt to maintain validated human agreement (92%, κ=0.75)
BIOMISTRAL_SYSTEM = "You are a helpful, respectful and honest assistant."
USER_TASK = "Discharge Summary:\n{note}\n\nQuestion: {question}\n\nAnswer:"

COT_EVIDENCE_SYSTEM = (
    "You are a medical expert answering questions about discharge summaries. "
    "First, extract the specific evidence from the discharge summary that answers "
    "this question. Then provide your answer based solely on that evidence."
)

COT_CONCLUSION_SYSTEM = (
    "You are a medical expert answering questions about discharge summaries. "
    "First state your answer, then explain your reasoning based on the discharge summary."
)


def build_llama2(system, user):
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def build_llama2_multiturn(system, ex_user, ex_asst, user):
    return (
        f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{ex_user} [/INST] {ex_asst} </s>"
        f"<s>[INST] {user} [/INST]"
    )


def build_llama3(system, user):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_llama3_multiturn(system, ex_user, ex_asst, user):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{ex_user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{ex_asst}<|eot_id|>"
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


def build_chatml_multiturn(system, ex_user, ex_asst, user):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{ex_user}<|im_end|>\n"
        f"<|im_start|>assistant\n{ex_asst}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Aliases
build_qwen25 = build_chatml
build_qwen25_multiturn = build_chatml_multiturn
build_qwen3 = build_chatml
build_qwen3_multiturn = build_chatml_multiturn


# =============================================================================
# MODEL CONFIGS
# =============================================================================

MODEL_CONFIGS = {
    "biomistral-7b": {
        "build_fn": build_llama2,
        "multiturn_fn": build_llama2_multiturn,
        "max_tokens": 512,
        "is_thinking": False,
        "index_source": "biomistral",
        "system_prompt": BIOMISTRAL_SYSTEM,
    },
    "deepseek-r1-distill-llama-8b": {
        "build_fn": build_llama3,
        "multiturn_fn": build_llama3_multiturn,
        "max_tokens": 2048,
        "is_thinking": True,
        "index_source": "qwen",
    },
    "qwen2.5-7b-instruct": {
        "build_fn": build_qwen25,
        "multiturn_fn": build_qwen25_multiturn,
        "max_tokens": 1024,
        "is_thinking": False,
        "index_source": "qwen",
    },
    "llama-3.1-8b-instruct": {
        "build_fn": build_llama3,
        "multiturn_fn": build_llama3_multiturn,
        "max_tokens": 1024,
        "is_thinking": False,
        "index_source": "qwen",
    },
    "qwen3-8b": {
        "build_fn": build_qwen3,
        "multiturn_fn": build_qwen3_multiturn,
        "max_tokens": 2048,
        "is_thinking": True,
        "index_source": "qwen",
    },
}


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_prompt(method, note, question, build_fn, multiturn_fn,
                 pos_ex=None, neg_ex=None, neg_examples=None, mixed_ex=None,
                 base_system=None):
    """Build model-specific prompt for a given condition."""
    sys_prompt = base_system or BASE_SYSTEM
    user = USER_TASK.format(note=note, question=question)

    if method == "zeroshot":
        return build_fn(sys_prompt, user)

    elif method == "gtr_note_pos_k1":
        system = (
            f"{sys_prompt}\n\n"
            "Here is an example of a good answer:\n"
            f"[Question]: {pos_ex['question']}\n"
            f"[Answer]: {pos_ex['openended_answer']}\n\n"
            "Apply the same precision and directness to your answer."
        )
        return build_fn(system, user)

    elif method == "gtr_note_neg_k1":
        system = (
            f"{sys_prompt}\n\n"
            "Here is an example of a common mistake to avoid:\n"
            f"[Question]: {neg_ex['question']}\n"
            f"[Incorrect Answer]: {neg_ex['openended_answer']}\n"
            f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
            "Learn from this mistake. Answer based only on what is explicitly stated "
            "in the discharge summary."
        )
        return build_fn(system, user)

    elif method == "gtr_note_posneg_k1":
        system = (
            f"{sys_prompt}\n\n"
            "EXAMPLE OF A MISTAKE:\n"
            f"[Question]: {neg_ex['question']}\n"
            f"[Incorrect Answer]: {neg_ex['openended_answer']}\n"
            f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
            "EXAMPLE OF A GOOD ANSWER:\n"
            f"[Question]: {pos_ex['question']}\n"
            f"[Answer]: {pos_ex['openended_answer']}\n\n"
            "Apply the same precision and avoid the same kinds of mistakes."
        )
        return build_fn(system, user)

    elif method == "cot_evidence":
        cot_ev = (
            f"{sys_prompt} "
            "First, extract the specific evidence from the discharge summary that answers "
            "this question. Then provide your answer based solely on that evidence."
        )
        return build_fn(cot_ev, user)

    elif method == "cot_conclusion":
        cot_cl = (
            f"{sys_prompt} "
            "First state your answer, then explain your reasoning based on the discharge summary."
        )
        return build_fn(cot_cl, user)

    elif method == "multiturn":
        system = (
            f"{sys_prompt} "
            "Answer concisely using only facts from the note."
        )
        ex_user = f"Question: {pos_ex['question']}"
        ex_asst = pos_ex["openended_answer"]
        return multiturn_fn(system, ex_user, ex_asst, user)

    elif method == "gtr_note_any_unlabeled_k1":
        system = (
            f"{sys_prompt}\n\n"
            "Here is an example from a similar patient case:\n"
            f"[Question]: {mixed_ex['question']}\n"
            f"[Answer]: {mixed_ex['openended_answer']}\n\n"
            "Now answer the following question based only on the discharge summary provided."
        )
        return build_fn(system, user)

    elif method == "random_pos_k1":
        # Same prompt as gtr_note_pos_k1 but with randomly selected example
        system = (
            f"{sys_prompt}\n\n"
            "Here is an example of a good answer:\n"
            f"[Question]: {pos_ex['question']}\n"
            f"[Answer]: {pos_ex['openended_answer']}\n\n"
            "Apply the same precision and directness to your answer."
        )
        return build_fn(system, user)

    elif method == "random_neg_k1":
        # Same prompt as gtr_note_neg_k1 but with randomly selected example
        system = (
            f"{sys_prompt}\n\n"
            "Here is an example of a common mistake to avoid:\n"
            f"[Question]: {neg_ex['question']}\n"
            f"[Incorrect Answer]: {neg_ex['openended_answer']}\n"
            f"[Correct Answer]: {neg_ex['ground_truth']}\n\n"
            "Learn from this mistake. Answer based only on what is explicitly stated "
            "in the discharge summary."
        )
        return build_fn(system, user)

    elif method.startswith("gtr_note_neg_k") and method != "gtr_note_neg_k1":
        # Negative multi-shot (k=2,3,4,5)
        examples_text = ""
        for i, ex in enumerate(neg_examples, 1):
            examples_text += (
                f"Mistake {i}:\n"
                f"[Question]: {ex['question']}\n"
                f"[Incorrect Answer]: {ex['openended_answer']}\n"
                f"[Correct Answer]: {ex['ground_truth']}\n\n"
            )
        system = (
            f"{sys_prompt}\n\n"
            f"Here are {len(neg_examples)} common mistakes to avoid:\n\n"
            f"{examples_text}"
            "Learn from these mistakes. Answer based only on what is explicitly stated "
            "in the discharge summary."
        )
        return build_fn(system, user)

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# HELPERS
# =============================================================================

VLLM_BASE_URL = None
VLLM_MODEL_NAME = None


def check_vllm(port):
    global VLLM_BASE_URL, VLLM_MODEL_NAME
    VLLM_BASE_URL = f"http://localhost:{port}/v1"
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            VLLM_MODEL_NAME = data["data"][0]["id"]
            return True
    except Exception:
        pass
    return False


def generate_vllm(prompt, max_tokens):
    resp = requests.post(
        f"{VLLM_BASE_URL}/completions",
        json={"model": VLLM_MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.1},
        timeout=300,
    )
    if resp.status_code != 200:
        raise Exception(f"vLLM error: {resp.text}")
    return resp.json()["choices"][0]["text"].strip()


def extract_thinking_answer(text):
    """Strip <think>...</think> block from thinking model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"</think>", "", cleaned).strip()
    return cleaned if cleaned else text


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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 8: Full-scale Multi-Model ICL (5-Fold CV)")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--conditions", nargs="+", default=STEP1_CONDITIONS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    if not check_vllm(args.port):
        print(f"ERROR: vLLM not running on port {args.port}")
        sys.exit(1)
    print(f"vLLM model: {VLLM_MODEL_NAME}")

    # Validate conditions
    for c in args.conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition: {c}")
            sys.exit(1)

    # Load notes
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    print(f"Loaded notes for {len(notes_df)} patients")

    # Determine index directory
    if cfg["index_source"] == "biomistral":
        index_base = BIO_INDEX_DIR
    else:
        index_base = QWEN_INDEX_DIR

    # Load GTR model if needed for retrieval
    any_needs_pos = any(c in NEEDS_POS for c in args.conditions)
    any_needs_neg = any(c in NEEDS_NEG for c in args.conditions)
    any_needs_mixed = any(c in NEEDS_MIXED for c in args.conditions)

    gtr_model = None
    if any_needs_pos or any_needs_neg or any_needs_mixed:
        print("Loading GTR model on CPU...")
        gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    folds_dir = PROJECT_ROOT / "output" / "folds"
    total_combos = len(args.conditions) * len(args.folds)
    completed = 0

    for fold_id in args.folds:
        # Load test data
        test_file = folds_dir / f"fold_{fold_id}" / "test.jsonl"
        test_data = []
        with open(test_file) as f:
            for line in f:
                test_data.append(json.loads(line))
        for i, s in enumerate(test_data):
            if "idx" not in s:
                s["idx"] = i

        # Load fold-specific retrievers
        fold_index_dir = index_base / f"fold_{fold_id}"
        pos_retriever = None
        neg_retriever = None
        mixed_retriever = None

        correct_pool = None
        incorrect_pool = None
        pos_embs = None
        neg_embs = None

        if any_needs_pos or any_needs_mixed:
            with open(fold_index_dir / "correct_pool.json") as f:
                correct_pool = json.load(f)
            pos_embs = np.load(fold_index_dir / "gtr_note_embeddings.npy")
            if any_needs_pos:
                pos_retriever = NoteRetriever(correct_pool, pos_embs, model=gtr_model)

        if any_needs_neg or any_needs_mixed:
            with open(fold_index_dir / "incorrect_pool.json") as f:
                incorrect_pool = json.load(f)
            neg_embs = np.load(fold_index_dir / "gtr_note_incorrect_embeddings.npy")
            if any_needs_neg:
                neg_retriever = NoteRetriever(incorrect_pool, neg_embs, model=gtr_model)

        if any_needs_mixed:
            mixed_pool = []
            for ex in correct_pool:
                mixed_pool.append({**ex, "is_correct": True})
            for ex in incorrect_pool:
                mixed_pool.append({**ex, "is_correct": False})
            mixed_embs = np.concatenate([pos_embs, neg_embs], axis=0)
            mixed_retriever = NoteRetriever(mixed_pool, mixed_embs, model=gtr_model)

        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}: {len(test_data)} test samples | Model: {args.model}")
        if pos_retriever:
            print(f"  Positive pool: {len(correct_pool)} examples")
        if neg_retriever:
            print(f"  Negative pool: {len(incorrect_pool)} examples")
        if mixed_retriever:
            print(f"  Mixed pool: {len(mixed_pool)} examples")
        print(f"{'='*60}")

        for condition in args.conditions:
            completed += 1
            model_out = OUTPUT_DIR / args.model / f"fold_{fold_id}"
            model_out.mkdir(parents=True, exist_ok=True)
            output_file = model_out / f"{condition}_generated.csv"

            # Skip if already complete
            if output_file.exists():
                existing = pd.read_csv(output_file)
                if len(existing) >= len(test_data):
                    print(f"\n  [{completed}/{total_combos}] fold_{fold_id}/{condition}: "
                          f"Already complete ({len(existing)} rows)")
                    continue

            print(f"\n  [{completed}/{total_combos}] fold_{fold_id}/{condition}")

            # Resume support
            results = []
            done_ids = set()
            if output_file.exists():
                existing = pd.read_csv(output_file)
                done_ids = set(existing["idx"].tolist())
                results = existing.to_dict("records")
                print(f"    Resuming from {len(results)}")

            for sample in test_data:
                idx = sample.get("idx", 0)
                if idx in done_ids:
                    continue

                note = assemble_note(sample, notes_df)
                question = str(sample.get("question", ""))
                gt = get_ground_truth(sample)

                # Retrieve examples as needed
                pos_ex = None
                neg_ex = None
                neg_examples = None
                mixed_ex = None
                sim_score = 0.0

                if condition == "random_pos_k1" and correct_pool:
                    rng = random.Random(fold_id * 100000 + idx)
                    pos_ex = rng.choice(correct_pool)
                    sim_score = 0.0
                elif condition in NEEDS_POS and pos_retriever:
                    retrieved = pos_retriever.retrieve(note, k=1)
                    pos_ex = retrieved[0][0]
                    sim_score = retrieved[0][1]

                if condition == "random_neg_k1" and incorrect_pool:
                    rng = random.Random(fold_id * 100000 + idx)
                    neg_ex = rng.choice(incorrect_pool)
                    sim_score = 0.0
                elif condition == "gtr_note_neg_k1" and neg_retriever:
                    retrieved = neg_retriever.retrieve(note, k=1)
                    neg_ex = retrieved[0][0]
                    sim_score = retrieved[0][1]
                elif condition == "gtr_note_posneg_k1" and neg_retriever:
                    neg_retrieved = neg_retriever.retrieve(note, k=1)
                    neg_ex = neg_retrieved[0][0]
                    if pos_ex:
                        sim_score = (sim_score + neg_retrieved[0][1]) / 2
                elif condition.startswith("gtr_note_neg_k") and condition != "gtr_note_neg_k1" and neg_retriever:
                    k = int(condition.split("_k")[1])
                    retrieved = neg_retriever.retrieve(note, k=k)
                    neg_examples = [r[0] for r in retrieved]
                    sim_score = sum(r[1] for r in retrieved) / len(retrieved)

                if condition in NEEDS_MIXED and mixed_retriever:
                    retrieved = mixed_retriever.retrieve(note, k=1)
                    mixed_ex = retrieved[0][0]
                    sim_score = retrieved[0][1]

                prompt = build_prompt(
                    condition, note, question,
                    cfg["build_fn"], cfg["multiturn_fn"],
                    pos_ex=pos_ex, neg_ex=neg_ex,
                    neg_examples=neg_examples, mixed_ex=mixed_ex,
                    base_system=cfg.get("system_prompt"),
                )

                raw_answer = generate_vllm(prompt, cfg["max_tokens"])

                if cfg["is_thinking"]:
                    clean_answer = extract_thinking_answer(raw_answer)
                else:
                    clean_answer = raw_answer

                results.append({
                    "idx": idx,
                    "patient_id": sample.get("patient_id", ""),
                    "fold_id": fold_id,
                    "question": question,
                    "question_type": sample.get("question_type", ""),
                    "ground_truth": gt,
                    "model_answer": clean_answer,
                    "model": args.model,
                    "method": condition,
                    "retrieval_sim_score": round(float(sim_score), 4),
                    "prompt_length": len(prompt),
                    "answer_length": len(clean_answer),
                })

                if len(results) % 20 == 0:
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    print(f"    Progress: {len(results)}/{len(test_data)}")

            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"    Done: {len(results)} predictions -> {output_file.name}")

    print(f"\nAll generation complete for {args.model}. Results in: {OUTPUT_DIR / args.model}")


if __name__ == "__main__":
    main()
