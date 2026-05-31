#!/usr/bin/env python3
"""
Phase 2: Fullscale Note-Retrieval ICL — 5-Fold CV

4 conditions using note-based retrieval across all 5 folds:
  1. gtr_note_pos_k1         — Note-retrieved positive example (Q+A only)
  2. gtr_note_fullctx_pos_k1 — Note-retrieved positive + full example notes
  3. gtr_note_neg_k1         — Note-retrieved negative example (mistake to avoid)
  4. gtr_note_posneg_k1      — Note-retrieved contrastive (negative + positive)

Usage:
    python generate_fullscale_pilot12.py                                    # All
    python generate_fullscale_pilot12.py --folds 0 --methods gtr_note_pos_k1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from retrieval_strategies import NoteRetriever, TypeNoteRetriever, classify_question

PROJECT_ROOT = Path(__file__).parent.parent.parent
VLLM_BASE_URL = "http://localhost:8001/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INDEX_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "indices"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "fullscale"

AVAILABLE_METHODS = [
    "gtr_note_pos_k1", "gtr_note_fullctx_pos_k1",
    "gtr_note_neg_k1", "gtr_note_posneg_k1",
    "gtr_type_note_pos_k1",
    "gtr_note_any_unlabeled_k1", "gtr_note_any_labeled_k1",
    "gtr_note_pos_k2", "gtr_note_pos_k3", "gtr_note_pos_k4", "gtr_note_pos_k5",
]

NEEDS_POS = {
    "gtr_note_pos_k1", "gtr_note_fullctx_pos_k1", "gtr_note_posneg_k1", "gtr_type_note_pos_k1",
    "gtr_note_pos_k2", "gtr_note_pos_k3", "gtr_note_pos_k4", "gtr_note_pos_k5",
}
NEEDS_NEG = {"gtr_note_neg_k1", "gtr_note_posneg_k1"}
NEEDS_MIXED = {"gtr_note_any_unlabeled_k1", "gtr_note_any_labeled_k1"}

# =============================================================================
# CHATML BUILDER & PROMPT TEMPLATES
# =============================================================================

BASE_SYSTEM = "You are a medical expert answering questions about discharge summaries."
USER_TASK = "Discharge Summary:\n{note}\n\nQuestion: {question}\n\nAnswer:"


def build_chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def prompt_positive(note, question, example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "Here is an example of a good answer:\n"
        f"[Question]: {example['question']}\n"
        f"[Answer]: {example['openended_answer']}\n\n"
        "Apply the same precision and directness to your answer."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def prompt_fullctx_positive(note, question, example, example_note):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "Here is an example showing how to correctly answer a question about a discharge summary.\n"
        "Study this example to understand what a correct answer looks like, "
        "then answer the NEW question below using ONLY the NEW discharge summary.\n\n"
        "=== EXAMPLE (for reference only) ===\n"
        f"[Example Discharge Summary]:\n{example_note}\n\n"
        f"[Example Question]: {example['question']}\n"
        f"[Correct Answer]: {example['openended_answer']}\n"
        "=== END OF EXAMPLE ===\n\n"
        "Now answer the following question using ONLY the discharge summary provided below. "
        "Do NOT use any information from the example above."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def prompt_negative(note, question, neg_example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "Here is an example of a common mistake to avoid:\n"
        f"[Question]: {neg_example['question']}\n"
        f"[Incorrect Answer]: {neg_example['openended_answer']}\n"
        f"[Correct Answer]: {neg_example['ground_truth']}\n\n"
        "Learn from this mistake. Answer based only on what is explicitly stated in the discharge summary."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def prompt_contrastive(note, question, pos_example, neg_example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "EXAMPLE OF A MISTAKE:\n"
        f"[Question]: {neg_example['question']}\n"
        f"[Incorrect Answer]: {neg_example['openended_answer']}\n"
        f"[Correct Answer]: {neg_example['ground_truth']}\n\n"
        "EXAMPLE OF A GOOD ANSWER:\n"
        f"[Question]: {pos_example['question']}\n"
        f"[Answer]: {pos_example['openended_answer']}\n\n"
        "Apply the same precision and avoid the same kinds of mistakes."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def prompt_positive_multi(note, question, examples):
    """Multi-shot: show k positive examples retrieved by note similarity."""
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += (
            f"Example {i}:\n"
            f"[Question]: {ex['question']}\n"
            f"[Answer]: {ex['openended_answer']}\n\n"
        )
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"Here are {len(examples)} examples of good answers:\n\n"
        f"{examples_text}"
        "Apply the same precision and directness to your answer."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def prompt_any_unlabeled(note, question, example):
    """Mixed pool: show example Q+A without labeling whether it's correct or incorrect."""
    system = (
        f"{BASE_SYSTEM}\n\n"
        "Here is an example from a similar patient case:\n"
        f"[Question]: {example['question']}\n"
        f"[Answer]: {example['openended_answer']}\n\n"
        "Now answer the following question based only on the discharge summary provided."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def prompt_any_labeled(note, question, example):
    """Mixed pool: show example Q+A and explicitly label whether it was correct or incorrect."""
    if example.get("is_correct", True):
        system = (
            f"{BASE_SYSTEM}\n\n"
            "Here is an example of a CORRECT answer from a similar patient case:\n"
            f"[Question]: {example['question']}\n"
            f"[Answer]: {example['openended_answer']}\n\n"
            "Apply the same precision and directness to your answer."
        )
    else:
        system = (
            f"{BASE_SYSTEM}\n\n"
            "Here is an example of an INCORRECT answer from a similar patient case:\n"
            f"[Question]: {example['question']}\n"
            f"[Incorrect Answer]: {example['openended_answer']}\n"
            f"[Correct Answer]: {example['ground_truth']}\n\n"
            "Learn from this mistake. Answer based only on what is explicitly stated in the discharge summary."
        )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# =============================================================================
# GENERATION HELPERS
# =============================================================================

def check_vllm():
    try:
        return requests.get(f"{VLLM_BASE_URL}/models", timeout=5).status_code == 200
    except Exception:
        return False


def generate_vllm(prompt: str) -> str:
    response = requests.post(
        f"{VLLM_BASE_URL}/completions",
        json={"model": MODEL_NAME, "prompt": prompt, "max_tokens": 2048, "temperature": 0.1},
        timeout=300,
    )
    if response.status_code != 200:
        raise Exception(f"vLLM error: {response.text}")
    return response.json()["choices"][0]["text"].strip()


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
    parser = argparse.ArgumentParser(description="Fullscale Pilot 12 generation (5-fold CV)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=AVAILABLE_METHODS)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    global VLLM_BASE_URL
    VLLM_BASE_URL = f"http://localhost:{args.port}/v1"

    if not check_vllm():
        print(f"ERROR: vLLM server not running on port {args.port}")
        print(f"Start with: vllm serve ./models/qwen2.5-7b-instruct --port {args.port} --max-model-len 16384")
        sys.exit(1)
    print("vLLM server running")

    # Load notes
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    print(f"Loaded notes for {len(notes_df)} patients")

    # Shared GTR model for all NoteRetrievers
    print("Loading GTR model on CPU...")
    gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    folds_dir = PROJECT_ROOT / "output" / "folds"
    any_needs_pos = any(m in NEEDS_POS for m in args.methods)
    any_needs_neg = any(m in NEEDS_NEG for m in args.methods)
    any_needs_mixed = any(m in NEEDS_MIXED for m in args.methods)

    total_combos = len(args.methods) * len(args.folds)
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

        fold_index_dir = INDEX_DIR / f"fold_{fold_id}"

        # Load retrievers
        pos_retriever = None
        neg_retriever = None
        type_note_retriever = None
        mixed_retriever = None

        if any_needs_pos or any_needs_mixed:
            with open(fold_index_dir / "correct_pool.json") as f:
                correct_pool = json.load(f)
            pos_embs = np.load(fold_index_dir / "gtr_note_embeddings.npy")
            if any_needs_pos:
                pos_retriever = NoteRetriever(correct_pool, pos_embs, model=gtr_model)
            if "gtr_type_note_pos_k1" in args.methods:
                type_note_retriever = TypeNoteRetriever(correct_pool, pos_embs, model=gtr_model)

        if any_needs_neg or any_needs_mixed:
            with open(fold_index_dir / "incorrect_pool.json") as f:
                incorrect_pool = json.load(f)
            neg_embs = np.load(fold_index_dir / "gtr_note_incorrect_embeddings.npy")
            if any_needs_neg:
                neg_retriever = NoteRetriever(incorrect_pool, neg_embs, model=gtr_model)

        if any_needs_mixed:
            # Build combined pool: mark each example with is_correct flag
            mixed_pool = []
            for ex in correct_pool:
                mixed_pool.append({**ex, "is_correct": True})
            for ex in incorrect_pool:
                mixed_pool.append({**ex, "is_correct": False})
            mixed_embs = np.concatenate([pos_embs, neg_embs], axis=0)
            mixed_retriever = NoteRetriever(mixed_pool, mixed_embs, model=gtr_model)
            print(f"  Mixed pool: {len(mixed_pool)} examples ({len(correct_pool)} correct + {len(incorrect_pool)} incorrect)")

        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}: {len(test_data)} test samples")
        if pos_retriever:
            print(f"  Positive pool: {len(pos_retriever.correct_pool)} examples")
        if neg_retriever:
            print(f"  Negative pool: {len(neg_retriever.correct_pool)} examples")
        if mixed_retriever:
            print(f"  Mixed pool: {len(mixed_retriever.correct_pool)} examples")
        print(f"{'='*60}")

        for method in args.methods:
            completed += 1
            fold_out = OUTPUT_DIR / f"fold_{fold_id}"
            fold_out.mkdir(parents=True, exist_ok=True)
            output_file = fold_out / f"{method}_generated.csv"

            # Skip if already complete
            if output_file.exists():
                existing = pd.read_csv(output_file)
                if len(existing) >= len(test_data):
                    print(f"\n  [{completed}/{total_combos}] fold_{fold_id}/{method}: Already complete ({len(existing)} rows)")
                    continue

            print(f"\n  [{completed}/{total_combos}] fold_{fold_id}/{method}")

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
                q_type = classify_question(question)

                if method == "gtr_note_pos_k1":
                    retrieved = pos_retriever.retrieve(note, k=1)
                    prompt = prompt_positive(note, question, retrieved[0][0])
                    sim_score = retrieved[0][1]

                elif method == "gtr_note_fullctx_pos_k1":
                    retrieved = pos_retriever.retrieve(note, k=1)
                    ex = retrieved[0][0]
                    ex_note = assemble_note(ex, notes_df)
                    prompt = prompt_fullctx_positive(note, question, ex, ex_note)
                    sim_score = retrieved[0][1]

                elif method == "gtr_note_neg_k1":
                    retrieved = neg_retriever.retrieve(note, k=1)
                    prompt = prompt_negative(note, question, retrieved[0][0])
                    sim_score = retrieved[0][1]

                elif method == "gtr_note_posneg_k1":
                    pos_retrieved = pos_retriever.retrieve(note, k=1)
                    neg_retrieved = neg_retriever.retrieve(note, k=1)
                    prompt = prompt_contrastive(
                        note, question, pos_retrieved[0][0], neg_retrieved[0][0]
                    )
                    sim_score = (pos_retrieved[0][1] + neg_retrieved[0][1]) / 2

                elif method == "gtr_type_note_pos_k1":
                    retrieved = type_note_retriever.retrieve(note, question, k=1)
                    prompt = prompt_positive(note, question, retrieved[0][0])
                    sim_score = retrieved[0][1]

                elif method == "gtr_note_any_unlabeled_k1":
                    retrieved = mixed_retriever.retrieve(note, k=1)
                    prompt = prompt_any_unlabeled(note, question, retrieved[0][0])
                    sim_score = retrieved[0][1]

                elif method == "gtr_note_any_labeled_k1":
                    retrieved = mixed_retriever.retrieve(note, k=1)
                    prompt = prompt_any_labeled(note, question, retrieved[0][0])
                    sim_score = retrieved[0][1]

                elif method.startswith("gtr_note_pos_k") and method != "gtr_note_pos_k1":
                    k = int(method.split("_k")[1])
                    retrieved = pos_retriever.retrieve(note, k=k)
                    examples = [r[0] for r in retrieved]
                    prompt = prompt_positive_multi(note, question, examples)
                    sim_score = sum(r[1] for r in retrieved) / len(retrieved)

                else:
                    raise ValueError(f"Unknown method: {method}")

                answer = generate_vllm(prompt)

                results.append({
                    "idx": idx,
                    "patient_id": sample.get("patient_id", ""),
                    "fold_id": fold_id,
                    "question": question,
                    "question_type": q_type,
                    "ground_truth": gt,
                    "model_answer": answer,
                    "method": method,
                    "retrieval_sim_score": round(float(sim_score), 4),
                    "prompt_length": len(prompt),
                })

                if len(results) % 20 == 0:
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    print(f"    Progress: {len(results)}/{len(test_data)}")

            pd.DataFrame(results).to_csv(output_file, index=False)
            correct_count = sum(1 for r in results if r.get("model_answer"))
            print(f"    Done: {len(results)} predictions -> {output_file.name}")

    print(f"\nAll generation complete. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
