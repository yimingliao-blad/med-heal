#!/usr/bin/env python3
"""
Phase 3: Composite ICL Pilot — Fold 0, 50 Samples

7 composite conditions combining note retrieval with guideline/annotation/full-context:
  1. gtr_note_guideline_pos_k1           — Guideline + note-retrieved positive (Q+A)
  2. gtr_note_guideline_pos_annotated_k1 — Guideline + annotation + note-retrieved positive
  3. gtr_note_neg_annotated_k1           — Note-retrieved negative + GPT-4o annotation
  4. gtr_note_neg_guideline_k1           — Guideline + note-retrieved negative
  5. gtr_note_neg_full_k1                — Note-retrieved negative + full example notes
  6. gtr_note_posneg_annotated_k1        — Annotated contrastive (neg+pos)
  7. gtr_note_posneg_guideline_k1        — Guideline + contrastive (neg+pos)

Usage:
    python generate_pilot12_phase3.py                                      # All
    python generate_pilot12_phase3.py --methods gtr_note_guideline_pos_k1  # Single
    python generate_pilot12_phase3.py --n_samples 20                       # Fewer
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from retrieval_strategies import NoteRetriever, classify_question

PROJECT_ROOT = Path(__file__).parent.parent.parent
VLLM_BASE_URL = "http://localhost:8001/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INDEX_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "indices"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "pilot_phase3" / "fold_0"

AVAILABLE_METHODS = [
    "gtr_note_guideline_pos_k1",
    "gtr_note_guideline_pos_annotated_k1",
    "gtr_note_neg_annotated_k1",
    "gtr_note_neg_guideline_k1",
    "gtr_note_neg_full_k1",
    "gtr_note_posneg_annotated_k1",
    "gtr_note_posneg_guideline_k1",
]

NEEDS_POS = {
    "gtr_note_guideline_pos_k1", "gtr_note_guideline_pos_annotated_k1",
    "gtr_note_posneg_annotated_k1", "gtr_note_posneg_guideline_k1",
}
NEEDS_NEG = {
    "gtr_note_neg_annotated_k1", "gtr_note_neg_guideline_k1", "gtr_note_neg_full_k1",
    "gtr_note_posneg_annotated_k1", "gtr_note_posneg_guideline_k1",
}
NEEDS_ANNOTATION = {
    "gtr_note_guideline_pos_annotated_k1", "gtr_note_neg_annotated_k1",
    "gtr_note_posneg_annotated_k1",
}

# =============================================================================
# CHATML BUILDER & PROMPT TEMPLATES
# =============================================================================

BASE_SYSTEM = "You are a medical expert answering questions about discharge summaries."
USER_TASK = "Discharge Summary:\n{note}\n\nQuestion: {question}\n\nAnswer:"

GUIDELINE_TEXT = (
    "When answering, follow these steps:\n"
    "1. LOCATE: Find the section(s) of the discharge summary relevant to the question.\n"
    "2. EXTRACT: Identify the specific facts, values, or descriptions that directly answer the question.\n"
    "3. VERIFY: Check that your answer is supported by explicit statements in the summary, not inferred or assumed.\n"
    "4. ANSWER: Provide a concise answer using only the information you found.\n\n"
    "Do not speculate, generalize, or add information not present in the discharge summary."
)


def build_chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# 1. Guideline + note-retrieved positive
def prompt_guideline_pos(note, question, example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"{GUIDELINE_TEXT}\n\n"
        "Here is an example demonstrating this approach:\n"
        f"[Question]: {example['question']}\n"
        f"[Answer]: {example['openended_answer']}"
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# 2. Guideline + annotation + note-retrieved positive
def prompt_guideline_pos_annotated(note, question, example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"{GUIDELINE_TEXT}\n\n"
        "Here is an example demonstrating this approach:\n\n"
        f"[Question]: {example['question']}\n"
        f"[Answer]: {example['openended_answer']}\n"
        f"[Why this answer is correct]: {example['annotation']}"
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# 3. Note-retrieved negative + annotation
def prompt_neg_annotated(note, question, neg_example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "Here is an example of a common mistake, with an analysis of what went wrong:\n\n"
        f"[Question]: {neg_example['question']}\n"
        f"[Incorrect Answer]: {neg_example['openended_answer']}\n"
        f"[Correct Answer]: {neg_example['ground_truth']}\n"
        f"[What went wrong]: {neg_example['annotation']}\n\n"
        "Avoid this type of mistake. Base your answer only on what is explicitly stated in the discharge summary."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# 4. Guideline + note-retrieved negative
def prompt_neg_guideline(note, question, neg_example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"{GUIDELINE_TEXT}\n\n"
        "Here is an example of a common mistake to avoid:\n"
        f"[Question]: {neg_example['question']}\n"
        f"[Incorrect Answer]: {neg_example['openended_answer']}\n"
        f"[Correct Answer]: {neg_example['ground_truth']}\n\n"
        "Learn from this mistake and follow the guidelines above."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# 5. Note-retrieved negative + full example notes
def prompt_neg_full(note, question, neg_example, example_note):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "Here is an example showing a common mistake when answering questions about a discharge summary.\n"
        "Study this example to understand what went wrong, "
        "then answer the NEW question below using ONLY the NEW discharge summary.\n\n"
        "=== EXAMPLE (for reference only) ===\n"
        f"[Example Discharge Summary]:\n{example_note}\n\n"
        f"[Example Question]: {neg_example['question']}\n"
        f"[Incorrect Answer]: {neg_example['openended_answer']}\n"
        f"[Correct Answer]: {neg_example['ground_truth']}\n"
        "=== END OF EXAMPLE ===\n\n"
        "Now answer the following question using ONLY the discharge summary provided below. "
        "Do NOT use any information from the example above. "
        "Avoid the same type of mistake shown in the example."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# 6. Annotated contrastive (neg annotation + pos annotation)
def prompt_posneg_annotated(note, question, pos_example, neg_example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "EXAMPLE OF A MISTAKE (and why it was wrong):\n"
        f"[Question]: {neg_example['question']}\n"
        f"[Incorrect Answer]: {neg_example['openended_answer']}\n"
        f"[Correct Answer]: {neg_example['ground_truth']}\n"
        f"[What went wrong]: {neg_example['annotation']}\n\n"
        "EXAMPLE OF A GOOD ANSWER (and why it was correct):\n"
        f"[Question]: {pos_example['question']}\n"
        f"[Answer]: {pos_example['openended_answer']}\n"
        f"[Why this answer is correct]: {pos_example['annotation']}\n\n"
        "Learn from both examples. Base your answer only on what is explicitly stated in the discharge summary."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# 7. Guideline + contrastive (neg + pos)
def prompt_posneg_guideline(note, question, pos_example, neg_example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"{GUIDELINE_TEXT}\n\n"
        "EXAMPLE OF A MISTAKE:\n"
        f"[Question]: {neg_example['question']}\n"
        f"[Incorrect Answer]: {neg_example['openended_answer']}\n"
        f"[Correct Answer]: {neg_example['ground_truth']}\n\n"
        "EXAMPLE OF A GOOD ANSWER:\n"
        f"[Question]: {pos_example['question']}\n"
        f"[Answer]: {pos_example['openended_answer']}\n\n"
        "Follow the guidelines above and avoid the same kinds of mistakes."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# =============================================================================
# ANNOTATION GENERATION (GPT-4o, cached)
# =============================================================================

POSITIVE_ANNOTATION_PROMPT = """You are analyzing why an AI model's answer to a clinical question was correct.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
MODEL'S ANSWER: {openended_answer}

Write a concise 1-2 sentence explanation of WHY this answer is correct.
Focus on what specific information the model correctly identified and what reasoning approach made the answer accurate.

Do not start with "The model..." — write as if explaining to another AI."""

NEGATIVE_ANNOTATION_PROMPT = """You are analyzing why an AI model's answer to a clinical question was incorrect.

QUESTION: {question}
GROUND TRUTH (correct answer): {ground_truth}
MODEL'S INCORRECT ANSWER: {openended_answer}

Write a concise 1-2 sentence explanation of WHAT WENT WRONG.
Focus on the specific type of error (fabricated detail, missed information, over-generalized, confused concepts) and how it relates to misusing or ignoring information in the discharge summary."""


class AnnotationCache:
    """Generate and cache GPT-4o annotations for retrieved examples."""

    def __init__(self, client, cache_file):
        self.client = client
        self.cache_file = Path(cache_file)
        self.cache = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def get_annotation(self, example, pool_type):
        key = f"{pool_type}_{example.get('idx', '')}_{example.get('patient_id', '')}"
        if key in self.cache:
            return self.cache[key]

        if pool_type == "positive":
            prompt = POSITIVE_ANNOTATION_PROMPT.format(
                question=example["question"],
                ground_truth=example.get("ground_truth", ""),
                openended_answer=example["openended_answer"],
            )
        else:
            prompt = NEGATIVE_ANNOTATION_PROMPT.format(
                question=example["question"],
                ground_truth=example.get("ground_truth", ""),
                openended_answer=example["openended_answer"],
            )

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                )
                annotation = resp.choices[0].message.content.strip()
                self.cache[key] = annotation
                self._save()
                time.sleep(0.3)
                return annotation
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    annotation = f"ERROR: {e}"
                    self.cache[key] = annotation
                    self._save()
                    return annotation

    def _save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)


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
    parser = argparse.ArgumentParser(description="Phase 3 Composite ICL Pilot (fold 0, 50 samples)")
    parser.add_argument("--methods", nargs="+", default=AVAILABLE_METHODS)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    global VLLM_BASE_URL
    VLLM_BASE_URL = f"http://localhost:{args.port}/v1"

    if not check_vllm():
        print(f"ERROR: vLLM server not running on port {args.port}")
        print(f"Start with: vllm serve ./models/qwen2.5-7b-instruct --port {args.port} --max-model-len 16384")
        sys.exit(1)
    print("vLLM server running")

    # Check if annotation methods need OpenAI
    needs_ann = any(m in NEEDS_ANNOTATION for m in args.methods)
    openai_client = None
    ann_cache = None
    if needs_ann:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set (needed for annotation methods)")
            sys.exit(1)
        openai_client = OpenAI(api_key=api_key)
        cache_file = OUTPUT_DIR / "annotation_cache.json"
        ann_cache = AnnotationCache(openai_client, cache_file)
        print("OpenAI client ready for annotations")

    # Load notes
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    print(f"Loaded notes for {len(notes_df)} patients")

    # Load test data (fold 0, first n_samples)
    test_file = PROJECT_ROOT / "output" / "folds" / "fold_0" / "test.jsonl"
    test_data = []
    with open(test_file) as f:
        for line in f:
            test_data.append(json.loads(line))
    for i, s in enumerate(test_data):
        if "idx" not in s:
            s["idx"] = i
    test_data = test_data[: args.n_samples]
    print(f"Test data: {len(test_data)} samples (fold 0)")

    # Load retrievers
    fold_index_dir = INDEX_DIR / "fold_0"

    any_needs_pos = any(m in NEEDS_POS for m in args.methods)
    any_needs_neg = any(m in NEEDS_NEG for m in args.methods)

    print("Loading GTR model on CPU...")
    gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    pos_retriever = None
    neg_retriever = None

    if any_needs_pos:
        with open(fold_index_dir / "correct_pool.json") as f:
            correct_pool = json.load(f)
        pos_embs = np.load(fold_index_dir / "gtr_note_embeddings.npy")
        pos_retriever = NoteRetriever(correct_pool, pos_embs, model=gtr_model)
        print(f"Positive pool: {len(correct_pool)} examples")

    if any_needs_neg:
        with open(fold_index_dir / "incorrect_pool.json") as f:
            incorrect_pool = json.load(f)
        neg_embs = np.load(fold_index_dir / "gtr_note_incorrect_embeddings.npy")
        neg_retriever = NoteRetriever(incorrect_pool, neg_embs, model=gtr_model)
        print(f"Negative pool: {len(incorrect_pool)} examples")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_methods = len(args.methods)

    for method_idx, method in enumerate(args.methods, 1):
        output_file = OUTPUT_DIR / f"{method}_generated.csv"

        # Skip if already complete
        if output_file.exists():
            existing = pd.read_csv(output_file)
            if len(existing) >= len(test_data):
                print(f"\n  [{method_idx}/{total_methods}] {method}: Already complete ({len(existing)} rows)")
                continue

        print(f"\n  [{method_idx}/{total_methods}] {method}")

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

            if method == "gtr_note_guideline_pos_k1":
                retrieved = pos_retriever.retrieve(note, k=1)
                prompt = prompt_guideline_pos(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_note_guideline_pos_annotated_k1":
                retrieved = pos_retriever.retrieve(note, k=1)
                ex = retrieved[0][0]
                annotation = ann_cache.get_annotation(ex, "positive")
                ex_with_ann = {**ex, "annotation": annotation}
                prompt = prompt_guideline_pos_annotated(note, question, ex_with_ann)
                sim_score = retrieved[0][1]

            elif method == "gtr_note_neg_annotated_k1":
                retrieved = neg_retriever.retrieve(note, k=1)
                ex = retrieved[0][0]
                annotation = ann_cache.get_annotation(ex, "negative")
                ex_with_ann = {**ex, "annotation": annotation}
                prompt = prompt_neg_annotated(note, question, ex_with_ann)
                sim_score = retrieved[0][1]

            elif method == "gtr_note_neg_guideline_k1":
                retrieved = neg_retriever.retrieve(note, k=1)
                prompt = prompt_neg_guideline(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_note_neg_full_k1":
                retrieved = neg_retriever.retrieve(note, k=1)
                ex = retrieved[0][0]
                ex_note = assemble_note(ex, notes_df)
                prompt = prompt_neg_full(note, question, ex, ex_note)
                sim_score = retrieved[0][1]

            elif method == "gtr_note_posneg_annotated_k1":
                pos_retrieved = pos_retriever.retrieve(note, k=1)
                neg_retrieved = neg_retriever.retrieve(note, k=1)
                pos_ex = pos_retrieved[0][0]
                neg_ex = neg_retrieved[0][0]
                pos_ann = ann_cache.get_annotation(pos_ex, "positive")
                neg_ann = ann_cache.get_annotation(neg_ex, "negative")
                pos_with_ann = {**pos_ex, "annotation": pos_ann}
                neg_with_ann = {**neg_ex, "annotation": neg_ann}
                prompt = prompt_posneg_annotated(note, question, pos_with_ann, neg_with_ann)
                sim_score = (pos_retrieved[0][1] + neg_retrieved[0][1]) / 2

            elif method == "gtr_note_posneg_guideline_k1":
                pos_retrieved = pos_retriever.retrieve(note, k=1)
                neg_retrieved = neg_retriever.retrieve(note, k=1)
                prompt = prompt_posneg_guideline(
                    note, question, pos_retrieved[0][0], neg_retrieved[0][0]
                )
                sim_score = (pos_retrieved[0][1] + neg_retrieved[0][1]) / 2

            else:
                raise ValueError(f"Unknown method: {method}")

            answer = generate_vllm(prompt)

            results.append({
                "idx": idx,
                "patient_id": sample.get("patient_id", ""),
                "fold_id": 0,
                "question": question,
                "question_type": q_type,
                "ground_truth": gt,
                "model_answer": answer,
                "method": method,
                "retrieval_sim_score": round(float(sim_score), 4),
                "prompt_length": len(prompt),
            })

            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(output_file, index=False)
                print(f"    Progress: {len(results)}/{len(test_data)}")

        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"    Done: {len(results)} predictions -> {output_file.name}")

    print(f"\nAll generation complete. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
