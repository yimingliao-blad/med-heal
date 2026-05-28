#!/usr/bin/env python3
"""
Phase 1: Generate Pilot 12 Predictions — RA-ICL Retriever Comparison

7 retrieval-based conditions on fold 0, first 50 test samples:
  R1. bm25_pos_k1         — BM25 retrieval, 1-shot positive
  R2. gtr_pos_k1          — GTR retrieval, 1-shot positive
  R3. kate_pos_k1         — KATE retrieval, 1-shot positive
  R4. gtr_pos_k2          — GTR retrieval, 2-shot positive
  R5. gtr_pos_k3          — GTR retrieval, 3-shot positive
  R6. gtr_type_pos_k1     — GTR within-type retrieval, 1-shot positive
  R7. gtr_guideline_pos_k1 — GTR + guideline prompt, 1-shot positive

Usage:
    python generate_pilot12.py                              # All conditions
    python generate_pilot12.py --methods gtr_pos_k1         # Single condition
    python generate_pilot12.py --n_samples 20               # Fewer samples
    python generate_pilot12.py --port 8001                  # Custom vLLM port
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from retrieval_strategies import (
    BM25Retriever,
    GTRRetriever,
    KATERetriever,
    NoteRetriever,
    TypeNoteRetriever,
    classify_question,
    load_retriever_from_index,
    load_type_filtered_retriever,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
VLLM_BASE_URL = "http://localhost:8001/v1"
MODEL_NAME = "./models/qwen2.5-7b-instruct"
INDEX_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "indices"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "pilot" / "fold_0"

AVAILABLE_METHODS = [
    "bm25_pos_k1", "gtr_pos_k1", "kate_pos_k1",
    "gtr_pos_k2", "gtr_pos_k3",
    "gtr_type_pos_k1", "gtr_guideline_pos_k1",
    "gtr_note_pos_k1", "gtr_type_note_pos_k1",
    "gtr_note_context_pos_k1", "gtr_note_fullctx_pos_k1",
]

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


def build_positive_k1_prompt(note, question, example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        "Here is an example of a good answer:\n"
        f"[Question]: {example['question']}\n"
        f"[Answer]: {example['openended_answer']}\n\n"
        "Apply the same precision and directness to your answer."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def build_positive_kn_prompt(note, question, examples):
    example_text = ""
    for i, ex in enumerate(examples, 1):
        example_text += (
            f"[Example {i}]\n"
            f"[Question]: {ex['question']}\n"
            f"[Answer]: {ex['openended_answer']}\n\n"
        )
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"Here are examples of good answers:\n\n"
        f"{example_text}"
        "Apply the same precision and directness to your answers."
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


def build_note_context_positive_prompt(note, question, example, example_note):
    """Include the example's discharge note so the model sees full context of why the answer is correct."""
    # Truncate example note to ~500 words to stay within context window
    words = example_note.split()
    if len(words) > 500:
        example_note = " ".join(words[:500]) + "\n[... truncated ...]"
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


def build_note_fullctx_positive_prompt(note, question, example, example_note):
    """Include the FULL example discharge note (no truncation) for maximum context."""
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


def build_guideline_positive_prompt(note, question, example):
    system = (
        f"{BASE_SYSTEM}\n\n"
        f"{GUIDELINE_TEXT}\n\n"
        "Here is an example demonstrating this approach:\n"
        f"[Question]: {example['question']}\n"
        f"[Answer]: {example['openended_answer']}"
    )
    return build_chatml(system, USER_TASK.format(note=note, question=question))


# =============================================================================
# GENERATION
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


def main():
    parser = argparse.ArgumentParser(description="Generate Pilot 12 RA-ICL predictions")
    parser.add_argument("--methods", nargs="+", default=AVAILABLE_METHODS)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    global VLLM_BASE_URL
    VLLM_BASE_URL = f"http://localhost:{args.port}/v1"

    if not check_vllm():
        print(f"ERROR: vLLM server not running on port {args.port}")
        sys.exit(1)
    print("vLLM server running")

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

    # Load correct pool
    fold_index_dir = INDEX_DIR / "fold_0"
    with open(fold_index_dir / "correct_pool.json") as f:
        correct_pool = json.load(f)
    print(f"Correct pool: {len(correct_pool)} examples")

    # Load retrievers (lazy — only load what we need)
    retrievers = {}

    def get_retriever(name):
        if name not in retrievers:
            if name == "bm25":
                retrievers[name] = BM25Retriever(correct_pool)
                print(f"  Loaded BM25 retriever")
            elif name == "gtr":
                retrievers[name] = load_retriever_from_index("gtr", fold_index_dir, correct_pool)
                print(f"  Loaded GTR retriever")
            elif name == "kate":
                retrievers[name] = load_retriever_from_index("kate", fold_index_dir, correct_pool)
                print(f"  Loaded KATE retriever")
            elif name == "gtr_type":
                retrievers[name] = load_type_filtered_retriever(fold_index_dir, correct_pool)
                print(f"  Loaded TypeFiltered GTR retriever")
            elif name == "gtr_note":
                note_emb_file = fold_index_dir / "gtr_note_embeddings.npy"
                note_embs = np.load(note_emb_file)
                retrievers[name] = NoteRetriever(correct_pool, note_embs)
                print(f"  Loaded Note retriever ({note_embs.shape})")
            elif name == "gtr_type_note":
                note_emb_file = fold_index_dir / "gtr_note_embeddings.npy"
                note_embs = np.load(note_emb_file)
                retrievers[name] = TypeNoteRetriever(correct_pool, note_embs)
                print(f"  Loaded TypeNote retriever")
        return retrievers[name]

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

            # Retrieve examples and build prompt based on method
            if method == "bm25_pos_k1":
                retriever = get_retriever("bm25")
                retrieved = retriever.retrieve(question, k=1)
                prompt = build_positive_k1_prompt(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_pos_k1":
                retriever = get_retriever("gtr")
                retrieved = retriever.retrieve(question, k=1)
                prompt = build_positive_k1_prompt(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "kate_pos_k1":
                retriever = get_retriever("kate")
                retrieved = retriever.retrieve(question, k=1)
                prompt = build_positive_k1_prompt(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_pos_k2":
                retriever = get_retriever("gtr")
                retrieved = retriever.retrieve(question, k=2)
                examples = [r[0] for r in retrieved]
                prompt = build_positive_kn_prompt(note, question, examples)
                sim_score = np.mean([r[1] for r in retrieved])

            elif method == "gtr_pos_k3":
                retriever = get_retriever("gtr")
                retrieved = retriever.retrieve(question, k=3)
                examples = [r[0] for r in retrieved]
                prompt = build_positive_kn_prompt(note, question, examples)
                sim_score = np.mean([r[1] for r in retrieved])

            elif method == "gtr_type_pos_k1":
                retriever = get_retriever("gtr_type")
                retrieved = retriever.retrieve(question, k=1)
                prompt = build_positive_k1_prompt(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_guideline_pos_k1":
                retriever = get_retriever("gtr")
                retrieved = retriever.retrieve(question, k=1)
                prompt = build_guideline_positive_prompt(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_note_pos_k1":
                retriever = get_retriever("gtr_note")
                retrieved = retriever.retrieve(note, k=1)
                prompt = build_positive_k1_prompt(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_type_note_pos_k1":
                retriever = get_retriever("gtr_type_note")
                retrieved = retriever.retrieve(note, question, k=1)
                prompt = build_positive_k1_prompt(note, question, retrieved[0][0])
                sim_score = retrieved[0][1]

            elif method == "gtr_note_context_pos_k1":
                retriever = get_retriever("gtr_note")
                retrieved = retriever.retrieve(note, k=1)
                ex = retrieved[0][0]
                ex_note = assemble_note(ex, notes_df)
                prompt = build_note_context_positive_prompt(note, question, ex, ex_note)
                sim_score = retrieved[0][1]

            elif method == "gtr_note_fullctx_pos_k1":
                retriever = get_retriever("gtr_note")
                retrieved = retriever.retrieve(note, k=1)
                ex = retrieved[0][0]
                ex_note = assemble_note(ex, notes_df)
                prompt = build_note_fullctx_positive_prompt(note, question, ex, ex_note)
                sim_score = retrieved[0][1]

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
                "retrieval_sim_score": round(sim_score, 4),
                "prompt_length": len(prompt),
            })

            if len(results) % 20 == 0:
                pd.DataFrame(results).to_csv(output_file, index=False)
                print(f"    Progress: {len(results)}/{len(test_data)}")

        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"    Done: {len(results)} predictions -> {output_file.name}")

    print(f"\nAll generation complete. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
