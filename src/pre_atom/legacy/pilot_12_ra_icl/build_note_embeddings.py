#!/usr/bin/env python3
"""
Build discharge note embeddings for note-based retrieval.

Encodes the first ~512 tokens of each patient's discharge notes using GTR-T5-Base.
Saves embeddings alongside existing question-based indices.

Usage:
    python build_note_embeddings.py                          # All folds, both pools
    python build_note_embeddings.py --fold 0                 # Single fold
    python build_note_embeddings.py --pool_type incorrect     # Incorrect pool only
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "indices"
FULLSCALE2_DIR = PROJECT_ROOT / "output" / "fullscale_2"


def assemble_note_text(patient_id, notes_lookup):
    """Get concatenated note text for a patient."""
    if str(patient_id) not in notes_lookup:
        return ""
    row = notes_lookup[str(patient_id)]
    parts = []
    for i in [1, 2, 3]:
        k = f"note_{i}"
        if k in row and row[k] and str(row[k]).strip().lower() != "nan":
            parts.append(str(row[k]).strip())
    return "\n\n".join(parts)


def build_pool_embeddings(pool, pool_name, fold_dir, model, notes_lookup):
    """Build and save note embeddings for a pool."""
    note_texts = []
    missing = 0
    for ex in pool:
        note = assemble_note_text(ex["patient_id"], notes_lookup)
        if not note:
            missing += 1
        note_texts.append(note)

    if missing > 0:
        print(f"  Warning: {missing} examples have no notes")

    print(f"  Encoding {len(note_texts)} {pool_name} note texts...")
    embeddings = model.encode(note_texts, batch_size=32, show_progress_bar=True)

    if pool_name == "correct":
        emb_file = fold_dir / "gtr_note_embeddings.npy"
    else:
        emb_file = fold_dir / "gtr_note_incorrect_embeddings.npy"

    np.save(emb_file, embeddings)
    print(f"  Saved: {emb_file} shape={embeddings.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Build note embeddings")
    parser.add_argument("--fold", default="all", help="Fold to build (0-4 or 'all')")
    parser.add_argument("--pool_type", default="both", choices=["correct", "incorrect", "both"],
                        help="Which pool to build embeddings for")
    args = parser.parse_args()

    print("Loading GTR model on CPU...")
    model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    # Load notes lookup
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    notes_lookup = {str(r["patient_id"]): r.to_dict() for _, r in notes_df.iterrows()}
    print(f"Loaded notes for {len(notes_lookup)} patients")

    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]
    build_correct = args.pool_type in ("correct", "both")
    build_incorrect = args.pool_type in ("incorrect", "both")

    for fold_id in folds:
        fold_dir = OUTPUT_DIR / f"fold_{fold_id}"

        # --- Correct pool ---
        if build_correct:
            emb_file = fold_dir / "gtr_note_embeddings.npy"
            if emb_file.exists():
                print(f"\nFold {fold_id} correct: Already exists, skipping")
            else:
                with open(fold_dir / "correct_pool.json") as f:
                    correct_pool = json.load(f)
                print(f"\nFold {fold_id} correct: {len(correct_pool)} examples")
                note_embeddings = build_pool_embeddings(
                    correct_pool, "correct", fold_dir, model, notes_lookup
                )
                # Type-partitioned note embeddings
                from retrieval_strategies import classify_question
                type_dir = fold_dir / "type_subindex"
                type_indices = {}
                for i, ex in enumerate(correct_pool):
                    q_type = classify_question(ex["question"])
                    type_indices.setdefault(q_type, []).append(i)
                for q_type, indices in type_indices.items():
                    type_note_embs = note_embeddings[indices]
                    np.save(type_dir / f"{q_type}_note_embeddings.npy", type_note_embs)
                    print(f"    {q_type}: {len(indices)} note embeddings")

        # --- Incorrect pool ---
        if build_incorrect:
            # Copy incorrect_pool.json from fullscale_2 if not present
            incorrect_pool_dst = fold_dir / "incorrect_pool.json"
            incorrect_pool_src = FULLSCALE2_DIR / f"fold_{fold_id}" / "incorrect_pool.json"
            if not incorrect_pool_dst.exists():
                shutil.copy2(incorrect_pool_src, incorrect_pool_dst)
                print(f"\nFold {fold_id}: Copied incorrect_pool.json from fullscale_2")

            with open(incorrect_pool_dst) as f:
                incorrect_pool = json.load(f)
            print(f"Fold {fold_id} incorrect: {len(incorrect_pool)} examples")
            build_pool_embeddings(incorrect_pool, "incorrect", fold_dir, model, notes_lookup)

    print("\nDone!")


if __name__ == "__main__":
    main()
