#!/usr/bin/env python3
"""Phase 2: Build error-typed retrieval sub-indices.

Partitions existing gtr_note_incorrect_embeddings.npy by primary error type,
creating sub-pools for targeted contrastive retrieval.

Usage:
    python build_error_indices.py --folds 0 1 2 3 4
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
BIO_INDEX_DIR = PROJECT_ROOT / "output" / "fullscale_4_biomistral" / "indices"
ERROR_DIR = PROJECT_ROOT / "output" / "step8" / "error_classification"

ERROR_TYPES = [
    "omission",
    "hallucination",
    "reasoning_failure",
    "specificity",
    "context_confusion",
    "temporal_error",
]


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Build error sub-indices")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    for fold_id in args.folds:
        fold_dir = BIO_INDEX_DIR / f"fold_{fold_id}"
        error_file = ERROR_DIR / f"fold_{fold_id}_errors.json"
        emb_file = fold_dir / "gtr_note_incorrect_embeddings.npy"

        if not error_file.exists() or not emb_file.exists():
            print(f"Fold {fold_id}: missing files, skipping")
            continue

        with open(error_file) as f:
            errors = json.load(f)
        embeddings = np.load(emb_file)

        assert len(errors) == embeddings.shape[0], (
            f"Fold {fold_id}: mismatch {len(errors)} errors vs {embeddings.shape[0]} embeddings"
        )

        # Create output directory
        sub_dir = fold_dir / "error_subindex"
        sub_dir.mkdir(exist_ok=True)

        print(f"Fold {fold_id}: {len(errors)} total incorrect examples")

        for etype in ERROR_TYPES:
            indices = [i for i, e in enumerate(errors) if e.get("primary_error") == etype]

            if not indices:
                print(f"  {etype}: 0 examples (skipping)")
                continue

            sub_pool = [errors[i] for i in indices]
            sub_emb = embeddings[indices]

            pool_file = sub_dir / f"{etype}_pool.json"
            emb_file_out = sub_dir / f"{etype}_embeddings.npy"

            with open(pool_file, "w") as f:
                json.dump(sub_pool, f)
            np.save(emb_file_out, sub_emb)

            print(f"  {etype}: {len(indices)} examples -> {emb_file_out.name}")

        # Also save unclassified if any
        unclassified = [i for i, e in enumerate(errors) if e.get("primary_error") is None]
        if unclassified:
            print(f"  unclassified: {len(unclassified)} examples (not indexed)")

    print("\nDone. Sub-indices saved to each fold's error_subindex/ directory.")


if __name__ == "__main__":
    main()
