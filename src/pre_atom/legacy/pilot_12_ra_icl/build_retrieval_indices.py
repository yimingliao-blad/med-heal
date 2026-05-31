#!/usr/bin/env python3
"""
Phase 0: Build Retrieval Indices for Pilot 12 RA-ICL

Builds BM25, GTR, and KATE indices for all 5 folds from the correct-only training pools.
Also builds type-partitioned sub-indices for the hybrid type+retrieval condition.

Usage:
    python build_retrieval_indices.py              # All folds
    python build_retrieval_indices.py --fold 0     # Single fold
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval_strategies import BM25Retriever, classify_question

PROJECT_ROOT = Path(__file__).parent.parent.parent
CORRECT_POOL_DIR = PROJECT_ROOT / "output" / "fullscale_2"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "indices"


def load_correct_pool(fold_id: int) -> list[dict]:
    pool_file = CORRECT_POOL_DIR / f"fold_{fold_id}" / "correct_pool.json"
    with open(pool_file) as f:
        return json.load(f)


def build_fold_indices(fold_id: int, gtr_model, kate_model):
    fold_dir = OUTPUT_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FOLD {fold_id}")
    print(f"{'='*60}")

    pool = load_correct_pool(fold_id)
    print(f"  Correct pool: {len(pool)} examples")

    questions = [ex["question"] for ex in pool]

    # 1. BM25 — just save the pool (BM25 is built at query time, fast enough)
    print("  Building BM25 index...")
    bm25_retriever = BM25Retriever(pool)
    with open(fold_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever.bm25, f)

    # 2. GTR embeddings
    print("  Computing GTR embeddings...")
    gtr_embeddings = gtr_model.encode(questions, batch_size=64, show_progress_bar=True)
    np.save(fold_dir / "gtr_correct_embeddings.npy", gtr_embeddings)

    # 3. KATE embeddings
    print("  Computing KATE embeddings...")
    kate_embeddings = kate_model.encode(questions, batch_size=64, show_progress_bar=True)
    np.save(fold_dir / "kate_correct_embeddings.npy", kate_embeddings)

    # 4. Type-partitioned sub-indices (GTR only, for R6 condition)
    print("  Building type-partitioned sub-indices...")
    type_dir = fold_dir / "type_subindex"
    type_dir.mkdir(exist_ok=True)

    type_pools = {}
    for ex in pool:
        q_type = classify_question(ex["question"])
        type_pools.setdefault(q_type, []).append(ex)

    for q_type, type_pool in type_pools.items():
        type_questions = [ex["question"] for ex in type_pool]
        type_embeddings = gtr_model.encode(type_questions, batch_size=64, show_progress_bar=False)
        np.save(type_dir / f"{q_type}_gtr_embeddings.npy", type_embeddings)
        with open(type_dir / f"{q_type}_pool.pkl", "wb") as f:
            pickle.dump(type_pool, f)
        print(f"    {q_type}: {len(type_pool)} examples")

    # 5. Save pool reference
    with open(fold_dir / "correct_pool.json", "w") as f:
        json.dump(pool, f, indent=2, default=str)

    # 6. Retrieval quality check — show top-3 for 5 sample test questions
    test_file = PROJECT_ROOT / "output" / "folds" / f"fold_{fold_id}" / "test.jsonl"
    test_data = []
    with open(test_file) as f:
        for line in f:
            test_data.append(json.loads(line))

    sample_indices = np.linspace(0, min(len(test_data) - 1, 49), 5, dtype=int)
    quality_lines = [f"Retrieval Quality Check — Fold {fold_id}\n{'='*80}\n"]

    for idx in sample_indices:
        test_q = test_data[idx]["question"]
        quality_lines.append(f"\nTEST QUESTION [{idx}]: {test_q}")
        quality_lines.append("-" * 60)

        # GTR retrieval
        q_emb = gtr_model.encode([test_q])
        from sklearn.neighbors import NearestNeighbors

        knn = NearestNeighbors(n_neighbors=3, metric="cosine")
        knn.fit(gtr_embeddings)
        distances, indices = knn.kneighbors(q_emb)
        quality_lines.append("  GTR top-3:")
        for rank, (dist, train_idx) in enumerate(zip(distances[0], indices[0])):
            sim = 1 - dist
            quality_lines.append(f"    [{rank+1}] sim={sim:.4f}: {pool[train_idx]['question'][:120]}")

        # BM25 retrieval
        results = bm25_retriever.retrieve(test_q, k=3)
        quality_lines.append("  BM25 top-3:")
        for rank, (ex, score) in enumerate(results):
            quality_lines.append(f"    [{rank+1}] score={score:.4f}: {ex['question'][:120]}")

        # KATE retrieval
        q_emb_kate = kate_model.encode([test_q])
        knn_kate = NearestNeighbors(n_neighbors=3, metric="cosine")
        knn_kate.fit(kate_embeddings)
        distances_k, indices_k = knn_kate.kneighbors(q_emb_kate)
        quality_lines.append("  KATE top-3:")
        for rank, (dist, train_idx) in enumerate(zip(distances_k[0], indices_k[0])):
            sim = 1 - dist
            quality_lines.append(f"    [{rank+1}] sim={sim:.4f}: {pool[train_idx]['question'][:120]}")

    quality_text = "\n".join(quality_lines)
    with open(fold_dir / "retrieval_quality_check.txt", "w") as f:
        f.write(quality_text)
    print(quality_text)

    # Save summary
    summary = {
        "fold": fold_id,
        "pool_size": len(pool),
        "gtr_embedding_dim": gtr_embeddings.shape[1],
        "kate_embedding_dim": kate_embeddings.shape[1],
        "type_distribution": {k: len(v) for k, v in type_pools.items()},
    }
    with open(fold_dir / "index_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done! Indices saved to: {fold_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices for Pilot 12")
    parser.add_argument("--fold", default="all", help="Fold to build (0-4 or 'all')")
    args = parser.parse_args()

    print("Loading embedding models...")
    gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base")
    kate_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Models loaded.")

    if args.fold == "all":
        folds = list(range(5))
    else:
        folds = [int(args.fold)]

    for fold_id in folds:
        build_fold_indices(fold_id, gtr_model, kate_model)

    # Update experiment log
    log_file = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "experiment_log.json"
    with open(log_file) as f:
        log = json.load(f)
    log["phases"]["phase_0"]["status"] = "completed"
    log["phases"]["phase_0"]["completed"] = "2026-02-11"
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nAll indices built. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
