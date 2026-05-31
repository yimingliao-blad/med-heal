"""Re-encode the 962-item pool index with gte-large-en-v1.5 (full notes, no truncation).

After 2026-04-27 full-note bake-off, gte-large beat nomic on every gold dimension
(\u03c1_clinical 0.566 vs 0.348, NDCG@3 0.905 vs 0.875). Switching the locked
embedding from nomic to gte-large.

Saves to:
  pool_index/gte_question.npy
  pool_index/gte_note.npy
  pool_index/gte_bm_zs.npy
Old nomic_*.npy stay in place for back-compat / comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = ROOT / "output" / "ichl" / "retrieval_study" / "pool_index"

GTE_MODEL = "Alibaba-NLP/gte-large-en-v1.5"


def main():
    items = [json.loads(l) for l in (POOL_DIR / "items.jsonl").open()]
    print(f"pool items: {len(items)}")

    notes  = [it["note_text"] for it in items]            # full notes (post 2026-04-27 fix)
    qs     = [it["question"]   for it in items]
    bm_zs  = [it.get("bm_zeroshot", "") for it in items]

    char_max = max(len(n) for n in notes)
    print(f"  full-note char counts: max={char_max}")

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    model = SentenceTransformer(GTE_MODEL, trust_remote_code=True, device=device)

    t0 = time.monotonic()
    print(f"\n[1/3] question embeddings ({len(qs)} items)...")
    emb_q = model.encode(qs, batch_size=16, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    print(f"  shape={emb_q.shape}  elapsed={time.monotonic()-t0:.0f}s")

    t1 = time.monotonic()
    print(f"\n[2/3] note embeddings ({len(notes)} items, FULL notes)...")
    emb_n = model.encode(notes, batch_size=4, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    print(f"  shape={emb_n.shape}  elapsed={time.monotonic()-t1:.0f}s")

    t2 = time.monotonic()
    print(f"\n[3/3] BM zeroshot embeddings ({len(bm_zs)} items)...")
    emb_bm = model.encode(bm_zs, batch_size=16, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    print(f"  shape={emb_bm.shape}  elapsed={time.monotonic()-t2:.0f}s")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    np.save(POOL_DIR / "gte_question.npy", emb_q)
    np.save(POOL_DIR / "gte_note.npy", emb_n)
    np.save(POOL_DIR / "gte_bm_zs.npy", emb_bm)
    print(f"\nSaved gte_* embeddings (3 .npy files). nomic_*.npy preserved as before.")
    print(f"Total elapsed: {time.monotonic()-t0:.0f}s")


if __name__ == "__main__":
    main()
