#!/usr/bin/env python3
"""
Module 3 helper — per-note sentence index.

For each item, build a sentence-level GTR-T5 index over the same patient's
discharge note(s) and return the top-K most relevant spans for a given query.

This is the *primary* evidence channel for correction (per user direction):
when detection identifies an error and produces a `correct_statement` /
`error_statement`, we look in the patient's own notes for the sentences that
support the claim. Real spans, not analogies from another patient.

Lazy global model load — first call instantiates GTR-T5 on CPU and caches.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
    return _embedder


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


def split_sentences(note: str) -> list[str]:
    """Light sentence splitter tuned for clinical notes (no NLTK dep)."""
    if not note:
        return []
    # Strip the [Note N] section markers; keep the content joined
    cleaned_lines = []
    for line in note.splitlines():
        s = line.strip()
        if not s or s.startswith("[Note "):
            continue
        cleaned_lines.append(s)
    text = " ".join(cleaned_lines)
    # Split into sentences
    raw = _SENT_SPLIT.split(text)
    # Drop very short fragments
    sents = [s.strip() for s in raw if len(s.strip()) >= 10]
    return sents


def topk_spans(note: str, queries: Sequence[str], k: int = 3,
               *, agreement_floor: float = 0.40,
               scoring: str = "agreement") -> list[dict]:
    """Return top-K sentences from `note`, scored by how many queries agree on
    them rather than the single best query.

    Two scoring modes:
      - "max"        legacy: take max(sim) across queries (single-query view)
      - "agreement"  default: score = sum_q max(0, sim_q - agreement_floor)
                     A sentence many queries agree on (each at sim ≥ floor)
                     beats a sentence one query loves at high sim.

    `queries` may include several rewordings of the error statement (one per
    K-vote sample) plus the question itself; deduplication of identical
    strings is the caller's job.

    Returns list of {sentence, score, max_sim, n_supporting, query_hits}:
        score          aggregated score by the chosen scoring mode
        max_sim        the highest single cosine over all queries
        n_supporting   number of queries with sim_q >= agreement_floor
        query_hits     list of (query_idx, sim) above the floor, descending
    """
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return []
    sents = split_sentences(note)
    if not sents:
        return []
    emb = get_embedder()
    sent_embs = emb.encode(sents, normalize_embeddings=True, show_progress_bar=False)
    q_embs = emb.encode(list(queries), normalize_embeddings=True, show_progress_bar=False)
    sims = sent_embs @ q_embs.T  # (num_sents, num_queries)

    max_per_sent = sims.max(axis=1)
    if scoring == "max":
        score_per_sent = max_per_sent
    elif scoring == "agreement":
        floored = np.clip(sims - agreement_floor, 0.0, None)
        score_per_sent = floored.sum(axis=1)
    else:
        raise ValueError(f"unknown scoring: {scoring}")

    n_supporting = (sims >= agreement_floor).sum(axis=1)
    order = np.argsort(-score_per_sent)
    out: list[dict] = []
    for i in order[:k]:
        i = int(i)
        # Per-query hits above the floor, sorted descending
        per_q = [(qi, float(sims[i, qi]))
                 for qi in range(len(queries))
                 if sims[i, qi] >= agreement_floor]
        per_q.sort(key=lambda t: -t[1])
        out.append({
            "sentence": sents[i],
            "score": float(score_per_sent[i]),
            "max_sim": float(max_per_sent[i]),
            "n_supporting": int(n_supporting[i]),
            "query_hits": per_q,
            # Backwards-compat alias so older callers that read 'similarity'
            # don't break:
            "similarity": float(max_per_sent[i]),
        })
    return out


# ---------- Self-test ----------

if __name__ == "__main__":
    note = """[Note 1]
Patient was admitted for elective first rib resection. She tolerated the
procedure well. Once she was transferred to the floor, her Foley was
discharged at midnight on POD0. She was unable to void and was bladder
scanned showing 700cc of retained urine. She was straight cathed in the
morning of POD1. The JP drain was removed and she was then able to void
after ambulating. Discharged in stable condition."""
    queries = [
        "patient experienced difficulty voiding after surgery",
        "no postoperative complications were noted",
    ]
    res = topk_spans(note, queries, k=3)
    for r in res:
        print(f"sim={r['similarity']:.3f}  q={r['query_idx']}  {r['sentence']}")
