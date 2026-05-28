#!/usr/bin/env python3
"""
Retrieval Strategies for Pilot 12: RA-ICL

Three off-the-shelf retrievers:
  - BM25: Lexical similarity (rank_bm25)
  - GTR: Dense semantic retrieval (gtr-t5-base)
  - KATE: kNN semantic similarity (all-MiniLM-L6-v2)

Plus a type-filtered wrapper that retrieves within question categories.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Question type classifier (same as all prior pilots)
TEMPORAL_KEYWORDS = [
    "before admission", "prior to", "at discharge", "during the first",
    "during the second", "between", "pre-admission", "post-admission",
    "after the", "before the surgery", "before the procedure",
    "upon discharge", "upon admission", "at the time of",
]
DOCUMENTATION_KEYWORDS = [
    "as noted", "as stated", "as documented", "according to",
    "exact cause", "what was the reason", "what reason was stated",
]
PRECISION_KEYWORDS = [
    "dose", "dosage", "mg", "what results", "test results",
    "what level", "what value", "what was prescribed",
]


def classify_question(question: str) -> str:
    q = question.lower()
    for kw in TEMPORAL_KEYWORDS:
        if kw in q:
            return "temporal"
    for kw in DOCUMENTATION_KEYWORDS:
        if kw in q:
            return "documentation"
    for kw in PRECISION_KEYWORDS:
        if kw in q:
            return "precision"
    return "selectivity"


class BM25Retriever:
    """Lexical similarity retrieval using BM25."""

    def __init__(self, correct_pool: list[dict]):
        from rank_bm25 import BM25Okapi

        self.correct_pool = correct_pool
        self.tokenized_corpus = [ex["question"].lower().split() for ex in correct_pool]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query_question: str, k: int = 1) -> list[tuple[dict, float]]:
        tokenized_query = query_question.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = scores.argsort()[-k:][::-1]
        return [(self.correct_pool[i], float(scores[i])) for i in top_k_idx]


class GTRRetriever:
    """Dense semantic retrieval using GTR-T5-Base."""

    def __init__(self, correct_pool: list[dict], embeddings: np.ndarray | None = None):
        self.correct_pool = correct_pool
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
            questions = [ex["question"] for ex in correct_pool]
            self.embeddings = self.model.encode(questions, batch_size=64, show_progress_bar=False)
        self.knn = NearestNeighbors(
            n_neighbors=min(10, len(correct_pool)), metric="cosine", algorithm="auto"
        )
        self.knn.fit(self.embeddings)
        # Keep model reference for query encoding
        if not hasattr(self, "model"):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    def retrieve(self, query_question: str, k: int = 1) -> list[tuple[dict, float]]:
        q_emb = self.model.encode([query_question])
        distances, indices = self.knn.kneighbors(q_emb, n_neighbors=min(k, len(self.correct_pool)))
        return [(self.correct_pool[idx], 1.0 - dist) for idx, dist in zip(indices[0], distances[0])]


class KATERetriever:
    """kNN semantic retrieval using all-MiniLM-L6-v2."""

    def __init__(self, correct_pool: list[dict], embeddings: np.ndarray | None = None):
        self.correct_pool = correct_pool
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            questions = [ex["question"] for ex in correct_pool]
            self.embeddings = self.model.encode(questions, batch_size=64, show_progress_bar=False)
        self.knn = NearestNeighbors(
            n_neighbors=min(10, len(correct_pool)), metric="cosine", algorithm="auto"
        )
        self.knn.fit(self.embeddings)
        if not hasattr(self, "model"):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def retrieve(self, query_question: str, k: int = 1) -> list[tuple[dict, float]]:
        q_emb = self.model.encode([query_question])
        distances, indices = self.knn.kneighbors(q_emb, n_neighbors=min(k, len(self.correct_pool)))
        return [(self.correct_pool[idx], 1.0 - dist) for idx, dist in zip(indices[0], distances[0])]


class TypeFilteredRetriever:
    """Retrieve within the same question type, then apply a sub-retriever (GTR)."""

    def __init__(self, correct_pool: list[dict], gtr_model=None):
        from sentence_transformers import SentenceTransformer

        self.model = gtr_model or SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
        self.sub_retrievers = {}
        self.full_pool = correct_pool

        # Partition pool by type
        type_pools = {}
        for ex in correct_pool:
            q_type = classify_question(ex["question"])
            type_pools.setdefault(q_type, []).append(ex)

        # Build sub-retriever per type
        for q_type, pool in type_pools.items():
            if len(pool) >= 1:
                questions = [ex["question"] for ex in pool]
                embeddings = self.model.encode(questions, batch_size=64, show_progress_bar=False)
                retriever = GTRRetriever(pool, embeddings=embeddings)
                retriever.model = self.model
                self.sub_retrievers[q_type] = retriever

        # Fallback: full pool
        all_questions = [ex["question"] for ex in correct_pool]
        all_emb = self.model.encode(all_questions, batch_size=64, show_progress_bar=False)
        self.fallback = GTRRetriever(correct_pool, embeddings=all_emb)
        self.fallback.model = self.model

    def retrieve(self, query_question: str, k: int = 1) -> list[tuple[dict, float]]:
        q_type = classify_question(query_question)
        retriever = self.sub_retrievers.get(q_type)
        # Fallback if type pool is too small
        if retriever is None or len(retriever.correct_pool) < k:
            retriever = self.fallback
        return retriever.retrieve(query_question, k)


class NoteRetriever:
    """Retrieve by discharge note similarity using GTR-T5-Base."""

    def __init__(self, correct_pool: list[dict], note_embeddings: np.ndarray, model=None):
        self.correct_pool = correct_pool
        self.embeddings = note_embeddings
        self.knn = NearestNeighbors(
            n_neighbors=min(10, len(correct_pool)), metric="cosine", algorithm="auto"
        )
        self.knn.fit(self.embeddings)
        if model is not None:
            self.model = model
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    def retrieve(self, query_note: str, k: int = 1) -> list[tuple[dict, float]]:
        q_emb = self.model.encode([query_note])
        distances, indices = self.knn.kneighbors(q_emb, n_neighbors=min(k, len(self.correct_pool)))
        return [(self.correct_pool[idx], 1.0 - dist) for idx, dist in zip(indices[0], distances[0])]


class TypeNoteRetriever:
    """Retrieve within question type by note similarity."""

    def __init__(self, correct_pool: list[dict], note_embeddings: np.ndarray, model=None):
        from sentence_transformers import SentenceTransformer
        self.model = model or SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
        self.full_pool = correct_pool
        self.sub_retrievers = {}

        # Partition pool and embeddings by type
        type_indices = {}
        for i, ex in enumerate(correct_pool):
            q_type = classify_question(ex["question"])
            type_indices.setdefault(q_type, []).append(i)

        for q_type, indices in type_indices.items():
            type_pool = [correct_pool[i] for i in indices]
            type_embs = note_embeddings[indices]
            sub = NoteRetriever(type_pool, type_embs, model=self.model)
            self.sub_retrievers[q_type] = sub

        # Fallback: full pool
        self.fallback = NoteRetriever(correct_pool, note_embeddings, model=self.model)

    def retrieve(self, query_note: str, query_question: str, k: int = 1) -> list[tuple[dict, float]]:
        q_type = classify_question(query_question)
        retriever = self.sub_retrievers.get(q_type)
        if retriever is None or len(retriever.correct_pool) < k:
            retriever = self.fallback
        return retriever.retrieve(query_note, k)


def load_retriever_from_index(
    retriever_type: str,
    index_dir: Path,
    correct_pool: list[dict],
) -> BM25Retriever | GTRRetriever | KATERetriever:
    """Load a pre-built retriever from saved indices."""
    if retriever_type == "bm25":
        return BM25Retriever(correct_pool)

    elif retriever_type == "gtr":
        emb_file = index_dir / "gtr_correct_embeddings.npy"
        embeddings = np.load(emb_file)
        return GTRRetriever(correct_pool, embeddings=embeddings)

    elif retriever_type == "kate":
        emb_file = index_dir / "kate_correct_embeddings.npy"
        embeddings = np.load(emb_file)
        return KATERetriever(correct_pool, embeddings=embeddings)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def load_type_filtered_retriever(
    index_dir: Path,
    correct_pool: list[dict],
) -> TypeFilteredRetriever:
    """Load a type-filtered retriever using pre-built type sub-indices."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    retriever = TypeFilteredRetriever.__new__(TypeFilteredRetriever)
    retriever.model = model
    retriever.full_pool = correct_pool
    retriever.sub_retrievers = {}

    type_dir = index_dir / "type_subindex"
    if type_dir.exists():
        for q_type in ["selectivity", "temporal", "documentation", "precision"]:
            emb_file = type_dir / f"{q_type}_gtr_embeddings.npy"
            pool_file = type_dir / f"{q_type}_pool.pkl"
            if emb_file.exists() and pool_file.exists():
                with open(pool_file, "rb") as f:
                    type_pool = pickle.load(f)
                embeddings = np.load(emb_file)
                sub = GTRRetriever(type_pool, embeddings=embeddings)
                sub.model = model
                retriever.sub_retrievers[q_type] = sub

    # Fallback: full pool
    all_emb = np.load(index_dir / "gtr_correct_embeddings.npy")
    retriever.fallback = GTRRetriever(correct_pool, embeddings=all_emb)
    retriever.fallback.model = model

    return retriever
