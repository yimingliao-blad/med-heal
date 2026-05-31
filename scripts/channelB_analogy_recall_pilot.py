#!/usr/bin/env python3
"""4B-1: GTR-vs-lexical analogy recall pilot.

User direction 2026-05-29: "have a test, so we know if it makes sense" — for the
cross-patient analogy retrieval in Channel B, compare the runner's current lexical
token-overlap mechanism against GTR-T5 cosine over the precomputed
bm_contrast_pool/fold_X_question_embeddings.npy.

For each of N Qwen2.5 zero-shot wrong cases:
  - Build the same query the runner builds: question + detection-payload-like fields
    (we use a synthetic payload here, since detection is not yet run; the synthetic
    payload uses question + ground_truth + a one-line error description from
    the audited error taxonomy if present, else empty).
  - Lexical top-1 from fold-safe bm_contrast_pool entries (set intersection).
  - GTR cosine top-1 from the same fold-safe pool, using the precomputed
    bm_contrast_pool/fold_X_question_embeddings.npy and a fresh GTR encoding
    of the query.
  - GPT-4o-mini single-call utility rating per analogy on a fixed yes/no rubric.

CPU-only. No vLLM. Cost: 2 ratings per case x N cases x gpt-4o-mini at ~$0.0002.
Default N=50 -> ~100 calls -> $0.02 oracle.

Outputs:
  runs/analogy_recall/qwen25_nw{N}_seed{SEED}/comparison.jsonl
  runs/analogy_recall/qwen25_nw{N}_seed{SEED}/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", PROJECT_ROOT.parent / "llm-ehr-hallucination"))
OUT_ROOT = PROJECT_ROOT / "runs" / "analogy_recall"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

POOL_DIR = SOURCE_REPO / "workspace" / "self_critique" / "data" / "bm_contrast_pool"


# ---------- data loading ----------

def load_pool(fold: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    pool = json.loads((POOL_DIR / f"fold_{fold}_pool.json").read_text())
    emb = np.load(POOL_DIR / f"fold_{fold}_question_embeddings.npy")
    if len(pool) != emb.shape[0]:
        raise RuntimeError(f"pool/embedding length mismatch for fold {fold}: pool={len(pool)} emb={emb.shape[0]}")
    return pool, emb


def load_qwen25_wrong_rows(n_wrong: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        df = pd.read_csv(SOURCE_REPO / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv")
        for _, r in df.iterrows():
            if int(r["binary_correct"]) == 0:
                rows.append({
                    "fold": fold,
                    "idx": int(r["idx"]),
                    "patient_id": int(r["patient_id"]),
                    "question": str(r["question"]),
                    "ground_truth": str(r["ground_truth"]),
                    "original_answer": str(r["model_answer"]),
                })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(n_wrong, len(rows))]


# ---------- retrieval ----------

_gtr_model = None


def gtr_encoder():
    global _gtr_model
    if _gtr_model is None:
        from sentence_transformers import SentenceTransformer
        _gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
    return _gtr_model


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", (s or "").lower()))


def build_query(row: dict[str, Any]) -> str:
    """Mirror the runner's retrieve_example query shape, minus detection output
    (which is not yet run in this CPU pilot). We use question + ground_truth +
    original_answer fragments as a stand-in for the detection-payload fields."""
    return " ".join([
        row["question"],
        row["original_answer"][:400],
        row["ground_truth"][:200],
    ])


def lexical_top1(query: str, pool: list[dict[str, Any]]) -> tuple[int, float, dict[str, Any]]:
    qt = toks(query)
    best_i, best_s = -1, -1
    for i, ex in enumerate(pool):
        text = " ".join([ex.get("question", ""), ex.get("what_was_wrong", ""), ex.get("ground_truth", "")])
        s = len(qt & toks(text))
        if s > best_s:
            best_i, best_s = i, s
    return best_i, float(best_s), pool[best_i]


def gtr_top1(query: str, pool: list[dict[str, Any]], emb: np.ndarray) -> tuple[int, float, dict[str, Any]]:
    q_emb = gtr_encoder().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    pool_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = pool_norm @ q_emb
    best_i = int(np.argmax(sims))
    return best_i, float(sims[best_i]), pool[best_i]


# ---------- judging ----------

JUDGE_SYSTEM = "You are evaluating whether a retrieved reference case would help correct another case's wrong answer."

JUDGE_USER_TEMPLATE = """Test case:
- Question: {q}
- Wrong answer: {wrong}
- Ground truth answer: {gt}

Retrieved analogy case:
- Question: {ex_q}
- Wrong answer: {ex_wrong}
- What was wrong: {ex_what_wrong}
- Ground truth answer: {ex_gt}

Rate the analogy on three dimensions, each 0 (no help), 1 (weak help), or 2 (clear help):
- Q_SIM: does the analogy's question target a similar clinical slot or topic as the test question?
- E_SIM: is the analogy's "what was wrong" pattern similar to the kind of mistake in the test wrong answer?
- A_SIM: does the analogy's ground truth answer model a shape/style that would guide the test correction?

Reply ONLY in this format, one number per line:
Q_SIM: <0|1|2>
E_SIM: <0|1|2>
A_SIM: <0|1|2>
USEFUL: <YES|NO>

USEFUL is YES only if the sum >= 3 (i.e. at least one clear-help and one weak-help across the three dimensions, or two clear-helps).
"""


def load_api_key() -> str:
    for env in (PROJECT_ROOT / ".env", SOURCE_REPO / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("OPENAI_API_KEY not found")


_openai_client = None


def openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=load_api_key())
    return _openai_client


def parse_judge_reply(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        m = re.match(r"^(Q_SIM|E_SIM|A_SIM)\s*:\s*([012])\b", line)
        if m:
            out[m.group(1)] = int(m.group(2))
        if line.upper().startswith("USEFUL"):
            m2 = re.search(r"(YES|NO)\b", line.upper())
            if m2:
                out["USEFUL"] = m2.group(1)
    return out


def rate_analogy(row: dict[str, Any], analogy: dict[str, Any]) -> dict[str, Any]:
    user = JUDGE_USER_TEMPLATE.format(
        q=row["question"][:800],
        wrong=row["original_answer"][:800],
        gt=row["ground_truth"][:600],
        ex_q=analogy.get("question", "")[:800],
        ex_wrong=analogy.get("wrong_answer", "")[:600],
        ex_what_wrong=analogy.get("what_was_wrong", "")[:600],
        ex_gt=analogy.get("ground_truth", "")[:600],
    )
    for attempt in range(4):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            raw = (r.choices[0].message.content or "").strip()
            parsed = parse_judge_reply(raw)
            return {"raw": raw, **parsed}
        except Exception as e:
            if attempt == 3:
                return {"raw": "", "error": str(e)}
            time.sleep(1 + attempt)
    return {"raw": ""}


# ---------- main ----------

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(side: str) -> dict[str, Any]:
        ratings = [r[side]["rating"] for r in rows if side in r]
        useful_yes = sum(1 for x in ratings if x.get("USEFUL") == "YES")
        usable = sum(1 for x in ratings if (x.get("Q_SIM", 0) + x.get("E_SIM", 0) + x.get("A_SIM", 0)) >= 3)
        mean_q = float(np.mean([x.get("Q_SIM", 0) for x in ratings])) if ratings else 0.0
        mean_e = float(np.mean([x.get("E_SIM", 0) for x in ratings])) if ratings else 0.0
        mean_a = float(np.mean([x.get("A_SIM", 0) for x in ratings])) if ratings else 0.0
        return {
            "n_rated": len(ratings),
            "useful_yes": useful_yes,
            "useful_rate": useful_yes / max(1, len(ratings)),
            "sum_ge_3_rate": usable / max(1, len(ratings)),
            "mean_Q_SIM": mean_q,
            "mean_E_SIM": mean_e,
            "mean_A_SIM": mean_a,
        }
    same_pool = sum(1 for r in rows if r.get("same_top1"))
    return {
        "n_cases": len(rows),
        "same_top1_count": same_pool,
        "same_top1_rate": same_pool / max(1, len(rows)),
        "lexical": agg("lexical"),
        "gtr": agg("gtr"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wrong", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sample = load_qwen25_wrong_rows(args.n_wrong, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"sample={len(sample)} out={out_dir}", flush=True)

    pool_cache: dict[int, tuple[list[dict[str, Any]], np.ndarray]] = {}

    results: list[dict[str, Any]] = []
    for i, row in enumerate(sample, 1):
        if row["fold"] not in pool_cache:
            pool_cache[row["fold"]] = load_pool(row["fold"])
        pool, emb = pool_cache[row["fold"]]
        query = build_query(row)
        lex_i, lex_score, lex_ex = lexical_top1(query, pool)
        gtr_i, gtr_score, gtr_ex = gtr_top1(query, pool, emb)
        same_top1 = (lex_i == gtr_i)
        lex_rating = rate_analogy(row, lex_ex)
        if same_top1:
            gtr_rating = lex_rating
        else:
            gtr_rating = rate_analogy(row, gtr_ex)
        results.append({
            "fold": row["fold"],
            "idx": row["idx"],
            "patient_id": row["patient_id"],
            "question": row["question"],
            "lexical": {
                "pool_index": lex_i,
                "score": lex_score,
                "analogy": {k: lex_ex.get(k) for k in ["fold", "idx", "patient_id", "question", "wrong_answer", "what_was_wrong", "ground_truth"]},
                "rating": lex_rating,
            },
            "gtr": {
                "pool_index": gtr_i,
                "cosine": gtr_score,
                "analogy": {k: gtr_ex.get(k) for k in ["fold", "idx", "patient_id", "question", "wrong_answer", "what_was_wrong", "ground_truth"]},
                "rating": gtr_rating,
            },
            "same_top1": same_top1,
        })
        if i % 5 == 0 or i == len(sample):
            print(f"rated {i}/{len(sample)}", flush=True)

    with (out_dir / "comparison.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
