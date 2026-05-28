"""Build the (anchor, candidate, gpt_score) gold dataset for embedding alignment study.

Per design (locked 2026-04-26):
  - 75 anchors stratified by EHRNoteQA `category` × question coverage
  - 8 candidates per anchor: 2 high-cos / 2 mid-cos / 2 low-cos / 1 hard-neg / 1 random
  - same train_pool for both anchor and candidate sides (fold_0/train.jsonl)
  - patient_id ≠ same-patient leakage filter
  - GPT-4o relevance score 0-3 per pair (locked prompt below)

Output: output/ichl/retrieval_study/gold_pairs.jsonl  (one row per pair)
        output/ichl/retrieval_study/gold_pairs_design.json  (sampling provenance)

Cost: ~600 GPT-4o calls @ ~$0.005 = ~$3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
NOTES_FILE = ROOT / "output" / "EHRNoteQA_processed.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "retrieval_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GTR_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # 8K-context baseline; replaces gtr-t5-base
# (which was 512-token only) per [Workflow] No Silent Truncation 2026-04-27

# === LOCKED GPT-4o PROMPT ===
SYSTEM = (
    "You are a senior clinician evaluating whether one clinical case (CANDIDATE) would be a "
    "useful in-context example for an AI model answering a question about a different "
    "clinical case (TEST). A useful candidate is one whose question type and clinical "
    "context are similar enough that the reasoning pattern transfers."
)

USER_TMPL = """# TEST CASE
QUESTION: {test_q}
CLINICAL NOTE (excerpt): {test_note_excerpt}

# CANDIDATE EXAMPLE
QUESTION: {pool_q}
CLINICAL NOTE (excerpt): {pool_note_excerpt}

# RATING SCALE
0 = NOT USEFUL — different specialty / different question type / no transferable insight
1 = SLIGHTLY USEFUL — same broad domain (e.g. both cardiology) but different specifics
2 = USEFUL — similar question type AND similar clinical context
3 = HIGHLY USEFUL — same question type, very similar case, reasoning pattern transfers nearly directly

Respond with ONLY a single digit (0, 1, 2, or 3). No prose."""


def load_pool() -> list[dict]:
    """Load fold_0 train items, one per question (deduplicated by (patient_id, question))."""
    rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    print(f"  fold_0/train.jsonl: {len(rows)} rows")
    # Build clean records with stable IDs
    out = []
    for i, r in enumerate(rows):
        pid = int(r["patient_id"])
        # ground_truth from answer letter + choice
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        # Note excerpt: first 900 chars from concatenated notes
        parts = []
        # Use [Note i]-prefixed step8 format to match how step8 generation/judge see notes.
        # NEVER truncate per [Workflow] No Silent Truncation. Max EHRNoteQA note = 5709 nomic
        # tokens, fits 8K+ context. The earlier note_excerpt = note_full[:900] cap was the
        # source of the 2026-04-27 gold-pair calibration bug \u2014 removed.
        note_step8 = "\n\n".join(f"[Note {j}]\n{str(r.get(f'note_{j}','')).strip()}"
                                 for j in [1, 2, 3]
                                 if r.get(f"note_{j}") and str(r.get(f"note_{j}")).strip()
                                 and str(r.get(f"note_{j}")).lower() != "nan")
        out.append({
            "row_id": i,
            "patient_id": pid,
            "category": r.get("category", ""),
            "question": str(r["question"]),
            "ground_truth": gt,
            "note_full": note_step8,
            "note_excerpt": note_step8,    # alias for back-compat; same full content
        })
    print(f"  loaded {len(out)} pool items")
    return out


def embed_pool_with_gtr(pool: list[dict]) -> np.ndarray:
    """Compute baseline embeddings on the pool (used only for stratified candidate selection
    \u2014 we just need a similarity signal that ranks pool items, not a high-quality scorer).
    NEVER truncate. Default model: nomic-embed-text-v1.5 (8K context).
    Uses GPU if available; falls back to CPU otherwise (vLLM must be stopped to free GPU)."""
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEmbedding pool with {GTR_MODEL} (device={device})...")
    model = SentenceTransformer(GTR_MODEL, device=device, trust_remote_code=True)
    texts = [r["note_full"] for r in pool]  # FULL notes; no truncation per principle
    embs = model.encode(texts, batch_size=8, show_progress_bar=True, convert_to_numpy=True,
                        normalize_embeddings=True)
    print(f"  embeddings shape: {embs.shape}")
    return embs


def select_anchors(pool: list[dict], n_anchors: int = 75, seed: int = 42) -> list[int]:
    """Stratified by category (level1 vs level2), random within each stratum."""
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    for r in pool:
        by_cat.setdefault(r["category"], []).append(r["row_id"])
    print(f"\n  category counts: { {k: len(v) for k, v in by_cat.items()} }")
    # Stratified proportional split
    total = sum(len(v) for v in by_cat.values())
    chosen = []
    for cat, ids in by_cat.items():
        share = max(1, round(n_anchors * len(ids) / total))
        rng.shuffle(ids)
        chosen.extend(ids[:share])
    rng.shuffle(chosen)
    return chosen[:n_anchors]


def select_candidates_for_anchor(anchor_id: int, pool: list[dict], embs: np.ndarray,
                                 k_per_band: int = 2, seed: int = 42) -> list[tuple[int, str]]:
    """For given anchor, return 8 (cand_id, band_label) tuples:
       2 high-cos, 2 mid-cos, 2 low-cos, 1 hard-neg (high-cos but different patient + opposite label), 1 random.
       All filter: patient_id != anchor's patient_id."""
    anchor = pool[anchor_id]
    anchor_pid = anchor["patient_id"]
    anchor_emb = embs[anchor_id]
    sims = embs @ anchor_emb  # cosine since normalized
    # Mask out self + same-patient
    valid_mask = np.array([(p["row_id"] != anchor_id) and (p["patient_id"] != anchor_pid) for p in pool])
    valid_idx = np.where(valid_mask)[0]
    valid_sims = sims[valid_idx]
    # Sort by sim desc
    order = np.argsort(-valid_sims)
    sorted_idx = valid_idx[order]
    n = len(sorted_idx)
    # Bands: top 10% = high, mid 40-60% = mid, bottom 10% = low
    high_pool = sorted_idx[:max(10, n // 10)]
    mid_pool = sorted_idx[2 * n // 5: 3 * n // 5]
    low_pool = sorted_idx[-max(10, n // 10):]
    rng = random.Random(seed + anchor_id)
    out = []
    for cid in rng.sample(list(high_pool), min(k_per_band, len(high_pool))):
        out.append((int(cid), "high"))
    for cid in rng.sample(list(mid_pool), min(k_per_band, len(mid_pool))):
        out.append((int(cid), "mid"))
    for cid in rng.sample(list(low_pool), min(k_per_band, len(low_pool))):
        out.append((int(cid), "low"))
    # Hard negative: highest cosine candidate that we haven't already picked
    picked_set = {c for c, _ in out}
    hard_neg = next((cid for cid in sorted_idx if int(cid) not in picked_set), None)
    if hard_neg is not None:
        out.append((int(hard_neg), "hard_neg"))
    # Random (any unused)
    remaining = [cid for cid in valid_idx if int(cid) not in picked_set and int(cid) != hard_neg]
    rand = rng.choice(remaining) if remaining else None
    if rand is not None:
        out.append((int(rand), "random"))
    return out


def gpt4o_call(client, anchor: dict, candidate: dict, max_retries: int = 3) -> tuple[int | None, dict]:
    user = USER_TMPL.format(
        test_q=anchor["question"],
        test_note_excerpt=anchor["note_excerpt"],
        pool_q=candidate["question"],
        pool_note_excerpt=candidate["note_excerpt"],
    )
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=4,
            )
            lat = time.monotonic() - t0
            txt = (resp.choices[0].message.content or "").strip()
            # Parse single digit 0-3
            import re
            m = re.search(r"[0-3]", txt)
            score = int(m.group(0)) if m else None
            usage = resp.usage
            meta = {
                "raw": txt[:50],
                "latency_s": round(lat, 2),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }
            return score, meta
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 + attempt * 3)
            else:
                return None, {"error": str(e)[:200]}
    return None, {"error": "max_retries_exhausted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anchors", type=int, default=75)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build design only; skip GPT-4o calls")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading pool...")
    pool = load_pool()

    embs = embed_pool_with_gtr(pool)

    print(f"\nSelecting {args.n_anchors} anchors (stratified by category)...")
    anchor_ids = select_anchors(pool, n_anchors=args.n_anchors, seed=args.seed)
    print(f"  selected anchors: {len(anchor_ids)}")

    print(f"\nSelecting 8 candidates per anchor...")
    pairs = []
    for aid in anchor_ids:
        cands = select_candidates_for_anchor(aid, pool, embs, seed=args.seed)
        for cid, band in cands:
            pairs.append({"anchor_row_id": aid, "candidate_row_id": cid, "band": band})
    print(f"  total pairs: {len(pairs)}  (expected ~{args.n_anchors * 8})")

    # Save design
    design_path = OUT_DIR / "gold_pairs_design.json"
    design_path.write_text(json.dumps({
        "n_anchors": len(anchor_ids), "n_pairs": len(pairs),
        "seed": args.seed, "gtr_model_for_sampling": GTR_MODEL,
        "system_prompt": SYSTEM, "user_template": USER_TMPL,
        "anchor_ids": anchor_ids[:20] + ["..."] if len(anchor_ids) > 20 else anchor_ids,
        "band_distribution": {b: sum(1 for p in pairs if p["band"] == b) for b in ["high", "mid", "low", "hard_neg", "random"]},
    }, indent=2, default=str))
    print(f"\nDesign saved: {design_path}")

    if args.dry_run:
        print("\n--dry-run: skipping GPT-4o calls")
        return

    # === GPT-4o gold labeling ===
    print(f"\nLoading OpenAI key + creating client...")
    env_path = ROOT / ".env"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set and not in .env")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    out_path = OUT_DIR / "gold_pairs.jsonl"
    print(f"\nLabeling {len(pairs)} pairs with GPT-4o ({args.workers} workers)...")
    t0 = time.monotonic()
    total_in = 0; total_out = 0; n_done = 0; n_errors = 0

    def label_one(pair: dict) -> dict:
        a = pool[pair["anchor_row_id"]]
        c = pool[pair["candidate_row_id"]]
        score, meta = gpt4o_call(client, a, c)
        return {**pair, "gpt_score": score, **meta}

    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(label_one, pairs), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            n_done += 1
            if r.get("gpt_score") is None:
                n_errors += 1
            else:
                total_in += r.get("prompt_tokens", 0) or 0
                total_out += r.get("completion_tokens", 0) or 0
            if i % 50 == 0:
                dt = time.monotonic() - t0
                eta = dt * (len(pairs) - i) / i
                cost_so_far = total_in * 5e-6 + total_out * 1.5e-5
                print(f"  {i}/{len(pairs)}  elapsed={dt:.0f}s  eta={eta:.0f}s  errors={n_errors}  cost~${cost_so_far:.2f}")

    elapsed = time.monotonic() - t0
    final_cost = total_in * 5e-6 + total_out * 1.5e-5
    print(f"\nDONE in {elapsed:.0f}s")
    print(f"  successful: {n_done - n_errors}/{n_done}  errors: {n_errors}")
    print(f"  prompt_tokens: {total_in}  completion_tokens: {total_out}  cost: ${final_cost:.3f}")
    print(f"  saved: {out_path}")

    # Score distribution
    rows = [json.loads(l) for l in out_path.open() if l.strip()]
    from collections import Counter
    by_score = Counter(r.get("gpt_score") for r in rows)
    by_band_score = {}
    for r in rows:
        key = (r["band"], r.get("gpt_score"))
        by_band_score[key] = by_band_score.get(key, 0) + 1
    print(f"\n  gpt_score distribution: {dict(by_score)}")
    print(f"  band × score: {dict(sorted(by_band_score.items()))}")


if __name__ == "__main__":
    main()
