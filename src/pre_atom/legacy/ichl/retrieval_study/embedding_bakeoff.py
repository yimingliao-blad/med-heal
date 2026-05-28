"""Phase B — 12-embedding bake-off with two golds.

Loads:
  - Pool: output/folds/fold_0/train.jsonl (769 items)
  - Multi-dim gold: output/ichl/retrieval_study/gold_pairs_multidim.jsonl (600 pairs × 5 scores)
  - Error taxonomy: output/step8/error_classification/all_errors_by_patient.json (BM 6-cat)

Per embedding:
  - Encode all 769 pool NOTES (truncated to ~2000 chars to be fair across max_seq_len)
  - For each of 600 pairs: cosine
  - Metrics:
      * per-dim Spearman ρ (5 dims from the GPT-4o gold)
      * NDCG@3 on `clinical_context_similarity` (production-relevant dim)
      * error-type gap: mean_cos(same_err) − mean_cos(diff_err) on BM-error subset
      * Mann-Whitney U p-value for the error-type gap

Output: output/ichl/retrieval_study/bakeoff_results.json
        output/ichl/retrieval_study/bakeoff_table.md (printable)

Caches: output/ichl/retrieval_study/emb_cache/{name}.npy

Cost: $0 except OpenAI text-embedding-3-large pass (~$0.50).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
GOLD_MULTIDIM = ROOT / "output" / "ichl" / "retrieval_study" / "gold_pairs_multidim.jsonl"
ERR_FILE = ROOT / "output" / "step8" / "error_classification" / "all_errors_by_patient.json"
OUT_DIR = ROOT / "output" / "ichl" / "retrieval_study"
EMB_CACHE = OUT_DIR / "emb_cache"
EMB_CACHE.mkdir(parents=True, exist_ok=True)

DIMS = ["question_type_match", "error_type_match", "clinical_context_similarity",
        "ground_truth_alignment", "critical_detail_overlap"]

# === EMBEDDING CANDIDATES ===
# Each: (alias, model_id, kind)
# kind: "st" (sentence-transformers symmetric), "asymmetric" (MedCPT q/a), "openai"
# Bake-off restricted to FREE models with \u22658K context, per
# [Workflow] No Silent Truncation 2026-04-27 (max EHRNoteQA note = 5709 nomic
# tokens; all 9 short-context candidates dropped: gtr-t5-base, bge-large,
# MedEmbed-large, MedCPT, e5-large-v2, mxbai, snowflake; gte-Qwen2-1.5B and
# openai-text-3-large dropped per-project rule excluding paid OpenAI from production).
CANDIDATES = [
    ("nomic-embed-text-v1.5",  "nomic-ai/nomic-embed-text-v1.5",             "st"),    # 8192 ctx
    ("bge-m3",                 "BAAI/bge-m3",                                 "st"),    # 8192 ctx
    ("gte-large-en-v1.5",      "Alibaba-NLP/gte-large-en-v1.5",              "st"),    # 8192 ctx
]


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        pid = int(r["patient_id"])
        # Step8 [Note i] format. NEVER truncate per [Workflow] No Silent Truncation.
        note = "\n\n".join(f"[Note {j}]\n{str(r.get(f'note_{j}','')).strip()}"
                           for j in [1, 2, 3]
                           if r.get(f"note_{j}") and str(r.get(f"note_{j}")).strip()
                           and str(r.get(f"note_{j}")).lower() != "nan")
        out.append({"row_id": i, "patient_id": pid,
                    "question": str(r["question"]),
                    "note_text": note})   # full note, no truncation
    return out


def encode_st(model_id: str, texts: list[str], device: str = "cuda") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"    loading {model_id} (device={device})...")
    kwargs = {"device": device}
    # Some models need trust_remote_code
    if "nomic" in model_id or "Qwen" in model_id or "gte" in model_id:
        model = SentenceTransformer(model_id, trust_remote_code=True, **kwargs)
    else:
        model = SentenceTransformer(model_id, **kwargs)
    embs = model.encode(texts, batch_size=8, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    del model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return embs


def encode_medcpt(q_model_id: str, a_model_id: str, items: list[dict], device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """MedCPT is asymmetric: Q-encoder for queries, A-encoder for documents.
    For our purposes we encode the QUESTION with Q-encoder, NOTE with A-encoder, return both.
    Pairwise cosine in eval uses Q (anchor) × A (candidate).
    """
    from transformers import AutoTokenizer, AutoModel
    import torch
    questions = [it["question"] for it in items]
    notes = [it["note_text"] for it in items]
    out = []
    for which, model_id, texts in [("Q", q_model_id, questions), ("A", a_model_id, notes)]:
        print(f"    loading {model_id} ({which}-encoder)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                batch = texts[i:i+8]
                enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt",
                                max_length=512).to(device)
                outputs = model(**enc)
                # CLS pooling per MedCPT convention
                embs = outputs.last_hidden_state[:, 0, :]
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                all_embs.append(embs.cpu().numpy())
        out.append(np.concatenate(all_embs, axis=0))
        del model, tokenizer
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out[0], out[1]


def encode_openai(model_id: str, texts: list[str]) -> np.ndarray:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_path = ROOT / ".env"
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip(); break
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    out = []
    bs = 100
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        r = client.embeddings.create(model=model_id, input=batch)
        out.extend(np.array(d.embedding) for d in r.data)
    arr = np.stack(out)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def encode_one_candidate(name: str, model_id, kind: str, pool: list[dict]) -> dict:
    """Returns dict with 'note_emb' or ('q_emb', 'a_emb') depending on kind."""
    cache_path = EMB_CACHE / f"{name}.npz"
    if cache_path.exists():
        print(f"    cache hit: {cache_path}")
        z = np.load(cache_path)
        return dict(z)
    notes = [it["note_text"] for it in pool]
    if kind == "st":
        embs = encode_st(model_id, notes, device="cuda")
        np.savez(cache_path, note_emb=embs)
        return {"note_emb": embs}
    elif kind == "asymmetric":
        q_emb, a_emb = encode_medcpt(model_id[0], model_id[1], pool, device="cuda")
        np.savez(cache_path, q_emb=q_emb, a_emb=a_emb)
        return {"q_emb": q_emb, "a_emb": a_emb}
    elif kind == "openai":
        embs = encode_openai(model_id, notes)
        np.savez(cache_path, note_emb=embs)
        return {"note_emb": embs}
    else:
        raise ValueError(f"unknown kind: {kind}")


def pair_cosines(emb_data: dict, pairs: list[dict]) -> np.ndarray:
    if "note_emb" in emb_data:
        e = emb_data["note_emb"]
        return np.array([float(e[p["anchor_row_id"]] @ e[p["candidate_row_id"]]) for p in pairs])
    elif "q_emb" in emb_data and "a_emb" in emb_data:
        # Asymmetric: anchor Q × candidate A (retrieval direction)
        q = emb_data["q_emb"]; a = emb_data["a_emb"]
        return np.array([float(q[p["anchor_row_id"]] @ a[p["candidate_row_id"]]) for p in pairs])
    else:
        raise ValueError(f"unrecognized emb_data shape: {list(emb_data.keys())}")


def metrics_for_candidate(name: str, cosines: np.ndarray, pairs: list[dict],
                          pool: list[dict], err_labels: dict[int, str]) -> dict:
    from scipy.stats import spearmanr, mannwhitneyu

    out = {"name": name, "n_pairs": len(pairs)}

    # Per-dim Spearman ρ
    for d in DIMS:
        gold = []
        cos = []
        for p, c in zip(pairs, cosines):
            scores = p.get("scores")
            if not isinstance(scores, dict): continue
            s = scores.get(d)
            if s is None: continue
            gold.append(s); cos.append(c)
        if len(gold) > 5:
            rho, p_val = spearmanr(gold, cos)
            out[f"rho_{d}"] = round(float(rho) if not np.isnan(rho) else 0.0, 3)
            out[f"p_{d}"] = round(float(p_val) if not np.isnan(p_val) else 1.0, 4)
        else:
            out[f"rho_{d}"] = None
            out[f"p_{d}"] = None

    # NDCG@3 on clinical_context_similarity per anchor
    from collections import defaultdict
    by_anchor: dict[int, list[tuple[float, int]]] = defaultdict(list)  # anchor_row_id → [(cos, gold_score)]
    for p, c in zip(pairs, cosines):
        scores = p.get("scores", {}) or {}
        gs = scores.get("clinical_context_similarity")
        if gs is None: continue
        by_anchor[p["anchor_row_id"]].append((c, int(gs)))
    ndcgs = []
    for aid, items in by_anchor.items():
        if len(items) < 3: continue
        # Sort by predicted cosine desc, take top-3
        items_sorted = sorted(items, key=lambda x: -x[0])
        gold_at_3 = [g for _, g in items_sorted[:3]]
        # DCG with binary-relevance treated as graded
        dcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gold_at_3))
        # Ideal DCG: sort by gold desc
        ideal_at_3 = sorted([g for _, g in items], reverse=True)[:3]
        idcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(ideal_at_3))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    out["ndcg3_clinical"] = round(float(np.mean(ndcgs)) if ndcgs else 0.0, 3)
    out["n_anchors_in_ndcg"] = len(ndcgs)

    # Error-type gap (only on pairs where BOTH anchor and candidate have err labels)
    same_cos = []; diff_cos = []
    for p, c in zip(pairs, cosines):
        a_pid = pool[p["anchor_row_id"]]["patient_id"]
        c_pid = pool[p["candidate_row_id"]]["patient_id"]
        a_err = err_labels.get(a_pid)
        c_err = err_labels.get(c_pid)
        if a_err is None or c_err is None: continue
        if a_err == c_err: same_cos.append(c)
        else: diff_cos.append(c)
    if same_cos and diff_cos:
        gap = float(np.mean(same_cos) - np.mean(diff_cos))
        try:
            u, p_u = mannwhitneyu(same_cos, diff_cos, alternative="greater")
        except Exception:
            p_u = 1.0
        out["err_type_gap"] = round(gap, 4)
        out["err_type_gap_p"] = round(float(p_u), 4)
        out["n_same_err"] = len(same_cos)
        out["n_diff_err"] = len(diff_cos)
    else:
        out["err_type_gap"] = None
        out["err_type_gap_p"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="alias names to skip")
    ap.add_argument("--only", nargs="*", default=[], help="alias names to only run")
    args = ap.parse_args()

    print("Loading pool, gold, errors...")
    pool = load_pool()
    pairs = [json.loads(l) for l in GOLD_MULTIDIM.open() if l.strip()]
    pairs = [p for p in pairs if isinstance(p.get("scores"), dict)]
    err_arr = json.loads(ERR_FILE.read_text())
    err_labels = {int(r["patient_id"]): r["primary_error"] for r in err_arr}
    print(f"  pool={len(pool)}  pairs={len(pairs)}  err_labels={len(err_labels)}")

    results = []
    for (name, model_id, kind) in CANDIDATES:
        if name in args.skip:
            print(f"\n[skip] {name}")
            continue
        if args.only and name not in args.only: continue
        print(f"\n=== {name} ({model_id if isinstance(model_id, str) else 'asymmetric'}) ===")
        t0 = time.monotonic()
        try:
            emb_data = encode_one_candidate(name, model_id, kind, pool)
            cos = pair_cosines(emb_data, pairs)
            m = metrics_for_candidate(name, cos, pairs, pool, err_labels)
            m["wall_s"] = round(time.monotonic() - t0, 1)
            results.append(m)
            print(f"  rhos: " + ", ".join(f"{d.split('_')[0]}={m.get(f'rho_{d}')}" for d in DIMS))
            print(f"  ndcg@3 (clinical) = {m['ndcg3_clinical']}  err_type_gap = {m['err_type_gap']}  p={m.get('err_type_gap_p')}")
            print(f"  wall: {m['wall_s']}s")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({"name": name, "error": str(e)[:300]})
        # Save incrementally
        (OUT_DIR / "bakeoff_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Build markdown table
    rows = [r for r in results if "error" not in r]
    if rows:
        df = pd.DataFrame(rows).set_index("name")
        cols = [f"rho_{d}" for d in DIMS] + ["ndcg3_clinical", "err_type_gap", "err_type_gap_p"]
        cols = [c for c in cols if c in df.columns]
        md = df[cols].to_markdown(floatfmt=".3f")
        (OUT_DIR / "bakeoff_table.md").write_text(md + "\n")
        print(f"\n=== TABLE ===\n{md}\n")
    print(f"\nResults: {OUT_DIR / 'bakeoff_results.json'}")
    print(f"Table:   {OUT_DIR / 'bakeoff_table.md'}")


if __name__ == "__main__":
    main()
