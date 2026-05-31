"""Phase 0 redo: validate locked retrieval scorer with gte-large + full-note 200-pair gold.

Inputs:
  - fold_0/train.jsonl (769 items \u2014 the pool / anchor side)
  - pool_index/gte_{question,note,bm_zs}.npy (962-item pool, full-note encodings)
  - gold_pairs_multidim.jsonl (200 pairs, 5-dim multidim scores under full notes)
  - step8/qwen2.5-7b-instruct/fold_0/zeroshot_evaluated_binary.csv (BM correctness)

Computes:
  - For each of the 200 (anchor, candidate) gold pairs:
      cos_q     = cos(anchor.Q,    candidate.Q)
      cos_note  = cos(anchor.note, candidate.note)
      cos_zs_zs = cos(anchor.X_zs, candidate.BM_zs)   \u2014 cross-model role-matched
  - Composite scorer = cos_q + cos_note + cos_zs_zs
  - Spearman \u03c1 of composite vs gold composite (mean of 5 dims)
  - Per-dim \u03c1
  - Component-only \u03c1 (cos_q alone, cos_note alone, etc.) for ablation

Output:
  output/ichl/retrieval_study/phase0_redo_gte_results.json
  output/ichl/retrieval_study/phase0_redo_gte_table.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
POOL_DIR = RS / "pool_index"
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
BM_CSV = ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / "fold_0" / "zeroshot_evaluated_binary.csv"
GOLD = RS / "gold_pairs_multidim.jsonl"

DIMS = ["question_type_match", "error_type_match", "clinical_context_similarity",
        "ground_truth_alignment", "critical_detail_overlap"]


def main():
    print("Loading inputs...")
    # Pool index (962 items) with gte embeddings
    pool_items = [json.loads(l) for l in (POOL_DIR / "items.jsonl").open()]
    gte_q  = np.load(POOL_DIR / "gte_question.npy")
    gte_n  = np.load(POOL_DIR / "gte_note.npy")
    gte_zs = np.load(POOL_DIR / "gte_bm_zs.npy")
    print(f"  pool_index items: {len(pool_items)}, gte_q={gte_q.shape}")

    # Map patient_id -> pool_index row
    pid_to_pool_row = {int(it["patient_id"]): i for i, it in enumerate(pool_items)}

    # fold_0/train (769 items) — gold pairs use these row_ids
    train_rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    print(f"  fold_0/train: {len(train_rows)} items")

    # For each fold_0/train item, find corresponding pool_index row
    train_to_pool = {}
    for ti, r in enumerate(train_rows):
        pid = int(r["patient_id"])
        if pid in pid_to_pool_row:
            train_to_pool[ti] = pid_to_pool_row[pid]
    print(f"  fold_0/train mapped to pool: {len(train_to_pool)}/{len(train_rows)}")

    # Gold pairs
    gold = [json.loads(l) for l in GOLD.open() if l.strip()]
    valid = [g for g in gold
             if isinstance(g.get("scores"), dict) and all(g["scores"].get(d) is not None for d in DIMS)]
    print(f"  gold pairs valid: {len(valid)}/{len(gold)}")

    # Qwen2.5 zs and step8 binary_correct (for X_zs side)
    df = pd.read_csv(BM_CSV)
    qwen25_zs_by_pid = {int(r["patient_id"]): str(r["model_answer"] or "") for _, r in df.iterrows()}
    bm_correct = {int(r["patient_id"]): int(r["binary_correct"]) for _, r in df.iterrows()
                  if r["binary_correct"] in (0, 1)}

    # Encode fold_0/train side: Qwen2.5 zs (X_zs anchor side) AND ground_truth (for cos_gt variants)
    print("\nEncoding fold_0/train Qwen2.5 zs + ground truth with gte-large...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device=device)
    train_zs_texts = [qwen25_zs_by_pid.get(int(r["patient_id"]), "") for r in train_rows]
    # Build GT in step8 format
    train_gt_texts = []
    for r in train_rows:
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        train_gt_texts.append(f"{letter}: {gt_text}" if (letter and gt_text) else gt_text)
    train_zs_emb = model.encode(train_zs_texts, batch_size=16, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
    train_gt_emb = model.encode(train_gt_texts, batch_size=16, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
    print(f"  train_zs_emb shape: {train_zs_emb.shape}")
    print(f"  train_gt_emb shape: {train_gt_emb.shape}")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Pool GT embeddings (encode FULL pool GTs; for the candidate side cos_*_gt terms)
    pool_gt_texts = [it.get("ground_truth", "") for it in pool_items]
    print("  encoding pool GT (962 items) with gte-large...")
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device=device)
    pool_gt_emb = model.encode(pool_gt_texts, batch_size=16, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
    print(f"  pool_gt_emb shape: {pool_gt_emb.shape}")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Compute per-pair features (multiple cosine variants)
    print("\nComputing pairwise cosines for gold pairs...")
    rows = []
    skipped = 0
    for g in valid:
        a_train = int(g["anchor_row_id"])
        c_train = int(g["candidate_row_id"])
        if a_train not in train_to_pool or c_train not in train_to_pool:
            skipped += 1
            continue
        a_pool = train_to_pool[a_train]
        c_pool = train_to_pool[c_train]
        # Anchor side: train_zs_emb[a_train] (X_zs), train_gt_emb[a_train] (anchor GT)
        # Candidate side: gte_zs[c_pool] (candidate BM_zs), pool_gt_emb[c_pool] (candidate GT)
        cos_q       = float(gte_q[a_pool]  @ gte_q[c_pool])
        cos_n       = float(gte_n[a_pool]  @ gte_n[c_pool])
        cos_zs_zs   = float(train_zs_emb[a_train] @ gte_zs[c_pool])      # Match #1: X_zs ↔ BM_zs
        cos_zs_gt   = float(train_zs_emb[a_train] @ pool_gt_emb[c_pool])  # Match #2: X_zs ↔ candidate GT
        cos_gt_gt   = float(train_gt_emb[a_train] @ pool_gt_emb[c_pool])  # GT ↔ GT (anchor GT vs cand GT)
        cos_gt_zs   = float(train_gt_emb[a_train] @ gte_zs[c_pool])       # GT ↔ BM_zs (less obvious; for completeness)
        composite = float(np.mean([g["scores"][d] for d in DIMS]))
        scores_each = {f"score_{d}": int(g["scores"][d]) for d in DIMS}
        rows.append({
            "anchor": a_train, "candidate": c_train,
            "cos_q": cos_q, "cos_note": cos_n,
            "cos_zs_zs": cos_zs_zs, "cos_zs_gt": cos_zs_gt,
            "cos_gt_gt": cos_gt_gt, "cos_gt_zs": cos_gt_zs,
            # Composite scorers
            "s_2comp_qnote":            cos_q + cos_n,
            "s_3comp_qnote_zszs":       cos_q + cos_n + cos_zs_zs,
            "s_3comp_qnote_zsgt":       cos_q + cos_n + cos_zs_gt,
            "s_3comp_qnote_gtgt":       cos_q + cos_n + cos_gt_gt,
            "s_4comp_qnote_zszs_zsgt":  cos_q + cos_n + cos_zs_zs + cos_zs_gt,
            "s_4comp_qnote_zszs_gtgt":  cos_q + cos_n + cos_zs_zs + cos_gt_gt,
            "composite": composite,
            **scores_each,
        })
    print(f"  computed {len(rows)} pairs (skipped {skipped})")
    feat = pd.DataFrame(rows)

    # Spearman ρ vs composite + per-dim
    print("\n=== Per-method ρ vs each gold dimension ===")
    methods = {
        # Single components
        "cos_q only":                       "cos_q",
        "cos_note only":                    "cos_note",
        "cos_zs_zs only":                   "cos_zs_zs",
        "cos_zs_gt only":                   "cos_zs_gt",
        "cos_gt_gt only":                   "cos_gt_gt",
        "cos_gt_zs only":                   "cos_gt_zs",
        # 2-comp (production-safe: no GT/zs)
        "2-comp (q + note)":                "s_2comp_qnote",
        # 3-comp variants
        "3-comp + cos(zs,zs)":              "s_3comp_qnote_zszs",
        "3-comp + cos(zs,GT)":              "s_3comp_qnote_zsgt",
        "3-comp + cos(GT,GT)":              "s_3comp_qnote_gtgt",
        # 4-comp combinations
        "4-comp + cos(zs,zs) + cos(zs,GT)": "s_4comp_qnote_zszs_zsgt",
        "4-comp + cos(zs,zs) + cos(GT,GT)": "s_4comp_qnote_zszs_gtgt",
    }
    out_rows = []
    print(f"  {'method':40s}  q_type  err     clin    gt      crit    COMP")
    for label, col in methods.items():
        d = {}
        for dim in DIMS:
            r, _ = spearmanr(feat[col], feat[f"score_{dim}"])
            d[dim] = round(float(r) if not np.isnan(r) else 0.0, 3)
        rc, _ = spearmanr(feat[col], feat["composite"])
        d["composite"] = round(float(rc) if not np.isnan(rc) else 0.0, 3)
        out_rows.append({"method": label, **d})
        print(f"  {label:40s}  {d['question_type_match']:.3f}   {d['error_type_match']:.3f}   "
              f"{d['clinical_context_similarity']:.3f}   {d['ground_truth_alignment']:.3f}   "
              f"{d['critical_detail_overlap']:.3f}   {d['composite']:.3f}")

    # Save
    summary = {
        "n_pairs": len(rows),
        "embedding": "gte-large-en-v1.5 (full notes, post 2026-04-27 bake-off)",
        "gold": "200-pair multi-dim, full-note GPT-4o re-elicitation",
        "scorer_formula": "cos_q + cos_note + cos(X_zs, BM_zs)",
        "per_method_rho": out_rows,
    }
    (RS / "phase0_redo_gte_results.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(out_rows).set_index("method").to_markdown(RS / "phase0_redo_gte_table.md", floatfmt=".3f")
    print(f"\nSaved: {RS}/phase0_redo_gte_results.json")
    print(f"       {RS}/phase0_redo_gte_table.md")


if __name__ == "__main__":
    main()
