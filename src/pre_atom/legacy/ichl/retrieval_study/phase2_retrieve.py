"""Phase 2 retrieval — top-1 pool candidate per fold_0 test item.

Scorer (locked, per Phase 0): cos_q + cos_note + cos(X_zs, BM_zs)
  X_zs = target's zero-shot answer (Qwen2.5 here, from step8 CSV)
  BM_zs = pool side's BioMistral zero-shot (cached in pool_index)

Pool: fold_0/train items (subset of 962 pool_index by patient_id)
Targets: fold_0/test items (193)

Output: output/ichl/retrieval_study/phase2/retrievals_qwen25_fold0.jsonl
        one row per test item: {test_idx, test_pid, top1_pool_rid, top1_pool_pid, score, components}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
POOL_DIR = RS / "pool_index"
OUT_DIR = RS / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "retrievals_qwen25_fold0.jsonl"

FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
FOLD_TEST = ROOT / "output" / "folds" / "fold_0" / "test.jsonl"
QWEN25_ZS = ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / "fold_0" / "zeroshot_generated.csv"

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def load_pool_index() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Load 962-item pool index + cached embeddings."""
    items = [json.loads(l) for l in (POOL_DIR / "items.jsonl").open()]
    q = np.load(POOL_DIR / "nomic_question.npy")
    n = np.load(POOL_DIR / "nomic_note.npy")
    bm = np.load(POOL_DIR / "nomic_bm_zs.npy")
    print(f"  pool_index: {len(items)} items, embeds q={q.shape} n={n.shape} bm={bm.shape}")
    return items, q, n, bm


def main():
    print("Loading fold_0 splits + pool index + Qwen2.5 zs...")
    train = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    test = [json.loads(l) for l in FOLD_TEST.open() if l.strip()]
    print(f"  fold_0/train: {len(train)}  fold_0/test: {len(test)}")

    items, q_emb, n_emb, bm_emb = load_pool_index()
    pid_to_item_idx = {int(it["patient_id"]): i for i, it in enumerate(items)}

    # Pool side: filter to fold_0/train patients
    train_pids = {int(r["patient_id"]) for r in train}
    pool_idx = [pid_to_item_idx[pid] for pid in sorted(train_pids) if pid in pid_to_item_idx]
    print(f"  pool (fold_0/train) mapped to pool_index: {len(pool_idx)}")
    pool_q = q_emb[pool_idx]
    pool_n = n_emb[pool_idx]
    pool_bm = bm_emb[pool_idx]
    pool_meta = [items[i] for i in pool_idx]  # row_id, patient_id, question, etc.

    # Target side: fold_0/test
    qwen_df = pd.read_csv(QWEN25_ZS)
    print(f"  Qwen2.5 zs csv: {len(qwen_df)} rows")
    qwen_by_pid = {int(r["patient_id"]): str(r["model_answer"] or "") for _, r in qwen_df.iterrows()}

    test_pids = [int(r["patient_id"]) for r in test]
    test_indices_in_pool = []  # pool_index row for each test item (for q_emb, n_emb lookup)
    test_zs_texts = []
    for tp in test_pids:
        if tp not in pid_to_item_idx:
            print(f"    WARNING: test pid {tp} not in pool_index")
            test_indices_in_pool.append(None)
            test_zs_texts.append("")
            continue
        test_indices_in_pool.append(pid_to_item_idx[tp])
        test_zs_texts.append(qwen_by_pid.get(tp, ""))

    # Encode test-side Qwen2.5 zs
    n_missing = sum(1 for t in test_zs_texts if not t)
    print(f"  test items with Qwen2.5 zs: {len(test_zs_texts) - n_missing}/{len(test_zs_texts)}")
    print(f"\nEncoding {len(test_zs_texts)} test Qwen2.5 zs answers via nomic-embed-text-v1.5...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.monotonic()
    model = SentenceTransformer(NOMIC_MODEL, trust_remote_code=True, device=device)
    test_zs_emb = model.encode(test_zs_texts, batch_size=16, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True)
    print(f"  done in {time.monotonic()-t0:.1f}s  shape={test_zs_emb.shape}")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # For each test item: score against pool, pick top-1
    print("\nScoring test items against pool with locked 3-cos scorer...")
    rows = []
    for ti, (test_item, pool_idx_for_test, zs_emb_t) in enumerate(zip(test, test_indices_in_pool, test_zs_emb)):
        if pool_idx_for_test is None:
            continue
        # Use pool_index's question/note embeddings for the test item
        tq = q_emb[pool_idx_for_test]
        tn = n_emb[pool_idx_for_test]
        cos_q_vec = pool_q @ tq
        cos_n_vec = pool_n @ tn
        cos_zs_vec = pool_bm @ zs_emb_t
        score_vec = cos_q_vec + cos_n_vec + cos_zs_vec
        # Top-1
        top1 = int(np.argmax(score_vec))
        top1_pool_meta = pool_meta[top1]
        rows.append({
            "test_idx": ti,
            "test_patient_id": int(test_item["patient_id"]),
            "test_question": test_item["question"][:120],
            "top1_pool_rid": int(top1_pool_meta["row_id"]),
            "top1_pool_patient_id": int(top1_pool_meta["patient_id"]),
            "top1_score": float(score_vec[top1]),
            "top1_cos_q": float(cos_q_vec[top1]),
            "top1_cos_note": float(cos_n_vec[top1]),
            "top1_cos_zs": float(cos_zs_vec[top1]),
        })

    with OUT_FILE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} top-1 retrievals to {OUT_FILE}")
    # Quick sanity check
    print("\nFirst 3 retrievals:")
    for r in rows[:3]:
        print(f"  test {r['test_idx']} (pid {r['test_patient_id']}) -> pool rid {r['top1_pool_rid']} "
              f"(pid {r['top1_pool_patient_id']})  score={r['top1_score']:.3f} "
              f"(q={r['top1_cos_q']:.2f} n={r['top1_cos_note']:.2f} zs={r['top1_cos_zs']:.2f})")
    # Patient leakage check: should be 0 (test pids should not appear in train pids)
    leak = sum(1 for r in rows if r["top1_pool_patient_id"] == r["test_patient_id"])
    print(f"\nPatient-leakage in top-1: {leak}/{len(rows)} (should be 0; test/train patients are disjoint)")


if __name__ == "__main__":
    main()
