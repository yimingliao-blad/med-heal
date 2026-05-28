"""Multi-component scorer for retrieval, using nomic-embed-text-v1.5 + (optional) NER.

Components:
  1. cos_note(a, c)      — clinical context similarity (nomic on note text)
  2. cos_question(a, c)  — question-type similarity (nomic on question text only)
  3. cos_gt(a, c)        — ground-truth alignment (nomic on GT text only)
  4. jaccard_ner(a, c)   — critical-detail overlap (NER on note, future step)

Pipeline:
  - Embed each pool item with nomic on (note | question | GT) — 3 vectors per item
  - For each gold pair, compute 3 component cosines
  - Regress weights against the composite gold (mean of 5 dims) with 5-fold CV
  - Report per-dim ρ and composite ρ for: each component alone, equal-weighted sum,
    fitted-weight sum, and (as ceiling reference) OpenAI text-3-large note cosine.

Output: output/ichl/retrieval_study/multi_component_results.json
        output/ichl/retrieval_study/multi_component_table.md

Cost: $0 (all local). NER component placeholder for future LLM-extracted entities.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[3]
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
GOLD_MULTIDIM = ROOT / "output" / "ichl" / "retrieval_study" / "gold_pairs_multidim.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "retrieval_study"
EMB_CACHE = OUT_DIR / "emb_cache"
EMB_CACHE.mkdir(parents=True, exist_ok=True)

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DIMS = ["question_type_match", "error_type_match", "clinical_context_similarity",
        "ground_truth_alignment", "critical_detail_overlap"]


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        pid = int(r["patient_id"])
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        parts = []
        for j in [1, 2, 3]:
            v = r.get(f"note_{j}")
            if v and str(v).strip() and str(v).lower() != "nan":
                parts.append(str(v))
        note = "\n\n".join(parts)[:2000]
        out.append({
            "row_id": i, "patient_id": pid,
            "question": str(r["question"]),
            "ground_truth": gt,
            "note_text": note,
        })
    return out


def encode_nomic(texts: list[str], cache_name: str) -> np.ndarray:
    cache = EMB_CACHE / f"nomic_{cache_name}.npy"
    if cache.exists():
        print(f"  cache hit: {cache.name}")
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    print(f"  encoding {len(texts)} texts with {NOMIC_MODEL} (cache={cache_name})...")
    t0 = time.monotonic()
    model = SentenceTransformer(NOMIC_MODEL, trust_remote_code=True, device="cuda")
    embs = model.encode(texts, batch_size=16, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    print(f"    done in {time.monotonic()-t0:.1f}s, shape={embs.shape}")
    np.save(cache, embs)
    del model
    import gc, torch; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return embs


def main():
    print("Loading pool + multidim gold...")
    pool = load_pool()
    pairs = [json.loads(l) for l in GOLD_MULTIDIM.open() if l.strip()]
    pairs = [p for p in pairs if isinstance(p.get("scores"), dict) and
             all(p["scores"].get(d) is not None for d in DIMS)]
    print(f"  pool={len(pool)}  valid_pairs={len(pairs)}")

    print("\nEncoding 3 nomic representations (note | question | GT)...")
    embs_note = encode_nomic([p["note_text"] for p in pool], "note")
    embs_q = encode_nomic([p["question"] for p in pool], "question")
    embs_gt = encode_nomic([p["ground_truth"] for p in pool], "ground_truth")

    # Build feature matrix X (4 columns: cos_note, cos_q, cos_gt, jaccard_NER_placeholder)
    # and gold matrix Y (5 dim scores) plus composite (mean of 5)
    print("\nComputing pair-level features...")
    X = []
    Y_dims = {d: [] for d in DIMS}
    for p in pairs:
        a = p["anchor_row_id"]; c = p["candidate_row_id"]
        cos_note = float(embs_note[a] @ embs_note[c])
        cos_q = float(embs_q[a] @ embs_q[c])
        cos_gt = float(embs_gt[a] @ embs_gt[c])
        # NER placeholder: 0 for now (future enhancement)
        ner = 0.0
        X.append([cos_note, cos_q, cos_gt, ner])
        for d in DIMS:
            Y_dims[d].append(int(p["scores"][d]))
    X = np.array(X)
    Y = {d: np.array(v) for d, v in Y_dims.items()}
    Y_composite = np.mean(np.stack([Y[d] for d in DIMS], axis=1), axis=1)
    print(f"  X shape: {X.shape}  Y_composite shape: {Y_composite.shape}")
    print(f"  composite gold range: [{Y_composite.min():.2f}, {Y_composite.max():.2f}]  "
          f"mean={Y_composite.mean():.2f}  std={Y_composite.std():.2f}")

    # === Per-component baseline: each cosine alone vs each gold dim ===
    print("\n=== Per-component Spearman \u03c1 (each component alone) ===")
    per_comp = {}
    comp_names = ["cos_note", "cos_question", "cos_gt", "ner_jaccard"]
    for ci, cn in enumerate(comp_names):
        if cn == "ner_jaccard":
            continue  # placeholder column
        col = X[:, ci]
        per_comp[cn] = {}
        for d in DIMS:
            rho, p = spearmanr(col, Y[d])
            per_comp[cn][d] = (round(float(rho), 3), round(float(p), 4))
        rho_c, p_c = spearmanr(col, Y_composite)
        per_comp[cn]["composite"] = (round(float(rho_c), 3), round(float(p_c), 4))

    # Print per-component table
    rows = []
    for cn in [c for c in comp_names if c != "ner_jaccard"]:
        row = {"component": cn}
        for d in DIMS + ["composite"]:
            r, p = per_comp[cn][d]
            row[d] = f"{r:.3f}" + ("*" if p < 0.001 else "")
        rows.append(row)
    df_comp = pd.DataFrame(rows).set_index("component")
    print(df_comp.to_string())

    # === Equal-weighted sum (no fitting) ===
    print("\n=== Equal-weighted sum of (cos_note + cos_question + cos_gt) ===")
    equal_score = X[:, :3].mean(axis=1)
    eq_results = {}
    for d in DIMS + (["composite"] if True else []):
        gold_vec = Y_composite if d == "composite" else Y[d]
        rho, p = spearmanr(equal_score, gold_vec)
        eq_results[d] = (round(float(rho), 3), round(float(p), 4))
        print(f"  vs {d:32s}: \u03c1={rho:.3f}  p={p:.4f}")

    # === Fitted weights with 5-fold CV ===
    print("\n=== Fitted weights (Ridge regression vs composite, 5-fold CV) ===")
    X_3 = X[:, :3]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rhos = []
    all_preds = np.zeros(len(Y_composite))
    fold_weights = []
    for fold_i, (tr, te) in enumerate(kf.split(X_3)):
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge.fit(X_3[tr], Y_composite[tr])
        pred_te = ridge.predict(X_3[te])
        all_preds[te] = pred_te
        rho_te, _ = spearmanr(pred_te, Y_composite[te])
        fold_rhos.append(round(float(rho_te), 3))
        fold_weights.append({"alpha": float(ridge.alpha_),
                              "intercept": round(float(ridge.intercept_), 4),
                              "coefs": [round(float(c), 4) for c in ridge.coef_]})
    print(f"  per-fold \u03c1 vs composite: {fold_rhos}  mean={np.mean(fold_rhos):.3f}")
    print(f"  fold weights:")
    for fi, fw in enumerate(fold_weights):
        print(f"    fold {fi}: a={fw['alpha']:.3f}  int={fw['intercept']:.3f}  coefs={fw['coefs']}")
    # Final fit on all data for reporting
    ridge_full = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(X_3, Y_composite)
    print(f"  final fit (all data): coefs={[round(float(c), 4) for c in ridge_full.coef_]}  "
          f"int={float(ridge_full.intercept_):.4f}")
    rho_full_oof, _ = spearmanr(all_preds, Y_composite)
    print(f"  OOF \u03c1 vs composite (5-fold concat): {rho_full_oof:.3f}")

    # === Per-dim performance of fitted score ===
    print("\n=== Fitted score (OOF) vs each dim ===")
    fitted_results = {}
    for d in DIMS + ["composite"]:
        gold_vec = Y_composite if d == "composite" else Y[d]
        rho, p = spearmanr(all_preds, gold_vec)
        fitted_results[d] = (round(float(rho), 3), round(float(p), 4))
        print(f"  vs {d:32s}: \u03c1={rho:.3f}  p={p:.4f}")

    # === Reference: ceiling from bake-off (note-only OpenAI text-3-large) ===
    bo = json.loads((OUT_DIR / "bakeoff_results.json").read_text())
    nomic_row = next((r for r in bo if r["name"] == "nomic-embed-text-v1.5"), None)
    openai_row = next((r for r in bo if r["name"] == "openai-text-3-large"), None)

    # Build summary table: rows = methods, cols = ρ per dim + composite
    summary_rows = []
    if nomic_row:
        summary_rows.append({
            "method": "nomic note-only (production baseline)",
            **{d: nomic_row.get(f"rho_{d}") for d in DIMS},
            "composite": "—",
        })
    if openai_row:
        summary_rows.append({
            "method": "openai-text-3-large note-only (research ceiling)",
            **{d: openai_row.get(f"rho_{d}") for d in DIMS},
            "composite": "—",
        })
    for cn in [c for c in comp_names if c != "ner_jaccard"]:
        summary_rows.append({
            "method": f"nomic {cn} only",
            **{d: per_comp[cn][d][0] for d in DIMS},
            "composite": per_comp[cn]["composite"][0],
        })
    summary_rows.append({
        "method": "equal-weighted sum (3 nomic cosines)",
        **{d: eq_results[d][0] for d in DIMS},
        "composite": eq_results["composite"][0],
    })
    summary_rows.append({
        "method": "fitted weights (Ridge, 5-fold CV)",
        **{d: fitted_results[d][0] for d in DIMS},
        "composite": fitted_results["composite"][0],
    })

    df = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY TABLE ===")
    print(df.to_string(index=False))

    # Save
    out_data = {
        "nomic_model": NOMIC_MODEL,
        "n_pairs": len(pairs),
        "per_component": per_comp,
        "equal_weighted": eq_results,
        "fitted": fitted_results,
        "fold_weights": fold_weights,
        "final_weights": [round(float(c), 4) for c in ridge_full.coef_],
        "final_intercept": round(float(ridge_full.intercept_), 4),
        "fold_rhos": fold_rhos,
        "oof_rho_composite": round(float(rho_full_oof), 3),
        "summary_rows": summary_rows,
    }
    (OUT_DIR / "multi_component_results.json").write_text(json.dumps(out_data, indent=2, default=str))
    md = df.to_markdown(index=False, floatfmt=".3f")
    (OUT_DIR / "multi_component_table.md").write_text(md + "\n")
    print(f"\nSaved: {OUT_DIR / 'multi_component_results.json'}")
    print(f"       {OUT_DIR / 'multi_component_table.md'}")


if __name__ == "__main__":
    main()
