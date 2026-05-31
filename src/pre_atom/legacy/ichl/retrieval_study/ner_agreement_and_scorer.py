"""1) Compare Magistral vs Qwen3-235B NER on the 50-overlap items (sanity check).
2) Build pair-level Jaccard feature from Magistral's full 769-pool extraction.
3) Integrate as 4th component of the multi-component scorer; re-run with NER added.

Output:
  output/ichl/retrieval_study/ner_agreement.json
  output/ichl/retrieval_study/multi_component_with_ner.json
  output/ichl/retrieval_study/multi_component_with_ner_table.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "ichl" / "retrieval_study"

NER_MAG = OUT / "ner_magistral.jsonl"
NER_Q3_SAMPLE = OUT / "ner_qwen3-235b_sample50.jsonl"
GOLD_MULTIDIM = OUT / "gold_pairs_multidim.jsonl"
EMB_CACHE = OUT / "emb_cache"

DIMS = ["question_type_match", "error_type_match", "clinical_context_similarity",
        "ground_truth_alignment", "critical_detail_overlap"]
NER_CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]


def normalize_entity(s: str) -> str:
    s = s.lower().strip()
    # Drop trailing punctuation, normalize whitespace
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,;:")
    return s


def get_entity_set(rec: dict, cats: list[str] = NER_CATS) -> dict[str, set[str]]:
    """Return per-category set of normalized entities."""
    out = {c: set() for c in cats}
    e = rec.get("entities", {}) or {}
    if not isinstance(e, dict):
        return out
    for c in cats:
        v = e.get(c, [])
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    n = normalize_entity(item)
                    if n: out[c].add(n)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(len(a | b), 1)


def jaccard_overall(setsA: dict[str, set], setsB: dict[str, set]) -> float:
    """Jaccard on the union across all categories."""
    a = set().union(*setsA.values())
    b = set().union(*setsB.values())
    return jaccard(a, b)


def main():
    # === 1. Magistral vs Qwen3-235B agreement on 50-overlap ===
    print("Loading NER extractions...")
    mag_rows = [json.loads(l) for l in NER_MAG.open() if l.strip()]
    q3_rows = [json.loads(l) for l in NER_Q3_SAMPLE.open() if l.strip()]
    mag_by_id = {r["row_id"]: r for r in mag_rows}
    q3_by_id = {r["row_id"]: r for r in q3_rows}
    overlap_ids = sorted(set(mag_by_id.keys()) & set(q3_by_id.keys()))
    print(f"  magistral rows: {len(mag_rows)}  qwen3 rows: {len(q3_rows)}  overlap ids: {len(overlap_ids)}")

    print("\n=== Per-item Magistral vs Qwen3-235B Jaccard agreement ===")
    per_cat_scores = {c: [] for c in NER_CATS}
    overall_scores = []
    for rid in overlap_ids:
        mag_sets = get_entity_set(mag_by_id[rid])
        q3_sets = get_entity_set(q3_by_id[rid])
        if any("_err" in (q3_by_id[rid].get("entities") or {}) for _ in [0]):
            continue
        for c in NER_CATS:
            per_cat_scores[c].append(jaccard(mag_sets[c], q3_sets[c]))
        overall_scores.append(jaccard_overall(mag_sets, q3_sets))
    if overall_scores:
        print(f"  Overall Jaccard (union of all cats):  mean={np.mean(overall_scores):.3f}  "
              f"median={np.median(overall_scores):.3f}  std={np.std(overall_scores):.3f}  n={len(overall_scores)}")
        for c in NER_CATS:
            s = per_cat_scores[c]
            if s:
                print(f"  {c:14s} Jaccard:  mean={np.mean(s):.3f}  median={np.median(s):.3f}  zeros={sum(1 for x in s if x == 0)}/{len(s)}")
    agreement_data = {
        "overall_jaccard_mean": float(np.mean(overall_scores)),
        "overall_jaccard_median": float(np.median(overall_scores)),
        "n_compared": len(overall_scores),
        "per_category": {c: {"mean": float(np.mean(s)), "median": float(np.median(s)),
                              "zeros": sum(1 for x in s if x == 0), "n": len(s)}
                          for c, s in per_cat_scores.items() if s},
    }
    (OUT / "ner_agreement.json").write_text(json.dumps(agreement_data, indent=2))

    # === 2. Build pair Jaccard feature from Magistral full pool ===
    print("\n=== Building pair-level Jaccard from Magistral (769 items) ===")
    pool_sets = {r["row_id"]: get_entity_set(r) for r in mag_rows}
    print(f"  pool entity sets ready for {len(pool_sets)} items")

    pairs = [json.loads(l) for l in GOLD_MULTIDIM.open() if l.strip()]
    pairs = [p for p in pairs if isinstance(p.get("scores"), dict) and
             all(p["scores"].get(d) is not None for d in DIMS)]
    print(f"  gold pairs: {len(pairs)}")

    # Load nomic embeddings
    embs_note = np.load(EMB_CACHE / "nomic_note.npy")
    embs_q = np.load(EMB_CACHE / "nomic_question.npy")
    embs_gt = np.load(EMB_CACHE / "nomic_ground_truth.npy")

    X = []  # [cos_note, cos_q, cos_gt, jaccard_overall, jaccard_meds, jaccard_procs, jaccard_diag, jaccard_lab, jaccard_dose]
    Y = {d: [] for d in DIMS}
    for p in pairs:
        a = p["anchor_row_id"]; c = p["candidate_row_id"]
        cos_note = float(embs_note[a] @ embs_note[c])
        cos_q = float(embs_q[a] @ embs_q[c])
        cos_gt = float(embs_gt[a] @ embs_gt[c])
        a_sets = pool_sets.get(a, {c: set() for c in NER_CATS})
        c_sets = pool_sets.get(c, {ct: set() for ct in NER_CATS})
        j_overall = jaccard_overall(a_sets, c_sets)
        j_per = [jaccard(a_sets[ct], c_sets[ct]) for ct in NER_CATS]
        X.append([cos_note, cos_q, cos_gt, j_overall] + j_per)
        for d in DIMS:
            Y[d].append(int(p["scores"][d]))
    X = np.array(X)
    Y = {d: np.array(v) for d, v in Y.items()}
    Y_composite = np.mean(np.stack([Y[d] for d in DIMS], axis=1), axis=1)
    print(f"  X shape: {X.shape}")

    feat_names = ["cos_note", "cos_q", "cos_gt", "jacc_overall"] + [f"jacc_{c}" for c in NER_CATS]

    # === 3. Per-component Spearman ρ ===
    print("\n=== Per-component Spearman ρ vs each gold dim ===")
    per_comp = {}
    for fi, fname in enumerate(feat_names):
        col = X[:, fi]
        per_comp[fname] = {}
        for d in DIMS:
            rho, p = spearmanr(col, Y[d])
            per_comp[fname][d] = (round(float(rho), 3), round(float(p), 4))
        rho_c, _ = spearmanr(col, Y_composite)
        per_comp[fname]["composite"] = round(float(rho_c), 3)

    df_comp = pd.DataFrame({fn: {d: per_comp[fn][d][0] for d in DIMS} | {"composite": per_comp[fn]["composite"]}
                            for fn in feat_names}).T
    print(df_comp.to_string())

    # === 4. Equal-weighted vs Fitted (Ridge CV) — both with and without NER ===
    print("\n=== Composite scorer comparison ===")

    methods = {}

    # Equal-weighted, 3 cosines (baseline from previous run)
    eq3 = X[:, :3].mean(axis=1)
    rho_eq3 = {d: round(float(spearmanr(eq3, Y[d])[0]), 3) for d in DIMS}
    rho_eq3["composite"] = round(float(spearmanr(eq3, Y_composite)[0]), 3)
    methods["3-nomic equal"] = rho_eq3

    # Equal-weighted, 3 cosines + jaccard_overall
    eq4 = X[:, :4].mean(axis=1)
    rho_eq4 = {d: round(float(spearmanr(eq4, Y[d])[0]), 3) for d in DIMS}
    rho_eq4["composite"] = round(float(spearmanr(eq4, Y_composite)[0]), 3)
    methods["3-nomic + jacc_overall equal"] = rho_eq4

    # Fitted Ridge: 3 cosines (original)
    def ridge_oof(X_sub, label="?"):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_p = np.zeros(len(Y_composite))
        coefs = []
        rhos = []
        for tr, te in kf.split(X_sub):
            r = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(X_sub[tr], Y_composite[tr])
            all_p[te] = r.predict(X_sub[te])
            coefs.append([round(float(c), 3) for c in r.coef_])
            rho_te, _ = spearmanr(all_p[te], Y_composite[te])
            rhos.append(round(float(rho_te), 3))
        out_rhos = {d: round(float(spearmanr(all_p, Y[d])[0]), 3) for d in DIMS}
        out_rhos["composite"] = round(float(spearmanr(all_p, Y_composite)[0]), 3)
        return out_rhos, coefs, rhos
    rho_fit3, coefs3, foldrhos3 = ridge_oof(X[:, :3], "3-nomic")
    methods["3-nomic Ridge CV"] = rho_fit3

    # Fitted Ridge: 3 cosines + jaccard_overall
    rho_fit4, coefs4, foldrhos4 = ridge_oof(X[:, :4], "3-nomic + jacc_overall")
    methods["3-nomic + jacc_overall Ridge CV"] = rho_fit4

    # Fitted Ridge: all 9 features (3 cos + jacc_overall + 5 per-cat jaccards)
    rho_fit9, coefs9, foldrhos9 = ridge_oof(X, "all 9 features")
    methods["all 9 features (3-nomic + jaccards) Ridge CV"] = rho_fit9

    # Print summary
    print()
    df = pd.DataFrame(methods).T
    df = df[DIMS + ["composite"]]
    print(df.to_string())

    # Coef inspection on the 9-feature fit
    print("\n=== Final fit (all 9 features) Ridge weights ===")
    ridge_full = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(X, Y_composite)
    for fn, w in zip(feat_names, ridge_full.coef_):
        print(f"  {fn:20s}  weight={w:+.3f}")
    print(f"  intercept: {ridge_full.intercept_:.3f}")
    print(f"  per-fold OOF ρ (9-feat): {foldrhos9}  mean={np.mean(foldrhos9):.3f}")

    out_data = {
        "agreement_data": agreement_data,
        "per_component_rho": per_comp,
        "method_comparison": methods,
        "fold_rhos_9feat": foldrhos9,
        "final_weights_9feat": [round(float(c), 4) for c in ridge_full.coef_],
        "final_intercept_9feat": round(float(ridge_full.intercept_), 4),
        "feature_names": feat_names,
    }
    (OUT / "multi_component_with_ner.json").write_text(json.dumps(out_data, indent=2, default=str))

    md = df.to_markdown(floatfmt=".3f")
    (OUT / "multi_component_with_ner_table.md").write_text(md + "\n")
    print(f"\nSaved: {OUT / 'multi_component_with_ner.json'}")
    print(f"       {OUT / 'multi_component_with_ner_table.md'}")


if __name__ == "__main__":
    main()
