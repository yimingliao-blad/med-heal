"""Phase 0: validate retrieval scorers for positive/error/full modes,
honoring role-matching principle (error_zs ↔ error_zs, correct_zs ↔ correct_zs/GT).

Key tests on the existing 600-pair gold:
  0.1  Input-only 3-comp scorer (cos_q + cos_note + jacc_NER, no GT, no zs)
       Baseline for ALL modes; works for positive-mode retrieval (input-only matching).

  0.2  Add cos(BM_zs_anchor, BM_zs_pool) — Match #1, role-stratified:
       0.2a  error-error pairs (both BM-wrong)        → pure error-pattern match
       0.2b  correct-correct pairs (both BM-right)    → pure correct-pattern match
       0.2c  cross-correctness pairs (error vs correct) → ROLE MISMATCH (expect noise)

  0.3  Add cos(BM_zs_anchor, GT_pool) — Match #2:
       0.3a  correct anchor (anchor BM-right)         → correct ↔ GT (should mirror 0.2b)
       0.3b  error anchor   (anchor BM-wrong)         → error ↔ GT (role mismatch; expect weak)

  0.4  cos(GT_anchor, GT_pool) — original 4-comp without per-target leakage
       Reference: this is what the validated 0.655 was using (cos_gt). Compare against 0.2/0.3.

Output: output/ichl/retrieval_study/phase0_results.json + table.md
Cost: $0, ~5 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "ichl" / "retrieval_study"
EMB_CACHE = OUT / "emb_cache"
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
BM_GEN = ROOT / "output" / "ours_biomistral-7b_EHRNoteQA_processed.csv"
NER_FILE = OUT / "ner_magistral.jsonl"
GOLD = OUT / "gold_pairs_multidim.jsonl"
ERR_FILE = ROOT / "output" / "step8" / "error_classification" / "all_errors_by_patient.json"

DIMS = ["question_type_match", "error_type_match", "clinical_context_similarity",
        "ground_truth_alignment", "critical_detail_overlap"]
NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        pid = int(r["patient_id"])
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        out.append({"row_id": i, "patient_id": pid,
                    "question": str(r["question"]),
                    "ground_truth": gt})
    return out


def load_bm_zs(pool: list[dict]) -> dict[int, str]:
    """Load BM zero-shot answer for each patient_id in pool."""
    df = pd.read_csv(BM_GEN)
    # Find the openended-answer column
    col = "openended_answer" if "openended_answer" in df.columns else None
    if col is None:
        candidates = [c for c in df.columns if "answer" in c.lower()]
        col = candidates[0] if candidates else None
    print(f"  BM answer column: {col!r}")
    bm_by_pid = {int(r["patient_id"]): str(r.get(col, "") or "") for _, r in df.iterrows()}
    out = {}
    n_missing = 0
    for p in pool:
        ans = bm_by_pid.get(p["patient_id"], "")
        if not ans or ans == "nan":
            n_missing += 1
        out[p["row_id"]] = ans
    print(f"  BM zs available for {len(out) - n_missing}/{len(pool)} pool items")
    return out


def load_bm_correctness(pool: list[dict]) -> dict[int, bool]:
    """Return {row_id: is_BM_correct}. BM-error if patient_id appears in error file."""
    err = json.loads(ERR_FILE.read_text())
    err_pids = {int(r["patient_id"]) for r in err}
    out = {p["row_id"]: (p["patient_id"] not in err_pids) for p in pool}
    n_correct = sum(out.values())
    print(f"  BM-correct: {n_correct}/{len(pool)}  BM-error: {len(pool) - n_correct}/{len(pool)}")
    return out


def encode_nomic(texts: list[str], cache_name: str) -> np.ndarray:
    cache = EMB_CACHE / f"nomic_{cache_name}.npy"
    if cache.exists():
        print(f"  cache hit: {cache.name}")
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    print(f"  encoding {len(texts)} with {NOMIC_MODEL} → {cache_name}...")
    t0 = time.monotonic()
    model = SentenceTransformer(NOMIC_MODEL, trust_remote_code=True, device="cuda")
    embs = model.encode(texts, batch_size=16, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    print(f"    done in {time.monotonic()-t0:.1f}s  shape={embs.shape}")
    np.save(cache, embs)
    del model
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return embs


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(len(a | b), 1)


def get_ner_set(rec: dict) -> set:
    """Flatten 5 categories of NER entities into a single set."""
    e = rec.get("entities", {})
    if not isinstance(e, dict): return set()
    out = set()
    for cat in ["medications", "doses", "procedures", "lab_values", "diagnoses"]:
        v = e.get(cat, [])
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    out.add(x.lower().strip().rstrip(".,;:"))
    return out


def main():
    print("Loading pool, BM zs, BM correctness, NER, gold pairs...")
    pool = load_pool()
    bm_zs = load_bm_zs(pool)
    bm_correct = load_bm_correctness(pool)

    ner_rows = [json.loads(l) for l in NER_FILE.open() if l.strip()]
    ner_by_id = {r["row_id"]: get_ner_set(r) for r in ner_rows}

    gold = [json.loads(l) for l in GOLD.open() if l.strip()]
    gold = [g for g in gold if isinstance(g.get("scores"), dict) and
            all(g["scores"].get(d) is not None for d in DIMS)]
    print(f"  pool={len(pool)}  gold pairs={len(gold)}")

    # Embeddings
    print("\nEncoding nomic embeddings...")
    embs_q = encode_nomic([p["question"] for p in pool], "question")
    embs_note = np.load(EMB_CACHE / "nomic_note.npy")
    embs_gt = np.load(EMB_CACHE / "nomic_ground_truth.npy")
    bm_zs_texts = [bm_zs.get(p["row_id"], "") for p in pool]
    embs_bm_zs = encode_nomic(bm_zs_texts, "bm_zs")

    # Per-pair feature computation
    print("\nComputing per-pair features for all 600 gold pairs...")
    rows = []
    for g in gold:
        a = g["anchor_row_id"]; c = g["candidate_row_id"]
        a_correct = bm_correct.get(a, False)
        c_correct = bm_correct.get(c, False)
        cos_q = float(embs_q[a] @ embs_q[c])
        cos_note = float(embs_note[a] @ embs_note[c])
        cos_gt_pool = float(embs_gt[a] @ embs_gt[c])  # original 4-comp uses this
        cos_bm_a_b = float(embs_bm_zs[a] @ embs_bm_zs[c])  # Match #1
        cos_bm_a_gt_c = float(embs_bm_zs[a] @ embs_gt[c])  # Match #2 (anchor zs ↔ pool GT)
        jac = jaccard(ner_by_id.get(a, set()), ner_by_id.get(c, set()))
        composite = float(np.mean([g["scores"][d] for d in DIMS]))
        rows.append({
            "anchor": a, "candidate": c,
            "a_correct": a_correct, "c_correct": c_correct,
            "stratum": ("correct-correct" if a_correct and c_correct
                        else "error-error" if not a_correct and not c_correct
                        else "correct-error" if a_correct and not c_correct
                        else "error-correct"),
            "cos_q": cos_q, "cos_note": cos_note,
            "cos_gt_pool": cos_gt_pool,
            "cos_bm_a_b": cos_bm_a_b,
            "cos_bm_a_gt_c": cos_bm_a_gt_c,
            "jacc_ner": jac,
            **{f"score_{d}": int(g["scores"][d]) for d in DIMS},
            "composite": composite,
        })
    df = pd.DataFrame(rows)
    print(f"  feature df: {df.shape}")
    print(f"\n  Stratum counts:")
    print(df["stratum"].value_counts().to_string())

    # Helper to compute Spearman ρ vs composite over a subset
    def rho_composite(sub: pd.DataFrame, score_col: str) -> tuple[float, int]:
        if len(sub) < 5: return (float("nan"), len(sub))
        rho, _ = spearmanr(sub[score_col], sub["composite"])
        return (round(float(rho) if not np.isnan(rho) else 0.0, 3), len(sub))

    print("\n=== Test 0.1: input-only 3-comp scorer (cos_q + cos_note + jacc_NER) ===")
    df["s_input_only"] = df["cos_q"] + df["cos_note"] + df["jacc_ner"]
    overall_rho, n = rho_composite(df, "s_input_only")
    print(f"  overall: \u03c1={overall_rho}  (n={n})")

    print("\n=== Test 0.2 + 0.3: stratified analysis ===")
    print(f"  {'stratum':18s}  {'n':>4}  {'cos_q':>6} {'cos_note':>9} {'cos_gt_pool':>12} {'cos_bm_a_b':>11} {'cos_bm_a_gt_c':>13} {'jacc':>6} {'input_only':>11}")
    for stratum in ["error-error", "correct-correct", "correct-error", "error-correct"]:
        sub = df[df["stratum"] == stratum]
        if len(sub) < 5:
            print(f"  {stratum:18s}  {len(sub):>4}  (too small)")
            continue
        cells = []
        for col in ["cos_q", "cos_note", "cos_gt_pool", "cos_bm_a_b", "cos_bm_a_gt_c", "jacc_ner", "s_input_only"]:
            r, _ = rho_composite(sub, col)
            cells.append(f"{r:>6.3f}" if col not in ["cos_gt_pool", "cos_bm_a_b", "cos_bm_a_gt_c"]
                         else (f"{r:>11.3f}" if col == "cos_bm_a_b"
                               else f"{r:>12.3f}" if col == "cos_gt_pool"
                               else f"{r:>13.3f}"))
        print(f"  {stratum:18s}  {len(sub):>4}  " + " ".join(cells[:6]) + " " + cells[6])

    # === Test 0.4: composite scorer with each match strategy ===
    print("\n=== Test 0.4: composite scorers (input + one answer-side) ===")
    df["s_input_plus_zs_zs"] = df["s_input_only"] + df["cos_bm_a_b"]
    df["s_input_plus_zs_gt"] = df["s_input_only"] + df["cos_bm_a_gt_c"]
    df["s_input_plus_gt_gt"] = df["s_input_only"] + df["cos_gt_pool"]
    df["s_full_5comp"] = df["s_input_only"] + df["cos_bm_a_b"] + df["cos_bm_a_gt_c"]

    def rho_per_dim(sub: pd.DataFrame, score_col: str) -> dict:
        out = {}
        for d in DIMS:
            r, _ = rho_composite(sub.assign(composite=sub[f"score_{d}"]), score_col)
            out[d] = r
        rc, _ = rho_composite(sub, score_col)
        out["composite"] = rc
        return out

    methods = {
        "0.1 input_only (cos_q+cos_note+jacc)": "s_input_only",
        "0.4a + cos_gt_pool (orig 4-comp)":     "s_input_plus_gt_gt",
        "0.4b + cos(BM_zs_a, BM_zs_b) [Match#1]": "s_input_plus_zs_zs",
        "0.4c + cos(BM_zs_a, GT_b) [Match#2]":   "s_input_plus_zs_gt",
        "0.4d + both (full 5-comp)":             "s_full_5comp",
    }

    print(f"\n  Per-method, ρ vs each dim (overall, all 600 pairs):")
    out_methods = {}
    for label, col in methods.items():
        d = rho_per_dim(df, col)
        out_methods[label] = d
        print(f"  {label:40s}  q={d['question_type_match']:.3f}  err={d['error_type_match']:.3f}  "
              f"clin={d['clinical_context_similarity']:.3f}  gt={d['ground_truth_alignment']:.3f}  "
              f"crit={d['critical_detail_overlap']:.3f}  COMP={d['composite']:.3f}")

    # === Test 0.5: same per-method but stratified ===
    print(f"\n=== Test 0.5: composite ρ per method, per stratum ===")
    print(f"  {'method':45s}  {'overall':>8}  {'err-err':>8}  {'cor-cor':>8}  {'cor-err':>8}  {'err-cor':>8}")
    strat_table = {}
    for label, col in methods.items():
        row = {"overall": rho_composite(df, col)[0]}
        for stratum in ["error-error", "correct-correct", "correct-error", "error-correct"]:
            sub = df[df["stratum"] == stratum]
            r, _ = rho_composite(sub, col)
            row[stratum] = r
        strat_table[label] = row
        print(f"  {label:45s}  {row['overall']:>8.3f}  {row['error-error']:>8.3f}  "
              f"{row['correct-correct']:>8.3f}  {row['correct-error']:>8.3f}  "
              f"{row['error-correct']:>8.3f}")

    # Save
    summary = {
        "n_pairs": len(df),
        "stratum_counts": df["stratum"].value_counts().to_dict(),
        "method_overall_rho": out_methods,
        "stratified_rho_by_method": strat_table,
    }
    (OUT / "phase0_results.json").write_text(json.dumps(summary, indent=2, default=str))
    df.to_csv(OUT / "phase0_features.csv", index=False)

    # Build markdown
    rows_md = []
    for label, m_dict in out_methods.items():
        rows_md.append({"method": label, **m_dict})
    df_md = pd.DataFrame(rows_md).set_index("method")
    md = df_md.to_markdown(floatfmt=".3f")
    (OUT / "phase0_table.md").write_text(md + "\n")
    print(f"\nSaved: {OUT/'phase0_results.json'}")
    print(f"       {OUT/'phase0_features.csv'}")
    print(f"       {OUT/'phase0_table.md'}")


if __name__ == "__main__":
    main()
