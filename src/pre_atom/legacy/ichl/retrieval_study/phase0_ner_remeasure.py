"""Remeasure composite ρ on the 600 gold pairs with 4 NER variants.

Locked scorer formula (unchanged): cos_q + cos_note + cos(BM_zs_a, BM_zs_b) + jacc_NER
We swap only jacc_NER, computed from each of:

  V0  Magistral (current locked baseline)         -> reproduces ρ=0.656
  V1  GPT-4o-mini soft-normalized (ner_pool.jsonl)
  V2  GPT-4o-mini PLAIN extraction (raw strings)
  V3  GPT-4o-mini PLAIN extraction + canonicalized hybrid CUI-or-string @ 0.70

Also reports the 3-cosine no-NER baseline for reference.

Output: output/ichl/retrieval_study/phase0_ner_remeasure.json + table.md
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
POOL_DIR = RS / "pool_index"
FEATURES_CSV = RS / "phase0_features.csv"

NER_MAGISTRAL = RS / "ner_magistral.jsonl"           # V0 — uses fold_0/train row_id
NER_SOFT_NORM = POOL_DIR / "ner_pool.jsonl"          # V1 — uses items row_id (current GPT-4o-mini soft-normalized)
NER_PLAIN     = POOL_DIR / "ner_gpt4o_mini_plain_962.jsonl"  # V2 (and source for V3)
ITEMS_FILE    = POOL_DIR / "items.jsonl"
FOLD_TRAIN    = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"

OUT_JSON = RS / "phase0_ner_remeasure.json"
OUT_MD   = RS / "phase0_ner_remeasure_table.md"
OUT_CANONICAL = POOL_DIR / "ner_pool_canonical.jsonl"  # V3 saved here

DIMS = ["question_type_match", "error_type_match", "clinical_context_similarity",
        "ground_truth_alignment", "critical_detail_overlap"]
CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]
SIM_THRESHOLD = 0.70


def normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip()).strip(".,;:")


def strip_for_lookup(s: str, c: str) -> str:
    if c not in ("doses", "lab_values"):
        return s
    return re.split(r"[—\-=:|/]", s.strip(), maxsplit=1)[0].strip() or s


def get_strings(rec: dict) -> list[str]:
    """Flatten 5 categories to a list of normalized strings (with head-extraction for doses/labs)."""
    e = rec.get("entities", {})
    if not isinstance(e, dict) or "_err" in e:
        return []
    out = []
    for c in CATS:
        v = e.get(c, [])
        if not isinstance(v, list): continue
        for x in v:
            if not isinstance(x, str): continue
            head = strip_for_lookup(x, c)
            n = normalize_str(head)
            if n:
                out.append(n)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(len(a | b), 1)


def main():
    print("Loading existing phase0_features.csv (cos_q, cos_note, cos_bm_a_b, gold composite)...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df)} pairs  cols={list(df.columns)[:12]}...")

    # === Build mapping: fold_0/train row_id -> patient_id -> items row_id ===
    fold_train = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    items = [json.loads(l) for l in ITEMS_FILE.open() if l.strip()]
    fold_rid_to_pid = {i: int(r["patient_id"]) for i, r in enumerate(fold_train)}
    pid_to_item_rid = {int(r["patient_id"]): int(r["row_id"]) for r in items}
    fold_rid_to_item_rid = {fr: pid_to_item_rid[pid] for fr, pid in fold_rid_to_pid.items()
                            if pid in pid_to_item_rid}
    print(f"  fold_0/train: {len(fold_train)}, items: {len(items)}, mappable fold_rids: {len(fold_rid_to_item_rid)}")

    # === Load NER variants ===
    print("\nLoading NER variants...")

    # V0: Magistral, indexed by fold_0/train row_id
    v0_strs = {}
    if NER_MAGISTRAL.exists():
        for l in NER_MAGISTRAL.open():
            r = json.loads(l)
            v0_strs[int(r["row_id"])] = set(get_strings(r))
        print(f"  V0 Magistral: {len(v0_strs)} (by fold_0/train rid)")

    # V1: GPT-4o-mini soft-normalized, indexed by items row_id
    v1_strs = {}
    if NER_SOFT_NORM.exists():
        for l in NER_SOFT_NORM.open():
            r = json.loads(l)
            v1_strs[int(r["row_id"])] = set(get_strings(r))
        print(f"  V1 soft-normalized: {len(v1_strs)} (by items rid)")

    # V2: plain (NEW)
    v2_strs = {}
    if NER_PLAIN.exists():
        for l in NER_PLAIN.open():
            r = json.loads(l)
            v2_strs[int(r["row_id"])] = set(get_strings(r))
        print(f"  V2 plain raw: {len(v2_strs)} (by items rid)")
    else:
        print(f"  V2 plain raw: NOT FOUND — skipping")
        v2_strs = None

    # === V3: canonicalize V2 strings via SciSpacy linker ===
    v3_sets = None
    if v2_strs is not None:
        print(f"\nV3: Canonicalizing V2 unique strings via SciSpacy linker (threshold {SIM_THRESHOLD})...")
        # Universe of all unique strings across V2
        universe = set()
        for s in v2_strs.values():
            universe.update(s)
        universe_list = sorted(universe)
        print(f"  unique strings to canonicalize: {len(universe_list)}")

        from scispacy.candidate_generation import CandidateGenerator
        t0 = time.monotonic()
        cg = CandidateGenerator(name="umls")
        print(f"  CandidateGenerator loaded in {time.monotonic()-t0:.1f}s")

        t0 = time.monotonic()
        cands = cg(universe_list, 1)
        print(f"  lookup done in {time.monotonic()-t0:.1f}s")
        sim_by_str = {}
        cui_by_str = {}
        for s, cs in zip(universe_list, cands):
            if cs and cs[0].similarities:
                sim_by_str[s] = cs[0].similarities[0]
                cui_by_str[s] = cs[0].concept_id
            else:
                sim_by_str[s] = 0.0
                cui_by_str[s] = None
        n_mapped = sum(1 for s in universe_list
                       if cui_by_str.get(s) and sim_by_str.get(s, 0) >= SIM_THRESHOLD)
        print(f"  mapped={n_mapped}/{len(universe_list)} ({100*n_mapped/max(len(universe_list),1):.1f}%) "
              f"at sim>={SIM_THRESHOLD}")

        def to_hybrid(strs):
            toks = set()
            for s in strs:
                if cui_by_str.get(s) and sim_by_str.get(s, 0) >= SIM_THRESHOLD:
                    toks.add(f"cui:{cui_by_str[s]}")
                else:
                    toks.add(f"str:{s}")
            return toks

        v3_sets = {rid: to_hybrid(strs) for rid, strs in v2_strs.items()}

        # Save the canonicalized pool for downstream use
        with OUT_CANONICAL.open("w") as f:
            for rid in sorted(v3_sets.keys()):
                f.write(json.dumps({
                    "row_id": rid,
                    "patient_id": next((int(it["patient_id"]) for it in items if int(it["row_id"]) == rid), None),
                    "tokens": sorted(v3_sets[rid]),
                }) + "\n")
        print(f"  saved canonical pool: {OUT_CANONICAL}")

    # === Compute Jaccard column for each variant on the 600 gold pairs ===
    print("\nComputing Jaccard for each variant on the 600 gold pairs...")
    def jacc_for_pair(a_fold_rid, c_fold_rid, source, by_fold=True):
        if source is None: return None
        if by_fold:
            a_key, c_key = a_fold_rid, c_fold_rid
        else:
            a_key = fold_rid_to_item_rid.get(a_fold_rid)
            c_key = fold_rid_to_item_rid.get(c_fold_rid)
            if a_key is None or c_key is None: return None
        return jaccard(source.get(a_key, set()), source.get(c_key, set()))

    df["jacc_v0_magistral"] = df.apply(lambda r: jacc_for_pair(int(r["anchor"]), int(r["candidate"]),
                                                                v0_strs, by_fold=True), axis=1)
    df["jacc_v1_softnorm"] = df.apply(lambda r: jacc_for_pair(int(r["anchor"]), int(r["candidate"]),
                                                               v1_strs, by_fold=False), axis=1)
    if v2_strs is not None:
        df["jacc_v2_plain_raw"] = df.apply(lambda r: jacc_for_pair(int(r["anchor"]), int(r["candidate"]),
                                                                    v2_strs, by_fold=False), axis=1)
    if v3_sets is not None:
        df["jacc_v3_plain_canonical"] = df.apply(lambda r: jacc_for_pair(int(r["anchor"]), int(r["candidate"]),
                                                                          v3_sets, by_fold=False), axis=1)

    # === Composite scorers ===
    df["s_3cos_only"] = df["cos_q"] + df["cos_note"] + df["cos_bm_a_b"]
    df["s_v0"] = df["s_3cos_only"] + df["jacc_v0_magistral"]
    df["s_v1"] = df["s_3cos_only"] + df["jacc_v1_softnorm"]
    if v2_strs is not None:
        df["s_v2"] = df["s_3cos_only"] + df["jacc_v2_plain_raw"]
    if v3_sets is not None:
        df["s_v3"] = df["s_3cos_only"] + df["jacc_v3_plain_canonical"]

    # === Rho calc helper ===
    def rho_per_dim(score_col):
        out = {}
        for d in DIMS:
            r, _ = spearmanr(df[score_col], df[f"score_{d}"])
            out[d] = round(float(r), 3) if not np.isnan(r) else 0.0
        r, _ = spearmanr(df[score_col], df["composite"])
        out["composite"] = round(float(r), 3) if not np.isnan(r) else 0.0
        return out

    # Also track ρ of jacc itself (independent contribution)
    def jacc_rho(jacc_col):
        if jacc_col not in df.columns: return None
        out = {"composite": round(float(spearmanr(df[jacc_col], df["composite"])[0]), 3)}
        for d in DIMS:
            r, _ = spearmanr(df[jacc_col], df[f"score_{d}"])
            out[d] = round(float(r), 3) if not np.isnan(r) else 0.0
        return out

    print("\n=== Composite ρ vs each gold dimension (per scorer variant) ===")
    print(f"  {'variant':50s}  q       err     clin    gt      crit    COMP")
    rows_md = []
    methods = [
        ("3-cos only (no NER baseline)",                 "s_3cos_only"),
        ("V0  + jacc Magistral (locked, current 0.656)", "s_v0"),
        ("V1  + jacc GPT-4o-mini soft-normalized",       "s_v1"),
    ]
    if v2_strs is not None: methods.append(("V2  + jacc GPT-4o-mini PLAIN raw",                "s_v2"))
    if v3_sets is not None: methods.append(("V3  + jacc PLAIN + canonicalized hybrid @0.70",   "s_v3"))

    out_methods = {}
    for label, col in methods:
        d = rho_per_dim(col)
        out_methods[label] = d
        print(f"  {label:50s}  {d['question_type_match']:.3f}   {d['error_type_match']:.3f}   "
              f"{d['clinical_context_similarity']:.3f}   {d['ground_truth_alignment']:.3f}   "
              f"{d['critical_detail_overlap']:.3f}   {d['composite']:.3f}")
        rows_md.append({"method": label, **d})

    # ρ of jacc column alone (how much info each Jaccard carries on its own)
    print("\n=== ρ of jacc COLUMN ALONE vs each gold dimension ===")
    jacc_methods = [
        ("V0 jacc Magistral",       "jacc_v0_magistral"),
        ("V1 jacc soft-normalized", "jacc_v1_softnorm"),
    ]
    if v2_strs is not None: jacc_methods.append(("V2 jacc PLAIN raw",          "jacc_v2_plain_raw"))
    if v3_sets is not None: jacc_methods.append(("V3 jacc PLAIN canonical",    "jacc_v3_plain_canonical"))
    out_jacc = {}
    for label, col in jacc_methods:
        d = jacc_rho(col)
        out_jacc[label] = d
        print(f"  {label:30s}  q={d['question_type_match']:.3f}  err={d['error_type_match']:.3f}  "
              f"clin={d['clinical_context_similarity']:.3f}  gt={d['ground_truth_alignment']:.3f}  "
              f"crit={d['critical_detail_overlap']:.3f}  COMP={d['composite']:.3f}")

    # Save
    summary = {
        "n_pairs": len(df),
        "scorer_formula": "cos_q + cos_note + cos(BM_zs_a, BM_zs_b) + jacc_NER",
        "sim_threshold_v3": SIM_THRESHOLD,
        "scorer_rho": out_methods,
        "jacc_only_rho": out_jacc,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(rows_md).set_index("method").to_markdown(OUT_MD, floatfmt=".3f")
    print(f"\nSaved: {OUT_JSON}")
    print(f"       {OUT_MD}")
    if v3_sets is not None:
        print(f"       {OUT_CANONICAL}")


if __name__ == "__main__":
    main()
