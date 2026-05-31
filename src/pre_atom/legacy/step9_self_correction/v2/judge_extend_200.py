#!/usr/bin/env python3
"""Build a 200-patient evaluation sample from the 493 human-annotated set:
  - All 175 patients annotated by ≥2 reviewers (full joint coverage)
  - +25 single-rater patients (13 Sara, 12 Jose), random with seed=42

Run GPT-4o Stage-1 binary judge on any patients not yet in
output/step9_v2/judge_agreement_step2_T0.json, append them, and report
per-reviewer agreement & kappa with GPT-4o on the 200-item sample.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.step9_self_correction.v2.judge import (  # noqa: E402
    OUTPUT_DIR,
    _load_biomistral_step2_rows,
    _load_notes_lookup,
    judge,
)

JSON_PATH = OUTPUT_DIR / "judge_agreement_step2_T0.json"
SAMPLE_PATH = OUTPUT_DIR / "sample200_patients.csv"
HUMAN_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "all_users_openended_BioMistral-7B_1775740232208.csv"
)

A, B, C = "Sara Saif", "Jose E. Lizarraga Mazab", "Caitlin Schwanke"
SEED = 42


def build_sample() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_CSV)
    df = df.sort_values("Timestamp").drop_duplicates(["User Name", "Patient ID"], keep="last")
    df["bin"] = (df["Answer Quality"] == 5).astype(int)

    counts = df.groupby("Patient ID")["User Name"].nunique()
    multi_pids = list(counts[counts >= 2].index)
    single_df = df[df["Patient ID"].isin(counts[counts == 1].index)]

    rng = np.random.default_rng(SEED)
    sara_only = single_df[single_df["User Name"] == A]["Patient ID"].tolist()
    jose_only = single_df[single_df["User Name"] == B]["Patient ID"].tolist()
    sara_pick = rng.choice(sara_only, size=13, replace=False).tolist()
    jose_pick = rng.choice(jose_only, size=12, replace=False).tolist()

    sample_pids = sorted(set(multi_pids) | set(sara_pick) | set(jose_pick))
    print(f"Multi-rater (≥2) patients (mandatory): {len(multi_pids)}", flush=True)
    print(f"Single-rater additions: 13 Sara + 12 Jose = 25", flush=True)
    print(f"Total sample: {len(sample_pids)}", flush=True)
    assert len(sample_pids) == 200, f"sample size = {len(sample_pids)}"

    piv = (
        df.pivot_table(index="Patient ID", columns="User Name", values="bin", aggfunc="first")
        .reindex(sample_pids)
        .reset_index()
        .rename(columns={"Patient ID": "patient_id"})
    )
    return piv


def load_existing_judgments() -> dict[int, int]:
    if not JSON_PATH.exists():
        return {}
    with open(JSON_PATH) as f:
        d = json.load(f)
    out: dict[int, int] = {}
    for it in d.get("per_item", []):
        if it.get("gpt4o_label_T0") is not None:
            out[int(it["patient_id"])] = int(it["gpt4o_label_T0"])
    for it in d.get("per_item_extension_trio", []):
        if it.get("gpt4o_label_T0") is not None:
            out[int(it["patient_id"])] = int(it["gpt4o_label_T0"])
    return out


def append_extension(item: dict) -> None:
    """Persist to per_item_extension_sample200, replacing if exists."""
    with open(JSON_PATH) as f:
        data = json.load(f)
    bucket = data.setdefault("per_item_extension_sample200", [])
    bucket = [x for x in bucket if int(x["patient_id"]) != int(item["patient_id"])]
    bucket.append(item)
    data["per_item_extension_sample200"] = bucket
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def fleiss_kappa_2cat(matrix: np.ndarray) -> float:
    """matrix: [N x 2] of category counts per item; assumes equal raters per item."""
    N, kcats = matrix.shape
    n_per_item = matrix.sum(axis=1)
    if not np.all(n_per_item == n_per_item[0]):
        # variable raters — use mean for a rough number
        pass
    k = int(n_per_item[0])
    p_j = matrix.sum(axis=0) / (N * k)
    P_i = ((matrix**2).sum(axis=1) - k) / (k * (k - 1))
    Pbar = P_i.mean()
    Pe = (p_j**2).sum()
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) > 0 else float("nan")


def cohen_kappa(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    po = (x == y).mean()
    cats = [0, 1]
    pe = sum(((x == c).mean()) * ((y == c).mean()) for c in cats)
    return po, ((po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan"))


def main() -> int:
    sample = build_sample()
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote sample to {SAMPLE_PATH}\n", flush=True)

    existing = load_existing_judgments()
    sample_pids = sample["patient_id"].astype(int).tolist()
    missing = [pid for pid in sample_pids if pid not in existing]
    print(f"GPT-4o judgments needed: {len(missing)} / {len(sample_pids)}", flush=True)

    if missing:
        print("Loading BioMistral step2 rows + notes...", flush=True)
        bm = _load_biomistral_step2_rows()
        bm["patient_id"] = bm["patient_id"].astype(int)
        bm_idx = bm.set_index("patient_id")
        notes = _load_notes_lookup()

        sample_idx = sample.set_index("patient_id")
        for i, pid in enumerate(missing, 1):
            if pid not in bm_idx.index:
                print(f"  !! pid {pid} missing from BM step2 rows, skipping", flush=True)
                continue
            row = bm_idx.loc[pid]
            note = notes.get(str(pid), "")
            if not note:
                print(f"  !! no note for pid {pid}, skipping", flush=True)
                continue
            res = judge(
                note, row["question"], row["ground_truth"], str(row["model_answer"]),
                n=1, temperature=0.0,
            )
            srow = sample_idx.loc[pid]
            item = {
                "patient_id": int(pid),
                "gpt4o_label_T0": res["label"],
                "gpt4o_raws": res["raws"],
                "unanimity": res["unanimity"],
                "source_note": "sample200_extension",
                "reviewer_labels": {
                    "A_Sara":    None if pd.isna(srow.get(A)) else int(srow[A]),
                    "B_Jose":    None if pd.isna(srow.get(B)) else int(srow[B]),
                    "C_Caitlin": None if pd.isna(srow.get(C)) else int(srow[C]),
                },
            }
            append_extension(item)
            existing[pid] = res["label"]
            print(
                f"  [{i}/{len(missing)}] pid {pid}: gpt={res['label']}  "
                f"A={item['reviewer_labels']['A_Sara']} "
                f"B={item['reviewer_labels']['B_Jose']} "
                f"C={item['reviewer_labels']['C_Caitlin']}",
                flush=True,
            )
            time.sleep(0.5)

    # ---------- Statistics on the 200-item sample ----------
    sample["gpt"] = sample["patient_id"].astype(int).map(existing)
    print(f"\nFinal sample: {len(sample)}, with GPT judgment: {sample['gpt'].notna().sum()}")
    s = sample.dropna(subset=["gpt"]).copy()
    s["gpt"] = s["gpt"].astype(int)

    print("\n" + "=" * 70)
    print(f"PER-REVIEWER STATS ON SAMPLE (N=200, GPT judged on {len(s)})")
    print("=" * 70)
    print(f"\nGPT-4o true rate (entire sample): {s['gpt'].mean():.3f}")

    rows = []
    for label, name in [("A_Sara", A), ("B_Jose", B), ("C_Caitlin", C)]:
        sub = s[s[name].notna()].copy()
        sub[name] = sub[name].astype(int)
        n = len(sub)
        if n == 0:
            continue
        true_rate = sub[name].mean()
        po, k = cohen_kappa(sub[name].values, sub["gpt"].values)
        fn = int(((sub[name] == 1) & (sub["gpt"] == 0)).sum())
        fp = int(((sub[name] == 0) & (sub["gpt"] == 1)).sum())
        rows.append((label.split("_")[1], n, true_rate, po, k, fn, fp))

    print(f"\n{'Reviewer':<10}{'N':>6}{'True%':>10}{'Agr%':>10}{'kappa':>10}{'FN':>6}{'FP':>6}")
    for r in rows:
        print(f"{r[0]:<10}{r[1]:>6}{r[2]*100:>9.1f}%{r[3]*100:>9.1f}%{r[4]:>10.3f}{r[5]:>6}{r[6]:>6}")

    # Pairwise human kappas (where both annotated)
    print("\nPairwise human-vs-human (within sample):")
    for u1, u2 in [(A, B), (A, C), (B, C)]:
        sub = s[s[u1].notna() & s[u2].notna()]
        if len(sub) == 0:
            continue
        po, k = cohen_kappa(sub[u1].astype(int).values, sub[u2].astype(int).values)
        print(f"  {u1[:14]:<14} vs {u2[:14]:<14}  N={len(sub):<4}  agr={po*100:5.1f}%  κ={k:.3f}")

    # Majority gold (only on items with ≥2 raters)
    print("\nMajority-of-≥2 gold vs GPT-4o:")
    multi = s[s[[A, B, C]].notna().sum(axis=1) >= 2].copy()
    def maj(row):
        vals = [int(row[u]) for u in [A, B, C] if pd.notna(row[u])]
        return int(sum(vals) > len(vals) / 2)
    multi["maj"] = multi.apply(maj, axis=1)
    po, k = cohen_kappa(multi["maj"].values, multi["gpt"].astype(int).values)
    fn = int(((multi["maj"] == 1) & (multi["gpt"] == 0)).sum())
    fp = int(((multi["maj"] == 0) & (multi["gpt"] == 1)).sum())
    print(f"  N={len(multi)}  agr={po*100:.1f}%  κ={k:.3f}  FN={fn}  FP={fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
