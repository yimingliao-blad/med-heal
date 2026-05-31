#!/usr/bin/env python3
"""Extend judge_agreement_step2_T0.json with the trio-overlap patients that
were excluded from the original Sara∩Jose (A==B) gold subset.

These are the items where Sara and Jose disagree but Caitlin also annotated
them, so a 3-rater majority gold can still be defined. Without GPT-4o
judgments for these items, we can't build a clean N=70 trio table.

Reuses the canonical Stage-1 judge from judge.py — same prompt, T=0, n=1.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
HUMAN_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "external"
    / "all_users_openended_BioMistral-7B_1775740232208.csv"
)

REVIEWERS = {
    "A": "Sara Saif",
    "B": "Jose E. Lizarraga Mazab",
    "C": "Caitlin Schwanke",
}


def load_trio_pids_with_majority() -> pd.DataFrame:
    """Return DataFrame of patient_ids annotated by all 3 reviewers, with
    majority-vote label. Columns: patient_id, A, B, C, majority."""
    df = pd.read_csv(HUMAN_CSV)
    df = df.sort_values("Timestamp").drop_duplicates(["User Name", "Patient ID"], keep="last")
    df["bin"] = (df["Answer Quality"] == 5).astype(int)
    piv = (
        df.pivot_table(index="Patient ID", columns="User Name", values="bin", aggfunc="first")
        [list(REVIEWERS.values())]
        .dropna()
        .astype(int)
        .reset_index()
        .rename(columns={"Patient ID": "patient_id"})
    )
    piv["majority"] = (piv[list(REVIEWERS.values())].sum(axis=1) >= 2).astype(int)
    return piv


def main() -> int:
    if not JSON_PATH.exists():
        print(f"!! {JSON_PATH} not found", flush=True)
        return 1
    with open(JSON_PATH) as f:
        data = json.load(f)
    existing_pids = {int(it["patient_id"]) for it in data["per_item"]}
    print(f"Existing per_item entries: {len(existing_pids)}", flush=True)

    trio = load_trio_pids_with_majority()
    print(f"Trio (3-rater) overlap: {len(trio)} patients", flush=True)

    missing = trio[~trio["patient_id"].isin(existing_pids)].copy()
    print(f"Missing GPT-4o judgments for: {len(missing)} patients", flush=True)
    if missing.empty:
        print("Nothing to do.", flush=True)
        return 0

    print("Loading BioMistral step2 rows + notes...", flush=True)
    bm = _load_biomistral_step2_rows()
    bm["patient_id"] = bm["patient_id"].astype(int)
    notes = _load_notes_lookup()

    bm_idx = bm.set_index("patient_id")
    new_items: list[dict] = []
    for i, row in missing.iterrows():
        pid = int(row["patient_id"])
        if pid not in bm_idx.index:
            print(f"  !! pid {pid} missing from BM step2 rows, skipping", flush=True)
            continue
        bm_row = bm_idx.loc[pid]
        note = notes.get(str(pid), "")
        if not note:
            print(f"  !! no note for pid {pid}, skipping", flush=True)
            continue
        result = judge(
            note,
            bm_row["question"],
            bm_row["ground_truth"],
            str(bm_row["model_answer"]),
            n=1,
            temperature=0.0,
        )
        item = {
            "patient_id": pid,
            "gold_label": int(row["majority"]),
            "gpt4o_label_T0": result["label"],
            "gpt4o_raws": result["raws"],
            "unanimity": result["unanimity"],
            "source_note": "trio_majority_extension",
            "reviewer_labels": {
                "A_Sara": int(row[REVIEWERS["A"]]),
                "B_Jose": int(row[REVIEWERS["B"]]),
                "C_Caitlin": int(row[REVIEWERS["C"]]),
            },
        }
        new_items.append(item)
        print(
            f"  judged pid {pid}: gpt={result['label']} "
            f"(A={item['reviewer_labels']['A_Sara']} "
            f"B={item['reviewer_labels']['B_Jose']} "
            f"C={item['reviewer_labels']['C_Caitlin']} "
            f"maj={item['gold_label']})",
            flush=True,
        )
        time.sleep(0.5)

        # Persist incrementally so a crash doesn't lose work
        data.setdefault("per_item_extension_trio", [])
        # Replace if exists
        data["per_item_extension_trio"] = [
            x for x in data["per_item_extension_trio"] if int(x["patient_id"]) != pid
        ] + [item]
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)

    print(f"\nDone. Added {len(new_items)} extension judgments.", flush=True)
    print(f"  Wrote {JSON_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
