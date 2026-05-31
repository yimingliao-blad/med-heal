"""Cell builder for Stage III Verdict.

Constructs verdict pairs from existing judgments (no new compute):
- step8 zeroshot answers + binary_correct labels for 5 targets (Qwen2.5/Qwen3/DS-R1/Llama/BioMistral)
- 4corrector_compare T0.a + v4 corrected outputs + judge labels (Qwen2.5/DS-R1/Llama)

Each pair has known-correct label per candidate. Verdict's job is to pick the labeled-correct one.
Cells: FIX (one wrong, one right) / BREAK (one right, one wrong) / stay-right (both right) / stay-wrong (both wrong).

Per user 2026-04-30: no synthetic gold gen, no new corrector runs. Reuse existing labels.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STEP8 = ROOT / "output" / "step8"
TRACK_LOC = ROOT / "output" / "ichl" / "correction" / "track_loc" / "4corrector_compare"
NOTES_FILE = ROOT / "output" / "EHRNoteQA_processed.jsonl"

STEP8_TARGETS = [
    "qwen2.5-7b-instruct",
    "qwen3-8b",
    "deepseek-r1-distill-llama-8b",
    "llama-3.1-8b-instruct",
    "biomistral-7b",
]
TRACK_LOC_CORRECTORS = ["qwen2.5-7b-instruct", "deepseek-r1-distill-llama-8b", "llama-3.1-8b-instruct"]


def load_notes() -> dict[int, dict]:
    out = {}
    for line in NOTES_FILE.open():
        if not line.strip(): continue
        r = json.loads(line)
        pid = int(r["patient_id"])
        parts = []
        for i in (1, 2, 3):
            n = r.get(f"note_{i}")
            if n: parts.append(f"[Note {i}]\n{n}")
        out[pid] = {
            "note": "\n\n".join(parts),
            "question": str(r.get("question", "")),
            "ground_truth": str(r.get(f"choice_{r.get('answer','')}", "")),
        }
    return out


def load_step8_zs(target: str) -> dict[int, dict]:
    """patient_id -> {fold_id, model_answer, binary_correct, source_id}"""
    out = {}
    for f in range(5):
        p = STEP8 / target / f"fold_{f}" / "zeroshot_evaluated_binary.csv"
        if not p.exists(): continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            pid = int(r["patient_id"])
            bc = r.get("binary_correct")
            if bc not in (0, 1): continue
            out[pid] = {
                "patient_id": pid, "fold_id": f,
                "answer_text": str(r["model_answer"] or ""),
                "label": int(bc),
                "source_id": f"{target}::zs",
            }
    return out


def load_track_loc(corrector: str, version: str) -> dict[int, dict]:
    """patient_id -> {fold_id, corrected_answer, corrected_correct, source_id}.
    version in {'t0a', 'v4'}. Corrector ran only on Qwen2.5's 109 wrong items.
    """
    out = {}
    for f in range(5):
        p_corr = TRACK_LOC / corrector / f"fold_{f}" / f"corrected_{version}.jsonl"
        p_judg = TRACK_LOC / "judge" / corrector / f"fold_{f}" / f"judged_{version}.jsonl"
        if not p_corr.exists() or not p_judg.exists(): continue
        corr = {int(json.loads(l)["patient_id"]): json.loads(l) for l in p_corr.open()}
        for line in p_judg.open():
            j = json.loads(line)
            pid = int(j["patient_id"])
            bc = j.get("corrected_correct")
            if bc not in (0, 1): continue
            ans = corr.get(pid, {}).get("corrected_answer", "")
            out[pid] = {
                "patient_id": pid, "fold_id": f,
                "answer_text": ans, "label": int(bc),
                "source_id": f"{corrector}::{version}",
            }
    return out


def all_candidates() -> dict[int, list[dict]]:
    """patient_id -> list of candidate records (each with answer_text, label, source_id)."""
    by_pid = defaultdict(list)
    for tgt in STEP8_TARGETS:
        for pid, rec in load_step8_zs(tgt).items():
            by_pid[pid].append(rec)
    for cor in TRACK_LOC_CORRECTORS:
        for ver in ("t0a", "v4"):
            for pid, rec in load_track_loc(cor, ver).items():
                by_pid[pid].append(rec)
    return by_pid


def build_pairs(
    pair_type: str = "zs_vs_v4",
    primary_target: str = "qwen2.5-7b-instruct",
    rng_seed: int = 42,
) -> list[dict]:
    """Build (A_text, B_text, A_label, B_label, A_source, B_source, gold_pick) pairs.

    pair_type:
      'zs_vs_t0a': Qwen2.5 zs vs Qwen2.5 T0.a regen (same target self-correction)
      'zs_vs_v4':  Qwen2.5 zs vs Qwen2.5 v4 corrected
      'cross_zs':  Qwen2.5 zs vs another model's zs (uses any target with available label)
                   — gives broadest cell coverage including BREAK / stay-right.
    """
    notes = load_notes()
    rng = random.Random(rng_seed)
    pairs = []

    if pair_type == "zs_vs_t0a":
        zs = load_step8_zs(primary_target)
        alt = load_track_loc(primary_target, "t0a")
        common = set(zs) & set(alt)
        for pid in common:
            pairs.append(_make_pair(notes, pid, zs[pid], alt[pid], rng))

    elif pair_type == "zs_vs_v4":
        zs = load_step8_zs(primary_target)
        alt = load_track_loc(primary_target, "v4")
        common = set(zs) & set(alt)
        for pid in common:
            pairs.append(_make_pair(notes, pid, zs[pid], alt[pid], rng))

    elif pair_type == "cross_zs":
        # For each item, pair primary_target zs with each other target's zs.
        primary = load_step8_zs(primary_target)
        for other in STEP8_TARGETS:
            if other == primary_target: continue
            other_zs = load_step8_zs(other)
            for pid, pr in primary.items():
                if pid not in other_zs: continue
                pairs.append(_make_pair(notes, pid, pr, other_zs[pid], rng))

    else:
        raise ValueError(f"Unknown pair_type: {pair_type}")

    return pairs


def _make_pair(notes, pid, A_rec, B_rec, rng):
    """Apply position randomization. A/B in output is shuffled; A_id/B_id record source."""
    n = notes.get(pid, {})
    if rng.random() < 0.5:
        A_text, B_text = A_rec["answer_text"], B_rec["answer_text"]
        A_label, B_label = A_rec["label"], B_rec["label"]
        A_source, B_source = A_rec["source_id"], B_rec["source_id"]
    else:
        A_text, B_text = B_rec["answer_text"], A_rec["answer_text"]
        A_label, B_label = B_rec["label"], A_rec["label"]
        A_source, B_source = B_rec["source_id"], A_rec["source_id"]

    # Cell classification + gold pick
    if A_label == 1 and B_label == 0:
        cell = "FIX_or_BREAK"; gold_pick = "A"
    elif A_label == 0 and B_label == 1:
        cell = "FIX_or_BREAK"; gold_pick = "B"
    elif A_label == 1 and B_label == 1:
        cell = "stay_right"; gold_pick = "EITHER"
    else:  # 0,0
        cell = "stay_wrong"; gold_pick = "NEITHER"

    return {
        "patient_id": pid, "fold_id": A_rec["fold_id"],
        "note": n.get("note", ""), "question": n.get("question", ""),
        "ground_truth": n.get("ground_truth", ""),
        "candidate_A": A_text, "candidate_B": B_text,
        "A_label": A_label, "B_label": B_label,
        "A_source": A_source, "B_source": B_source,
        "A_length": len(A_text), "B_length": len(B_text),
        "cell": cell, "gold_pick": gold_pick,
        "primary_zs_label": A_rec["label"] if "::zs" in A_rec["source_id"] else B_rec["label"],
    }


def stratified_pilot(pairs: list[dict], n_per_cell: int = 10, rng_seed: int = 42) -> list[dict]:
    """Sample n_per_cell items from each detailed cell:
    FIX  (zs=0, alt=1)
    BREAK(zs=1, alt=0)
    stay_right (both=1)
    stay_wrong (both=0)
    """
    rng = random.Random(rng_seed)
    by_cell = defaultdict(list)
    for p in pairs:
        zs_label = p["primary_zs_label"]
        # alt label = whichever isn't the zs source
        alt_label = p["B_label"] if "::zs" in p["A_source"] else p["A_label"]
        if zs_label == 0 and alt_label == 1:
            cell = "FIX"
        elif zs_label == 1 and alt_label == 0:
            cell = "BREAK"
        elif zs_label == 1 and alt_label == 1:
            cell = "stay_right"
        else:
            cell = "stay_wrong"
        p["detail_cell"] = cell
        by_cell[cell].append(p)

    out = []
    for cell, lst in by_cell.items():
        rng.shuffle(lst)
        out.extend(lst[:n_per_cell])
    return out


if __name__ == "__main__":
    import sys
    pair_type = sys.argv[1] if len(sys.argv) > 1 else "zs_vs_v4"
    pairs = build_pairs(pair_type)
    print(f"{pair_type}: {len(pairs)} pairs total")
    pilot = stratified_pilot(pairs, n_per_cell=10)
    from collections import Counter
    print(f"  cells: {Counter(p['detail_cell'] for p in pairs)}")
    print(f"  pilot 10/cell: {Counter(p['detail_cell'] for p in pilot)}")
