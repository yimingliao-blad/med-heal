"""Data loader for T0 correction — join Step-8 zeroshot with detection JSONL.

Produces a list of per-item dicts ready for the correction runner:
    {
        pilot_item_id, patient_id, fold,
        note, question, A0, A0_binary_correct,
        target_model,
    }

Filters to `chosen_verdict == "INCORRECT"` from the detection JSONL.

The detection JSONL file path is target-specific — callers pass the exact
path. This keeps Stage II decoupled from Stage I cell selection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ichl.common.pilot_loader import load_detection_fullscale


def _load_detection_flags(path: Path, only_verdict: str = "INCORRECT") -> set[tuple[int, int]]:
    """Return set of (fold, patient_id) with chosen_verdict == only_verdict."""
    flagged: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("chosen_verdict") == only_verdict:
                flagged.add((int(r["fold"]), int(r["patient_id"])))
    return flagged


def load_correction_items(
    target_model: str,
    detection_jsonl_path: Path,
    only_verdict: str = "INCORRECT",
) -> list[dict[str, Any]]:
    """Load items flagged by the detection cell for correction.

    Args:
        target_model: target internal key (e.g. 'deepseek-r1-distill-llama-8b').
        detection_jsonl_path: Stage-I fullscale result JSONL for the chosen cell.
        only_verdict: keep rows where `chosen_verdict == this`. Default INCORRECT.

    Returns:
        list of dicts, one per flagged item, in deterministic (fold, patient_id) order.
    """
    flagged_keys = _load_detection_flags(Path(detection_jsonl_path), only_verdict=only_verdict)
    if not flagged_keys:
        return []

    # Pull the full EHRNoteQA+Step-8 join (962 items) for the target.
    fullscale = load_detection_fullscale(target_model)
    # Index by (fold, patient_id).
    by_key = {(int(it["fold"]), int(it["patient_id"])): it for it in fullscale}

    items: list[dict[str, Any]] = []
    missing: list[tuple[int, int]] = []
    for key in sorted(flagged_keys):
        src = by_key.get(key)
        if src is None:
            missing.append(key)
            continue
        items.append({
            "pilot_item_id": src["pilot_item_id"],
            "patient_id": src["patient_id"],
            "fold": src["fold"],
            "note": src["note"],
            "question": src["question"],
            "A0": src["model_answer"],
            "A0_binary_correct": int(src["binary_correct"]),
            "target_model": target_model,
        })

    if missing:
        # Informational — shouldn't happen since both sources derive from the same 962.
        print(f"[data_loader] WARN: {len(missing)} detection keys not found in fullscale "
              f"join (first 3: {missing[:3]})")
    return items
