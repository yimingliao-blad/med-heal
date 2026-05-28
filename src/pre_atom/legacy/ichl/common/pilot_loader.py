"""Pilot-data loader for the detection verdict-only experiment.

Loads the Step-8 binary-judged zeroshot outputs for one target model, joins
with EHRNoteQA notes + choices from `output/EHRNoteQA_processed.jsonl`, and
returns a deterministic stratified sample as `list[PilotItem]`.

Two entry points:
  - `load_detection_pilot(target_model, n_per_fold=8, n_incorrect_per_fold=4)`
    — the main-pilot sampler (40 items by default, stratified per fold).
  - `load_detection_pilot_sub(target_model, n_total=5, n_incorrect=2)`
    — the parser-design sub-pilot sampler (5 items across folds).

Both sample the SAME pool but with different stratification semantics. Same
`seed` (default `20260422`) is used so two calls with the same args return
the same patient_ids (reproducibility + re-use across target models).

Data-source conventions (see Notion 'Claude: Plan: Detection — Pilot Runner
Design' § Inputs):
  - Ground truth: `output/step8/<target_model>/fold_*/zeroshot_evaluated_binary.csv`
  - Notes + choices: `output/EHRNoteQA_processed.jsonl`

Joined by `patient_id`. Each pilot item is a dict matching the runner
schema; see PilotItem below.
"""
from __future__ import annotations

import glob
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP8_DIR = PROJECT_ROOT / "output" / "step8"
EHRNOTEQA_PATH = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"

# Target-model key → step8 subdir name. Keep in one place so config
# adapters don't re-invent it.
TARGET_SUBDIR: dict[str, str] = {
    "qwen3-8b": "qwen3-8b",
    "qwen2.5-7b": "qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "biomistral-7b": "biomistral-7b",
    "llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8b",
}


class PilotItem(TypedDict):
    """Shape of one pilot record. Kept as TypedDict so JSONL round-trips are trivial."""
    pilot_item_id: str
    fold: int
    patient_id: int
    question: str
    question_type: str
    note: str              # pre-concatenated [Note 1]...[Note 2]... block
    choices: str           # formatted 'A) ...\nB) ...\n...' block or '' if openended
    model_answer: str
    binary_correct: int
    target_model: str


# ─────────────────────── data loading helpers ───────────────────────

def _load_target_df(target_model: str) -> pd.DataFrame:
    """Concat all 5 folds of zeroshot_evaluated_binary.csv for a target model."""
    subdir = TARGET_SUBDIR.get(target_model, target_model)
    pattern = str(STEP8_DIR / subdir / "fold_*" / "zeroshot_evaluated_binary.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Step-8 files for target '{target_model}' at {pattern}"
        )
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    # Canonicalise column name: Step 8 uses 'fold_id'; we expose 'fold'.
    if "fold_id" in df.columns and "fold" not in df.columns:
        df["fold"] = df["fold_id"].astype(int)
    df["patient_id"] = df["patient_id"].astype(int)
    df["binary_correct"] = df["binary_correct"].astype(int)
    return df


def _load_notes_and_choices() -> dict[int, dict[str, Any]]:
    """patient_id → {'note': concatenated, 'choices': formatted_block or ''}."""
    df = pd.read_json(EHRNOTEQA_PATH, lines=True)
    out: dict[int, dict[str, Any]] = {}
    for _, r in df.iterrows():
        pid = int(r.get("patient_id", 0))
        note_parts = []
        for i in (1, 2, 3):
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    note_parts.append(f"[Note {i}]\n{t}")
        choice_parts = []
        for letter in ("A", "B", "C", "D", "E"):
            col = f"choice_{letter}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    choice_parts.append(f"{letter}) {t}")
        out[pid] = {
            "note": "\n\n".join(note_parts),
            "choices": "\n".join(choice_parts),
        }
    return out


def _row_to_pilot_item(
    row: pd.Series, notes_and_choices: dict[int, dict[str, Any]], target_model: str,
) -> PilotItem | None:
    pid = int(row["patient_id"])
    entry = notes_and_choices.get(pid)
    if not entry or not entry["note"]:
        return None
    fold = int(row["fold"])
    qtype = str(row.get("question_type", "")).lower()
    # 'choices' only for multichoice questions; otherwise empty so the template
    # collapses cleanly.
    choices_block = entry["choices"] if "multichoice" in qtype or qtype == "" else ""
    return PilotItem(
        pilot_item_id=f"f{fold}_{pid}",
        fold=fold,
        patient_id=pid,
        question=str(row["question"]),
        question_type=qtype,
        note=entry["note"],
        choices=choices_block,
        model_answer=str(row.get("model_answer", "")),
        binary_correct=int(row["binary_correct"]),
        target_model=target_model,
    )


# ─────────────────────── public entry points ───────────────────────

def load_detection_pilot(
    target_model: str,
    n_per_fold: int = 8,
    n_incorrect_per_fold: int = 4,
    seed: int = 20260422,
) -> list[PilotItem]:
    """Main-pilot sampler: stratified per fold by `binary_correct`.

    Returns `n_per_fold * 5` items (default 40), with `n_incorrect_per_fold`
    `binary_correct == 0` rows per fold and the remainder `binary_correct == 1`.

    Raises ValueError if any fold is short on incorrect items (rare: folds
    typically have 15-20 incorrect rows each).
    """
    if n_incorrect_per_fold > n_per_fold:
        raise ValueError(
            f"n_incorrect_per_fold ({n_incorrect_per_fold}) > n_per_fold ({n_per_fold})"
        )
    df = _load_target_df(target_model)
    notes = _load_notes_and_choices()
    rng = random.Random(seed)

    items: list[PilotItem] = []
    for fold in sorted(df["fold"].unique()):
        fold_df = df[df["fold"] == fold]
        incorrect = fold_df[fold_df["binary_correct"] == 0]
        correct = fold_df[fold_df["binary_correct"] == 1]
        if len(incorrect) < n_incorrect_per_fold:
            raise ValueError(
                f"Fold {fold}: only {len(incorrect)} incorrect rows, "
                f"need {n_incorrect_per_fold}"
            )
        n_correct_needed = n_per_fold - n_incorrect_per_fold
        if len(correct) < n_correct_needed:
            raise ValueError(
                f"Fold {fold}: only {len(correct)} correct rows, need {n_correct_needed}"
            )
        # Sample deterministically.
        incorrect_sample = incorrect.sample(n=n_incorrect_per_fold, random_state=seed + fold)
        correct_sample = correct.sample(n=n_correct_needed, random_state=seed + fold + 100)
        picked = pd.concat([incorrect_sample, correct_sample], ignore_index=True)
        # Stable within-fold ordering: sort by patient_id to keep traces deterministic.
        picked = picked.sort_values("patient_id").reset_index(drop=True)

        for _, row in picked.iterrows():
            pi = _row_to_pilot_item(row, notes, target_model)
            if pi is not None:
                items.append(pi)

    # Final shuffle so candidates don't see all-fold-0 first.
    rng.shuffle(items)
    return items


def load_detection_fullscale(target_model: str) -> list[PilotItem]:
    """Full 962-item set for a target model (no stratification).

    Returns every row from the target's Step-8 zeroshot_evaluated_binary CSVs,
    joined with notes + choices. Use for full-scale runs where pilot-stratified
    subsampling is not wanted.
    """
    df = _load_target_df(target_model)
    notes = _load_notes_and_choices()
    # Stable order: fold, patient_id
    df = df.sort_values(["fold", "patient_id"]).reset_index(drop=True)
    items: list[PilotItem] = []
    for _, row in df.iterrows():
        pi = _row_to_pilot_item(row, notes, target_model)
        if pi is not None:
            items.append(pi)
    return items


def load_detection_pilot_sub(
    target_model: str,
    n_total: int = 5,
    n_incorrect: int = 2,
    seed: int = 20260422,
) -> list[PilotItem]:
    """Parser-design sub-pilot sampler: `n_total` items across all folds.

    `n_incorrect` items with `binary_correct == 0` + `(n_total - n_incorrect)`
    items with `binary_correct == 1`. Attempts one-per-fold but falls back
    to within-fold repeats if folds run out.
    """
    if n_incorrect > n_total:
        raise ValueError(f"n_incorrect ({n_incorrect}) > n_total ({n_total})")
    df = _load_target_df(target_model)
    notes = _load_notes_and_choices()
    rng = random.Random(seed)

    # Pick folds round-robin; first n_incorrect folds give an incorrect row,
    # the rest give a correct row. Using fold 0..4 in order.
    folds = sorted(df["fold"].unique().tolist())
    if not folds:
        raise ValueError("No folds found in Step-8 data")
    items: list[PilotItem] = []
    for i in range(n_total):
        fold = folds[i % len(folds)]
        want_incorrect = i < n_incorrect
        fold_df = df[(df["fold"] == fold) &
                     (df["binary_correct"] == (0 if want_incorrect else 1))]
        if fold_df.empty:
            # Fallback: any other fold with the desired label.
            fallback = df[df["binary_correct"] == (0 if want_incorrect else 1)]
            if fallback.empty:
                raise ValueError(
                    f"No rows with binary_correct={0 if want_incorrect else 1}"
                )
            row = fallback.sample(n=1, random_state=seed + i).iloc[0]
        else:
            row = fold_df.sample(n=1, random_state=seed + i).iloc[0]
        pi = _row_to_pilot_item(row, notes, target_model)
        if pi is not None:
            items.append(pi)
    rng.shuffle(items)
    return items
