#!/usr/bin/env python3
"""
Step 8: Copy reusable results from existing experiments into step8 output structure.

Copies:
1. BioMistral zeroshot + RA-ICL from fullscale_4_biomistral
2. Qwen2.5 zeroshot from step9_qwen_evaluated.csv
3. Qwen2.5 RA-ICL from pilot_12_ra_icl fullscale

Usage:
    python copy_reusable_results.py
"""

import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "step8"

FOLD_SIZES = {0: 193, 1: 193, 2: 192, 3: 192, 4: 192}


def copy_with_verify(src, dst, expected_rows=None):
    """Copy a CSV file and verify row count."""
    if not src.exists():
        print(f"  WARNING: Source not found: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    df = pd.read_csv(dst)
    if expected_rows and len(df) != expected_rows:
        print(f"  WARNING: {dst.name} has {len(df)} rows, expected {expected_rows}")
    return True


def copy_biomistral():
    """Copy BioMistral zeroshot + RA-ICL from fullscale_4."""
    print("\n=== BioMistral-7B ===")
    bio_dir = PROJECT_ROOT / "output" / "fullscale_4_biomistral" / "fullscale"

    conditions = ["zeroshot", "gtr_note_pos_k1", "gtr_note_neg_k1", "gtr_note_posneg_k1"]
    copied = 0

    for fold_id in range(5):
        for condition in conditions:
            src_gen = bio_dir / f"fold_{fold_id}" / f"{condition}_generated.csv"
            src_eval = bio_dir / f"fold_{fold_id}" / f"{condition}_evaluated.csv"
            dst_gen = OUTPUT_DIR / "biomistral-7b" / f"fold_{fold_id}" / f"{condition}_generated.csv"
            dst_eval = OUTPUT_DIR / "biomistral-7b" / f"fold_{fold_id}" / f"{condition}_evaluated.csv"

            if dst_eval.exists():
                print(f"  Already exists: biomistral-7b/fold_{fold_id}/{condition}")
                copied += 1
                continue

            if copy_with_verify(src_gen, dst_gen, FOLD_SIZES[fold_id]):
                if copy_with_verify(src_eval, dst_eval, FOLD_SIZES[fold_id]):
                    # Verify the evaluated file has correct column
                    df = pd.read_csv(dst_eval)
                    if "openended_correct" not in df.columns and "correct" in df.columns:
                        df = df.rename(columns={"correct": "openended_correct", "reason": "openended_reason"})
                        df.to_csv(dst_eval, index=False)
                    copied += 1

    print(f"  Copied: {copied}/{len(conditions) * 5} condition-folds")


def _get_ground_truth(row):
    gt = row.get("ground_truth", "")
    if gt and str(gt).strip() and str(gt).lower() != "nan":
        return str(gt)
    al = str(row.get("answer", row.get("ground_truth_letter", ""))).strip()
    choice_col = f"choice_{al}"
    if choice_col in row.index and pd.notna(row.get(choice_col)):
        return f"{al}: {row[choice_col]}"
    return al


def copy_qwen25_zeroshot():
    """Copy Qwen2.5 zeroshot from step9_qwen_evaluated.csv, split by fold."""
    print("\n=== Qwen2.5-7B-Instruct: Zeroshot ===")

    step9_file = PROJECT_ROOT / "output" / "fullscale" / "step9_qwen_evaluated.csv"
    if not step9_file.exists():
        print(f"  WARNING: step9_qwen_evaluated.csv not found")
        return

    df = pd.read_csv(step9_file)
    copied = 0

    for fold_id in range(5):
        dst_gen = OUTPUT_DIR / "qwen2.5-7b-instruct" / f"fold_{fold_id}" / "zeroshot_generated.csv"
        dst_eval = OUTPUT_DIR / "qwen2.5-7b-instruct" / f"fold_{fold_id}" / "zeroshot_evaluated.csv"

        if dst_eval.exists():
            print(f"  Already exists: qwen2.5-7b-instruct/fold_{fold_id}/zeroshot")
            copied += 1
            continue

        fold_df = df[df["fold_id"] == fold_id].reset_index(drop=True)
        if len(fold_df) == 0:
            print(f"  WARNING: No fold_id={fold_id} samples in step9")
            continue

        # Create generated CSV (standardized columns)
        gen_cols = {
            "idx": range(len(fold_df)),
            "patient_id": fold_df["patient_id"],
            "fold_id": fold_id,
            "question": fold_df["question"],
            "question_type": fold_df.get("question_type", ""),
            "ground_truth": fold_df.apply(
                lambda r: _get_ground_truth(r), axis=1,
            ),
            "model_answer": fold_df["openended_answer"],
            "model": "qwen2.5-7b-instruct",
            "method": "zeroshot",
            "retrieval_sim_score": 0.0,
            "prompt_length": 0,
            "answer_length": fold_df["openended_answer"].astype(str).str.len(),
        }
        gen_df = pd.DataFrame(gen_cols)

        # Create evaluated CSV
        eval_df = gen_df.copy()
        eval_df["openended_correct"] = fold_df["openended_correct"].values
        if "openended_reason" in fold_df.columns:
            eval_df["openended_reason"] = fold_df["openended_reason"].values
        else:
            eval_df["openended_reason"] = ""

        dst_gen.parent.mkdir(parents=True, exist_ok=True)
        gen_df.to_csv(dst_gen, index=False)
        eval_df.to_csv(dst_eval, index=False)

        correct = (eval_df["openended_correct"] == "yes").sum()
        print(f"  fold_{fold_id}: {correct}/{len(eval_df)} = {correct/len(eval_df):.1%}")
        copied += 1

    print(f"  Copied: {copied}/5 folds")


def copy_qwen25_ra_icl():
    """Copy Qwen2.5 RA-ICL results from pilot_12 fullscale."""
    print("\n=== Qwen2.5-7B-Instruct: RA-ICL ===")

    p12_dir = PROJECT_ROOT / "output" / "pilot_12_ra_icl" / "fullscale"
    conditions = ["gtr_note_pos_k1", "gtr_note_neg_k1", "gtr_note_posneg_k1", "gtr_note_any_unlabeled_k1"]
    copied = 0

    for fold_id in range(5):
        for condition in conditions:
            src_gen = p12_dir / f"fold_{fold_id}" / f"{condition}_generated.csv"
            src_eval = p12_dir / f"fold_{fold_id}" / f"{condition}_evaluated.csv"
            dst_gen = OUTPUT_DIR / "qwen2.5-7b-instruct" / f"fold_{fold_id}" / f"{condition}_generated.csv"
            dst_eval = OUTPUT_DIR / "qwen2.5-7b-instruct" / f"fold_{fold_id}" / f"{condition}_evaluated.csv"

            if dst_eval.exists():
                print(f"  Already exists: qwen2.5-7b-instruct/fold_{fold_id}/{condition}")
                copied += 1
                continue

            if copy_with_verify(src_gen, dst_gen, FOLD_SIZES[fold_id]):
                if copy_with_verify(src_eval, dst_eval, FOLD_SIZES[fold_id]):
                    # Normalize column names if needed
                    df = pd.read_csv(dst_eval)
                    if "openended_correct" not in df.columns and "correct" in df.columns:
                        df = df.rename(columns={"correct": "openended_correct", "reason": "openended_reason"})
                        df.to_csv(dst_eval, index=False)
                    copied += 1

    print(f"  Copied: {copied}/{len(conditions) * 5} condition-folds")


def main():
    print("Step 8: Copying reusable results")
    print("=" * 60)

    copy_biomistral()
    copy_qwen25_zeroshot()
    copy_qwen25_ra_icl()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model in ["biomistral-7b", "qwen2.5-7b-instruct"]:
        model_dir = OUTPUT_DIR / model
        if not model_dir.exists():
            continue
        eval_files = list(model_dir.rglob("*_evaluated.csv"))
        print(f"\n  {model}: {len(eval_files)} evaluated CSVs")
        for ef in sorted(eval_files):
            df = pd.read_csv(ef)
            correct = (df["openended_correct"] == "yes").sum()
            print(f"    {ef.parent.name}/{ef.name}: {correct}/{len(df)} = {correct/len(df):.1%}")


if __name__ == "__main__":
    main()
