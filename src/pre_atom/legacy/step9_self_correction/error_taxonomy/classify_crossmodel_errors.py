#!/usr/bin/env python3
"""Cross-model zeroshot error classification.

Reuses the exact WRONG_ANALYSIS_PROMPT from analyze_qwen25_errors.py to
maintain taxonomic consistency with the existing Qwen2.5 annotations. Runs
GPT-4o classification on a stratified random sample of wrong answers from
BioMistral-7B, Llama-3.1-8B, Qwen3-8B, and DeepSeek-R1-Distill-Llama-8B.

Sampling: up to MAX_PER_MODEL wrong answers per model (default 80), stratified
proportionally across 5 folds. Each call takes ~3-5s with rate-limit spacing.

Output: output/step9_v2/crossmodel_error_taxonomy.json
  { model: [ {patient_id, fold, idx, primary_error, confidence, severity,
              description, question, model_answer, ground_truth}, ... ] }
"""
from __future__ import annotations

import glob
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.step9_self_correction.error_taxonomy.analyze_qwen25_errors import (  # noqa: E402
    WRONG_ANALYSIS_PROMPT,
    load_notes,
    parse_wrong_analysis,
)
from src.step9_self_correction.v2.judge import client  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output" / "step9_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUTPUT_DIR / "crossmodel_error_taxonomy.json"
SEED = 42
MAX_PER_MODEL = 80

MODELS = [
    "biomistral-7b",
    "llama-3.1-8b-instruct",
    "qwen3-8b",
    "deepseek-r1-distill-llama-8b",
]


def load_wrong(model: str) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(PROJECT_ROOT / "output" / "step8" / model /
                                   "fold_*" / "zeroshot_evaluated_binary.csv"))):
        df = pd.read_csv(f)
        if "fold" not in df.columns:
            # extract fold from path
            df["fold"] = int(Path(f).parent.name.split("_")[1])
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    wrong = df[df["binary_correct"] == 0].copy()
    return wrong


def stratified_sample(wrong: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratify by fold so every fold contributes proportionally."""
    if len(wrong) <= n:
        return wrong.copy()
    rng = random.Random(seed)
    out = []
    per_fold = n // 5
    remainder = n - per_fold * 5
    for fold in sorted(wrong["fold"].unique()):
        sub = wrong[wrong["fold"] == fold]
        k = per_fold + (1 if fold < remainder else 0)
        k = min(k, len(sub))
        idxs = rng.sample(list(sub.index), k)
        out.append(sub.loc[idxs])
    return pd.concat(out).reset_index(drop=True)


def gpt4o_classify(note: str, question: str, ground_truth: str, model_answer: str,
                   *, max_retries: int = 3, sleep: float = 5.0) -> tuple[dict, str]:
    """Call GPT-4o with the canonical taxonomy prompt."""
    prompt = WRONG_ANALYSIS_PROMPT.format(
        note=note, question=question,
        ground_truth=ground_truth, model_answer=model_answer[:1000],
    )
    for attempt in range(1, max_retries + 1):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical expert analyzing errors in clinical AI answers."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.1,
            )
            raw = r.choices[0].message.content.strip()
            return parse_wrong_analysis(raw), raw
        except Exception as e:
            if attempt < max_retries:
                print(f"    retry {attempt}: {e}", flush=True)
                time.sleep(sleep)
            else:
                return {"PRIMARY_ERROR": "OTHER", "CONFIDENCE": "UNKNOWN",
                        "SEVERITY": "UNKNOWN", "ERROR_DESCRIPTION": f"ERR: {e}"}, ""
    return {}, ""


def load_existing() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {}


def save(data: dict) -> None:
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> int:
    notes = load_notes()
    data = load_existing()

    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        wrong = load_wrong(model)
        if wrong.empty:
            print(f"  no wrong answers, skipping", flush=True)
            continue
        print(f"  total wrong: {len(wrong)}", flush=True)
        sample = stratified_sample(wrong, MAX_PER_MODEL, SEED)
        print(f"  sampled: {len(sample)}", flush=True)

        existing = {(int(it["fold"]), int(it["idx"])): it
                    for it in data.get(model, [])}
        print(f"  already classified: {len(existing)}", flush=True)

        data.setdefault(model, list(existing.values()))
        for i, row in sample.iterrows():
            key = (int(row["fold"]), int(row["idx"]))
            if key in existing:
                continue
            note = notes.get(str(row["patient_id"]), "")
            if not note:
                continue
            answer = str(row.get("model_answer", row.get("openended_answer", "")))
            parsed, raw = gpt4o_classify(note, row["question"], row["ground_truth"], answer)
            item = {
                "model": model,
                "fold": int(row["fold"]),
                "idx": int(row["idx"]),
                "patient_id": int(row["patient_id"]),
                "question": str(row["question"])[:300],
                "ground_truth": str(row["ground_truth"])[:300],
                "model_answer": answer[:400],
                "primary_error": parsed.get("PRIMARY_ERROR", "OTHER"),
                "confidence": parsed.get("CONFIDENCE", "UNKNOWN"),
                "severity": parsed.get("SEVERITY", "UNKNOWN"),
                "description": parsed.get("ERROR_DESCRIPTION", "")[:400],
            }
            data[model].append(item)
            existing[key] = item
            print(f"    [{len(existing)}/{len(sample)}] fold={key[0]} idx={key[1]} "
                  f"→ {item['primary_error']}", flush=True)
            if len(data[model]) % 10 == 0:
                save(data)
            time.sleep(0.5)
        save(data)

    save(data)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model in MODELS:
        items = data.get(model, [])
        if not items:
            continue
        c = Counter(it["primary_error"] for it in items)
        n = len(items)
        print(f"\n{model} (n={n})")
        for t, k in c.most_common():
            print(f"  {t:<22} {k:>4}  ({k/n*100:5.1f}%)")
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
