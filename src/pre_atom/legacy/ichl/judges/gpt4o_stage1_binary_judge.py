#!/usr/bin/env python3
"""GPT-4o Stage-1 binary judge — the calibrated 92%-human-agreement evaluator.

Origin: Stage 1 binary prompt validated by Sara+Jose human gold (N=112,
agreement 92%, kappa=0.75). Replaced the abandoned step8 strict-reason prompt
(72.3% agreement, kappa=0.36).

Calibration: passes the discharge note IN the user prompt (essential for the
92%; without note the judge degrades to literal string matching against GT).

Per [Workflow] Implementation Discipline Rule 1: this is the canonical
inheritable judge. Do NOT reimplement the prompt elsewhere.

Sequential API calls (no parallelism per user preference). max_tokens=10.

Predecessor location (deprecated): src/step9_self_correction/evaluate_step9_binary.py
                                   src/step8_multimodel_icl/evaluate_step8_binary.py

Usage:
    # Evaluate all step8 models
    python -m ichl.judges.gpt4o_stage1_binary_judge --scope step8

    # Evaluate specific model/conditions
    python -m ichl.judges.gpt4o_stage1_binary_judge --scope step8 --model qwen3-8b --folds 0 1

    # Evaluate cross-dataset pilot
    python -m ichl.judges.gpt4o_stage1_binary_judge --scope crossdataset_pilot

    # Evaluate specific files
    python -m ichl.judges.gpt4o_stage1_binary_judge --files /path/to/generated.csv
"""


import argparse
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent.parent
STEP8_DIR = PROJECT_ROOT / "output" / "step8"
PILOT_DIR = PROJECT_ROOT / "output" / "other_benchmark_icl" / "crossdataset_pilot"
NOTES_FILE = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"

ALL_MODELS = [
    "biomistral-7b",
    "deepseek-r1-distill-llama-8b",
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "qwen3-8b",
]


def load_notes():
    notes_df = pd.read_json(NOTES_FILE, lines=True)
    lookup = {}
    for _, r in notes_df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(f"[Note {i}]\n{t}")
        lookup[pid] = "\n\n".join(parts)
    return lookup


def evaluate_one_binary(client, note, question, ground_truth, model_answer, gpt_model="gpt-4o"):
    """Evaluate with Stage 1 binary prompt. Returns 1, 0, or None."""
    messages = [
        {
            "role": "system",
            "content": "You are a medical expert evaluating an AI model's answer to a clinical question.",
        },
        {
            "role": "user",
            "content": (
                f"DISCHARGE SUMMARY:\n{note}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
                f"MODEL'S ANSWER:\n{model_answer}\n\n"
                f"Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
                f"Respond with ONLY a single digit:\n"
                f"1 = Correct\n"
                f"0 = Incorrect"
            ),
        },
    ]

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                max_tokens=10,
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            if "1" in content and "0" not in content:
                return 1
            elif "0" in content:
                return 0
            else:
                return None
        except Exception as e:
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"    ERROR after 5 attempts: {e}")
                return None
    return None


def evaluate_file(client, gen_file, eval_file, notes_lookup, gpt_model="gpt-4o"):
    """Evaluate a single generated CSV file with binary prompt."""
    gen_df = pd.read_csv(gen_file)

    # Skip if already complete
    if eval_file.exists():
        existing = pd.read_csv(eval_file)
        if len(existing) >= len(gen_df):
            correct = (existing["binary_correct"] == 1).sum()
            total = len(existing)
            print(f"  DONE {gen_file.name}: {correct}/{total} = {correct/total:.1%}")
            return True

    # Resume support
    results = []
    done_ids = set()
    if eval_file.exists():
        existing = pd.read_csv(eval_file)
        done_ids = set(existing["idx"].tolist())
        results = existing.to_dict("records")
        print(f"  Resuming {gen_file.name} from {len(results)}")

    for _, row in gen_df.iterrows():
        idx = row.get("idx", 0)
        if idx in done_ids:
            continue

        pid = str(row.get("patient_id", ""))
        note = notes_lookup.get(pid, "")
        question = str(row.get("question", ""))
        gt = str(row.get("ground_truth", ""))
        model_answer = str(row.get("model_answer", ""))

        score = evaluate_one_binary(client, note, question, gt, model_answer, gpt_model)

        result_row = row.to_dict()
        result_row["binary_correct"] = score
        results.append(result_row)

        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(eval_file, index=False)
            correct_so_far = sum(1 for r in results if r.get("binary_correct") == 1)
            print(f"    Progress: {len(results)}/{len(gen_df)} "
                  f"({correct_so_far}/{len(results)} correct)")

        time.sleep(0.2)  # rate limit

    pd.DataFrame(results).to_csv(eval_file, index=False)
    correct = sum(1 for r in results if r.get("binary_correct") == 1)
    print(f"  Done: {correct}/{len(results)} = {correct/len(results):.1%} -> {eval_file.name}")
    return True


def collect_step8_files(model=None, folds=None, conditions=None):
    """Collect all _generated.csv files from step8 output."""
    files = []
    models = [model] if model else [m for m in ALL_MODELS if (STEP8_DIR / m).exists()]
    fold_ids = folds or list(range(5))

    for m in models:
        model_dir = STEP8_DIR / m
        if not model_dir.exists():
            continue
        for fold_id in fold_ids:
            fold_dir = model_dir / f"fold_{fold_id}"
            if not fold_dir.exists():
                continue
            for gen_file in sorted(fold_dir.glob("*_generated.csv")):
                cond = gen_file.stem.replace("_generated", "")
                if conditions and cond not in conditions:
                    continue
                eval_file = gen_file.parent / f"{cond}_evaluated_binary.csv"
                files.append((gen_file, eval_file))
    return files


def collect_pilot_files():
    """Collect all _generated.csv files from crossdataset pilot."""
    files = []
    if not PILOT_DIR.exists():
        return files
    for gen_file in sorted(PILOT_DIR.glob("*_generated.csv")):
        cond = gen_file.stem.replace("_generated", "")
        eval_file = PILOT_DIR / f"{cond}_evaluated_binary.csv"
        files.append((gen_file, eval_file))
    return files


def main():
    parser = argparse.ArgumentParser(description="Binary GPT-4o evaluation")
    parser.add_argument("--scope", choices=["step8", "crossdataset_pilot", "files"],
                        required=True)
    parser.add_argument("--model", default=None, help="Specific step8 model")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--files", nargs="+", default=None, help="Specific CSV files")
    parser.add_argument("--gpt_model", default="gpt-4o")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    client = OpenAI(api_key=api_key)

    print("Loading notes...")
    notes_lookup = load_notes()
    print(f"  {len(notes_lookup)} patients loaded")

    # Collect files to evaluate
    if args.scope == "step8":
        file_pairs = collect_step8_files(args.model, args.folds, args.conditions)
    elif args.scope == "crossdataset_pilot":
        file_pairs = collect_pilot_files()
    elif args.scope == "files":
        file_pairs = []
        for f in (args.files or []):
            gen = Path(f)
            eval_f = gen.parent / gen.name.replace("_generated.csv", "_evaluated_binary.csv")
            file_pairs.append((gen, eval_f))

    print(f"\nFiles to evaluate: {len(file_pairs)}")
    total_rows = 0
    for gen_file, _ in file_pairs:
        if gen_file.exists():
            total_rows += len(pd.read_csv(gen_file))
    print(f"Total rows: ~{total_rows}")
    print(f"Estimated cost: ~${total_rows * 0.005:.0f}")

    evaluated = 0
    skipped = 0
    for gen_file, eval_file in file_pairs:
        if not gen_file.exists():
            print(f"  SKIP (not found): {gen_file}")
            continue

        result = evaluate_file(client, gen_file, eval_file, notes_lookup, args.gpt_model)
        if result:
            evaluated += 1

    print(f"\nDone. Evaluated: {evaluated}, Total files: {len(file_pairs)}")


if __name__ == "__main__":
    main()
