"""For-fun: Magistral-Small-2509 as a TARGET model on EHRNoteQA.

Compares Magistral-24B's answer accuracy against the known 7B/8B baselines
(BioMistral, Qwen2.5-7B, Llama-3.1-8B, Qwen3-8B, DeepSeek-R1-8B).

Pipeline:
  1. For each item: generate Magistral's open-ended answer using the canonical
     zeroshot prompt (Discharge Summary / Question / Answer).
  2. Judge with our shipped MagistralJudge (M4 prompt, same model, free, ~1.3 s).
     Note: self-judging introduces a confound — Magistral may be biased toward
     its own outputs. For a real headline number we'd want GPT-4o, but for
     "fun" this is good enough at ~85% calibrated agreement vs gold.

Output:
  output/ichl/mlx_judge/magistral_target/answers.jsonl  — per-item Q/GT/MA/judge
  output/ichl/mlx_judge/magistral_target/summary.json   — aggregate stats
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
NOTES_FILE = ROOT / "output" / "EHRNoteQA_processed.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "mlx_judge" / "magistral_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL = "http://localhost:8003/v1"
MODEL = "Magistral-Small-2509-AWQ"

# Same canonical prompt the 5 baseline models used (per generate_crossdataset.py)
TARGET_SYSTEM = "You are a medical expert answering questions about discharge summaries."
TARGET_USER = "Discharge Summary:\n{note}\n\nQuestion: {question}\n\nAnswer:"


def load_notes() -> dict[str, str]:
    df = pd.read_json(NOTES_FILE, lines=True)
    out = {}
    for _, r in df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1, 2, 3]:
            v = r.get(f"note_{i}")
            if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                parts.append(f"[Note {i}]\n{v}")
        out[pid] = "\n\n".join(parts)
    return out


def generate_one(args):
    client, item, note, max_tokens = args
    user = TARGET_USER.format(note=note, question=item["question"])
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": TARGET_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0.0, max_tokens=max_tokens,
        )
        lat = time.monotonic() - t0
        msg = resp.choices[0].message
        content = (getattr(msg, "content", None) or "").strip()
        # Strip <think> if present
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()
        return {
            "patient_id": item["patient_id"],
            "fold_id": item.get("fold_id"),
            "target_baseline_label": int(item.get("binary_correct", -1)),
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "magistral_answer": content,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "finish_reason": resp.choices[0].finish_reason,
            "latency_s": round(lat, 2),
        }
    except Exception as e:
        return {"patient_id": item["patient_id"], "error": str(e)[:200],
                "latency_s": round(time.monotonic() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--full-dataset", action="store_true",
                    help="Load all 962 unique questions from EHRNoteQA_processed.jsonl instead of dev split")
    ap.add_argument("--judge-only", action="store_true",
                    help="Skip generation, only run judge on already-generated answers.json")
    ap.add_argument("--gen-only", action="store_true",
                    help="Generate Magistral answers and stop (defer judging to a better judge later)")
    args = ap.parse_args()

    answers_path = OUT_DIR / "answers.jsonl"
    summary_path = OUT_DIR / "summary.json"

    if not args.judge_only:
        # === Phase 1: Generate ===
        notes = load_notes()
        if args.full_dataset:
            print("Loading FULL EHRNoteQA dataset (962 unique questions)…")
            df = pd.read_json(NOTES_FILE, lines=True)
            sample = []
            for _, r in df.iterrows():
                letter = str(r.get("answer", "")).strip().upper()
                col = f"choice_{letter}"
                gt_text = str(r.get(col, "")).strip() if col in r.index else ""
                gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
                sample.append({
                    "patient_id": int(r["patient_id"]),
                    "fold_id": "full",
                    "question": str(r["question"]),
                    "ground_truth": gt,
                    "binary_correct": -1,  # unknown — Magistral is the target
                })
            print(f"  generating: n={len(sample)} unique questions (Magistral as target)")
        else:
            print("Loading dev + notes…")
            dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
            # Stratified sample on baseline label so we can compare to baselines fairly
            g0 = [d for d in dev if d["binary_correct"] == 0]
            g1 = [d for d in dev if d["binary_correct"] == 1]
            if args.n >= len(dev):
                sample = dev
            else:
                half = args.n // 2
                sample = g0[:half] + g1[:args.n - half]
            print(f"  generating: n={len(sample)} items (Magistral as target, dev sample)")

        client = OpenAI(base_url=VLLM_URL, api_key="not-needed")
        tasks = [(client, it, notes.get(str(it["patient_id"]), ""), args.max_tokens) for it in sample]
        t0 = time.monotonic()
        rows = []
        with answers_path.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(generate_one, tasks), 1):
                f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                rows.append(r)
                if i % 25 == 0:
                    dt = time.monotonic() - t0
                    eta = dt * (len(sample) - i) / i
                    print(f"  gen {i}/{len(sample)}  elapsed={dt:.0f}s  eta={eta:.0f}s")
        print(f"  GEN DONE in {time.monotonic()-t0:.0f}s → {answers_path}")
    else:
        rows = [json.loads(l) for l in answers_path.open() if l.strip()]
        print(f"Skipping generation; loaded {len(rows)} existing answers")
        notes = load_notes()

    if args.gen_only:
        print(f"\n--gen-only set; skipping judge phase. Answers saved at {answers_path}")
        print(f"  Run later with: python -m ichl.prompt_engineering.mlx_judge.magistral_target_run --judge-only")
        return

    # === Phase 2: Judge with MagistralJudge (M4) ===
    print("\nJudging with MagistralJudge (M4 prompt)…")
    sys.path.insert(0, str(ROOT / "src"))
    from ichl.judges import MagistralJudge

    judge = MagistralJudge()
    t0 = time.monotonic()
    judged_path = OUT_DIR / "answers_judged.jsonl"
    n_correct = 0
    n_judged = 0
    n_none = 0

    def judge_one(idx_row):
        idx, r = idx_row
        if "error" in r:
            return idx, None
        note = notes.get(str(r["patient_id"]), "")
        jr = judge.judge(
            question=r["question"],
            ground_truth=r["ground_truth"],
            model_answer=r["magistral_answer"],
            note=note,
        )
        return idx, jr

    judged_rows = list(rows)  # mutable copy
    with ThreadPoolExecutor(max_workers=4) as ex:
        for idx, jr in ex.map(judge_one, list(enumerate(rows))):
            if jr is None:
                judged_rows[idx]["judge_label"] = None
                continue
            judged_rows[idx]["judge_label"] = jr.label
            judged_rows[idx]["judge_latency_s"] = jr.latency_s
            judged_rows[idx]["judge_truncation"] = jr.truncation_certain
            n_judged += 1
            if jr.label is None:
                n_none += 1
            elif jr.label == 1:
                n_correct += 1

    with judged_path.open("w") as f:
        for r in judged_rows:
            f.write(json.dumps(r, default=str) + "\n")
    judge_elapsed = time.monotonic() - t0
    print(f"  JUDGE DONE in {judge_elapsed:.0f}s → {judged_path}")

    # === Summary ===
    n_total = len(judged_rows)
    n_errors = sum(1 for r in judged_rows if r.get("error"))
    accuracy = n_correct / n_judged if n_judged else 0
    print(f"\n=== Magistral-Small-2509 on EHRNoteQA dev (n={n_total}) ===")
    print(f"  generation errors:  {n_errors}")
    print(f"  judged successfully: {n_judged}")
    print(f"  judge None (parse fail): {n_none}")
    print(f"  Magistral judge says CORRECT: {n_correct}/{n_judged} = {100*accuracy:.2f}%")
    print()
    print(f"  Compare to known baselines on full 962 (judged by GPT-4o, NOT Magistral):")
    print(f"    Qwen3-8B          : 92.41%")
    print(f"    Qwen2.5-7B        : 88.67%")
    print(f"    Llama-3.1-8B      : 89.09%")
    print(f"    DeepSeek-R1-8B    : 76.92%")
    print(f"    BioMistral-7B     : 53.85%")
    print()
    print(f"  Caveat: this run is self-judged by Magistral (M4 ~85% calibrated vs gold).")

    summary = {
        "n_total": n_total, "n_judged": n_judged, "n_none": n_none, "n_errors": n_errors,
        "accuracy_self_judged": accuracy,
        "model": MODEL,
        "judge": "MagistralJudge (M4)",
        "baselines_for_reference (GPT-4o judged, n=962)": {
            "qwen3-8b": 0.9241, "qwen2.5-7b": 0.8867, "llama-3.1-8b": 0.8909,
            "deepseek-r1-distill-llama-8b": 0.7692, "biomistral-7b": 0.5385,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
