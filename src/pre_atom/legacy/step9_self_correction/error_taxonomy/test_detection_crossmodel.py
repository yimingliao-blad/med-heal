#!/usr/bin/env python3
"""
Cross-model detection pilot: Test BC prompt on all 4 models.
Must swap vLLM model for each. Run one model at a time.

Usage:
    python test_detection_crossmodel.py --model qwen25 --port 8003
    python test_detection_crossmodel.py --model llama3 --port 8003
    python test_detection_crossmodel.py --model deepseek --port 8003
    python test_detection_crossmodel.py --model qwen3 --port 8003
"""
import json, os, random, re, sys, time, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

MODEL_MAP = {
    "qwen25": {"dir": "qwen2.5-7b-instruct", "template": "chatml", "stop": ["<|im_end|>", "<|endoftext|>"]},
    "llama3": {"dir": "llama-3.1-8b-instruct", "template": "llama3", "stop": ["<|eot_id|>", "<|end_of_text|>"]},
    "deepseek": {"dir": "deepseek-r1-distill-llama-8b", "template": "llama3", "stop": ["<|eot_id|>", "<｜end▁of▁sentence｜>"]},
    "qwen3": {"dir": "qwen3-8b", "template": "qwen3", "stop": ["<|im_end|>", "<|endoftext|>"]},
}

BC_COT_FEWSHOT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Review this answer for errors. Common error patterns in clinical answers include:
- MISREADING: confusing medications, dosages, or visits that ARE in the notes
- FABRICATION: stating details NOT found anywhere in the notes
- OMISSION: missing critical information that changes the answer
- QUESTION_MISALIGNMENT: answering about the wrong visit, time period, or clinical focus

STEP 1 — Does the answer address the right question?
Check: correct visit, correct time period, correct clinical focus.
ALIGNMENT: OK or PROBLEM

STEP 2 — Is every claim supported by the notes?
For each key claim, find the supporting passage in the notes.
EVIDENCE: OK or PROBLEM — <specific issue>

STEP 3 — Are critical details included?
Only flag omissions that change the answer's conclusion.
COMPLETENESS: OK or PROBLEM — <what's missing>

VERDICT: CORRECT or INCORRECT
IF INCORRECT — ERROR_TYPE: <MISREADING, FABRICATION, OMISSION, or QUESTION_MISALIGNMENT>"""


def build_prompt(template, system, user):
    if template == "chatml" or template == "qwen3":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n")
    elif template == "llama3":
        return ("<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return f"{system}\n\n{user}\n\nAssistant:"


def vllm_generate(port, prompt, stop_tokens, max_tokens=1024, temperature=0.0):
    try:
        model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
        resp = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature, "stop": stop_tokens},
            timeout=120,
        )
        raw = resp.json()["choices"][0]["text"].strip()
        # Strip thinking tags
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"</think>", "", raw).strip()
        return raw
    except Exception as e:
        print(f"  vLLM error: {e}")
        return ""


def load_notes():
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n-wrong", type=int, default=25)
    parser.add_argument("--n-correct", type=int, default=25)
    args = parser.parse_args()

    cfg = MODEL_MAP[args.model]
    notes = load_notes()

    # Load model data
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / cfg["dir"] / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    wrong_df = all_df[all_df["binary_correct"] == 0]
    correct_df = all_df[all_df["binary_correct"] == 1]

    random.seed(42)
    wrong_sample = wrong_df.sample(n=min(args.n_wrong, len(wrong_df)), random_state=42)
    correct_sample = correct_df.sample(n=min(args.n_correct, len(correct_df)), random_state=42)

    test_items = []
    for _, row in wrong_sample.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]),
                           "label": "wrong", "row": row})
    for _, row in correct_sample.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]),
                           "label": "correct", "row": row})

    n_wrong = sum(1 for t in test_items if t["label"] == "wrong")
    n_correct = sum(1 for t in test_items if t["label"] == "correct")
    print(f"BC Detection Pilot: {args.model} ({n_wrong} wrong + {n_correct} correct)")
    print("=" * 60)

    results = []
    for i, ti in enumerate(test_items):
        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note:
            continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        msg = BC_COT_FEWSHOT.format(note=note, question=row["question"], answer=answer[:800])
        system = "You are a strict medical expert verifying clinical answers against discharge notes."
        prompt = build_prompt(cfg["template"], system, msg)

        max_tok = 2048 if args.model in ("deepseek", "qwen3") else 1024
        raw = vllm_generate(args.port, prompt, cfg["stop"], max_tokens=max_tok)
        raw_upper = raw.upper()

        detected = "VERDICT: INCORRECT" in raw_upper or \
                   ("VERDICT" in raw_upper and "INCORRECT" in raw_upper.split("VERDICT")[-1][:20])

        results.append({"idx": ti["idx"], "fold": ti["fold"],
                        "label": ti["label"], "detected": detected})

        if (i + 1) % 10 == 0:
            w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
            c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
            wt = sum(1 for r in results if r["label"] == "wrong")
            ct = sum(1 for r in results if r["label"] == "correct")
            print(f"  [{i+1}/{len(test_items)}] wrong={w_det}/{wt} correct={c_det}/{ct}")

    # Summary
    w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
    c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
    wt = sum(1 for r in results if r["label"] == "wrong")
    ct = sum(1 for r in results if r["label"] == "correct")
    sel = (w_det/max(wt,1)) / max(c_det/max(ct,1), 0.01)

    print(f"\n{'='*60}")
    print(f"BC Detection — {args.model}")
    print(f"  Wrong: {w_det}/{wt} ({100*w_det/wt:.0f}%)")
    print(f"  Correct: {c_det}/{ct} ({100*c_det/ct:.0f}%)")
    print(f"  Selectivity: {sel:.1f}x")

    # Save
    out_file = OUTPUT_DIR / f"detection_crossmodel_{args.model}.json"
    with open(out_file, "w") as f:
        json.dump({"model": args.model, "results": results,
                   "wrong_det": w_det, "correct_det": c_det,
                   "wrong_total": wt, "correct_total": ct,
                   "selectivity": sel}, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
