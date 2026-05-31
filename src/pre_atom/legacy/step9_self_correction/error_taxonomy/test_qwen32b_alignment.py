#!/usr/bin/env python3
"""Test Qwen3-32B agreement with GPT-4o error classifications on 10 stratified cases."""
import json, random, re, sys
from pathlib import Path
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

WRONG_ANALYSIS_PROMPT = """You are a medical expert analyzing why an AI model's answer to a clinical question is incorrect.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER (Incorrect):
{model_answer}

Analyze the model's answer step by step:

1. WHAT THE QUESTION ASKS: What specific information is the question looking for?
2. WHAT THE NOTES SAY: What does the discharge summary say about this topic?
3. WHAT THE MODEL SAID: Summarize the model's key claims.
4. WHERE IT WENT WRONG: Identify the specific error(s).

5. ERROR CLASSIFICATION: Classify the PRIMARY error as exactly ONE of:
   a. HEDGING — provides multiple possible answers instead of committing
   b. FABRICATION — states something NOT in the discharge notes at all
   c. MISREADING — misread/misinterpreted information that IS in the notes
   d. OMISSION — failed to mention critical information from the notes
   e. QUESTION_MISALIGNMENT — answered a different question than asked
   f. OTHER — none of the above

6. CONFIDENCE: DEFINITIVE / HEDGED / MULTI_ANSWER

Respond in this exact format:
PRIMARY_ERROR: <one of: HEDGING, FABRICATION, MISREADING, OMISSION, QUESTION_MISALIGNMENT, OTHER>
CONFIDENCE: <DEFINITIVE, HEDGED, or MULTI_ANSWER>
ERROR_DESCRIPTION: <brief description>"""


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
    # Load GPT-4o results
    with open(OUTPUT_DIR / "phase1_wrong_gpt4o.json") as f:
        gpt_results = json.load(f)

    # Stratified sample: 2 per error type
    by_type = {}
    for r in gpt_results:
        by_type.setdefault(r["PRIMARY_ERROR"], []).append(r)

    random.seed(42)
    sample = []
    for t, items in by_type.items():
        sample.extend(random.sample(items, min(2, len(items))))
    sample = sample[:10]

    notes = load_notes()

    # Load Qwen2.5 data
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    print("Qwen3-32B vs GPT-4o Agreement (10 stratified cases)")
    print("=" * 60)

    results = []
    agree = 0
    for i, r in enumerate(sample):
        row = all_df[(all_df["fold"] == r["fold"]) & (all_df["idx"] == r["idx"])]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        note = notes.get(str(row["patient_id"]), "")
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        msg = WRONG_ANALYSIS_PROMPT.format(
            note=note, question=row["question"],
            ground_truth=row["ground_truth"],
            model_answer=answer[:1000],
        )

        try:
            resp = requests.post(QWEN32B_URL, json={
                "model": "Qwen/Qwen3-32B-MLX-bf16",
                "messages": [
                    {"role": "system", "content": "You are a medical expert analyzing errors in clinical AI answers."},
                    {"role": "user", "content": msg},
                ],
                "max_tokens": 1000, "temperature": 0.1,
            }, timeout=180)
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            m = re.search(r"PRIMARY_ERROR:\s*(\w+)", raw)
            qwen_pe = m.group(1).upper() if m else "UNKNOWN"
        except Exception as e:
            qwen_pe = "ERROR"
            print(f"  Error: {e}")

        gpt_pe = r["PRIMARY_ERROR"]
        match = qwen_pe == gpt_pe
        if match:
            agree += 1

        results.append({"idx": r["idx"], "fold": r["fold"], "gpt4o": gpt_pe, "qwen32b": qwen_pe, "agree": match})
        print(f"  [{i+1}/10] idx={r['idx']:3d} GPT4o={gpt_pe:25s} Qwen32B={qwen_pe:25s} {'✓' if match else '✗'}")

    print(f"\nAgreement: {agree}/10 ({100*agree/10:.0f}%)")

    with open(OUTPUT_DIR / "qwen32b_gpt4o_alignment.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
