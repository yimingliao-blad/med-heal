#!/usr/bin/env python3
"""
Cross-model error taxonomy validation.
Sample 20 wrong answers each from DeepSeek, Llama3, Qwen3 zeroshot.
Run GPT-4o error taxonomy analysis. Check if categories generalize.
"""
import json, os, random, re, sys, time
from pathlib import Path
from collections import Counter
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

api_key = None
for line in open(PROJECT_ROOT / ".env"):
    if line.startswith("OPENAI_API_KEY="):
        api_key = line.strip().split("=", 1)[1]
        break

from openai import OpenAI
client = OpenAI(api_key=api_key)
spending = {"calls": 0, "cost": 0.0}

WRONG_ANALYSIS_PROMPT = """You are a medical expert analyzing why an AI model's answer to a clinical question is incorrect.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER (Incorrect):
{model_answer}

Analyze briefly, then classify the PRIMARY error as exactly ONE of:
a. HEDGING — provides multiple possible answers instead of committing
b. FABRICATION — states something NOT in the discharge notes at all
c. MISREADING — misread/misinterpreted information that IS in the notes
d. OMISSION — failed to mention critical information from the notes
e. QUESTION_MISALIGNMENT — answered a different question than asked
f. OTHER — none of the above

Also assess confidence: DEFINITIVE / HEDGED / MULTI_ANSWER
And severity: CRITICAL / PARTIAL

Respond in this exact format:
PRIMARY_ERROR: <HEDGING, FABRICATION, MISREADING, OMISSION, QUESTION_MISALIGNMENT, or OTHER>
CONFIDENCE: <DEFINITIVE, HEDGED, or MULTI_ANSWER>
SEVERITY: <CRITICAL or PARTIAL>
ERROR_DESCRIPTION: <one sentence>"""


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


def load_model_zeroshot(model_dir):
    dfs = []
    base = PROJECT_ROOT / "output" / "step8" / model_dir
    for fold in range(5):
        f = base / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def gpt4o_analyze(note, question, gt, answer):
    time.sleep(1.5)
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical expert analyzing errors in clinical AI answers."},
                {"role": "user", "content": WRONG_ANALYSIS_PROMPT.format(
                    note=note, question=question, ground_truth=gt, model_answer=answer[:1000],
                )},
            ],
            max_tokens=300, temperature=0.1,
        )
        text = r.choices[0].message.content.strip()
        cost = r.usage.prompt_tokens * 2.5 / 1e6 + r.usage.completion_tokens * 10.0 / 1e6
        spending["calls"] += 1
        spending["cost"] += cost
        return text
    except Exception as e:
        print(f"  GPT-4o error: {e}")
        time.sleep(5)
        return ""


def parse_response(text):
    result = {}
    for field in ["PRIMARY_ERROR", "CONFIDENCE", "SEVERITY", "ERROR_DESCRIPTION"]:
        m = re.search(rf"{field}:\s*(.+?)(?:\n|$)", text)
        result[field] = m.group(1).strip() if m else ""

    pe = result.get("PRIMARY_ERROR", "").upper()
    valid = {"HEDGING", "FABRICATION", "MISREADING", "OMISSION", "QUESTION_MISALIGNMENT", "OTHER"}
    if pe not in valid:
        for v in valid:
            if v in pe:
                pe = v
                break
        else:
            pe = "OTHER"
    result["PRIMARY_ERROR"] = pe
    return result


MODELS = {
    "deepseek": "deepseek-r1-distill-llama-8b",
    "llama3": "llama-3.1-8b-instruct",
    "qwen3": "qwen3-8b",
}


def main():
    notes = load_notes()
    random.seed(42)

    all_results = {}

    for model_key, model_dir in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_key} ({model_dir})")
        print(f"{'='*60}")

        df = load_model_zeroshot(model_dir)
        if df.empty:
            print(f"  No data found for {model_dir}")
            continue

        wrong = df[df["binary_correct"] == 0]
        correct = df[df["binary_correct"] == 1]
        print(f"  Total: {len(df)} ({len(wrong)} wrong, {len(correct)} correct)")

        sample = wrong.sample(n=min(20, len(wrong)), random_state=42)
        results = []

        for i, (_, row) in enumerate(sample.iterrows()):
            note = notes.get(str(row["patient_id"]), "")
            if not note:
                continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))
            raw = gpt4o_analyze(note, row["question"], row["ground_truth"], answer)
            parsed = parse_response(raw)

            results.append({
                "idx": int(row["idx"]), "fold": int(row["fold"]),
                "primary_error": parsed["PRIMARY_ERROR"],
                "confidence": parsed.get("CONFIDENCE", ""),
                "severity": parsed.get("SEVERITY", ""),
                "description": parsed.get("ERROR_DESCRIPTION", "")[:150],
            })
            print(f"  [{i+1}/20] idx={row['idx']:3d} → {parsed['PRIMARY_ERROR']} ({parsed.get('CONFIDENCE','')}) ${spending['cost']:.2f}")

        all_results[model_key] = results

        # Summary per model
        pe_counts = Counter(r["primary_error"] for r in results)
        print(f"\n  Distribution:")
        for pe, count in pe_counts.most_common():
            print(f"    {pe:25s}: {count:2d} ({100*count/len(results):.0f}%)")

    # Save
    with open(OUTPUT_DIR / "cross_model_error_taxonomy.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Cross-model comparison table
    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Error Type':25s} {'Qwen2.5':>10s} {'DeepSeek':>10s} {'Llama3':>10s} {'Qwen3':>10s}")
    print("-" * 60)

    # Load Qwen2.5 for comparison
    with open(OUTPUT_DIR / "phase1_wrong_gpt4o.json") as f:
        qwen25 = json.load(f)
    qwen25_counts = Counter(r["PRIMARY_ERROR"] for r in qwen25)

    for pe in ["MISREADING", "QUESTION_MISALIGNMENT", "OMISSION", "FABRICATION", "HEDGING", "OTHER"]:
        q25 = f"{100*qwen25_counts.get(pe,0)/len(qwen25):.0f}%"
        vals = [q25]
        for mk in ["deepseek", "llama3", "qwen3"]:
            if mk in all_results:
                c = Counter(r["primary_error"] for r in all_results[mk])
                n = len(all_results[mk])
                vals.append(f"{100*c.get(pe,0)/max(n,1):.0f}%")
            else:
                vals.append("—")
        print(f"  {pe:25s} {''.join(f'{v:>10s}' for v in vals)}")

    print(f"\nGPT-4o: {spending['calls']} calls, ${spending['cost']:.3f}")


if __name__ == "__main__":
    main()
