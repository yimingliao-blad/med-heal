#!/usr/bin/env python3
"""Classify human reviewer rationales (Reasoning field) into the 5-category
hallucination taxonomy used for LLM error analysis.

Input:  datasets/external/all_users_openended_BioMistral-7B_1775740232208.csv
Output: output/step9_v2/human_rationales_classified.json

Uses GPT-4o @ T=0 with a tight classification prompt that mirrors the
WRONG_ANALYSIS_PROMPT categories from analyze_qwen25_errors.py.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src.step9_self_correction.v2.judge import client  # noqa: E402

CSV = PROJECT_ROOT / "datasets" / "external" / "all_users_openended_BioMistral-7B_1775740232208.csv"
OUT = PROJECT_ROOT / "output" / "step9_v2" / "human_rationales_classified.json"

CLASSIFY_PROMPT = """A medical reviewer has marked an AI model's answer as INCORRECT and written a brief
rationale explaining why. Your task is to classify the *type of error* the
reviewer is describing into exactly one category:

a. MISREADING — The reviewer says the AI got a fact wrong, misread, misinterpreted,
   confused two things, or stated something that contradicts the discharge note.
   The information IS in the notes but was read incorrectly.

b. FABRICATION — The reviewer says the AI invented or hallucinated something
   not present in the discharge note at all.

c. OMISSION — The reviewer says the AI failed to mention / include / note /
   describe critical information that IS in the notes. The answer is incomplete.
   Phrases like "doesn't include", "didn't mention", "left out", "missing".

d. QUESTION_MISALIGNMENT — The reviewer says the AI answered a different
   question, focused on the wrong visit/admission/topic, or included irrelevant
   information instead of what was asked.

e. HEDGING — The reviewer says the AI gave multiple possible answers, was
   vague, or hedged instead of committing to what the notes clearly state.

f. OTHER — None of the above fit.

REVIEWER RATIONALE:
{rationale}

Respond with EXACTLY one line in this format:
CATEGORY: <one of: MISREADING, FABRICATION, OMISSION, QUESTION_MISALIGNMENT, HEDGING, OTHER>"""


def classify(text: str) -> str:
    for attempt in range(3):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You categorize error rationales into a fixed taxonomy."},
                    {"role": "user", "content": CLASSIFY_PROMPT.format(rationale=text)},
                ],
                max_tokens=20,
                temperature=0.0,
            )
            raw = r.choices[0].message.content.strip().upper()
            for cat in ["MISREADING", "FABRICATION", "OMISSION", "QUESTION_MISALIGNMENT", "HEDGING", "OTHER"]:
                if cat in raw:
                    return cat
            return "OTHER"
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"  fail: {e}")
                return "ERROR"


def main() -> int:
    df = pd.read_csv(CSV)
    df = df.sort_values("Timestamp").drop_duplicates(["User Name", "Patient ID"], keep="last")
    wrong = df[df["Answer Quality"] == 1].copy()
    wrong["Reasoning"] = wrong["Reasoning"].fillna("").astype(str)
    wrong = wrong[wrong["Reasoning"].str.strip().str.len() >= 5].reset_index(drop=True)
    print(f"Classifying {len(wrong)} rationales...", flush=True)

    existing = {}
    if OUT.exists():
        for it in json.loads(OUT.read_text()):
            existing[(it["user"], it["patient_id"])] = it
        print(f"Resuming with {len(existing)} already classified", flush=True)

    out = list(existing.values())
    for i, row in wrong.iterrows():
        key = (row["User Name"], int(row["Patient ID"]))
        if key in existing:
            continue
        cat = classify(row["Reasoning"])
        item = {
            "user": row["User Name"],
            "patient_id": int(row["Patient ID"]),
            "rationale": row["Reasoning"],
            "category": cat,
        }
        out.append(item)
        existing[key] = item
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(wrong)}] {cat}  ::  {row['Reasoning'][:80]}", flush=True)
            OUT.write_text(json.dumps(out, indent=2))
        time.sleep(0.3)
    OUT.write_text(json.dumps(out, indent=2))

    from collections import Counter
    c = Counter(it["category"] for it in out)
    print("\n=== DISTRIBUTION ===")
    for k, v in c.most_common():
        print(f"  {k:<22} {v:>4}  ({v/len(out)*100:.1f}%)")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
