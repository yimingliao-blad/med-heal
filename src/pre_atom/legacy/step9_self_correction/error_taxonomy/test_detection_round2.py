#!/usr/bin/env python3
"""
Round 2: Iterate on BC prompt to improve MISREADING detection.
Problem: BC catches only 17% of MISREADING errors (5/30).
The model misreads the notes and self-confirms the misreading.

New strategies to break self-confirmation for MISREADING:
  BC2: Add explicit "re-read" instruction — force model to quote the exact note passage
  BC3: Add misreading-specific examples showing subtle confusions (dosage, laterality, visit)
  BC4: Reverse-check — instead of checking answer against notes, extract facts from notes first,
       then compare with answer (avoid anchoring on the answer)
  BC5: BC + A2 combined — run both, flag if either detects
"""
import json, os, random, re, sys, time
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
PORT = 8003

# ============================================================
# ROUND 2 PROMPTS
# ============================================================

# BC2: Force exact quote from notes for each claim
BC2_QUOTE_CHECK = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Review this answer for errors. Common error patterns include:
- MISREADING: confusing medications, dosages, or visits that ARE in the notes
- FABRICATION: stating details NOT found anywhere in the notes
- OMISSION: missing critical information that changes the answer
- QUESTION_MISALIGNMENT: answering about the wrong visit or clinical focus

STEP 1 — For each KEY CLAIM in the answer, find and QUOTE the exact passage from the discharge notes that supports it. If you cannot find a supporting passage, the claim may be fabricated or misread.

CLAIM 1: <claim from answer>
NOTES QUOTE: <exact quote, or "NOT FOUND">
MATCH: SUPPORTED / CONTRADICTED / NOT_FOUND

CLAIM 2: <claim from answer>
NOTES QUOTE: <exact quote, or "NOT FOUND">
MATCH: SUPPORTED / CONTRADICTED / NOT_FOUND

(continue for all key claims)

STEP 2 — Does the answer address the correct aspect of the question?
ALIGNMENT: OK or PROBLEM

STEP 3 — Is critical information from the notes missing?
COMPLETENESS: OK or PROBLEM

VERDICT: CORRECT or INCORRECT
IF INCORRECT — ERROR_TYPE: <MISREADING, FABRICATION, OMISSION, or QUESTION_MISALIGNMENT>"""

# BC3: Misreading-focused examples
BC3_MISREAD_EXAMPLES = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Review this answer for errors. Pay special attention to MISREADING errors — these are the most common type where the model gets a detail wrong that IS in the notes.

Examples of MISREADING errors:
1. Notes say "Lisinopril increased from 10mg to 20mg" → Answer says "Lisinopril decreased to 10mg" (wrong direction)
2. Notes say "LEFT knee replacement" → Answer says "right knee replacement" (wrong laterality)
3. Notes describe Visit 1 medications → Answer attributes them to Visit 2 (wrong visit)
4. Notes say "Hemoglobin improved from 8.2 to 10.1" → Answer says "Hemoglobin declined" (opposite trend)
5. Notes say "discontinued Metoprolol, started Atenolol" → Answer says "continued on Metoprolol" (missed the change)

STEP 1 — Check alignment: Does the answer address the right question?
ALIGNMENT: OK or PROBLEM

STEP 2 — Check each factual claim: Does it match EXACTLY what the notes say? Watch for:
- Wrong medication names or dosages
- Wrong laterality (left vs right)
- Wrong visit attribution
- Wrong direction of change (increased vs decreased, improved vs worsened)
- Wrong procedure or diagnosis name
EVIDENCE: OK or PROBLEM — <specific mismatch>

STEP 3 — Check completeness: Is critical info missing?
COMPLETENESS: OK or PROBLEM

VERDICT: CORRECT or INCORRECT
IF INCORRECT — ERROR_TYPE: <MISREADING, FABRICATION, OMISSION, or QUESTION_MISALIGNMENT>"""

# BC4: Notes-first extraction (avoid anchoring on answer)
BC4_NOTES_FIRST = """Discharge summary:
{note}

Question: {question}

STEP 1 — Before looking at the answer, extract the key facts from the discharge notes that are relevant to this question:
RELEVANT FACTS FROM NOTES:
- <fact 1>
- <fact 2>
- <fact 3>

Now look at the answer:
Answer: {answer}

STEP 2 — Compare the answer against the facts you extracted:
- Does the answer match the facts from the notes?
- Does the answer include anything that contradicts the notes?
- Does the answer miss any critical facts?

COMPARISON:
- <fact 1>: MATCHES / CONTRADICTS / MISSING
- <fact 2>: MATCHES / CONTRADICTS / MISSING
- <fact 3>: MATCHES / CONTRADICTS / MISSING

VERDICT: CORRECT or INCORRECT
IF INCORRECT — ERROR_TYPE: <MISREADING, FABRICATION, OMISSION, or QUESTION_MISALIGNMENT>"""

# BC5: BC original + A2 evidence check (run both, combine)
# This uses the original BC prompt — we'll combine with A2 in the runner

PROMPTS = {
    "BC2_quote": BC2_QUOTE_CHECK,
    "BC3_misread": BC3_MISREAD_EXAMPLES,
    "BC4_notes_first": BC4_NOTES_FIRST,
}


def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")


def vllm_generate(prompt, max_tokens=1024, temperature=0.0):
    try:
        model = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
        resp = requests.post(
            f"http://localhost:{PORT}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]},
            timeout=120,
        )
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"  vLLM error: {e}")
        return ""


def detect_error(prompt_key, prompt_template, note, question, answer):
    msg = prompt_template.format(note=note, question=question, answer=answer[:800])
    system = "You are a strict medical expert verifying clinical answers against discharge notes."
    prompt = build_chatml(system, msg)
    raw = vllm_generate(prompt)
    raw_upper = raw.upper()

    detected = "VERDICT: INCORRECT" in raw_upper or \
               ("VERDICT" in raw_upper and "INCORRECT" in raw_upper.split("VERDICT")[-1][:20])

    m = re.search(r'ERROR_TYPE:\s*(.+?)(?:\n|$)', raw, re.I)
    detail = m.group(1).strip()[:150] if m else ""

    # For BC2, also check CONTRADICTED
    if prompt_key == "BC2_quote" and not detected:
        if "CONTRADICTED" in raw_upper:
            detected = True
            detail = "Claim contradicted by notes"

    return detected, detail


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
    # Load same test set as round 1
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    notes = load_notes()

    random.seed(42)
    wrong_sample = all_df[all_df["binary_correct"] == 0].sample(n=50, random_state=42)
    correct_sample = all_df[all_df["binary_correct"] == 1].sample(n=50, random_state=42)

    test_items = []
    for _, row in wrong_sample.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]),
                           "label": "wrong", "row": row})
    for _, row in correct_sample.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]),
                           "label": "correct", "row": row})

    # Load GPT-4o error types for analysis
    with open(OUTPUT_DIR / "phase1_wrong_gpt4o.json") as f:
        gpt_errors = json.load(f)
    gpt_lookup = {(r["fold"], r["idx"]): r.get("PRIMARY_ERROR", "?") for r in gpt_errors}

    print(f"Round 2: Testing {len(PROMPTS)} prompts on {len(test_items)} items")

    all_results = {}
    for pkey, ptemplate in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Prompt: {pkey}")
        print(f"{'='*60}")

        results = []
        for i, ti in enumerate(test_items):
            row = ti["row"]
            note = notes.get(str(row["patient_id"]), "")
            if not note:
                continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))
            detected, detail = detect_error(pkey, ptemplate, note, row["question"], answer)
            results.append({
                "idx": ti["idx"], "fold": ti["fold"],
                "label": ti["label"], "detected": detected, "detail": detail[:100],
            })

            if (i + 1) % 25 == 0:
                w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
                c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
                w_tot = sum(1 for r in results if r["label"] == "wrong")
                c_tot = sum(1 for r in results if r["label"] == "correct")
                print(f"  [{i+1}/{len(test_items)}] wrong={w_det}/{w_tot} correct={c_det}/{c_tot}")

        all_results[pkey] = results

        # Summary with error type breakdown
        w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
        c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
        w_tot = sum(1 for r in results if r["label"] == "wrong")
        c_tot = sum(1 for r in results if r["label"] == "correct")
        sel = (w_det/max(w_tot,1)) / max(c_det/max(c_tot,1), 0.01)
        print(f"\n  Wrong: {w_det}/{w_tot} ({100*w_det/w_tot:.0f}%)  Correct: {c_det}/{c_tot} ({100*c_det/c_tot:.0f}%)  Select: {sel:.1f}x")

        # Per error type
        wrong_results = [r for r in results if r["label"] == "wrong"]
        for et in ["MISREADING", "QUESTION_MISALIGNMENT", "OMISSION", "FABRICATION"]:
            et_items = [r for r in wrong_results if gpt_lookup.get((r["fold"], r["idx"]), "?") == et]
            et_det = sum(1 for r in et_items if r["detected"])
            if et_items:
                print(f"    {et}: {et_det}/{len(et_items)} ({100*et_det/len(et_items):.0f}%)")

    # Save
    with open(OUTPUT_DIR / "detection_round2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Comparison with round 1 BC
    print(f"\n{'='*60}")
    print("ROUND 2 vs ROUND 1 BC")
    print(f"{'='*60}")
    print(f"  {'Prompt':<20} {'Wrong':>10} {'Correct':>10} {'Select':>10} {'MISREAD':>10}")
    print(f"  {'BC (round 1)':<20} {'12/50':>10} {'1/50':>10} {'12.0x':>10} {'5/30':>10}")
    for pkey in PROMPTS:
        r = all_results[pkey]
        w = sum(1 for x in r if x["label"] == "wrong" and x["detected"])
        c = sum(1 for x in r if x["label"] == "correct" and x["detected"])
        wr = [x for x in r if x["label"] == "wrong"]
        mr = [x for x in wr if gpt_lookup.get((x["fold"], x["idx"]), "?") == "MISREADING"]
        mr_det = sum(1 for x in mr if x["detected"])
        sel = (w/50) / max(c/50, 0.01)
        print(f"  {pkey:<20} {w}/50{' ':>5} {c}/50{' ':>5} {sel:>8.1f}x {mr_det}/{len(mr)}{' ':>5}")


if __name__ == "__main__":
    main()
