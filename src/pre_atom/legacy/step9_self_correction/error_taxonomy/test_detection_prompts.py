#!/usr/bin/env python3
"""
Phase 2: Test detection prompts on Qwen2.5 balanced set (50 wrong + 50 correct).
Uses the model's own answers — self-critique scenario (no ground truth given).

Prompts based on 3 principles:
  P1: Answer ties to question (catches QUESTION_MISALIGNMENT)
  P2: Faithful to evidence (catches MISREADING + FABRICATION)
  P3: Covers key details (catches OMISSION)

Strategies:
  A: Rule-by-rule (3 separate prompts)
  B: CoT single prompt
  C: Few-shot with error examples
  B+C: CoT with few-shot examples

Each outputs ERROR_FOUND: YES/NO
"""
import json, os, random, re, sys, time
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

VLLM_URL = "http://localhost:8003/v1/completions"
PORT = 8003

# ============================================================
# PROMPTS
# ============================================================

# --- Strategy A: Rule-by-rule ---

A1_QUESTION_ALIGNMENT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check: Does this answer directly address what the question asks?
- Does it answer about the correct hospital visit or time period?
- Does it focus on what the question specifically asks (medication, procedure, diagnosis, etc.)?
- Does it avoid answering a different question than what was asked?

ADDRESSES_QUESTION: YES or NO
IF NO — ISSUE: <what the answer addresses vs what the question asks>"""

A2_EVIDENCE_FAITHFUL = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check each factual claim in the answer against the discharge notes:
- Is every medication, dosage, procedure, diagnosis, and date mentioned in the answer supported by the notes?
- Does the answer correctly represent what the notes say (not reversed, not confused with a different detail)?

List any claims that contradict or are not supported by the notes.

ALL_CLAIMS_SUPPORTED: YES or NO
IF NO — UNSUPPORTED_CLAIM: <the specific claim and what the notes actually say>"""

A3_KEY_DETAILS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check: Does the answer include the critical details from the discharge notes needed to fully answer the question?
- Only flag missing information that would change the answer's conclusion
- Do NOT flag minor details (exact dates, non-critical medications) unless the question specifically asks for them

COVERS_KEY_DETAILS: YES or NO
IF NO — MISSING: <what critical detail from the notes is missing>"""

# --- Strategy B: CoT single prompt ---

B_COT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Review this answer step by step:

STEP 1 — QUESTION ALIGNMENT: Does the answer address what the question specifically asks? Check the correct visit, time period, and clinical focus.
ALIGNMENT: OK or PROBLEM — <explanation>

STEP 2 — EVIDENCE CHECK: Is every factual claim in the answer supported by the discharge notes? Check medications, dosages, procedures, diagnoses, dates.
EVIDENCE: OK or PROBLEM — <the specific unsupported or contradicted claim>

STEP 3 — COMPLETENESS: Does the answer include the critical information from the notes needed to answer the question? Only flag missing info that would change the conclusion.
COMPLETENESS: OK or PROBLEM — <what's missing>

FINAL VERDICT: Based on the above, is this answer likely correct?
VERDICT: CORRECT or INCORRECT"""

# --- Strategy C: Few-shot ---

C_FEWSHOT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Your task: check if this answer contains errors by comparing it against the discharge notes.

Here are examples of common errors in clinical answers:

Example 1 (MISREADING):
  Question: "What medications were changed between visits?"
  Notes say: "Lisinopril increased from 10mg to 20mg. Metoprolol was discontinued."
  Wrong answer: "Metoprolol was increased to 20mg."
  Error: Confused lisinopril with metoprolol — misread which medication changed.

Example 2 (QUESTION_MISALIGNMENT):
  Question: "What was the discharge diagnosis from the SECOND visit?"
  Notes contain two visits with different diagnoses.
  Wrong answer: Lists the diagnosis from the first visit.
  Error: Answered about the wrong visit.

Example 3 (OMISSION):
  Question: "What surgical procedures were performed?"
  Notes say: "Patient underwent appendectomy AND cholecystectomy."
  Wrong answer: "The patient underwent an appendectomy."
  Error: Missed the cholecystectomy — incomplete answer.

Example 4 (FABRICATION):
  Question: "What antibiotic was prescribed at discharge?"
  Notes say: "Discharged on cephalexin 500mg."
  Wrong answer: "The patient was prescribed amoxicillin 500mg at discharge."
  Error: Amoxicillin is not mentioned anywhere in the notes.

Now check the answer above. Does it contain any of these types of errors?

ERROR_FOUND: YES or NO
IF YES — ERROR_TYPE: <MISREADING, QUESTION_MISALIGNMENT, OMISSION, or FABRICATION>
EXPLANATION: <brief description of the error>"""

# --- Strategy B+C: CoT with few-shot ---

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

PROMPTS = {
    "A1_alignment": A1_QUESTION_ALIGNMENT,
    "A2_evidence": A2_EVIDENCE_FAITHFUL,
    "A3_details": A3_KEY_DETAILS,
    "B_cot": B_COT,
    "C_fewshot": C_FEWSHOT,
    "BC_cot_fewshot": BC_COT_FEWSHOT,
}


# ============================================================
# vLLM GENERATION
# ============================================================

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
    """Run detection prompt, return (detected, details)."""
    msg = prompt_template.format(note=note, question=question, answer=answer[:800])
    system = "You are a strict medical expert verifying clinical answers against discharge notes."
    prompt = build_chatml(system, msg)
    raw = vllm_generate(prompt)
    raw_upper = raw.upper()

    if prompt_key.startswith("A1"):
        detected = "ADDRESSES_QUESTION: NO" in raw_upper or \
                   ("ADDRESSES_QUESTION" in raw_upper and "NO" in raw_upper.split("ADDRESSES_QUESTION")[-1][:15])
        m = re.search(r'ISSUE:\s*(.+?)(?:\n|$)', raw, re.I)
        detail = m.group(1).strip()[:150] if m else ""
    elif prompt_key.startswith("A2"):
        detected = "ALL_CLAIMS_SUPPORTED: NO" in raw_upper or \
                   ("ALL_CLAIMS_SUPPORTED" in raw_upper and "NO" in raw_upper.split("ALL_CLAIMS_SUPPORTED")[-1][:15])
        m = re.search(r'UNSUPPORTED_CLAIM:\s*(.+?)(?:\n|$)', raw, re.I)
        detail = m.group(1).strip()[:150] if m else ""
    elif prompt_key.startswith("A3"):
        detected = "COVERS_KEY_DETAILS: NO" in raw_upper or \
                   ("COVERS_KEY_DETAILS" in raw_upper and "NO" in raw_upper.split("COVERS_KEY_DETAILS")[-1][:15])
        m = re.search(r'MISSING:\s*(.+?)(?:\n|$)', raw, re.I)
        detail = m.group(1).strip()[:150] if m else ""
    elif prompt_key.startswith("B_") or prompt_key.startswith("BC"):
        detected = "VERDICT: INCORRECT" in raw_upper or \
                   ("VERDICT" in raw_upper and "INCORRECT" in raw_upper.split("VERDICT")[-1][:20])
        m = re.search(r'ERROR_TYPE:\s*(.+?)(?:\n|$)', raw, re.I)
        detail = m.group(1).strip()[:150] if m else ""
    elif prompt_key.startswith("C_"):
        detected = "ERROR_FOUND: YES" in raw_upper or \
                   ("ERROR_FOUND" in raw_upper and "YES" in raw_upper.split("ERROR_FOUND")[-1][:15])
        m = re.search(r'EXPLANATION:\s*(.+?)(?:\n|$)', raw, re.I)
        detail = m.group(1).strip()[:150] if m else ""
    else:
        detected = False
        detail = ""

    return detected, detail


# ============================================================
# MAIN
# ============================================================

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wrong", type=int, default=50)
    parser.add_argument("--n-correct", type=int, default=50)
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()))
    args = parser.parse_args()

    # Load data
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    notes = load_notes()

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

    print(f"Testing {len(args.prompts)} prompts on {len(test_items)} items "
          f"({sum(1 for t in test_items if t['label']=='wrong')} wrong + "
          f"{sum(1 for t in test_items if t['label']=='correct')} correct)")

    all_results = {}

    for pkey in args.prompts:
        ptemplate = PROMPTS[pkey]
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
                "label": ti["label"],
                "detected": detected,
                "detail": detail[:100],
            })

            if (i + 1) % 20 == 0:
                w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
                c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
                w_tot = sum(1 for r in results if r["label"] == "wrong")
                c_tot = sum(1 for r in results if r["label"] == "correct")
                print(f"  [{i+1}/{len(test_items)}] wrong={w_det}/{w_tot} correct={c_det}/{c_tot}")

        all_results[pkey] = results

        # Summary
        w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
        c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
        w_tot = sum(1 for r in results if r["label"] == "wrong")
        c_tot = sum(1 for r in results if r["label"] == "correct")
        selectivity = (w_det / max(w_tot, 1)) / max(c_det / max(c_tot, 1), 0.01)
        print(f"\n  Wrong detected: {w_det}/{w_tot} ({100*w_det/w_tot:.0f}%)")
        print(f"  Correct detected: {c_det}/{c_tot} ({100*c_det/c_tot:.0f}%)")
        print(f"  Selectivity: {selectivity:.1f}x")

    # Save
    with open(OUTPUT_DIR / "detection_prompt_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Final comparison
    print(f"\n{'='*60}")
    print("DETECTION PROMPT COMPARISON")
    print(f"{'='*60}")
    print(f"{'Prompt':<20} {'Wrong det':>12} {'Correct det':>12} {'Select':>10}")
    print("-" * 56)
    for pkey in args.prompts:
        r = all_results[pkey]
        w = sum(1 for x in r if x["label"] == "wrong" and x["detected"])
        c = sum(1 for x in r if x["label"] == "correct" and x["detected"])
        wt = sum(1 for x in r if x["label"] == "wrong")
        ct = sum(1 for x in r if x["label"] == "correct")
        sel = (w/max(wt,1)) / max(c/max(ct,1), 0.01)
        print(f"  {pkey:<20} {w}/{wt:>5} ({100*w/wt:.0f}%) {c}/{ct:>5} ({100*c/ct:.0f}%) {sel:>8.1f}x")

    # Strategy A combined: any of A1/A2/A3 detected
    if all(k in all_results for k in ["A1_alignment", "A2_evidence", "A3_details"]):
        print(f"\n  Strategy A (any):  ", end="")
        w_any = sum(1 for i in range(len(test_items))
                    if test_items[i]["label"] == "wrong" and
                    any(all_results[k][i]["detected"] for k in ["A1_alignment", "A2_evidence", "A3_details"]))
        c_any = sum(1 for i in range(len(test_items))
                    if test_items[i]["label"] == "correct" and
                    any(all_results[k][i]["detected"] for k in ["A1_alignment", "A2_evidence", "A3_details"]))
        wt = sum(1 for t in test_items if t["label"] == "wrong")
        ct = sum(1 for t in test_items if t["label"] == "correct")
        sel = (w_any/max(wt,1)) / max(c_any/max(ct,1), 0.01)
        print(f"{w_any}/{wt} ({100*w_any/wt:.0f}%) wrong, {c_any}/{ct} ({100*c_any/ct:.0f}%) correct, {sel:.1f}x")


if __name__ == "__main__":
    main()
