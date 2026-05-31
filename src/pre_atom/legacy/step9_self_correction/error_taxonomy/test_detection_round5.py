#!/usr/bin/env python3
"""
Round 5: D2 variations — keep claim verification + add completeness.

D2 original: extract claims, verify each → catches MISREADING (30x selectivity, 30% recall)
Problem: misses OMISSION errors (doesn't check what's missing)

Variations:
  D2a: Claims + "what does the question need that the answer doesn't cover?"
  D2b: Claims + extract key facts from notes FIRST, then check if answer covers them
  D2c: Claims + at the end, list what the question asks for and check each is answered
  D2d: More claims (5 instead of 3) + completeness check

Usage:
    python test_detection_round5.py --port 8003
"""
import json, random, re, sys, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

# ============================================================
# D2 VARIATIONS
# ============================================================

D2_ORIGINAL = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Extract the key factual claims from this answer. For each claim, find the EXACT supporting or contradicting evidence in the discharge notes.

For each claim:
- State the claim
- Quote the relevant passage from the notes
- Is it SUPPORTED, CONTRADICTED, or NOT IN NOTES?

After checking all claims, state whether the answer is correct or has errors."""

D2A_CLAIMS_PLUS_GAPS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

PART 1: Extract the key factual claims from this answer. For each claim, find the supporting or contradicting evidence in the discharge notes.

For each claim:
- State the claim
- Quote the relevant passage from the notes
- Is it SUPPORTED, CONTRADICTED, or NOT IN NOTES?

PART 2: Now re-read the question. What specific information does the question ask for? Check if the answer covers EACH required piece of information. List anything the question asks for that the answer does not address.

Based on Parts 1 and 2, state whether the answer is correct or has errors."""

D2B_NOTES_THEN_CLAIMS = """Discharge summary:
{note}

Question: {question}

PART 1: Before looking at the answer, list the 3-5 key facts from the discharge notes that are needed to answer this question. Quote each fact.

Now read the answer:
Answer: {answer}

PART 2: For each key fact you listed above, does the answer include it? Does the answer say anything that contradicts it?

PART 3: Does the answer include any claims NOT supported by the notes?

Based on your analysis, state whether the answer is correct or has errors."""

D2C_REQUIREMENTS_CHECK = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Step 1: Extract each factual claim from the answer. Verify each against the discharge notes.

Claim 1: <claim> → SUPPORTED / CONTRADICTED / NOT IN NOTES
Claim 2: <claim> → SUPPORTED / CONTRADICTED / NOT IN NOTES
Claim 3: <claim> → SUPPORTED / CONTRADICTED / NOT IN NOTES

Step 2: What does the question specifically ask for? List each requirement.
- Requirement 1: <what the question needs>
  Covered by answer? YES / NO
- Requirement 2: <what the question needs>
  Covered by answer? YES / NO

Step 3: Based on Steps 1 and 2, is this answer correct or does it have errors?"""

D2D_MORE_CLAIMS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Extract ALL factual claims from this answer (aim for 5 or more). For each claim, find the EXACT evidence in the discharge notes.

Claim 1: <claim>
Notes: <exact quote or "NOT FOUND">
Verdict: SUPPORTED / CONTRADICTED / NOT IN NOTES

Claim 2: <claim>
Notes: <exact quote or "NOT FOUND">
Verdict: SUPPORTED / CONTRADICTED / NOT IN NOTES

[continue for all claims]

Also check: is there critical information in the notes relevant to the question that the answer does NOT mention?

Final assessment: is this answer correct or does it have errors?"""

PROMPTS = {
    "D2_original": D2_ORIGINAL,
    "D2a_claims_gaps": D2A_CLAIMS_PLUS_GAPS,
    "D2b_notes_first": D2B_NOTES_THEN_CLAIMS,
    "D2c_requirements": D2C_REQUIREMENTS_CHECK,
    "D2d_more_claims": D2D_MORE_CLAIMS,
}

# ============================================================
# HELPERS (same as round4)
# ============================================================

EXTRACT_PROMPT = """/nothink
Read the following self-critique output from a medical AI (Qwen2.5-7B) checking its own answer.

SELF-CRITIQUE OUTPUT:
{raw_output}

Extract as JSON:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the specific wrong/missing claim as one sentence", "correct_statement": "what the notes say as one sentence", "explanation": "brief explanation"}}"""

def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def vllm_generate(port, prompt, max_tokens=2048, temperature=0.0):
    try:
        model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
        resp = requests.post(f"http://localhost:{port}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]},
            timeout=180)
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        return f"ERROR: {e}"

def qwen32b_extract(raw_output):
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Extract structured info. Output ONLY valid JSON."},
                {"role": "user", "content": EXTRACT_PROMPT.format(raw_output=raw_output)},
            ],
            "max_tokens": 400, "temperature": 0.0,
        }, timeout=90)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        return f'{{"error": "{e}"}}'

def try_parse_json(text):
    try: return json.loads(text)
    except: pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

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
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n-wrong", type=int, default=10)
    parser.add_argument("--n-correct", type=int, default=10)
    args = parser.parse_args()

    notes = load_notes()
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    random.seed(42)
    wrong = all_df[all_df["binary_correct"]==0].sample(n=min(args.n_wrong, (all_df["binary_correct"]==0).sum()), random_state=42)
    correct = all_df[all_df["binary_correct"]==1].sample(n=min(args.n_correct, (all_df["binary_correct"]==1).sum()), random_state=42)

    test_items = []
    for _, row in wrong.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "wrong", "row": row})
    for _, row in correct.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "correct", "row": row})

    n_w = sum(1 for t in test_items if t["label"] == "wrong")
    n_c = sum(1 for t in test_items if t["label"] == "correct")
    print(f"Round 5: D2 variations ({n_w} wrong + {n_c} correct)")
    print("=" * 70)

    all_results = {}
    for pkey, ptemplate in PROMPTS.items():
        print(f"\n--- {pkey} ---")
        results = []
        for ti in test_items:
            row = ti["row"]
            note = notes.get(str(row["patient_id"]), "")
            if not note: continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))

            msg = ptemplate.format(note=note, question=row["question"], answer=answer[:800])
            prompt = build_chatml("You are a strict medical expert verifying clinical answers against discharge notes.", msg)
            raw = vllm_generate(args.port, prompt)

            q32_raw = qwen32b_extract(raw)
            obj = try_parse_json(q32_raw)

            if obj and isinstance(obj, dict):
                verdict = str(obj.get("verdict", "UNCLEAR")).upper()
                error_type = str(obj.get("error_type", "NONE")).upper()
                error_stmt = str(obj.get("error_statement", ""))[:250]
                correct_stmt = str(obj.get("correct_statement", ""))[:250]
                parse_ok = True
            else:
                verdict = "PARSE_FAIL"; error_type = "NONE"
                error_stmt = ""; correct_stmt = ""
                parse_ok = False

            results.append({
                "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                "verdict": verdict, "error_type": error_type,
                "error_statement": error_stmt, "correct_statement": correct_stmt,
                "parse_ok": parse_ok, "raw_output": raw, "raw_output_len": len(raw),
            })

        all_results[pkey] = results
        w = sum(1 for r in results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
        c = sum(1 for r in results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
        pf = sum(1 for r in results if not r["parse_ok"])
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        print(f"  wrong={w}/{n_w} correct={c}/{n_c} sel={sel:.1f}x pfail={pf}")

        # Show detections
        for r in results:
            if r["verdict"] == "INCORRECT":
                print(f"    [{r['label']:>7}] idx={r['idx']} {r['error_type']}: {r['error_statement'][:100]}")

    # Save
    out_file = OUTPUT_DIR / "detection_round5.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c}, f, indent=2)

    # Final table
    print(f"\n{'='*70}")
    print(f"{'Prompt':<22} {'Wrong':>8} {'Correct':>8} {'Select':>8} {'PFail':>6}")
    print("-" * 56)
    for pkey in PROMPTS:
        r = all_results[pkey]
        w = sum(1 for x in r if x["label"]=="wrong" and x["verdict"]=="INCORRECT")
        c = sum(1 for x in r if x["label"]=="correct" and x["verdict"]=="INCORRECT")
        pf = sum(1 for x in r if not x["parse_ok"])
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        print(f"  {pkey:<22} {w}/{n_w:>3}    {c}/{n_c:>3}    {sel:>5.1f}x  {pf:>4}")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
