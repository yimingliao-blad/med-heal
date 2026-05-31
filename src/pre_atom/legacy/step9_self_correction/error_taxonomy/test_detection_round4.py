#!/usr/bin/env python3
"""
Round 4: Free-form Qwen2.5 + Qwen3-32B extraction pipeline.
- Qwen2.5 outputs free-form verbose critique (max_tokens=2048)
- Qwen3-32B extracts structured JSON with atom-compatible error description
- All raw outputs saved for audit
- 10 wrong + 10 correct, same seed=42

Usage:
    python test_detection_round4.py --port 8003
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
# DETECTION PROMPTS (free-form, no output format constraint)
# ============================================================

# D1: CoT + few-shot (our validated best)
D1_COT_FEWSHOT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check this answer for errors. Common error patterns:
- MISREADING: Answer says "Lisinopril 20mg" but notes say "Lisinopril 10mg"
- FABRICATION: Answer mentions "CT scan" but notes never mention any CT scan
- OMISSION: Question asks about medication changes but answer only mentions one of three changes
- QUESTION_MISALIGNMENT: Question asks about second visit but answer describes the first visit

Check step by step:
1. Does it address the right question?
2. Is every claim supported by the notes?
3. Are critical details included?

If you find an error, explain what is wrong, what the notes actually say, and why this makes the answer incorrect. If the answer is correct, explain why."""

# D2: Claim extraction + verification
D2_CLAIMS_VERIFY = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Extract the key factual claims from this answer. For each claim, find the EXACT supporting or contradicting evidence in the discharge notes.

For each claim:
- State the claim
- Quote the relevant passage from the notes
- Is it SUPPORTED, CONTRADICTED, or NOT IN NOTES?

After checking all claims, state whether the answer is correct or has errors."""

# D3: Strict teacher grading
D3_STRICT_TEACHER = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

You are an experienced medical professor grading this answer. Be strict — check every detail.

Grade on three criteria:
1. ACCURACY: Does every fact match the discharge notes? Check medication names, dosages, procedures, diagnoses, and dates.
2. RELEVANCE: Does it answer what the question specifically asks? Not a related question, but THIS question.
3. COMPLETENESS: Does it include all critical information needed to fully answer the question?

Give your grade: PASS or FAIL. Explain your reasoning."""

# D4: Devil's advocate with evidence requirements
D4_DEVILS_EVIDENCE = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Your task: try to find errors in this answer. Be thorough and skeptical.

For each claim in the answer, search the discharge notes for the relevant passage. If the claim doesn't match the notes, explain the discrepancy. If you can't find supporting evidence for a claim, flag it.

Pay special attention to:
- Medication names and dosages
- Which hospital visit or admission the information comes from
- Whether all parts of the question are addressed

Report what you find."""

# D5: Question-focused verification
D5_QUESTION_FOCUS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

First, carefully re-read the question. What EXACTLY is it asking for?
- Which hospital visit? (first, second, both?)
- What clinical aspect? (medication, procedure, diagnosis, outcome, complication?)
- What time period?

Now check: does the answer address exactly what the question asks?
Then check: are the facts in the answer supported by the discharge notes?

Report your findings."""

PROMPTS = {
    "D1_cot_fewshot": D1_COT_FEWSHOT,
    "D2_claims_verify": D2_CLAIMS_VERIFY,
    "D3_strict_teacher": D3_STRICT_TEACHER,
    "D4_devils_evidence": D4_DEVILS_EVIDENCE,
    "D5_question_focus": D5_QUESTION_FOCUS,
}

# ============================================================
# QWEN3-32B EXTRACTION (atom-compatible)
# ============================================================

EXTRACT_PROMPT = """/nothink
Read the following self-critique output from a medical AI (Qwen2.5-7B) that was checking its own answer against discharge notes.

SELF-CRITIQUE OUTPUT:
{raw_output}

Extract the following as a JSON object:

1. "verdict": Did the AI conclude the answer was CORRECT or INCORRECT?
   - If the AI identified specific factual errors, contradictions, or critical omissions → INCORRECT
   - If the AI said "mostly correct but lacks critical details" with specific issues → INCORRECT
   - If the AI confirmed all claims are supported → CORRECT
   - If unclear → UNCLEAR

2. "error_type": MISREADING / FABRICATION / OMISSION / QUESTION_MISALIGNMENT / NONE

3. "error_statement": Rewrite the specific error as ONE atomic factual statement that is WRONG.
   Example: "The patient was prescribed Metoprolol 50mg at discharge" (when notes say Lisinopril 10mg)
   This should be a single claim that can be verified against the notes.

4. "correct_statement": What the notes actually say, as ONE atomic factual statement.
   Example: "The patient was prescribed Lisinopril 10mg at discharge"

5. "explanation": Brief explanation of why this is an error.

Respond with ONLY a JSON object:
{{"verdict": "...", "error_type": "...", "error_statement": "...", "correct_statement": "...", "explanation": "..."}}"""


# ============================================================
# HELPERS
# ============================================================

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
                {"role": "system", "content": "Extract structured info from text. Output ONLY valid JSON."},
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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n-wrong", type=int, default=10)
    parser.add_argument("--n-correct", type=int, default=10)
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()))
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
    print(f"Round 4: Qwen2.5 free-form + Qwen32B extraction ({n_w} wrong + {n_c} correct)")
    print(f"Prompts: {args.prompts}")
    print(f"Max tokens: 2048")
    print("=" * 70)

    all_results = {}
    for pkey in args.prompts:
        ptemplate = PROMPTS[pkey]
        print(f"\n--- {pkey} ---")

        results = []
        for ti in test_items:
            row = ti["row"]
            note = notes.get(str(row["patient_id"]), "")
            if not note: continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))

            # Qwen2.5 free-form critique
            msg = ptemplate.format(note=note, question=row["question"], answer=answer[:800])
            prompt = build_chatml("You are a strict medical expert verifying clinical answers against discharge notes.", msg)
            raw = vllm_generate(args.port, prompt, max_tokens=2048)

            # Qwen3-32B extraction
            q32_raw = qwen32b_extract(raw)
            obj = try_parse_json(q32_raw)

            if obj and isinstance(obj, dict):
                verdict = str(obj.get("verdict", "UNCLEAR")).upper()
                error_type = str(obj.get("error_type", "NONE")).upper()
                error_stmt = str(obj.get("error_statement", ""))[:250]
                correct_stmt = str(obj.get("correct_statement", ""))[:250]
                explanation = str(obj.get("explanation", ""))[:250]
                parse_ok = True
            else:
                verdict = "PARSE_FAIL"
                error_type = "NONE"
                error_stmt = ""; correct_stmt = ""; explanation = ""
                parse_ok = False

            results.append({
                "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                "verdict": verdict,
                "error_type": error_type,
                "error_statement": error_stmt,
                "correct_statement": correct_stmt,
                "explanation": explanation,
                "parse_ok": parse_ok,
                "raw_output": raw,  # FULL output
                "raw_output_len": len(raw),
                "q32_raw": q32_raw[:400],
            })

        all_results[pkey] = results

        # Summary
        w_det = sum(1 for r in results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
        c_det = sum(1 for r in results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
        pf = sum(1 for r in results if not r["parse_ok"])
        avg_len = sum(r["raw_output_len"] for r in results) / max(len(results), 1)
        sel = (w_det/max(n_w,1)) / max(c_det/max(n_c,1), 0.01)
        print(f"  wrong={w_det}/{n_w} correct={c_det}/{n_c} sel={sel:.1f}x parse_fail={pf} avg_len={avg_len:.0f}")

        # Show detected errors with atom statements
        detected = [r for r in results if r["verdict"] == "INCORRECT"]
        if detected:
            print(f"  Detected errors:")
            for r in detected:
                print(f"    [{r['label']:>7}] idx={r['idx']} {r['error_type']}")
                if r["error_statement"]:
                    print(f"      WRONG: {r['error_statement'][:120]}")
                if r["correct_statement"]:
                    print(f"      RIGHT: {r['correct_statement'][:120]}")

    # Save everything
    out_file = OUTPUT_DIR / "detection_round4.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c}, f, indent=2)

    # Final table
    print(f"\n{'='*70}")
    print(f"{'Prompt':<22} {'Wrong':>8} {'Correct':>8} {'Select':>8} {'PFail':>6} {'AvgLen':>7}")
    print("-" * 62)
    for pkey in args.prompts:
        r = all_results[pkey]
        w = sum(1 for x in r if x["label"]=="wrong" and x["verdict"]=="INCORRECT")
        c = sum(1 for x in r if x["label"]=="correct" and x["verdict"]=="INCORRECT")
        pf = sum(1 for x in r if not x["parse_ok"])
        avg = sum(x["raw_output_len"] for x in r) / max(len(r), 1)
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        print(f"  {pkey:<22} {w}/{n_w:>3}    {c}/{n_c:>3}    {sel:>5.1f}x  {pf:>4}  {avg:>6.0f}")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
