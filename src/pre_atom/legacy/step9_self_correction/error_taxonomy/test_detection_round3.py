#!/usr/bin/env python3
"""
Round 3: Fresh prompt exploration with JSON output, 10+10 items.

Prompts from GPT-4o ideas + our taxonomy + P6 baseline.
All require strict JSON output. Parse success tracked.

Usage:
    python test_detection_round3.py --port 8003
"""
import json, random, re, sys, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

# ============================================================
# PROMPTS
# ============================================================

# P6 baseline (best from round 1)
P6_BASELINE = """Discharge summary:
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

If you find an error, you MUST specify exactly:
- What claim in the answer is wrong
- What the discharge notes actually say
- Why this makes the answer incorrect

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the specific wrong claim from the answer", "notes_say": "exact quote or paraphrase from the notes", "why_wrong": "brief explanation of the error"}}"""

# G1: Counterfactual — assume answer is wrong, find evidence
G1_COUNTERFACTUAL = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

TASK: Assume this answer contains an error. Search the discharge notes for evidence that could prove any part of this answer WRONG.

Look specifically for:
- Claims about medications, dosages, or procedures that don't match the notes
- Details attributed to the wrong visit or time period
- Information that the notes directly contradict
- Key details from the notes that are missing from the answer

If you find genuine evidence of an error, report it.
If after thorough searching you find NO contradicting evidence, the answer is likely correct.

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the specific claim that is wrong, or empty string", "notes_say": "the contradicting evidence from notes, or empty string", "why_wrong": "explanation, or empty string"}}"""

# G2: Claim decomposition — extract claims, verify each
G2_CLAIMS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

TASK: Extract the 3 most important factual claims from this answer. For each claim, search the discharge notes for the EXACT supporting evidence.

Claim 1: [extract from answer]
Notes evidence: [quote from notes, or "NOT FOUND"]
Match: [SUPPORTED / CONTRADICTED / NOT_IN_NOTES]

Claim 2: [extract from answer]
Notes evidence: [quote from notes, or "NOT FOUND"]
Match: [SUPPORTED / CONTRADICTED / NOT_IN_NOTES]

Claim 3: [extract from answer]
Notes evidence: [quote from notes, or "NOT FOUND"]
Match: [SUPPORTED / CONTRADICTED / NOT_IN_NOTES]

Based on your verification, respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the contradicted or unsupported claim", "notes_say": "what the notes actually say", "why_wrong": "explanation"}}"""

# G3: Reverse inference — what question does this answer fit?
G3_REVERSE = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

TASK: First, read the answer and determine what question it is actually answering. Then compare with the original question.

Step 1: What question does this answer appear to be answering? (Be specific — which visit, what clinical aspect, what time period?)

Step 2: Does that match the ORIGINAL question above? Pay attention to:
- Which visit or admission the question asks about (first? second? both?)
- What specific clinical detail the question focuses on (medication? procedure? diagnosis? outcome?)
- What time period the question covers

Step 3: Now check if the factual claims in the answer are supported by the discharge notes.

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "what is wrong", "notes_say": "what notes say", "why_wrong": "explanation"}}"""

# G4: Adversarial devil's advocate with high bar for CORRECT
G4_DEVILS_ADVOCATE = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

You are a devil's advocate reviewer. Your job is to find ANY reason this answer could be wrong. Be skeptical and thorough.

CHECK EACH OF THESE (answer honestly):
1. Does every medication name, dosage, and frequency in the answer exactly match the notes?
2. Does every procedure, diagnosis, and clinical finding in the answer exactly match the notes?
3. Does the answer address the specific visit/admission the question asks about?
4. Does the answer include ALL critical information the question asks for?

You may ONLY mark as CORRECT if ALL four checks pass with no issues.

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the specific issue found", "notes_say": "evidence from notes", "why_wrong": "explanation"}}"""

# G5: Notes-first extraction (answer last)
G5_NOTES_FIRST = """Question: {question}

Discharge summary:
{note}

TASK: Based ONLY on the discharge notes above, what are the key facts needed to answer this question? List them.

Key facts from notes:
1. ...
2. ...
3. ...

Now here is the model's answer:
{answer}

Compare the answer against your extracted facts. Does the answer match?

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "discrepancy found", "notes_say": "your extracted fact", "why_wrong": "explanation"}}"""

PROMPTS = {
    "P6_baseline": P6_BASELINE,
    "G1_counterfactual": G1_COUNTERFACTUAL,
    "G2_claims": G2_CLAIMS,
    "G3_reverse": G3_REVERSE,
    "G4_devils_advocate": G4_DEVILS_ADVOCATE,
    "G5_notes_first": G5_NOTES_FIRST,
}


# ============================================================
# GENERATION + PARSING
# ============================================================

def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def vllm_generate(port, prompt, max_tokens=1024, temperature=0.0):
    try:
        model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
        resp = requests.post(f"http://localhost:{port}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]},
            timeout=120)
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        return f"ERROR: {e}"

def try_parse_json(text):
    try: return json.loads(text), "direct"
    except: pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group()), "extracted"
        except: pass
    cleaned = re.sub(r'^```\w*\n?', '', text.strip())
    cleaned = re.sub(r'\n?```$', '', cleaned).strip()
    try: return json.loads(cleaned), "cleaned"
    except: pass
    return None, "failed"


# ============================================================
# DATA
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
    wrong = all_df[all_df["binary_correct"] == 0].sample(n=min(args.n_wrong, (all_df["binary_correct"]==0).sum()), random_state=42)
    correct = all_df[all_df["binary_correct"] == 1].sample(n=min(args.n_correct, (all_df["binary_correct"]==1).sum()), random_state=42)

    test_items = []
    for _, row in wrong.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "wrong", "row": row})
    for _, row in correct.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "correct", "row": row})

    n_w = sum(1 for t in test_items if t["label"] == "wrong")
    n_c = sum(1 for t in test_items if t["label"] == "correct")
    print(f"Round 3: Qwen2.5 ({n_w} wrong + {n_c} correct), {len(args.prompts)} prompts")
    print("=" * 70)

    all_results = {}
    for pkey in args.prompts:
        ptemplate = PROMPTS[pkey]
        results = []
        for ti in test_items:
            row = ti["row"]
            note = notes.get(str(row["patient_id"]), "")
            if not note: continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))
            msg = ptemplate.format(note=note, question=row["question"], answer=answer[:800])
            system = "You are a strict medical expert. Respond with ONLY a JSON object."
            prompt = build_chatml(system, msg)
            raw = vllm_generate(args.port, prompt)
            obj, parse_method = try_parse_json(raw)

            if obj and isinstance(obj, dict):
                verdict = str(obj.get("verdict", "")).upper()
                error_type = str(obj.get("error_type", "NONE")).upper()
                wrong_claim = str(obj.get("wrong_claim", ""))[:200]
                notes_say = str(obj.get("notes_say", ""))[:200]
                why_wrong = str(obj.get("why_wrong", ""))[:200]
            else:
                verdict = "PARSE_FAIL"; error_type = "NONE"
                wrong_claim = ""; notes_say = ""; why_wrong = ""
                parse_method = "failed"

            results.append({
                "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                "verdict": verdict, "detected": verdict == "INCORRECT",
                "error_type": error_type,
                "wrong_claim": wrong_claim, "notes_say": notes_say, "why_wrong": why_wrong,
                "parse_method": parse_method, "raw_output": raw[:400],
            })

        all_results[pkey] = results
        w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
        c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
        pf = sum(1 for r in results if r["verdict"] == "PARSE_FAIL")
        parse_ok = sum(1 for r in results if r["parse_method"] != "failed")
        sel = (w_det/max(n_w,1)) / max(c_det/max(n_c,1), 0.01)
        print(f"  {pkey:<25} wrong={w_det}/{n_w} ({100*w_det/n_w:>2.0f}%)  correct={c_det}/{n_c} ({100*c_det/n_c:>2.0f}%)  "
              f"sel={sel:>5.1f}x  parse={parse_ok}/{len(results)} fail={pf}")

    # Save with all raw outputs
    out_file = OUTPUT_DIR / "detection_round3.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c}, f, indent=2)

    # Show detected items details for best prompt
    print(f"\n{'='*70}")
    best_pkey = max(all_results, key=lambda k: sum(1 for r in all_results[k] if r["label"]=="wrong" and r["detected"]) - sum(1 for r in all_results[k] if r["label"]=="correct" and r["detected"]))
    print(f"Best prompt: {best_pkey}")
    for r in all_results[best_pkey]:
        if r["detected"]:
            print(f"  [{r['label'].upper():>7}] idx={r['idx']} {r['error_type']:>25} | {r['wrong_claim'][:80]}")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
