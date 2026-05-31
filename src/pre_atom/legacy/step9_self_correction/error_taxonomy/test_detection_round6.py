#!/usr/bin/env python3
"""
Round 6: Contradiction-focused detection with 3-category taxonomy.

New taxonomy:
  CONTRADICTION: Answer says X, notes say Y (includes misreading + fabrication)
  OMISSION: Critical info missing that changes the conclusion
  QUESTION_MISALIGNMENT: Answer addresses wrong visit/aspect

Key design: push model to find CONTRADICTIONS, not default to OMISSION.

Usage:
    python test_detection_round6.py --port 8003 --n-wrong 10 --n-correct 10
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
# DETECTION PROMPTS — contradiction-focused
# ============================================================

# E1: Claim extraction + strict contradiction check (GPT-4o inspired)
E1_CONTRADICTION_STRICT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Your task: find factual CONTRADICTIONS between the answer and the discharge notes.

A contradiction means: the answer states something that DIRECTLY CONFLICTS with the notes. For example:
- Answer says "prescribed Lisinopril 20mg" but notes say "Lisinopril 10mg"
- Answer says "no procedure was performed" but notes describe a surgical procedure
- Answer says "during the second visit" but the information is from the first visit

Step 1: Extract each factual claim from the answer.
Step 2: For each claim, find the relevant passage in the notes.
Step 3: Does the claim CONFLICT with what the notes say? Only flag genuine conflicts where the answer and notes provide OPPOSING information.

Important: missing details are NOT contradictions. Only flag when the answer says something that the notes directly disagree with.

Report your findings."""

# E2: Question-first + contradiction check
E2_QUESTION_FIRST = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Step 1 — QUESTION CHECK: What exactly does the question ask? Which visit, which clinical aspect, which time period? Does the answer address the right thing?

Step 2 — FACT CHECK: For each specific claim in the answer (medication names, dosages, procedures, diagnoses, dates, lab values), find the matching information in the discharge notes. Does the answer get any of these details WRONG?

A wrong detail means: the answer states a fact that contradicts the notes. This is different from a missing detail — only flag facts that conflict.

Step 3 — CRITICAL OMISSION CHECK: Is there information in the notes that is absolutely essential to answer the question but completely absent from the answer? Only flag if the omission makes the answer fundamentally wrong, not just incomplete.

Report what you find."""

# E3: Adversarial contradiction hunter
E3_ADVERSARIAL_CONTRA = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

You are a fact-checker. Your ONLY job is to find claims in the answer that CONTRADICT the discharge notes.

For every medication, dosage, procedure, diagnosis, date, and lab value mentioned in the answer:
- Find where the notes discuss the same topic
- Check if the answer matches the notes EXACTLY
- If they disagree, report the discrepancy

DO NOT flag missing information — that is a different task.
DO NOT flag interpretation differences — only flag factual conflicts.

Only report genuine contradictions where the answer says one thing and the notes say another."""

# E4: Structured claim-by-claim with contradiction emphasis
E4_STRUCTURED_CLAIMS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Extract the key factual claims from this answer. For EACH claim:

CLAIM: <state the claim>
NOTES: <quote the relevant passage from the notes>
STATUS: MATCHES / CONTRADICTS / NOT IN NOTES

Focus on claims about:
- Medication names and dosages
- Procedures performed
- Diagnoses and conditions
- Dates and timeline
- Lab values and vital signs

After checking all claims, answer:
1. Does the answer address the right question (correct visit, time period)?
2. Does any claim CONTRADICT the notes?
3. Is any CRITICAL information missing that would change the answer?"""

PROMPTS = {
    "E1_contra_strict": E1_CONTRADICTION_STRICT,
    "E2_question_first": E2_QUESTION_FIRST,
    "E3_adversarial_contra": E3_ADVERSARIAL_CONTRA,
    "E4_structured_claims": E4_STRUCTURED_CLAIMS,
}

# ============================================================
# QWEN32B EXTRACTION — new 3-category taxonomy
# ============================================================

EXTRACT_PROMPT = """/nothink
Read this medical AI self-critique output.

SELF-CRITIQUE:
{raw_output}

Extract the verdict using this taxonomy:
- CONTRADICTION: the answer states something that CONFLICTS with the notes (wrong medication, wrong value, wrong procedure, fabricated detail)
- OMISSION: critical information is missing that changes the answer
- QUESTION_MISALIGNMENT: the answer addresses the wrong visit, time period, or clinical focus
- NONE: no significant errors found

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the specific wrong claim as one sentence", "correct_statement": "what the notes actually say as one sentence", "explanation": "brief"}}"""


# ============================================================
# HELPERS
# ============================================================

def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def vllm_gen(port, prompt):
    model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 2048,
              "temperature": 0.0, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=180)
    return resp.json()["choices"][0]["text"].strip()

def q32_extract(raw):
    resp = requests.post(QWEN32B_URL, json={
        "model": "Qwen/Qwen3-32B-MLX-bf16",
        "messages": [
            {"role": "system", "content": "Extract structured info. Output ONLY valid JSON."},
            {"role": "user", "content": EXTRACT_PROMPT.format(raw_output=raw)},
        ],
        "max_tokens": 400, "temperature": 0.0,
    }, timeout=90)
    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
    parser.add_argument("--fold", type=int, default=None, help="Specific fold, or all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    notes = load_notes()
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    if args.fold is not None:
        all_df = all_df[all_df["fold"] == args.fold]

    random.seed(args.seed)
    wrong = all_df[all_df["binary_correct"]==0].sample(n=min(args.n_wrong, (all_df["binary_correct"]==0).sum()), random_state=args.seed)
    correct = all_df[all_df["binary_correct"]==1].sample(n=min(args.n_correct, (all_df["binary_correct"]==1).sum()), random_state=args.seed)

    test_items = []
    for _, row in wrong.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "wrong", "row": row})
    for _, row in correct.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "correct", "row": row})

    n_w = sum(1 for t in test_items if t["label"] == "wrong")
    n_c = sum(1 for t in test_items if t["label"] == "correct")
    print(f"Round 6: Contradiction-focused ({n_w} wrong + {n_c} correct, seed={args.seed})", flush=True)
    print("=" * 70, flush=True)

    # Load GPT-4o error types for analysis
    try:
        with open(OUTPUT_DIR / "phase1_wrong_gpt4o.json") as f:
            gpt_errors = json.load(f)
        gpt_lookup = {(r["fold"], r["idx"]): r.get("PRIMARY_ERROR", "?") for r in gpt_errors}
    except:
        gpt_lookup = {}

    # Progress file — written after EACH item
    fold_label = f"fold{args.fold}" if args.fold is not None else "all"
    progress_file = OUTPUT_DIR / f"round6_{fold_label}_seed{args.seed}_progress.json"

    all_results = {}
    for pkey, ptemplate in PROMPTS.items():
        print(f"\n--- {pkey} ---", flush=True)
        results = []
        for ti in test_items:
            row = ti["row"]
            note = notes.get(str(row["patient_id"]), "")
            if not note: continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))

            msg = ptemplate.format(note=note, question=row["question"], answer=answer[:800])
            prompt = build_chatml("You are a strict medical fact-checker verifying clinical answers against discharge notes.", msg)
            raw = vllm_gen(args.port, prompt)

            obj = q32_extract(raw)
            if obj:
                verdict = str(obj.get("verdict", "UNCLEAR")).upper()
                error_type = str(obj.get("error_type", "NONE")).upper()
                error_stmt = str(obj.get("error_statement", ""))[:250]
                correct_stmt = str(obj.get("correct_statement", ""))[:250]
                explanation = str(obj.get("explanation", ""))[:250]
                parse_ok = True
            else:
                verdict = "PARSE_FAIL"; error_type = "NONE"
                error_stmt = ""; correct_stmt = ""; explanation = ""
                parse_ok = False

            entry = {
                "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                "verdict": verdict, "error_type": error_type,
                "error_statement": error_stmt, "correct_statement": correct_stmt,
                "explanation": explanation, "parse_ok": parse_ok,
                "raw_output": raw, "raw_output_len": len(raw),
            }
            results.append(entry)

            # Print per-item with flush
            gpt_type = gpt_lookup.get((ti["fold"], ti["idx"]), "?") if ti["label"] == "wrong" else "correct"
            print(f"  [{len(results)}/{len(test_items)}] {ti['label']:>7} idx={ti['idx']} → {verdict} {error_type} (gpt={gpt_type})", flush=True)

            # Save progress after each item
            all_results[pkey] = results
            with open(progress_file, "w") as pf:
                json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c, "current_prompt": pkey}, pf)

        all_results[pkey] = results

        # Summary
        w = sum(1 for r in results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
        c = sum(1 for r in results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
        pfail = sum(1 for r in results if not r["parse_ok"])
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)

        # Error type breakdown
        w_types = Counter(r["error_type"] for r in results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
        c_types = Counter(r["error_type"] for r in results if r["label"]=="correct" and r["verdict"]=="INCORRECT")

        print(f"  wrong={w}/{n_w} correct={c}/{n_c} sel={sel:.1f}x pfail={pfail}", flush=True)
        print(f"  TP types: {dict(w_types)}", flush=True)
        if c_types:
            print(f"  FP types: {dict(c_types)}", flush=True)

        # Show detections with GPT comparison
        for r in results:
            if r["verdict"] == "INCORRECT":
                gpt_type = gpt_lookup.get((r["fold"], r["idx"]), "?") if r["label"] == "wrong" else "correct"
                print(f"    [{r['label']:>7}] idx={r['idx']} det={r['error_type']:>20} gpt={gpt_type:>20} | {r['error_statement'][:80]}", flush=True)

    # Save
    out_file = OUTPUT_DIR / f"detection_round6_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c, "seed": args.seed}, f, indent=2)

    # Final table
    print(f"\n{'='*70}", flush=True)
    print(f"{'Prompt':<25} {'Wrong':>8} {'Correct':>8} {'Select':>8} {'CONTRA':>7} {'OMIS':>5} {'QMIS':>5}", flush=True)
    print("-" * 68, flush=True)
    for pkey in PROMPTS:
        r = all_results[pkey]
        w = sum(1 for x in r if x["label"]=="wrong" and x["verdict"]=="INCORRECT")
        c = sum(1 for x in r if x["label"]=="correct" and x["verdict"]=="INCORRECT")
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        wt = Counter(x["error_type"] for x in r if x["label"]=="wrong" and x["verdict"]=="INCORRECT")
        print(f"  {pkey:<25} {w}/{n_w:>3}    {c}/{n_c:>3}    {sel:>5.1f}x  {wt.get('CONTRADICTION',0):>5}  {wt.get('OMISSION',0):>4}  {wt.get('QUESTION_MISALIGNMENT',0):>4}", flush=True)

    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
