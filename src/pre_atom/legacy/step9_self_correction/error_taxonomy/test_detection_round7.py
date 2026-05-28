#!/usr/bin/env python3
"""
Round 7: Improved prompts focusing on:
  1. Question alignment (P1 blind spot)
  2. Only flag errors critical to the conclusion (reduce FP from minor omissions)

Key changes from Round 6:
  - Explicit question decomposition BEFORE checking facts
  - "Would this error change the answer?" gate
  - Reasoning chain focus: does the error break the logic to the conclusion?

Usage:
    python test_detection_round7.py --port 8003 --fold 2
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
# PROMPTS
# ============================================================

# F1: Three principles — structured, explicit
F1_THREE_PRINCIPLES = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check this answer using THREE principles. Go through each one carefully.

PRINCIPLE 1 — TIED TO THE QUESTION:
Re-read the question. What EXACTLY does it ask?
- Which hospital visit? (first, second, both, most recent?)
- What clinical focus? (medication changes, procedures, diagnoses, complications, outcomes?)
- What time frame?
Now: does the answer address the RIGHT visit, the RIGHT clinical focus, and the RIGHT time frame? If the answer discusses the wrong visit or wrong aspect, that is a critical error.

PRINCIPLE 2 — FAITHFUL TO THE NOTES:
For each KEY factual claim in the answer (medications, dosages, procedures, diagnoses, lab values), find the matching information in the discharge notes. Does the answer contradict the notes on any key fact? Examples of contradictions:
- Answer says "prescribed Metoprolol" but notes say "prescribed Lisinopril"
- Answer says "no procedure performed" but notes describe a surgery
- Answer says "glucose was normal" but notes show glucose was 253

PRINCIPLE 3 — CRUCIAL TO THE CONCLUSION:
If you found any issue in Principles 1 or 2, ask: would correcting this issue CHANGE THE ANSWER to the question? If the answer's conclusion would stay the same after correction, it is NOT a critical error. Only flag errors that break the reasoning from notes to conclusion.

Report your findings for each principle."""

# F2: Reasoning chain — conclusion backward
F2_CONCLUSION_BACKWARD = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Work BACKWARD from the answer's conclusion:

Step 1: What is the answer's main conclusion? State it in one sentence.

Step 2: What facts from the notes must be true for this conclusion to be correct? List them.

Step 3: Are those facts actually in the notes? For each:
- Find the relevant passage in the notes
- Does the answer state the fact correctly, or does it contradict the notes?

Step 4: Does the answer address what the question actually asks? Check the specific visit, time period, and clinical focus.

Step 5: Based on Steps 2-4, is the conclusion WRONG? Only report errors that make the conclusion wrong — not minor missing details."""

# F3: Devil's advocate with principle gates
F3_DEVILS_PRINCIPLE = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Try to find errors in this answer. But apply THREE strict gates before flagging:

GATE 1 — Is the answer about the right thing?
Parse the question: what visit, what aspect, what time period does it ask about?
Does the answer match? If the answer discusses the wrong visit or wrong aspect, flag immediately — this is the most critical error type.

GATE 2 — Do the facts match the notes?
Check each medication name, dosage, procedure, diagnosis, and lab value against the notes. Flag ONLY when the answer directly contradicts what the notes say. Do NOT flag if something is just missing.

GATE 3 — Does it change the conclusion?
For any error found in Gates 1-2, ask: if I corrected this error, would the answer to the question be DIFFERENT? If yes, report it. If the answer would remain essentially correct, do not flag it.

Report only errors that pass all three gates."""

# F4: Question decomposition + fact extraction + alignment
F4_DECOMPOSE_MATCH = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

STEP 1: DECOMPOSE THE QUESTION
- Visit: which hospital visit does the question ask about?
- Focus: what clinical aspect does the question ask about?
- Scope: is it asking about a specific event, a comparison, or a summary?
Write your decomposition.

STEP 2: EXTRACT KEY FACTS FROM THE ANSWER
List the 3-5 most important factual claims the answer makes to answer the question.

STEP 3: VERIFY EACH FACT
For each claim from Step 2:
- Find the relevant passage in the discharge notes
- Does the claim match? Or does it contradict the notes?
- Is this claim about the right visit (matching Step 1)?

STEP 4: VERDICT
Based on Steps 1-3:
- Does the answer address the right question? (check Step 1 vs answer)
- Are the key facts correct? (check Step 3)
- Would any errors change the answer's conclusion?

Only flag the answer as wrong if errors would change the conclusion."""

PROMPTS = {
    "F1_three_principles": F1_THREE_PRINCIPLES,
    "F2_conclusion_backward": F2_CONCLUSION_BACKWARD,
    "F3_devils_principle": F3_DEVILS_PRINCIPLE,
    "F4_decompose_match": F4_DECOMPOSE_MATCH,
}

# ============================================================
# QWEN32B EXTRACTION
# ============================================================

EXTRACT_PROMPT = """/nothink
Read this self-critique from a medical AI checking its own answer.

SELF-CRITIQUE:
{raw_output}

Extract as JSON. The verdict should be INCORRECT only if the self-critique found errors that would CHANGE THE ANSWER'S CONCLUSION.

{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the wrong claim or missing info as one sentence", "correct_statement": "what the notes say as one sentence", "explanation": "brief"}}"""

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
    try:
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
            return json.loads(m.group())
    except:
        pass
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
    parser.add_argument("--fold", type=int, default=2)
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
    print(f"Round 7: Question-anchored + conclusion-critical ({n_w} wrong + {n_c} correct, fold={args.fold})", flush=True)
    print("=" * 70, flush=True)

    try:
        with open(OUTPUT_DIR / "phase1_wrong_gpt4o.json") as f:
            gpt_errors = json.load(f)
        gpt_lookup = {(r["fold"], r["idx"]): r.get("PRIMARY_ERROR", "?") for r in gpt_errors}
    except:
        gpt_lookup = {}

    progress_file = OUTPUT_DIR / f"round7_fold{args.fold}_progress.json"

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
            prompt = build_chatml("You are a strict medical expert. Only flag errors that change the answer's conclusion.", msg)
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

            gpt_type = gpt_lookup.get((ti["fold"], ti["idx"]), "?") if ti["label"] == "wrong" else "correct"
            print(f"  [{len(results)}/{len(test_items)}] {ti['label']:>7} idx={ti['idx']} → {verdict} {error_type} (gpt={gpt_type})", flush=True)

            all_results[pkey] = results
            with open(progress_file, "w") as pf:
                json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c}, pf)

        all_results[pkey] = results
        w = sum(1 for r in results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
        c = sum(1 for r in results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
        pfail = sum(1 for r in results if not r["parse_ok"])
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        w_types = Counter(r["error_type"] for r in results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
        c_types = Counter(r["error_type"] for r in results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
        print(f"  wrong={w}/{n_w} correct={c}/{n_c} sel={sel:.1f}x pfail={pfail}", flush=True)
        print(f"  TP types: {dict(w_types)}", flush=True)
        if c_types:
            print(f"  FP types: {dict(c_types)}", flush=True)
        for r in results:
            if r["verdict"] == "INCORRECT":
                gpt_type = gpt_lookup.get((r["fold"], r["idx"]), "?") if r["label"] == "wrong" else "correct"
                print(f"    [{r['label']:>7}] idx={r['idx']} {r['error_type']:>20} gpt={gpt_type:>20} | {r['error_statement'][:80]}", flush=True)

    # Final save
    out_file = OUTPUT_DIR / f"detection_round7_fold{args.fold}.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c}, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"{'Prompt':<28} {'Wrong':>8} {'Correct':>8} {'Select':>8} {'CONTRA':>7} {'OMIS':>5} {'QMIS':>5}", flush=True)
    print("-" * 72, flush=True)
    for pkey in PROMPTS:
        r = all_results[pkey]
        w = sum(1 for x in r if x["label"]=="wrong" and x["verdict"]=="INCORRECT")
        c = sum(1 for x in r if x["label"]=="correct" and x["verdict"]=="INCORRECT")
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        wt = Counter(x["error_type"] for x in r if x["label"]=="wrong" and x["verdict"]=="INCORRECT")
        print(f"  {pkey:<28} {w}/{n_w:>3}    {c}/{n_c:>3}    {sel:>5.1f}x  {wt.get('CONTRADICTION',0):>5}  {wt.get('OMISSION',0):>4}  {wt.get('QUESTION_MISALIGNMENT',0):>4}", flush=True)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
