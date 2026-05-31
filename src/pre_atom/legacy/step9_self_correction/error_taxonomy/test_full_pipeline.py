#!/usr/bin/env python3
"""
Full pipeline test: F3 detect → multiple correction methods → verdict comparison.

Tests on fold 1 (not fold 0). Includes FP items in correction to measure break rate.
Tests multiple temperatures for correction.
Saves all outputs for analysis.

Usage:
    python test_full_pipeline.py --port 8003
"""
import json, random, re, sys, os, time, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

# API key
key = None
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
        key = line.split("=", 1)[1]; break
from openai import OpenAI
gpt_client = OpenAI(api_key=key)
spending = {"calls": 0, "cost": 0.0}

PORT = 8003

# ============================================================
# F3 DETECTION PROMPT (best from Round 7)
# ============================================================

F3_DETECT = """Discharge summary:
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

# ============================================================
# CORRECTION PROMPTS — properly guided
# ============================================================

# COR_GUIDED: Use detected error type to guide correction
COR_GUIDED_CONTRA = """Discharge summary:
{note}

Question: {question}

Your previous answer contained a factual error:
YOUR ANSWER SAID: {error_statement}
BUT THE NOTES SAY: {correct_statement}

Re-read the relevant section of the discharge notes. Then answer the question correctly based on what the notes actually say.
Answer in 1-3 direct sentences."""

COR_GUIDED_OMISSION = """Discharge summary:
{note}

Question: {question}

Your previous answer was missing critical information:
MISSING: {error_statement}
THE NOTES SAY: {correct_statement}

Re-answer the question, making sure to include this information.
Answer in 1-3 direct sentences."""

COR_GUIDED_QMIS = """Discharge summary:
{note}

Question: {question}

Your previous answer addressed the wrong aspect of the question:
ISSUE: {error_statement}

Re-read the question carefully. Pay attention to which visit, time period, and clinical focus it asks about.
Answer in 1-3 direct sentences."""

# COR_POOL: Guided + BM error pool examples
COR_POOL = """Discharge summary:
{note}

Question: {question}

Your previous answer had this error:
{error_statement}
THE NOTES SAY: {correct_statement}

Here are examples of similar errors and their corrections:
{pool_examples}

Fix the error and re-answer the question correctly.
Answer in 1-3 direct sentences."""

# COR_REGEN: Plain regeneration (baseline)
COR_REGEN = """Discharge summary:
{note}

Question: {question}

Answer this question carefully based on the discharge notes. Be specific and accurate.
Answer in 1-3 direct sentences."""

# ============================================================
# QWEN32B EXTRACTION
# ============================================================

EXTRACT_PROMPT = """/nothink
Read this self-critique from a medical AI checking its own answer.

SELF-CRITIQUE:
{raw_output}

Extract as JSON. Only mark INCORRECT if the self-critique found errors that would CHANGE THE ANSWER.

{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the wrong claim as one sentence", "correct_statement": "what notes say as one sentence", "explanation": "brief"}}"""

# ============================================================
# HELPERS
# ============================================================

def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def vllm_gen(prompt, max_tokens=2048, temperature=0.0):
    model = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{PORT}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=180)
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
        if m: return json.loads(m.group())
    except: pass
    return None

def gpt4o_eval(note, question, gt, answer):
    time.sleep(1.5)
    try:
        r = gpt_client.chat.completions.create(
            model="gpt-4o", messages=[
                {"role": "system", "content": "You are a medical expert evaluating an AI model's answer."},
                {"role": "user", "content": (
                    f"DISCHARGE SUMMARY:\n{note}\n\nQUESTION:\n{question}\n\n"
                    f"CORRECT ANSWER (Ground Truth):\n{gt}\n\nMODEL'S ANSWER:\n{answer}\n\n"
                    f"Respond with ONLY a single digit: 1 = Correct, 0 = Incorrect"
                )},
            ], max_tokens=10, temperature=0.1,
        )
        text = r.choices[0].message.content.strip()
        cost = r.usage.prompt_tokens * 2.5 / 1e6 + r.usage.completion_tokens * 10.0 / 1e6
        spending["calls"] += 1; spending["cost"] += cost
        return 1 if text.startswith("1") else 0 if text.startswith("0") else -1
    except Exception as e:
        print(f"  GPT-4o error: {e}", flush=True); time.sleep(5); return -1

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

def load_pool(fold_id):
    f = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool" / f"fold_{fold_id}_atoms.json"
    if f.exists():
        with open(f) as fh: return json.load(fh)
    return []

def get_pool_examples(error_type, fold_id, k=2):
    pool = load_pool(fold_id)
    if not pool: return ""
    type_map = {"OMISSION": "omission", "CONTRADICTION": "factual_error"}
    target = type_map.get(error_type, "factual_error")
    matching = [a for a in pool if a.get("main_error_type") == target and a.get("gt_atom_raw")]
    if not matching: matching = [a for a in pool if a.get("gt_atom_raw")]
    random.shuffle(matching)
    return "\n".join(f'  Wrong: "{a["text_raw"][:120]}"\n  Correct: "{a["gt_atom_raw"][:120]}"' for a in matching[:k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n-wrong", type=int, default=15)
    parser.add_argument("--n-correct", type=int, default=15)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    global PORT; PORT = args.port
    notes = load_notes()

    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists(): df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    fold_df = all_df[all_df["fold"] == args.fold]

    random.seed(args.seed)
    wrong = fold_df[fold_df["binary_correct"]==0].sample(n=min(args.n_wrong, (fold_df["binary_correct"]==0).sum()), random_state=args.seed)
    correct = fold_df[fold_df["binary_correct"]==1].sample(n=min(args.n_correct, (fold_df["binary_correct"]==1).sum()), random_state=args.seed)

    test_items = []
    for _, row in wrong.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": args.fold, "label": "wrong", "row": row})
    for _, row in correct.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": args.fold, "label": "correct", "row": row})

    n_w = sum(1 for t in test_items if t["label"] == "wrong")
    n_c = sum(1 for t in test_items if t["label"] == "correct")
    print(f"Full Pipeline: fold={args.fold}, {n_w} wrong + {n_c} correct, seed={args.seed}", flush=True)
    print("=" * 70, flush=True)

    progress_file = OUTPUT_DIR / f"full_pipeline_fold{args.fold}_progress.json"

    # ============================
    # STAGE 1: F3 DETECTION
    # ============================
    print("\n=== STAGE 1: F3 DETECTION ===", flush=True)
    detection_results = []
    for ti in test_items:
        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        msg = F3_DETECT.format(note=note, question=row["question"], answer=answer[:800])
        prompt = build_chatml("You are a strict medical expert. Only flag errors that change the answer's conclusion.", msg)
        raw = vllm_gen(prompt, max_tokens=2048, temperature=0.0)

        obj = q32_extract(raw)
        verdict = str(obj.get("verdict", "UNCLEAR")).upper() if obj else "PARSE_FAIL"
        error_type = str(obj.get("error_type", "NONE")).upper() if obj else "NONE"
        error_stmt = str(obj.get("error_statement", ""))[:250] if obj else ""
        correct_stmt = str(obj.get("correct_statement", ""))[:250] if obj else ""

        detection_results.append({
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            "verdict": verdict, "error_type": error_type,
            "error_statement": error_stmt, "correct_statement": correct_stmt,
        })
        print(f"  {ti['label']:>7} idx={ti['idx']} → {verdict} {error_type}", flush=True)

    w_det = sum(1 for r in detection_results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
    c_det = sum(1 for r in detection_results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
    print(f"\nDetection: wrong={w_det}/{n_w} correct(FP)={c_det}/{n_c}", flush=True)

    # ============================
    # STAGE 2: CORRECTION (on ALL detected, including FP)
    # ============================
    detected = [r for r in detection_results if r["verdict"] == "INCORRECT"]
    print(f"\n=== STAGE 2: CORRECTION ({len(detected)} detected items) ===", flush=True)

    correction_methods = ["guided", "pool", "regen_t0", "regen_t1"]
    correction_results = {m: [] for m in correction_methods}

    for det in detected:
        row_match = all_df[(all_df["fold"]==det["fold"]) & (all_df["idx"]==det["idx"])]
        if len(row_match) == 0: continue
        row = row_match.iloc[0]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        gt = row["ground_truth"]
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        pool_examples = get_pool_examples(det["error_type"], det["fold"])

        for method in correction_methods:
            if method == "guided":
                if det["error_type"] == "CONTRADICTION":
                    msg = COR_GUIDED_CONTRA.format(note=note, question=row["question"],
                        error_statement=det["error_statement"], correct_statement=det["correct_statement"])
                elif det["error_type"] == "QUESTION_MISALIGNMENT":
                    msg = COR_GUIDED_QMIS.format(note=note, question=row["question"],
                        error_statement=det["error_statement"])
                else:
                    msg = COR_GUIDED_OMISSION.format(note=note, question=row["question"],
                        error_statement=det["error_statement"], correct_statement=det["correct_statement"])
                temp = 1.0
            elif method == "pool":
                msg = COR_POOL.format(note=note, question=row["question"],
                    error_statement=det["error_statement"], correct_statement=det["correct_statement"],
                    pool_examples=pool_examples)
                temp = 1.0
            elif method == "regen_t0":
                msg = COR_REGEN.format(note=note, question=row["question"])
                temp = 0.0
            elif method == "regen_t1":
                msg = COR_REGEN.format(note=note, question=row["question"])
                temp = 1.0

            prompt = build_chatml("You are a medical expert answering questions about discharge summaries.", msg)
            corrected = vllm_gen(prompt, max_tokens=512, temperature=temp)
            ev = gpt4o_eval(note, row["question"], gt, corrected)

            correction_results[method].append({
                "idx": det["idx"], "fold": det["fold"], "label": det["label"],
                "error_type": det["error_type"],
                "eval_orig": int(row["binary_correct"]),
                "eval_corrected": ev,
                "corrected_answer": corrected[:200],
            })

        status_parts = []
        for m in correction_methods:
            ev = correction_results[m][-1]["eval_corrected"]
            status_parts.append(f"{m[:5]}={'FIX' if ev==1 else 'FAIL'}")
        print(f"  [{det['label']:>7}] idx={det['idx']} {det['error_type']:>15} | {' '.join(status_parts)} ${spending['cost']:.2f}", flush=True)

        # Save progress
        with open(progress_file, "w") as f:
            json.dump({"detection": detection_results, "correction": correction_results, "spending": spending}, f)

    # ============================
    # SUMMARY
    # ============================
    print(f"\n{'='*70}", flush=True)
    print(f"FULL PIPELINE RESULTS — fold={args.fold}", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\nDetection (F3): wrong={w_det}/{n_w} FP={c_det}/{n_c}", flush=True)

    print(f"\n{'Method':<15} {'TP FIX':>8} {'TP FAIL':>8} {'FP BRK':>8} {'FP SAFE':>8} {'Net':>6}", flush=True)
    print("-" * 55, flush=True)
    for m in correction_methods:
        cr = correction_results[m]
        tp_fix = sum(1 for r in cr if r["label"]=="wrong" and r["eval_corrected"]==1)
        tp_fail = sum(1 for r in cr if r["label"]=="wrong" and r["eval_corrected"]==0)
        fp_brk = sum(1 for r in cr if r["label"]=="correct" and r["eval_corrected"]==0)
        fp_safe = sum(1 for r in cr if r["label"]=="correct" and r["eval_corrected"]==1)
        net = tp_fix - fp_brk
        print(f"  {m:<15} {tp_fix:>6}   {tp_fail:>6}   {fp_brk:>6}   {fp_safe:>6}  {net:>+5}", flush=True)

    # Best-of-4
    if detected:
        any_fix = 0; any_brk = 0
        for i in range(len(detected)):
            fixes = [correction_results[m][i]["eval_corrected"]==1 for m in correction_methods]
            if detected[i]["label"] == "wrong" and any(fixes): any_fix += 1
            if detected[i]["label"] == "correct" and not all(correction_results[m][i]["eval_corrected"]==1 for m in correction_methods):
                any_brk += 1  # at least one method broke it
        print(f"  {'best-of-4':<15} {any_fix:>6}   {'':>6}   {'':>6}   {'':>6}", flush=True)

    print(f"\nGPT-4o: {spending['calls']} calls, ${spending['cost']:.3f}", flush=True)

    # Final save
    with open(progress_file, "w") as f:
        json.dump({"detection": detection_results, "correction": correction_results, "spending": spending}, f, indent=2)
    print(f"Saved to {progress_file}", flush=True)


if __name__ == "__main__":
    main()
