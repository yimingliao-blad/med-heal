#!/usr/bin/env python3
"""
Test correction methods × verdict methods on Round 8 S1 detected items.

Correction: COR-A (most critical), COR-B (all errors), COR-C (plain regen baseline)
Verdict: V1 (contradiction count), V2 (principle comparison), V3 (error-specific verify)

All self-correction by Qwen2.5. Qwen3-32B only for extraction.
GPT-4o only for ground truth eval.

Usage:
    python test_correction_verdict.py --port 8003
"""
import json, random, re, sys, os, time, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

# GPT-4o
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
# CORRECTION PROMPTS
# ============================================================

COR_A_CRITICAL = """Discharge summary:
{note}

Question: {question}

Your previous answer had this error:
ERROR TYPE: {error_type}
ERROR: {error_statement}
THE NOTES SAY: {correct_statement}

Re-answer the question, fixing this specific error. Base your answer on the discharge notes.
Answer in 1-3 direct sentences."""

COR_B_ALL = """Discharge summary:
{note}

Question: {question}

Your previous answer had these issues:
{all_errors}

Re-answer the question, fixing ALL the issues above. Base your answer on the discharge notes.
Answer in 1-3 direct sentences."""

COR_C_REGEN = """Discharge summary:
{note}

Question: {question}

Answer this question carefully based on the discharge notes. Be specific and accurate.
Answer in 1-3 direct sentences."""

# ============================================================
# VERDICT PROMPTS (all Qwen2.5 self-judge)
# ============================================================

V1_CONTRA_COUNT = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count how many factual claims in each answer CONTRADICT the discharge notes. Different wording for the same fact is NOT a contradiction.

Answer A contradictions: <number>
Answer B contradictions: <number>
Better answer: A or B"""

V2_PRINCIPLE = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Compare these two answers on three criteria:
1. Which better addresses what the question specifically asks? (correct visit, time period, clinical focus)
2. Which has fewer factual conflicts with the discharge notes?
3. Which better covers the critical information needed to answer the question?

Better answer: A or B"""

V3_ERROR_VERIFY = """Discharge summary:
{note}

Question: {question}

An error was detected in the original answer:
ERROR: {error_statement}

ORIGINAL ANSWER:
{original}

CORRECTED ANSWER:
{corrected}

Does the corrected answer fix the detected error without introducing new errors?
If yes, the corrected answer is better. If no, keep the original.

Better answer: ORIGINAL or CORRECTED"""

# ============================================================
# HELPERS
# ============================================================

def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def vllm_gen(prompt, max_tokens=1024, temperature=1.0):
    model = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{PORT}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=120)
    return resp.json()["choices"][0]["text"].strip()

QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

def q32_extract_verdict(raw):
    """Extract which answer was picked: A, B, ORIGINAL, or CORRECTED."""
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Extract the verdict. Output ONLY JSON."},
                {"role": "user", "content": f'/nothink\nFrom this comparison output, which answer was chosen as better?\n\nTEXT:\n{raw[:1500]}\n\nRespond: {{"pick": "A" or "B" or "ORIGINAL" or "CORRECTED"}}'},
            ],
            "max_tokens": 100, "temperature": 0.0,
        }, timeout=60)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return str(obj.get("pick", "UNCLEAR")).upper()
    except: pass
    return "UNCLEAR"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    global PORT; PORT = args.port

    notes = load_notes()

    # Load S1 detected items from Round 8
    with open(OUTPUT_DIR / "detection_round8_fold1.json") as f:
        r8 = json.load(f)
    s1 = r8["results"]["S1_multirun"]
    detected = [r for r in s1 if r["verdict"] == "INCORRECT"]

    # Load original data
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists(): df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    print(f"Correction × Verdict test: {len(detected)} detected items ({sum(1 for d in detected if d['label']=='wrong')} TP, {sum(1 for d in detected if d['label']=='correct')} FP)", flush=True)
    print("=" * 70, flush=True)

    progress_file = OUTPUT_DIR / "correction_verdict_progress.json"
    results = []

    for det in detected:
        row = all_df[(all_df["fold"]==det["fold"]) & (all_df["idx"]==det["idx"])]
        if len(row) == 0: continue
        row = row.iloc[0]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        gt = row["ground_truth"]
        original_answer = str(row.get("openended_answer", row.get("model_answer", "")))

        # Build all_errors string for COR-B
        sub = det.get("sub_results", {})
        error_lines = []
        if det.get("error_statement"):
            error_lines.append(f"- {det['error_type']}: {det['error_statement']}")
        all_errors = "\n".join(error_lines) if error_lines else f"- {det['error_type']}: {det['error_statement']}"

        # === CORRECTION ===
        corrections = {}
        for cor_key in ["COR_A", "COR_B", "COR_C"]:
            if cor_key == "COR_A":
                msg = COR_A_CRITICAL.format(note=note, question=row["question"],
                    error_type=det["error_type"], error_statement=det["error_statement"][:200],
                    correct_statement=det["correct_statement"][:200])
            elif cor_key == "COR_B":
                msg = COR_B_ALL.format(note=note, question=row["question"], all_errors=all_errors)
            else:
                msg = COR_C_REGEN.format(note=note, question=row["question"])

            prompt = build_chatml("You are a medical expert answering questions about discharge summaries.", msg)
            corrected = vllm_gen(prompt, max_tokens=512, temperature=1.0)
            corrections[cor_key] = corrected

        # === VERDICT (for each correction) ===
        verdicts = {}
        for cor_key, corrected in corrections.items():
            # Randomize position for V1/V2
            rng = random.Random(42 + hash(str(det["idx"])) + hash(cor_key))
            orig_is_a = rng.random() > 0.5

            if orig_is_a:
                ans_a, ans_b = original_answer[:500], corrected[:500]
            else:
                ans_a, ans_b = corrected[:500], original_answer[:500]

            for v_key in ["V1", "V2", "V3"]:
                if v_key == "V1":
                    msg = V1_CONTRA_COUNT.format(note=note, question=row["question"],
                        answer_a=ans_a, answer_b=ans_b)
                elif v_key == "V2":
                    msg = V2_PRINCIPLE.format(note=note, question=row["question"],
                        answer_a=ans_a, answer_b=ans_b)
                else:
                    msg = V3_ERROR_VERIFY.format(note=note, question=row["question"],
                        error_statement=det["error_statement"][:200],
                        original=original_answer[:500], corrected=corrected[:500])

                prompt = build_chatml("You are a medical expert comparing clinical answers.", msg)
                raw = vllm_gen(prompt, max_tokens=512, temperature=0.0)
                pick = q32_extract_verdict(raw)

                # Map pick to decision
                if v_key in ["V1", "V2"]:
                    if orig_is_a:
                        accept = pick in ("B",)
                    else:
                        accept = pick in ("A",)
                else:
                    accept = pick in ("CORRECTED",)

                verdicts[f"{cor_key}_{v_key}"] = accept

        # === GPT-4o EVAL ===
        eval_orig = int(row["binary_correct"])
        eval_corrections = {}
        for cor_key, corrected in corrections.items():
            eval_corrections[cor_key] = gpt4o_eval(note, row["question"], gt, corrected)

        entry = {
            "idx": det["idx"], "fold": det["fold"], "label": det["label"],
            "error_type": det["error_type"],
            "eval_orig": eval_orig,
            "eval_corrections": eval_corrections,
            "verdicts": verdicts,
            "corrections": {k: v[:200] for k, v in corrections.items()},
        }
        results.append(entry)

        # Print summary
        parts = []
        for cor_key in ["COR_A", "COR_B", "COR_C"]:
            ev = eval_corrections[cor_key]
            accepted_by = [vk.split("_")[1] for vk in verdicts if vk.startswith(cor_key) and verdicts[vk]]
            parts.append(f"{cor_key}={'FIX' if ev==1 else 'FAIL'} acc=[{','.join(accepted_by) or 'none'}]")
        print(f"  [{det['label']:>7}] idx={det['idx']} {det['error_type']:>15} | {' | '.join(parts)} ${spending['cost']:.2f}", flush=True)

        with open(progress_file, "w") as f:
            json.dump(results, f)

    # === FINAL ANALYSIS ===
    print(f"\n{'='*70}", flush=True)
    print("CORRECTION × VERDICT RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    for cor_key in ["COR_A", "COR_B", "COR_C"]:
        print(f"\n--- {cor_key} ---", flush=True)
        tp_items = [r for r in results if r["label"] == "wrong"]
        fp_items = [r for r in results if r["label"] == "correct"]

        tp_fix = sum(1 for r in tp_items if r["eval_corrections"][cor_key] == 1)
        fp_brk = sum(1 for r in fp_items if r["eval_corrections"][cor_key] == 0)
        print(f"  Raw: TP fix={tp_fix}/{len(tp_items)} FP brk={fp_brk}/{len(fp_items)}", flush=True)

        for v_key in ["V1", "V2", "V3"]:
            combo = f"{cor_key}_{v_key}"
            # After verdict: accepted corrections
            tp_accepted_fix = sum(1 for r in tp_items if r["verdicts"].get(combo) and r["eval_corrections"][cor_key] == 1)
            tp_accepted_fail = sum(1 for r in tp_items if r["verdicts"].get(combo) and r["eval_corrections"][cor_key] == 0)
            tp_rejected_fix = sum(1 for r in tp_items if not r["verdicts"].get(combo) and r["eval_corrections"][cor_key] == 1)
            fp_accepted_brk = sum(1 for r in fp_items if r["verdicts"].get(combo) and r["eval_corrections"][cor_key] == 0)
            fp_accepted_safe = sum(1 for r in fp_items if r["verdicts"].get(combo) and r["eval_corrections"][cor_key] == 1)
            fp_rejected = sum(1 for r in fp_items if not r["verdicts"].get(combo))

            net = tp_accepted_fix - fp_accepted_brk
            print(f"  +{v_key}: accept_fix={tp_accepted_fix} accept_fail={tp_accepted_fail} reject_fix={tp_rejected_fix} | FP_brk={fp_accepted_brk} FP_safe={fp_accepted_safe} FP_reject={fp_rejected} | net={net:+d}", flush=True)

    print(f"\nGPT-4o: {spending['calls']} calls, ${spending['cost']:.3f}", flush=True)

    with open(OUTPUT_DIR / "correction_verdict_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved.", flush=True)


if __name__ == "__main__":
    main()
