#!/usr/bin/env python3
"""
Full pipeline test: detect → correct → verdict → eval.

One script, 4 stages, each saves progress. Resume-safe.

Stage 1: S1 multi-run detection (3 sub-prompts per item), save ALL error details
Stage 2: Two correction pipelines:
  P1: correct with most critical error only
  P2: correct with ALL detected errors
Stage 3: Three verdict methods per correction:
  V1: contradiction count comparison
  V2: principle-based comparison
  V3: error-specific verification
Stage 4: GPT-4o eval all corrections

Usage:
    python run_pipeline_test.py --port 8003 --fold 1 --seed 99
"""
import json, random, re, os, time, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

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
# DETECTION PROMPTS (S1 multi-run)
# ============================================================

DET_CONTRA = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CONTRADICTION: Does the answer state any fact that DIRECTLY CONFLICTS with the discharge notes?

For each key claim in the answer (medication names, dosages, procedures, diagnoses, lab values, dates):
1. Find the matching information in the notes
2. Does the answer say something DIFFERENT from the notes?

Only flag when the answer and notes provide OPPOSING information.
If you find a contradiction, explain: what does the answer say vs what do the notes say?"""

DET_QMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR QUESTION MISALIGNMENT: Does the answer address the WRONG thing?

Parse the question:
- Which hospital visit? (first, second, third, most recent?)
- What clinical aspect? (medications, procedures, diagnoses, complications?)
- What time period? (during admission, at discharge, between visits?)

Does the answer match? If misaligned, explain what the question asks vs what the answer discusses."""

DET_OMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CRITICAL OMISSION: Is there information in the notes ESSENTIAL to answer the question but COMPLETELY ABSENT from the answer?

Only flag omissions that would CHANGE the answer's conclusion. Do NOT flag minor details.
If you find a critical omission, explain what is missing and why it would change the answer."""

# ============================================================
# CORRECTION PROMPTS
# ============================================================

COR_P1 = """Discharge summary:
{note}

Question: {question}

Your previous answer had this error:
ERROR TYPE: {error_type}
ERROR: {error_statement}
THE NOTES SAY: {correct_statement}

Re-answer the question, fixing this specific error. Base your answer on the discharge notes.
Answer in 1-3 direct sentences."""

COR_P2 = """Discharge summary:
{note}

Question: {question}

Your previous answer had these errors:
{all_errors}

Re-answer the question, fixing ALL the errors above. Base your answer on the discharge notes.
Answer in 1-3 direct sentences."""

COR_REGEN = """Discharge summary:
{note}

Question: {question}

Answer this question carefully based on the discharge notes. Be specific and accurate.
Answer in 1-3 direct sentences."""

# ============================================================
# VERDICT PROMPTS
# ============================================================

V1_CONTRA_COUNT = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count how many factual claims in each answer CONTRADICT the discharge notes.

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

Compare on three criteria:
1. Which better addresses what the question asks?
2. Which has fewer factual conflicts with the notes?
3. Which better covers critical information?

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

Does the corrected answer fix the error without introducing new errors?
Better answer: ORIGINAL or CORRECTED"""

# ============================================================
# HELPERS
# ============================================================

EXTRACT_PROMPT = """/nothink
Read this self-critique. Extract as JSON.
Only INCORRECT if critical errors found.

TEXT:
{raw}

{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the error as one sentence", "correct_statement": "what notes say"}}"""

EXTRACT_VERDICT = """/nothink
Which answer was picked as better?

TEXT:
{raw}

{{"pick": "A" or "B" or "ORIGINAL" or "CORRECTED"}}"""

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

def q32_extract(raw, extract_template):
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Extract info. Output ONLY JSON."},
                {"role": "user", "content": extract_template.format(raw=raw[:2000])},
            ],
            "max_tokens": 300, "temperature": 0.0,
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

    save_file = OUTPUT_DIR / f"pipeline_fold{args.fold}_seed{args.seed}.json"
    print(f"Pipeline test: fold={args.fold}, {n_w} wrong + {n_c} correct", flush=True)
    print("=" * 70, flush=True)

    # Resume
    all_items = []
    done_keys = set()
    if save_file.exists():
        all_items = json.load(open(save_file))
        done_keys = {(r["fold"], r["idx"]) for r in all_items}
        print(f"Resuming: {len(done_keys)} done", flush=True)

    for ti in test_items:
        if (ti["fold"], ti["idx"]) in done_keys:
            continue

        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))
        gt = row["ground_truth"]
        eval_orig = int(row["binary_correct"])
        system_det = "You are a strict medical expert checking clinical answers against discharge notes."
        system_cor = "You are a medical expert answering questions about discharge summaries."
        system_ver = "You are a medical expert comparing clinical answers."

        # ====== STAGE 1: DETECTION (3 sub-prompts) ======
        det_results = {}
        for det_key, det_prompt in [("contra", DET_CONTRA), ("qmis", DET_QMIS), ("omis", DET_OMIS)]:
            msg = det_prompt.format(note=note, question=row["question"], answer=answer[:800])
            raw = vllm_gen(build_chatml(system_det, msg))
            obj = q32_extract(raw, EXTRACT_PROMPT) or {}
            det_results[det_key] = {
                "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
                "error_type": str(obj.get("error_type", "NONE")).upper(),
                "error_statement": str(obj.get("error_statement", ""))[:250],
                "correct_statement": str(obj.get("correct_statement", ""))[:250],
                "raw": raw,
            }

        # Aggregate: any detected?
        detected_types = [k for k in det_results if det_results[k]["verdict"] == "INCORRECT"]
        any_detected = len(detected_types) > 0

        # Most critical: qmis > contra > omis
        priority = ["qmis", "contra", "omis"]
        most_critical = None
        for p in priority:
            if det_results[p]["verdict"] == "INCORRECT":
                most_critical = det_results[p]
                break

        # All errors string
        all_error_lines = []
        for k in detected_types:
            d = det_results[k]
            all_error_lines.append(f"- {d['error_type']}: {d['error_statement']}")
            if d["correct_statement"]:
                all_error_lines.append(f"  NOTES SAY: {d['correct_statement']}")
        all_errors_str = "\n".join(all_error_lines) if all_error_lines else ""

        print(f"\n  [{ti['label']:>7}] idx={ti['idx']} detected=[{','.join(detected_types) or 'none'}]", flush=True)

        if not any_detected:
            # No detection → keep original, no correction needed
            entry = {
                "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                "eval_orig": eval_orig, "detected": False,
                "det_types": [], "det_details": {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in det_results.items()},
            }
            all_items.append(entry)
            with open(save_file, "w") as f: json.dump(all_items, f)
            print(f"    → no detection, skip correction", flush=True)
            continue

        # ====== STAGE 2: CORRECTION ======
        corrections = {}

        # P1: most critical error only
        msg = COR_P1.format(note=note, question=row["question"],
            error_type=most_critical["error_type"],
            error_statement=most_critical["error_statement"][:200],
            correct_statement=most_critical["correct_statement"][:200])
        corrections["P1_critical"] = vllm_gen(build_chatml(system_cor, msg), max_tokens=512, temperature=1.0)

        # P2: all detected errors
        msg = COR_P2.format(note=note, question=row["question"], all_errors=all_errors_str)
        corrections["P2_all"] = vllm_gen(build_chatml(system_cor, msg), max_tokens=512, temperature=1.0)

        # P3: plain regen (baseline)
        msg = COR_REGEN.format(note=note, question=row["question"])
        corrections["P3_regen"] = vllm_gen(build_chatml(system_cor, msg), max_tokens=512, temperature=1.0)

        # ====== STAGE 3: VERDICT (per correction) ======
        verdicts = {}
        for cor_key, corrected in corrections.items():
            rng = random.Random(42 + hash(str(ti["idx"])) + hash(cor_key))
            orig_is_a = rng.random() > 0.5
            ans_a = answer[:500] if orig_is_a else corrected[:500]
            ans_b = corrected[:500] if orig_is_a else answer[:500]

            for v_key, v_prompt in [("V1", V1_CONTRA_COUNT), ("V2", V2_PRINCIPLE)]:
                msg = v_prompt.format(note=note, question=row["question"], answer_a=ans_a, answer_b=ans_b)
                raw = vllm_gen(build_chatml(system_ver, msg), max_tokens=512, temperature=0.0)
                obj = q32_extract(raw, EXTRACT_VERDICT) or {}
                pick = str(obj.get("pick", "UNCLEAR")).upper()
                accept = (pick == "B") if orig_is_a else (pick == "A")
                verdicts[f"{cor_key}_{v_key}"] = accept

            # V3: error-specific
            msg = V3_ERROR_VERIFY.format(note=note, question=row["question"],
                error_statement=most_critical["error_statement"][:200],
                original=answer[:500], corrected=corrected[:500])
            raw = vllm_gen(build_chatml(system_ver, msg), max_tokens=512, temperature=0.0)
            obj = q32_extract(raw, EXTRACT_VERDICT) or {}
            pick = str(obj.get("pick", "UNCLEAR")).upper()
            verdicts[f"{cor_key}_V3"] = pick in ("CORRECTED", "B")

        # ====== STAGE 4: GPT-4o EVAL ======
        eval_corrections = {}
        for cor_key, corrected in corrections.items():
            eval_corrections[cor_key] = gpt4o_eval(note, row["question"], gt, corrected)

        entry = {
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            "eval_orig": eval_orig, "detected": True,
            "det_types": detected_types,
            "det_details": {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in det_results.items()},
            "corrections": {k: v[:200] for k, v in corrections.items()},
            "eval_corrections": eval_corrections,
            "verdicts": verdicts,
        }
        all_items.append(entry)
        with open(save_file, "w") as f: json.dump(all_items, f)

        # Print summary
        parts = []
        for ck in ["P1_critical", "P2_all", "P3_regen"]:
            ev = eval_corrections[ck]
            acc = [vk.split("_")[-1] for vk in verdicts if vk.startswith(ck) and verdicts[vk]]
            parts.append(f"{ck[:2]}={'FIX' if ev==1 else 'FAIL'}[{','.join(acc) or '-'}]")
        print(f"    {' | '.join(parts)} ${spending['cost']:.2f}", flush=True)

    # ====== FINAL SUMMARY ======
    detected_items = [r for r in all_items if r.get("detected")]
    tp = [r for r in detected_items if r["label"] == "wrong"]
    fp = [r for r in detected_items if r["label"] == "correct"]

    print(f"\n{'='*70}", flush=True)
    print(f"PIPELINE RESULTS — fold={args.fold}", flush=True)
    print(f"Detection: {len(tp)} TP + {len(fp)} FP out of {n_w}+{n_c}", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Correction+Verdict':<25} {'TP fix':>7} {'TP kept':>8} {'FP brk':>7} {'FP safe':>8} {'Net':>5}", flush=True)
    print("-" * 62, flush=True)

    for ck in ["P1_critical", "P2_all", "P3_regen"]:
        # Raw (no verdict)
        tp_fix = sum(1 for r in tp if r["eval_corrections"].get(ck) == 1)
        fp_brk = sum(1 for r in fp if r["eval_corrections"].get(ck) == 0)
        print(f"  {ck+' (raw)':<25} {tp_fix:>5}   {'':>6}   {fp_brk:>5}   {'':>6}  {tp_fix-fp_brk:>+4}", flush=True)

        # With each verdict
        for vk in ["V1", "V2", "V3"]:
            combo = f"{ck}_{vk}"
            tp_acc_fix = sum(1 for r in tp if r["verdicts"].get(combo) and r["eval_corrections"][ck]==1)
            tp_acc_fail = sum(1 for r in tp if r["verdicts"].get(combo) and r["eval_corrections"][ck]==0)
            tp_rej = sum(1 for r in tp if not r["verdicts"].get(combo))
            fp_acc_brk = sum(1 for r in fp if r["verdicts"].get(combo) and r["eval_corrections"][ck]==0)
            fp_acc_safe = sum(1 for r in fp if r["verdicts"].get(combo) and r["eval_corrections"][ck]==1)
            fp_rej = sum(1 for r in fp if not r["verdicts"].get(combo))
            net = tp_acc_fix - fp_acc_brk
            print(f"  {ck[:2]+'+'+vk:<25} {tp_acc_fix:>5}   {tp_rej:>6}r  {fp_acc_brk:>5}   {fp_rej:>6}r {net:>+4}", flush=True)

    print(f"\nGPT-4o: {spending['calls']} calls, ${spending['cost']:.3f}", flush=True)

    # Final save with indent
    with open(save_file, "w") as f:
        json.dump(all_items, f, indent=2)
    print(f"Saved to {save_file}", flush=True)


if __name__ == "__main__":
    main()
