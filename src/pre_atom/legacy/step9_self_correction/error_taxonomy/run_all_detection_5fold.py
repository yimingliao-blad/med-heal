#!/usr/bin/env python3
"""
Run ALL detection methods on 10 TP + 10 FP × 5 folds = 100 items.
Save all results for offline analysis.
Resume-safe.

Detection methods:
  S1_contra: contradiction sub-prompt only
  S1_qmis: question misalignment sub-prompt only
  S1_omis: omission sub-prompt only
  S1_all: aggregate of all 3 (any detected → detected)
  S1_no_omis: contra + qmis only
  S3_balanced: single balanced prompt (rules+fewshot+severity)
  D1_cot_fs: CoT + few-shot (original best single prompt)

Usage:
    python run_all_detection_5fold.py --port 8003
"""
import json, random, re, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"
PORT = 8003

# ============================================================
# DETECTION PROMPTS
# ============================================================

DET_CONTRA = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CONTRADICTION: Does the answer state any fact that DIRECTLY CONFLICTS with the discharge notes?

For each key claim (medication names, dosages, procedures, diagnoses, lab values, dates):
1. Find the matching information in the notes
2. Does the answer say something DIFFERENT?

Only flag OPPOSING information. If you find a contradiction, explain what the answer says vs what the notes say."""

DET_QMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR QUESTION MISALIGNMENT: Does the answer address the WRONG thing?

Parse the question: which visit, what aspect, what time period?
Does the answer match? If misaligned, explain what the question asks vs what the answer discusses."""

DET_OMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CRITICAL OMISSION: Is there information in the notes ESSENTIAL to answer the question but COMPLETELY ABSENT?

Only flag omissions that would CHANGE the conclusion. Do NOT flag minor details."""

S3_BALANCED = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

You must check THREE things. For each, state whether you found an issue and how SEVERE it is (CRITICAL = changes conclusion, MINOR = doesn't change conclusion).

CHECK 1 — QUESTION ALIGNMENT:
Parse the question: which visit, which clinical aspect, which time period?
Does the answer address the right thing?
Example error: Question asks about "second admission" but answer describes the first admission.
Finding: OK / ISSUE (severity: CRITICAL or MINOR)

CHECK 2 — FACTUAL ACCURACY:
For each key claim (medications, dosages, procedures, diagnoses), verify against the notes.
Does the answer contradict the notes on any key fact?
Example error: Answer says "prescribed Haloperidol" but notes say "prescribed Risperdal."
Finding: OK / ISSUE (severity: CRITICAL or MINOR)

CHECK 3 — CRITICAL COMPLETENESS:
Is there essential information from the notes that the answer completely ignores?
Only flag if the missing info would change the answer's conclusion.
Example error: Answer mentions "hives" but omits "anaphylaxis" — missing the severe reaction changes the answer.
Finding: OK / ISSUE (severity: CRITICAL or MINOR)

Final: Report only CRITICAL issues."""

D1_COT_FS = """Discharge summary:
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

PROMPTS = {
    "contra": DET_CONTRA,
    "qmis": DET_QMIS,
    "omis": DET_OMIS,
    "S3": S3_BALANCED,
    "D1": D1_COT_FS,
}

EXTRACT_DET = """/nothink
Read this self-critique. Extract as JSON. Only INCORRECT if critical errors found.

TEXT:
{raw}

{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the error as one sentence", "correct_statement": "what notes say"}}"""

# ============================================================
# HELPERS
# ============================================================

def build_chatml(s, u):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"

def vllm_gen(prompt):
    model = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{PORT}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 2048,
              "temperature": 0.0, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=180)
    return resp.json()["choices"][0]["text"].strip()

def q32(raw):
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [{"role": "system", "content": "Extract info. JSON only."},
                         {"role": "user", "content": EXTRACT_DET.format(raw=raw[:2000])}],
            "max_tokens": 300, "temperature": 0.0,
        }, timeout=90)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m: return json.loads(m.group())
    except: pass
    return None

def load_notes():
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    lookup = {}
    for _, r in notes_df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1,2,3]:
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
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists(): df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # 10 TP + 10 FP per fold
    test_items = []
    for fold in range(5):
        fold_df = all_df[all_df["fold"] == fold]
        random.seed(fold * 100 + 42)
        wrong = fold_df[fold_df["binary_correct"]==0].sample(n=min(10, (fold_df["binary_correct"]==0).sum()), random_state=fold*100+42)
        correct = fold_df[fold_df["binary_correct"]==1].sample(n=min(10, (fold_df["binary_correct"]==1).sum()), random_state=fold*100+42)
        for _, row in wrong.iterrows():
            test_items.append({"idx": int(row["idx"]), "fold": fold, "label": "wrong", "row": row})
        for _, row in correct.iterrows():
            test_items.append({"idx": int(row["idx"]), "fold": fold, "label": "correct", "row": row})

    n_w = sum(1 for t in test_items if t["label"] == "wrong")
    n_c = sum(1 for t in test_items if t["label"] == "correct")

    save_file = OUTPUT_DIR / "all_detection_5fold.json"
    results = []
    done_keys = set()
    if save_file.exists():
        results = json.load(open(save_file))
        done_keys = {(r["fold"], r["idx"]) for r in results}
        print(f"Resuming: {len(done_keys)} done", flush=True)

    print(f"All detection methods: {n_w} TP + {n_c} FP across 5 folds", flush=True)
    print("=" * 70, flush=True)

    sys_det = "You are a strict medical expert checking clinical answers."

    for ti in test_items:
        if (ti["fold"], ti["idx"]) in done_keys: continue

        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        # Run all 5 prompts
        det = {}
        for pk, pt in PROMPTS.items():
            raw = vllm_gen(build_chatml(sys_det, pt.format(note=note, question=row["question"], answer=answer[:800])))
            obj = q32(raw) or {}
            det[pk] = {
                "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
                "error_type": str(obj.get("error_type", "NONE")).upper(),
                "error_statement": str(obj.get("error_statement", ""))[:250],
                "correct_statement": str(obj.get("correct_statement", ""))[:250],
            }

        # Compute aggregated methods
        entry = {
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            "eval_orig": int(row["binary_correct"]),
            "det": det,
            # Derived
            "S1_all": det["contra"]["verdict"]=="INCORRECT" or det["qmis"]["verdict"]=="INCORRECT" or det["omis"]["verdict"]=="INCORRECT",
            "S1_no_omis": det["contra"]["verdict"]=="INCORRECT" or det["qmis"]["verdict"]=="INCORRECT",
            "S1_contra_only": det["contra"]["verdict"]=="INCORRECT",
            "S1_qmis_only": det["qmis"]["verdict"]=="INCORRECT",
            "S1_omis_only": det["omis"]["verdict"]=="INCORRECT",
            "S3_det": det["S3"]["verdict"]=="INCORRECT",
            "D1_det": det["D1"]["verdict"]=="INCORRECT",
        }
        results.append(entry)
        with open(save_file, "w") as f: json.dump(results, f)

        if len(results) % 10 == 0:
            w_t = sum(1 for r in results if r["label"]=="wrong")
            c_t = sum(1 for r in results if r["label"]=="correct")
            # Quick stats for S1_no_omis
            w_d = sum(1 for r in results if r["label"]=="wrong" and r["S1_no_omis"])
            c_d = sum(1 for r in results if r["label"]=="correct" and r["S1_no_omis"])
            print(f"  [{len(results)}/{len(test_items)}] S1_no_omis: w={w_d}/{w_t} c={c_d}/{c_t}", flush=True)

    # FINAL SAVE
    with open(save_file, "w") as f: json.dump(results, f, indent=2)

    # SUMMARY TABLE
    print(f"\n{'='*70}", flush=True)
    print(f"ALL DETECTION METHODS — {n_w} wrong + {n_c} correct", flush=True)
    print(f"{'='*70}", flush=True)

    methods = ["S1_all", "S1_no_omis", "S1_contra_only", "S1_qmis_only", "S1_omis_only", "S3_det", "D1_det"]
    print(f"\n{'Method':<20} {'TP det':>8} {'FP det':>8} {'Select':>8}", flush=True)
    print("-" * 46, flush=True)
    for mk in methods:
        w = sum(1 for r in results if r["label"]=="wrong" and r[mk])
        c = sum(1 for r in results if r["label"]=="correct" and r[mk])
        sel = (w/n_w) / max(c/n_c, 0.01)
        print(f"  {mk:<20} {w}/{n_w:>3} ({100*w/n_w:>2.0f}%) {c}/{n_c:>3} ({100*c/n_c:>2.0f}%) {sel:>6.1f}x", flush=True)

    # Per-fold breakdown for top methods
    print(f"\nPER-FOLD BREAKDOWN:", flush=True)
    for mk in ["S1_all", "S1_no_omis", "S3_det", "D1_det"]:
        print(f"\n  {mk}:", flush=True)
        for fold in range(5):
            fw = [r for r in results if r["fold"]==fold and r["label"]=="wrong"]
            fc = [r for r in results if r["fold"]==fold and r["label"]=="correct"]
            wd = sum(1 for r in fw if r[mk])
            cd = sum(1 for r in fc if r[mk])
            print(f"    Fold {fold}: TP={wd}/{len(fw)} FP={cd}/{len(fc)}", flush=True)

    print(f"\nSaved to {save_file}", flush=True)


if __name__ == "__main__":
    main()
