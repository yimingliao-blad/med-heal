#!/usr/bin/env python3
"""
Round 8: Systematic test of detection design space.

4 combinations from 3 dimensions:
  S1: Multi-run + rules + severity (3 separate calls)
  S2: Single CoT + few-shot + conclusion-impact (1 call)
  S3: Single CoT + rules+few-shot + severity (1 call, balanced)
  S4: Multi-run + few-shot + aggregate (3 calls, pick most critical)

Tests on fold 1, 15 wrong + 15 correct.
Free-form output → Qwen3-32B extraction.

Usage:
    python test_detection_round8.py --port 8003 --fold 1
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
# S1: MULTI-RUN — one prompt per type, rules-based
# ============================================================

S1_CONTRADICTION = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CONTRADICTION: Does the answer state any fact that DIRECTLY CONFLICTS with the discharge notes?

For each key claim in the answer (medication names, dosages, procedures, diagnoses, lab values, dates):
1. Find the matching information in the notes
2. Does the answer say something DIFFERENT from the notes?

Only flag when the answer and notes provide OPPOSING information — not when something is merely missing.

If you find a contradiction, explain: what does the answer say vs what do the notes say? Would correcting this change the answer's conclusion?"""

S1_QMISALIGN = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR QUESTION MISALIGNMENT: Does the answer address the WRONG thing?

Parse the question carefully:
- Which hospital visit does it ask about? (first, second, third, most recent?)
- What clinical aspect? (medications, procedures, diagnoses, complications, outcomes?)
- What time period? (during admission, at discharge, between visits?)

Does the answer match? Specifically:
- Does it discuss the correct hospital visit?
- Does it focus on the right clinical aspect?
- Does it cover the right time period?

If misaligned, explain: what does the question ask vs what does the answer discuss?"""

S1_OMISSION = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CRITICAL OMISSION: Is there information in the notes that is ESSENTIAL to answer the question but COMPLETELY ABSENT from the answer?

Important: only flag omissions that would CHANGE the answer's conclusion. Do NOT flag:
- Minor details (exact dates, non-critical medications)
- Background information not directly relevant to the question
- Details that would add completeness but don't change the conclusion

If you find a critical omission, explain: what is missing and why would adding it change the answer?"""

# ============================================================
# S2: SINGLE CoT + FEW-SHOT + conclusion-impact
# ============================================================

S2_COT_FEWSHOT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check this answer for errors that would make the conclusion wrong.

Examples of errors in clinical answers:

CONTRADICTION example:
  Q: "What medication adjustments were made for psychotic symptoms?"
  Error: Answer said "Haloperidol was added during second admission" but notes show Risperdal was added, not Haloperidol. This changes which medication was actually prescribed.

OMISSION example:
  Q: "What was the allergic reaction to Ativan?"
  Error: Answer only mentioned "hives" but notes also document "anaphylaxis". Missing the severe reaction changes the clinical picture.

QUESTION MISALIGNMENT example:
  Q: "What complications occurred postoperatively at the outside hospital?"
  Error: Answer included complications from the SECOND hospital, not the outside hospital. Answering about the wrong facility.

Now check the answer above:

Step 1: Does the answer address the correct visit, time period, and clinical focus that the question asks about?

Step 2: For each key fact in the answer, does it match the notes? Look for contradictions — the answer says X but notes say Y.

Step 3: Is there essential information in the notes needed to answer the question that the answer completely ignores?

For any error found: would correcting it CHANGE the answer's conclusion?

Report what you find."""

# ============================================================
# S3: SINGLE CoT + RULES + FEW-SHOT + severity
# ============================================================

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

# ============================================================
# S4: MULTI-RUN + FEW-SHOT + aggregate
# ============================================================

S4_CONTRA_FS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check for CONTRADICTIONS between the answer and the notes.

Example: A clinical answer said "Haloperidol was added during the second admission" but the notes showed Risperdal was added instead. The wrong medication name is a contradiction.

Another example: An answer said "no surgical procedure was performed" but the notes clearly describe a laparoscopic procedure. Denying a documented procedure is a contradiction.

For each key claim in the answer, find the matching fact in the notes. Report any contradictions — where the answer says one thing and the notes say another."""

S4_QMIS_FS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check if the answer addresses the WRONG thing.

Example: A question asked "What complications occurred at the outside hospital?" but the answer included complications from the second hospital instead. Answering about the wrong facility is a critical misalignment.

Another example: A question asked about "discharge medications after surgery" but the answer listed admission medications instead.

Re-read the question: what visit, what aspect, what time period? Does the answer match?"""

S4_OMISSION_FS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check if the answer is missing ESSENTIAL information.

Example: A question asked about allergic reactions. The answer mentioned "hives" but the notes also documented "anaphylaxis" — a much more severe reaction. Missing this changes the clinical picture.

Only flag missing information that would CHANGE the answer's conclusion. Do NOT flag minor missing details."""


# ============================================================
# HELPERS
# ============================================================

EXTRACT_PROMPT = """/nothink
Read this self-critique output. Extract as JSON.
Only mark INCORRECT if CRITICAL errors were found that change the answer's conclusion.

SELF-CRITIQUE:
{raw_output}

{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the specific error as one sentence", "correct_statement": "what notes say as one sentence", "explanation": "brief"}}"""

def build_chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")

def vllm_gen(port, prompt, max_tokens=2048):
    model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
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
        if m: return json.loads(m.group())
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

def run_detect(port, note, question, answer, prompt_template):
    """Run one detection prompt, return extracted result."""
    msg = prompt_template.format(note=note, question=question, answer=answer[:800])
    prompt = build_chatml("You are a strict medical expert checking clinical answers against discharge notes.", msg)
    raw = vllm_gen(port, prompt)
    obj = q32_extract(raw)
    if obj:
        return {
            "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
            "error_type": str(obj.get("error_type", "NONE")).upper(),
            "error_statement": str(obj.get("error_statement", ""))[:250],
            "correct_statement": str(obj.get("correct_statement", ""))[:250],
            "raw": raw,
        }
    return {"verdict": "PARSE_FAIL", "error_type": "NONE", "error_statement": "", "correct_statement": "", "raw": raw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n-wrong", type=int, default=15)
    parser.add_argument("--n-correct", type=int, default=15)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

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

    # Load GPT annotations for comparison
    try:
        with open(OUTPUT_DIR / "correction_oriented_annotations.json") as f:
            gpt_ann = json.load(f)
        gpt_lookup = {(r["fold"], r["idx"]): r for r in gpt_ann}
    except: gpt_lookup = {}

    progress_file = OUTPUT_DIR / f"round8_fold{args.fold}_progress.json"
    print(f"Round 8: fold={args.fold}, {n_w} wrong + {n_c} correct, seed={args.seed}", flush=True)
    print("=" * 70, flush=True)

    all_results = {}

    # --- S1: Multi-run (3 calls per item) ---
    print(f"\n=== S1: Multi-run (3 calls per item) ===", flush=True)
    s1_results = []
    for ti in test_items:
        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        r_contra = run_detect(args.port, note, row["question"], answer, S1_CONTRADICTION)
        r_qmis = run_detect(args.port, note, row["question"], answer, S1_QMISALIGN)
        r_omis = run_detect(args.port, note, row["question"], answer, S1_OMISSION)

        # Aggregate: pick most critical (QMIS > CONTRA > OMISSION)
        if r_qmis["verdict"] == "INCORRECT":
            final = r_qmis
        elif r_contra["verdict"] == "INCORRECT":
            final = r_contra
        elif r_omis["verdict"] == "INCORRECT":
            final = r_omis
        else:
            final = {"verdict": "CORRECT", "error_type": "NONE", "error_statement": "", "correct_statement": ""}

        gpt = gpt_lookup.get((ti["fold"], ti["idx"]), {})
        gpt_type = gpt.get("correction_type", "?") if ti["label"] == "wrong" else "correct"

        s1_results.append({
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            "verdict": final["verdict"], "error_type": final["error_type"],
            "error_statement": final.get("error_statement", "")[:200],
            "correct_statement": final.get("correct_statement", "")[:200],
            "sub_results": {"contra": r_contra["verdict"], "qmis": r_qmis["verdict"], "omis": r_omis["verdict"]},
        })
        print(f"  {ti['label']:>7} idx={ti['idx']} → {final['verdict']} {final['error_type']} (gpt={gpt_type}) [C={r_contra['verdict'][:3]} Q={r_qmis['verdict'][:3]} O={r_omis['verdict'][:3]}]", flush=True)

        with open(progress_file, "w") as f:
            json.dump({"S1": s1_results}, f)

    all_results["S1_multirun"] = s1_results
    w = sum(1 for r in s1_results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
    c = sum(1 for r in s1_results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
    sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
    print(f"  → wrong={w}/{n_w} FP={c}/{n_c} sel={sel:.1f}x", flush=True)

    # --- S2: Single CoT + few-shot ---
    print(f"\n=== S2: Single CoT + few-shot ===", flush=True)
    s2_results = []
    for ti in test_items:
        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        r = run_detect(args.port, note, row["question"], answer, S2_COT_FEWSHOT)
        gpt = gpt_lookup.get((ti["fold"], ti["idx"]), {})
        gpt_type = gpt.get("correction_type", "?") if ti["label"] == "wrong" else "correct"

        s2_results.append({
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            **{k: v for k, v in r.items() if k != "raw"},
        })
        print(f"  {ti['label']:>7} idx={ti['idx']} → {r['verdict']} {r['error_type']} (gpt={gpt_type})", flush=True)

    all_results["S2_cot_fewshot"] = s2_results
    w = sum(1 for r in s2_results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
    c = sum(1 for r in s2_results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
    sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
    print(f"  → wrong={w}/{n_w} FP={c}/{n_c} sel={sel:.1f}x", flush=True)

    # --- S3: Single balanced (rules+fewshot+severity) ---
    print(f"\n=== S3: Balanced (rules+fewshot+severity) ===", flush=True)
    s3_results = []
    for ti in test_items:
        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        r = run_detect(args.port, note, row["question"], answer, S3_BALANCED)
        gpt = gpt_lookup.get((ti["fold"], ti["idx"]), {})
        gpt_type = gpt.get("correction_type", "?") if ti["label"] == "wrong" else "correct"

        s3_results.append({
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            **{k: v for k, v in r.items() if k != "raw"},
        })
        print(f"  {ti['label']:>7} idx={ti['idx']} → {r['verdict']} {r['error_type']} (gpt={gpt_type})", flush=True)

    all_results["S3_balanced"] = s3_results
    w = sum(1 for r in s3_results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
    c = sum(1 for r in s3_results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
    sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
    print(f"  → wrong={w}/{n_w} FP={c}/{n_c} sel={sel:.1f}x", flush=True)

    # --- S4: Multi-run + few-shot ---
    print(f"\n=== S4: Multi-run + few-shot ===", flush=True)
    s4_results = []
    for ti in test_items:
        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        r_contra = run_detect(args.port, note, row["question"], answer, S4_CONTRA_FS)
        r_qmis = run_detect(args.port, note, row["question"], answer, S4_QMIS_FS)
        r_omis = run_detect(args.port, note, row["question"], answer, S4_OMISSION_FS)

        if r_qmis["verdict"] == "INCORRECT":
            final = r_qmis
        elif r_contra["verdict"] == "INCORRECT":
            final = r_contra
        elif r_omis["verdict"] == "INCORRECT":
            final = r_omis
        else:
            final = {"verdict": "CORRECT", "error_type": "NONE", "error_statement": "", "correct_statement": ""}

        gpt = gpt_lookup.get((ti["fold"], ti["idx"]), {})
        gpt_type = gpt.get("correction_type", "?") if ti["label"] == "wrong" else "correct"

        s4_results.append({
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            "verdict": final["verdict"], "error_type": final["error_type"],
            "error_statement": final.get("error_statement", "")[:200],
            "correct_statement": final.get("correct_statement", "")[:200],
            "sub_results": {"contra": r_contra["verdict"], "qmis": r_qmis["verdict"], "omis": r_omis["verdict"]},
        })
        print(f"  {ti['label']:>7} idx={ti['idx']} → {final['verdict']} {final['error_type']} (gpt={gpt_type}) [C={r_contra['verdict'][:3]} Q={r_qmis['verdict'][:3]} O={r_omis['verdict'][:3]}]", flush=True)

    all_results["S4_multirun_fs"] = s4_results
    w = sum(1 for r in s4_results if r["label"]=="wrong" and r["verdict"]=="INCORRECT")
    c = sum(1 for r in s4_results if r["label"]=="correct" and r["verdict"]=="INCORRECT")
    sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
    print(f"  → wrong={w}/{n_w} FP={c}/{n_c} sel={sel:.1f}x", flush=True)

    # Save
    out_file = OUTPUT_DIR / f"detection_round8_fold{args.fold}.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c}, f, indent=2)

    # Final table
    print(f"\n{'='*70}", flush=True)
    print(f"{'Strategy':<20} {'Wrong':>8} {'FP':>8} {'Select':>8} {'Calls':>6}", flush=True)
    print("-" * 52, flush=True)
    for skey in all_results:
        r = all_results[skey]
        w = sum(1 for x in r if x["label"]=="wrong" and x["verdict"]=="INCORRECT")
        c = sum(1 for x in r if x["label"]=="correct" and x["verdict"]=="INCORRECT")
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        calls = "3x" if "multi" in skey.lower() else "1x"
        print(f"  {skey:<20} {w}/{n_w:>3}    {c}/{n_c:>3}    {sel:>5.1f}x  {calls:>5}", flush=True)

    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
