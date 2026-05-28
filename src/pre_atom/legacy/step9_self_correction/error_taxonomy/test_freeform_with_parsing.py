#!/usr/bin/env python3
"""
Free-form detection + Qwen3-32B structured parsing.

Step 1: Qwen2.5 outputs free-form self-critique (no JSON constraint)
Step 2: Qwen3-32B reads the output and extracts structured JSON
Step 3: Compare: does Qwen3-32B's interpretation match what Qwen2.5 actually said?

This validates whether Qwen3-32B parsing is reliable for the prompt engineering loop.

Usage:
    python test_freeform_with_parsing.py --port 8003
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
# FREE-FORM DETECTION PROMPTS (no JSON constraint)
# ============================================================

# Same prompts as before but WITHOUT "respond with ONLY a JSON object"

F_P6 = """Discharge summary:
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

F_G2 = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Extract the 3 most important factual claims from this answer. For each claim, search the discharge notes for the EXACT supporting evidence.

Claim 1: <extract from answer>
Notes evidence: <quote from notes, or "NOT FOUND">
Match: SUPPORTED / CONTRADICTED / NOT_IN_NOTES

Claim 2: <extract from answer>
Notes evidence: <quote from notes, or "NOT FOUND">
Match: SUPPORTED / CONTRADICTED / NOT_IN_NOTES

Claim 3: <extract from answer>
Notes evidence: <quote from notes, or "NOT FOUND">
Match: SUPPORTED / CONTRADICTED / NOT_IN_NOTES

Based on your verification above, is this answer correct or incorrect? Explain."""

F_G4 = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

You are a devil's advocate reviewer. Your job is to find ANY reason this answer could be wrong.

CHECK EACH:
1. Does every medication name, dosage, and frequency exactly match the notes?
2. Does every procedure, diagnosis, and clinical finding exactly match the notes?
3. Does the answer address the specific visit/admission the question asks about?
4. Does the answer include ALL critical information the question asks for?

Report your findings for each check. Only mark as correct if ALL checks pass."""

F_COMBINED = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Review this answer carefully against the discharge notes.

Step 1 - QUESTION ALIGNMENT: What does the question ask about? Does the answer address the right visit, time period, and clinical focus?

Step 2 - CLAIM VERIFICATION: List the key factual claims in the answer. For each, find the supporting or contradicting passage in the notes.

Step 3 - COMPLETENESS: Is any critical information from the notes missing that would change the answer?

Step 4 - VERDICT: Based on the above, is this answer correct or incorrect? If incorrect, what specifically is wrong?"""

PROMPTS = {
    "F_P6": F_P6,
    "F_G2_claims": F_G2,
    "F_G4_devils": F_G4,
    "F_combined": F_COMBINED,
}

# ============================================================
# QWEN3-32B PARSING PROMPT
# ============================================================

QWEN32B_PARSE_PROMPT = """/nothink
Read the following self-critique output from a medical AI that was checking its own answer against discharge notes.

SELF-CRITIQUE OUTPUT:
{raw_output}

Extract the following information from this output:

1. VERDICT: Did the AI conclude its answer was CORRECT or INCORRECT? Look for the final conclusion, not intermediate steps. If the AI found specific errors or contradictions, the verdict is INCORRECT even if it didn't explicitly say so.

2. ERROR_TYPE: If incorrect, what type? Choose ONE:
   - MISREADING: misinterpreted info that IS in the notes (wrong value, confused entities)
   - FABRICATION: stated something NOT in the notes
   - OMISSION: missed critical info from the notes
   - QUESTION_MISALIGNMENT: answered the wrong question/visit/time period
   - NONE: if correct

3. WRONG_CLAIM: The specific claim from the answer that is wrong (quote it). Empty string if correct.

4. NOTES_SAY: What the discharge notes actually say about this topic (quote or paraphrase). Empty string if correct.

5. WHY_WRONG: Brief explanation of the error. Empty string if correct.

Respond with ONLY a JSON object, no other text:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "...", "notes_say": "...", "why_wrong": "..."}}"""


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

def qwen32b_parse(raw_output):
    """Qwen3-32B extracts structured JSON from free-form output."""
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "You extract structured information from text. Output ONLY valid JSON."},
                {"role": "user", "content": QWEN32B_PARSE_PROMPT.format(raw_output=raw_output[:2500])},
            ],
            "max_tokens": 300, "temperature": 0.0,
        }, timeout=90)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"</think>", "", text).strip()
        return text
    except Exception as e:
        return f'{{"error": "{e}"}}'

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

def human_read_verdict(raw):
    """Simple heuristic: does the free-form output lean CORRECT or INCORRECT?"""
    raw_lower = raw.lower()
    # Count signals
    incorrect_signals = sum(1 for w in ["incorrect", "wrong", "error", "contradicted",
                                         "not supported", "not found", "misread", "fabricat",
                                         "omit", "misalign", "does not match"]
                           if w in raw_lower)
    correct_signals = sum(1 for w in ["correct", "supported", "accurate", "matches",
                                       "no error", "all claims"]
                         if w in raw_lower)
    # Check final lines
    last_100 = raw_lower[-200:]
    if "incorrect" in last_100:
        return "INCORRECT"
    if "correct" in last_100 and "incorrect" not in last_100:
        return "CORRECT"
    if incorrect_signals > correct_signals:
        return "INCORRECT"
    if correct_signals > incorrect_signals:
        return "CORRECT"
    return "UNCLEAR"


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
    print(f"Free-form + Qwen32B parsing: Qwen2.5 ({n_w} wrong + {n_c} correct)")
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

            # Step 1: Free-form generation
            msg = ptemplate.format(note=note, question=row["question"], answer=answer[:800])
            system = "You are a strict medical expert verifying clinical answers against discharge notes."
            prompt = build_chatml(system, msg)
            raw = vllm_generate(args.port, prompt)

            # Step 2: Heuristic verdict from raw text
            heuristic_verdict = human_read_verdict(raw)

            # Step 3: Qwen3-32B structured parsing
            qwen32b_raw = qwen32b_parse(raw)
            obj, parse_method = try_parse_json(qwen32b_raw)

            if obj and isinstance(obj, dict):
                q32_verdict = str(obj.get("verdict", "")).upper()
                q32_error = str(obj.get("error_type", "NONE")).upper()
                q32_wrong_claim = str(obj.get("wrong_claim", ""))[:200]
                q32_notes_say = str(obj.get("notes_say", ""))[:200]
                q32_why = str(obj.get("why_wrong", ""))[:200]
                q32_parse_ok = True
            else:
                q32_verdict = "PARSE_FAIL"
                q32_error = "NONE"
                q32_wrong_claim = ""; q32_notes_say = ""; q32_why = ""
                q32_parse_ok = False

            # Agreement check
            agree = heuristic_verdict == q32_verdict

            results.append({
                "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                "heuristic_verdict": heuristic_verdict,
                "qwen32b_verdict": q32_verdict,
                "qwen32b_error_type": q32_error,
                "qwen32b_wrong_claim": q32_wrong_claim,
                "qwen32b_notes_say": q32_notes_say,
                "qwen32b_why_wrong": q32_why,
                "qwen32b_parse_ok": q32_parse_ok,
                "agree": agree,
                "raw_output": raw,  # FULL output, no truncation
                "qwen32b_raw": qwen32b_raw[:300],
            })

        all_results[pkey] = results

        # Summary
        w_det_h = sum(1 for r in results if r["label"] == "wrong" and r["heuristic_verdict"] == "INCORRECT")
        c_det_h = sum(1 for r in results if r["label"] == "correct" and r["heuristic_verdict"] == "INCORRECT")
        w_det_q = sum(1 for r in results if r["label"] == "wrong" and r["qwen32b_verdict"] == "INCORRECT")
        c_det_q = sum(1 for r in results if r["label"] == "correct" and r["qwen32b_verdict"] == "INCORRECT")
        n_agree = sum(1 for r in results if r["agree"])
        q32_ok = sum(1 for r in results if r["qwen32b_parse_ok"])
        sel_q = (w_det_q/max(n_w,1)) / max(c_det_q/max(n_c,1), 0.01)

        print(f"  Heuristic:  wrong={w_det_h}/{n_w} correct={c_det_h}/{n_c}")
        print(f"  Qwen32B:    wrong={w_det_q}/{n_w} correct={c_det_q}/{n_c}  sel={sel_q:.1f}x")
        print(f"  Agreement:  {n_agree}/{len(results)} ({100*n_agree/len(results):.0f}%)")
        print(f"  Q32B parse: {q32_ok}/{len(results)}")

    # Save
    out_file = OUTPUT_DIR / "detection_freeform_parsed.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "n_wrong": n_w, "n_correct": n_c}, f, indent=2)

    # Final comparison
    print(f"\n{'='*70}")
    print(f"{'Prompt':<18} {'Heur W':>8} {'Heur C':>8} {'Q32B W':>8} {'Q32B C':>8} {'Agree':>7} {'Q32B sel':>9}")
    print("-" * 70)
    for pkey in args.prompts:
        r = all_results[pkey]
        hw = sum(1 for x in r if x["label"]=="wrong" and x["heuristic_verdict"]=="INCORRECT")
        hc = sum(1 for x in r if x["label"]=="correct" and x["heuristic_verdict"]=="INCORRECT")
        qw = sum(1 for x in r if x["label"]=="wrong" and x["qwen32b_verdict"]=="INCORRECT")
        qc = sum(1 for x in r if x["label"]=="correct" and x["qwen32b_verdict"]=="INCORRECT")
        ag = sum(1 for x in r if x["agree"])
        sel = (qw/max(n_w,1)) / max(qc/max(n_c,1), 0.01)
        print(f"  {pkey:<18} {hw}/{n_w:>3}    {hc}/{n_c:>3}    {qw}/{n_w:>3}    {qc}/{n_c:>3}    {ag}/{len(r):>3}  {sel:>7.1f}x")

    # Show disagreements
    print(f"\n--- DISAGREEMENTS (heuristic vs Qwen32B) ---")
    for pkey in args.prompts:
        for r in all_results[pkey]:
            if not r["agree"]:
                print(f"  {pkey} idx={r['idx']} [{r['label']}] heur={r['heuristic_verdict']} q32b={r['qwen32b_verdict']}")
                print(f"    raw (last 200): ...{r['raw_output'][-200:]}")
                print()

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
