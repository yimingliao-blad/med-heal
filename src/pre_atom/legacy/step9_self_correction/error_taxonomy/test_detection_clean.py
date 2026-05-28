#!/usr/bin/env python3
"""
Clean detection prompt test — JSON output, verified parsing.

All prompts instruct the model to output JSON. Parse success is tracked.
Qwen3-32B validates ambiguous outputs.

Prompts redesigned from scratch based on error taxonomy:
  P1: Evidence check (claim-by-claim against notes) — targets MISREADING + FABRICATION
  P2: Question alignment check — targets QUESTION_MISALIGNMENT
  P3: Completeness check — targets OMISSION
  P4: Combined CoT (all 3 checks in one prompt)
  P5: Combined CoT + few-shot error patterns
  P6: Combined CoT + few-shot + error location required

Usage:
    python test_detection_clean.py --port 8003 --n-wrong 50 --n-correct 50
"""
import json, random, re, sys, time, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

# ============================================================
# PROMPTS — all require JSON output
# ============================================================

P1_EVIDENCE = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

For each key factual claim in the answer, check if it is supported by the discharge notes.

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "NONE", "wrong_claim": "the specific wrong claim or empty string", "notes_say": "what the notes actually say or empty string"}}"""

P2_ALIGNMENT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check: Does this answer address what the question specifically asks? Does it answer about the correct visit, time period, and clinical focus?

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "what the answer addresses vs what the question asks, or empty string", "notes_say": ""}}"""

P3_COMPLETENESS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check: Does the answer include the critical information from the discharge notes needed to fully answer the question? Only flag missing information that would change the answer's conclusion.

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "OMISSION" or "NONE", "wrong_claim": "what critical detail is missing, or empty string", "notes_say": "the relevant passage from the notes, or empty string"}}"""

P4_COMBINED = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Check this answer for errors:
1. Does it address the right question? (correct visit, time period, clinical focus)
2. Is every factual claim supported by the notes? (medications, dosages, procedures, dates)
3. Does it include critical details needed to answer the question?

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the specific error or empty string", "notes_say": "what the notes say or empty string"}}"""

P5_COMBINED_FEWSHOT = """Discharge summary:
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

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the specific error or empty string", "notes_say": "what the notes say or empty string"}}"""

P6_COMBINED_FEWSHOT_DETAIL = """Discharge summary:
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

PROMPTS = {
    "P1_evidence": P1_EVIDENCE,
    "P2_alignment": P2_ALIGNMENT,
    "P3_completeness": P3_COMPLETENESS,
    "P4_combined": P4_COMBINED,
    "P5_combined_fs": P5_COMBINED_FEWSHOT,
    "P6_combined_fs_detail": P6_COMBINED_FEWSHOT_DETAIL,
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
        resp = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]},
            timeout=120,
        )
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        return f"VLLM_ERROR: {e}"


def try_parse_json(text):
    """Try to extract and parse JSON from text."""
    # Direct parse
    try:
        return json.loads(text), "direct"
    except:
        pass
    # Find JSON object in text
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), "extracted"
        except:
            pass
    # Strip markdown code fences
    cleaned = re.sub(r'^```\w*\n?', '', text.strip())
    cleaned = re.sub(r'\n?```$', '', cleaned).strip()
    try:
        return json.loads(cleaned), "cleaned"
    except:
        pass
    return None, "failed"


def qwen32b_reparse(raw_output):
    """Use Qwen3-32B to extract verdict from unparseable output."""
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Extract the verdict from the text. Output ONLY a JSON object."},
                {"role": "user", "content": f"/nothink\nFrom this medical AI output, extract the final verdict.\n\nTEXT:\n{raw_output[:1500]}\n\nRespond with ONLY: {{\"verdict\": \"CORRECT\" or \"INCORRECT\", \"error_type\": \"...\" or \"NONE\"}}"},
            ],
            "max_tokens": 200, "temperature": 0.0,
        }, timeout=60)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        obj, method = try_parse_json(text)
        return obj, f"qwen32b_{method}" if obj else None, "qwen32b_failed"
    except Exception as e:
        return None, f"qwen32b_error: {e}"


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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n-wrong", type=int, default=50)
    parser.add_argument("--n-correct", type=int, default=50)
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()))
    args = parser.parse_args()

    notes = load_notes()

    # Load Qwen2.5 data
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
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
    print(f"Clean Detection Test: Qwen2.5 ({n_w} wrong + {n_c} correct)")
    print(f"Prompts: {args.prompts}")
    print("=" * 70)

    # Check Qwen32B
    try:
        requests.get(QWEN32B_URL.replace("/chat/completions", "/models"), timeout=5)
        print("Qwen3-32B: available for fallback parsing")
    except:
        print("Qwen3-32B: NOT available")

    all_results = {}

    for pkey in args.prompts:
        ptemplate = PROMPTS[pkey]
        print(f"\n{'='*70}")
        print(f"Prompt: {pkey}")
        print(f"{'='*70}")

        results = []
        for i, ti in enumerate(test_items):
            row = ti["row"]
            note = notes.get(str(row["patient_id"]), "")
            if not note:
                continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))

            msg = ptemplate.format(note=note, question=row["question"], answer=answer[:800])
            system = "You are a strict medical expert verifying clinical answers against discharge notes. Respond with ONLY a JSON object."
            prompt = build_chatml(system, msg)
            raw = vllm_generate(args.port, prompt)

            # Parse
            obj, parse_method = try_parse_json(raw)
            if obj is None:
                # Fallback to Qwen32B
                obj, parse_method = qwen32b_reparse(raw)

            if obj and isinstance(obj, dict):
                verdict = str(obj.get("verdict", "")).upper()
                error_type = str(obj.get("error_type", "NONE")).upper()
                wrong_claim = str(obj.get("wrong_claim", ""))[:200]
                notes_say = str(obj.get("notes_say", ""))[:200]
                why_wrong = str(obj.get("why_wrong", ""))[:200]
            else:
                verdict = "PARSE_FAIL"
                error_type = "NONE"
                wrong_claim = ""
                notes_say = ""
                why_wrong = ""
                parse_method = "failed"

            detected = verdict == "INCORRECT"

            entry = {
                "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                "detected": detected, "verdict": verdict,
                "error_type": error_type,
                "wrong_claim": wrong_claim, "notes_say": notes_say, "why_wrong": why_wrong,
                "parse_method": parse_method,
                "raw_output": raw[:500],
            }
            results.append(entry)

            if (i + 1) % 20 == 0:
                w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
                c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
                wt = sum(1 for r in results if r["label"] == "wrong")
                ct = sum(1 for r in results if r["label"] == "correct")
                pf = sum(1 for r in results if r["verdict"] == "PARSE_FAIL")
                print(f"  [{i+1}/{len(test_items)}] wrong={w_det}/{wt} correct={c_det}/{ct} parse_fail={pf}")

        all_results[pkey] = results

        # Summary
        w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
        c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
        wt = sum(1 for r in results if r["label"] == "wrong")
        ct = sum(1 for r in results if r["label"] == "correct")
        pf = sum(1 for r in results if r["verdict"] == "PARSE_FAIL")
        sel = (w_det/max(wt,1)) / max(c_det/max(ct,1), 0.01)

        parse_dist = Counter(r["parse_method"] for r in results)
        print(f"\n  Parse: {dict(parse_dist)}")
        print(f"  Parse fail: {pf}/{len(results)} ({100*pf/len(results):.0f}%)")
        print(f"  Wrong detected: {w_det}/{wt} ({100*w_det/wt:.0f}%)")
        print(f"  Correct detected: {c_det}/{ct} ({100*c_det/ct:.0f}%)")
        print(f"  Selectivity: {sel:.1f}x")

        # Error type distribution
        if w_det > 0:
            et_dist = Counter(r["error_type"] for r in results if r["label"] == "wrong" and r["detected"])
            print(f"  Error types: {dict(et_dist)}")

    # Save
    out_file = OUTPUT_DIR / "detection_clean_qwen25.json"
    with open(out_file, "w") as f:
        json.dump({"model": "qwen2.5-7b-instruct", "results": all_results}, f, indent=2)

    # Final table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Prompt':<25} {'Wrong':>8} {'Correct':>8} {'ParseFail':>10} {'Select':>10}")
    print("-" * 65)
    for pkey in args.prompts:
        r = all_results[pkey]
        w = sum(1 for x in r if x["label"] == "wrong" and x["detected"])
        c = sum(1 for x in r if x["label"] == "correct" and x["detected"])
        pf = sum(1 for x in r if x["verdict"] == "PARSE_FAIL")
        sel = (w/max(n_w,1)) / max(c/max(n_c,1), 0.01)
        print(f"  {pkey:<25} {w}/{n_w:>3} ({100*w/n_w:>2.0f}%) {c}/{n_c:>3} ({100*c/n_c:>2.0f}%) {pf:>8} {sel:>8.1f}x")

    # Combined: any of P1+P2+P3
    if all(k in all_results for k in ["P1_evidence", "P2_alignment", "P3_completeness"]):
        w_any = 0; c_any = 0
        for i in range(len(test_items)):
            det = any(all_results[k][i]["detected"] for k in ["P1_evidence", "P2_alignment", "P3_completeness"])
            if test_items[i]["label"] == "wrong" and det: w_any += 1
            if test_items[i]["label"] == "correct" and det: c_any += 1
        sel = (w_any/max(n_w,1)) / max(c_any/max(n_c,1), 0.01)
        print(f"  {'P1+P2+P3 (any)':<25} {w_any}/{n_w:>3} ({100*w_any/n_w:>2.0f}%) {c_any}/{n_c:>3} ({100*c_any/n_c:>2.0f}%) {'':>8} {sel:>8.1f}x")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
