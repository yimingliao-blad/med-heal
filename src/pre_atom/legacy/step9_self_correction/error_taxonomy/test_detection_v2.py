#!/usr/bin/env python3
"""
Detection V2: Robust testing with LLM-assisted parsing.

Changes from V1:
1. Updated prompt: requires error LOCATION + EXPLANATION (not just type)
2. Saves ALL raw outputs for audit
3. Uses Qwen3-32B to parse/reformat raw outputs before regex parsing
4. Reports parse success rate alongside detection rates
5. DeepSeek uses max_tokens=4096 for thinking
6. Tests both old (BC) and new (BC_v2) prompts for comparison

Usage:
    python test_detection_v2.py --model qwen25 --port 8003
"""
import json, os, random, re, sys, time, argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

MODEL_MAP = {
    "qwen25": {"dir": "qwen2.5-7b-instruct", "template": "chatml", "stop": ["<|im_end|>", "<|endoftext|>"], "max_tokens": 1024, "think": False},
    "qwen3_nothink": {"dir": "qwen3-8b", "template": "qwen3_nothink", "stop": ["<|im_end|>", "<|endoftext|>"], "max_tokens": 1024, "think": False},
    "qwen3_think": {"dir": "qwen3-8b", "template": "qwen3", "stop": ["<|im_end|>", "<|endoftext|>"], "max_tokens": 2048, "think": True},
    "llama3": {"dir": "llama-3.1-8b-instruct", "template": "llama3", "stop": ["<|eot_id|>", "<|end_of_text|>"], "max_tokens": 1024, "think": False},
    "deepseek": {"dir": "deepseek-r1-distill-llama-8b", "template": "llama3", "stop": ["<|eot_id|>", "<｜end▁of▁sentence｜>"], "max_tokens": 4096, "think": True},
}

# ============================================================
# PROMPTS
# ============================================================

# Original BC prompt (for comparison)
BC_ORIGINAL = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Review this answer for errors. Common error patterns in clinical answers include:
- MISREADING: confusing medications, dosages, or visits that ARE in the notes
- FABRICATION: stating details NOT found anywhere in the notes
- OMISSION: missing critical information that changes the answer
- QUESTION_MISALIGNMENT: answering about the wrong visit, time period, or clinical focus

STEP 1 — Does the answer address the right question?
Check: correct visit, correct time period, correct clinical focus.
ALIGNMENT: OK or PROBLEM

STEP 2 — Is every claim supported by the notes?
For each key claim, find the supporting passage in the notes.
EVIDENCE: OK or PROBLEM — <specific issue>

STEP 3 — Are critical details included?
Only flag omissions that change the answer's conclusion.
COMPLETENESS: OK or PROBLEM — <what's missing>

VERDICT: CORRECT or INCORRECT
IF INCORRECT — ERROR_TYPE: <MISREADING, FABRICATION, OMISSION, or QUESTION_MISALIGNMENT>"""

# Updated BC_v2 prompt — JSON output + error location/explanation
BC_V2 = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Review this answer for errors. Common error patterns in clinical answers include:
- MISREADING: confusing medications, dosages, or visits that ARE in the notes
- FABRICATION: stating details NOT found anywhere in the notes
- OMISSION: missing critical information that changes the answer
- QUESTION_MISALIGNMENT: answering about the wrong visit, time period, or clinical focus

Check step by step:
1. Does the answer address the right question? (correct visit, time period, clinical focus)
2. Is every claim supported by the notes? (check medications, dosages, procedures, dates)
3. Are critical details included? (only flag omissions that change the conclusion)

Respond with ONLY a JSON object (no other text):
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the specific wrong claim or empty string", "notes_say": "what the notes actually say or empty string", "why_wrong": "brief explanation or empty string"}}"""

PROMPTS = {
    "BC_original": BC_ORIGINAL,
    "BC_v2": BC_V2,
}


# ============================================================
# TEMPLATE BUILDERS
# ============================================================

def build_prompt(template, system, user):
    if template == "chatml":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n")
    elif template == "qwen3_nothink":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n/nothink\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n")
    elif template == "qwen3":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n")
    elif template == "llama3":
        return ("<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return f"{system}\n\n{user}\n\nAssistant:"


def vllm_generate(port, prompt, stop_tokens, max_tokens=1024, temperature=0.0):
    try:
        model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
        resp = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature, "stop": stop_tokens},
            timeout=180,
        )
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"  vLLM error: {e}")
        return ""


# ============================================================
# QWEN3-32B PARSING
# ============================================================

PARSE_PROMPT = """Read the following text from a medical AI self-critique. The AI was asked to review its own clinical answer.

TEXT:
{raw_output}

Extract the final conclusion as a JSON object. Did the AI conclude the answer is correct or incorrect?

/nothink
Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT" or "UNCLEAR", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "wrong_claim": "the flagged wrong claim or empty string", "notes_say": "what notes say or empty string"}}"""


def qwen32b_parse(raw_output):
    """Use Qwen3-32B to parse raw model output into structured JSON."""
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "You extract structured information from text and output JSON only."},
                {"role": "user", "content": PARSE_PROMPT.format(raw_output=raw_output[:2000])},
            ],
            "max_tokens": 500, "temperature": 0.0,
        }, timeout=90)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip thinking tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"</think>", "", text).strip()
        return text
    except Exception as e:
        return f'{{"error": "{e}"}}'


def try_parse_json(text):
    """Try to parse JSON from text, handling common issues."""
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    # Try to find JSON in the text
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    # Try fixing common issues (single quotes, trailing commas)
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except:
        pass
    return None


def parse_structured(text):
    """Parse Qwen3-32B output into structured result."""
    result = {"verdict": "UNCLEAR", "error_type": "NONE", "wrong_claim": "", "notes_say": ""}

    parsed = try_parse_json(text)
    if parsed and isinstance(parsed, dict):
        v = str(parsed.get("verdict", "")).upper()
        if v in ("CORRECT", "INCORRECT", "UNCLEAR"):
            result["verdict"] = v
        et = str(parsed.get("error_type", "NONE")).upper()
        if et in ("MISREADING", "FABRICATION", "OMISSION", "QUESTION_MISALIGNMENT"):
            result["error_type"] = et
        result["wrong_claim"] = str(parsed.get("wrong_claim", ""))[:200]
        result["notes_say"] = str(parsed.get("notes_say", ""))[:200]
        result["json_parsed"] = True
    else:
        # Fallback: try regex on raw text
        m = re.search(r'"verdict"\s*:\s*"(CORRECT|INCORRECT|UNCLEAR)"', text, re.I)
        if m:
            result["verdict"] = m.group(1).upper()
        result["json_parsed"] = False

    return result


# ============================================================
# DATA LOADING
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


def load_model_data(model_dir):
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / model_dir / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n-wrong", type=int, default=25)
    parser.add_argument("--n-correct", type=int, default=25)
    parser.add_argument("--prompts", nargs="+", default=["BC_original", "BC_v2"])
    args = parser.parse_args()

    cfg = MODEL_MAP[args.model]
    notes = load_notes()
    all_df = load_model_data(cfg["dir"])

    if all_df.empty:
        print(f"No data for {cfg['dir']}")
        return

    random.seed(42)
    wrong = all_df[all_df["binary_correct"] == 0].sample(n=min(args.n_wrong, (all_df["binary_correct"]==0).sum()), random_state=42)
    correct = all_df[all_df["binary_correct"] == 1].sample(n=min(args.n_correct, (all_df["binary_correct"]==1).sum()), random_state=42)

    test_items = []
    for _, row in wrong.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "wrong", "row": row})
    for _, row in correct.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "correct", "row": row})

    n_wrong = sum(1 for t in test_items if t["label"] == "wrong")
    n_correct = sum(1 for t in test_items if t["label"] == "correct")

    print(f"Detection V2: {args.model} ({n_wrong} wrong + {n_correct} correct)")
    print(f"Prompts: {args.prompts}")
    print(f"Max tokens: {cfg['max_tokens']}, Think: {cfg['think']}")
    print("=" * 60)

    # Check Qwen3-32B availability
    try:
        requests.get(QWEN32B_URL.replace("/chat/completions", "/models"), timeout=5)
        qwen32b_available = True
        print("Qwen3-32B: available for parsing")
    except:
        qwen32b_available = False
        print("Qwen3-32B: NOT available — will use regex only")

    all_results = {}

    for pkey in args.prompts:
        ptemplate = PROMPTS[pkey]
        print(f"\n{'='*60}")
        print(f"Prompt: {pkey}")
        print(f"{'='*60}")

        results = []
        for i, ti in enumerate(test_items):
            row = ti["row"]
            note = notes.get(str(row["patient_id"]), "")
            if not note:
                continue
            answer = str(row.get("openended_answer", row.get("model_answer", "")))

            msg = ptemplate.format(note=note, question=row["question"], answer=answer[:800])
            system = "You are a strict medical expert verifying clinical answers against discharge notes."
            prompt = build_prompt(cfg["template"], system, msg)

            # Generate
            raw = vllm_generate(args.port, prompt, cfg["stop"], max_tokens=cfg["max_tokens"])

            # Strip thinking tags for display (keep raw for saving)
            raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw_clean = re.sub(r"</think>", "", raw_clean).strip()

            # Step 1: Try direct JSON parse (for BC_v2 which asks for JSON output)
            direct_json = try_parse_json(raw_clean)
            if direct_json and isinstance(direct_json, dict):
                direct_verdict = str(direct_json.get("verdict", "")).upper()
                direct_error = str(direct_json.get("error_type", "NONE")).upper()
                direct_parsed = True
            else:
                direct_verdict = "UNCLEAR"
                direct_error = "NONE"
                direct_parsed = False

            # Step 2: Regex fallback (for BC_original which uses free text)
            raw_upper = raw_clean.upper()
            regex_verdict = "UNCLEAR"
            if "VERDICT: INCORRECT" in raw_upper or \
               ('"verdict": "incorrect"' in raw_clean.lower()) or \
               ('"verdict":"incorrect"' in raw_clean.lower()):
                regex_verdict = "INCORRECT"
            elif "VERDICT: CORRECT" in raw_upper or \
                 ('"verdict": "correct"' in raw_clean.lower()) or \
                 ('"verdict":"correct"' in raw_clean.lower()):
                regex_verdict = "CORRECT"

            # Step 3: Qwen3-32B parse (always, for validation)
            qwen32b_parsed = {"verdict": "UNCLEAR", "error_type": "NONE", "json_parsed": False}
            qwen32b_raw = ""
            if qwen32b_available:
                qwen32b_raw = qwen32b_parse(raw_clean[:2000])
                qwen32b_parsed = parse_structured(qwen32b_raw)

            # Final verdict: prefer direct JSON > Qwen32B > regex
            if direct_parsed and direct_verdict in ("CORRECT", "INCORRECT"):
                final_verdict = direct_verdict
                parse_method = "direct_json"
            elif qwen32b_parsed.get("json_parsed") and qwen32b_parsed["verdict"] in ("CORRECT", "INCORRECT"):
                final_verdict = qwen32b_parsed["verdict"]
                parse_method = "qwen32b"
            elif regex_verdict != "UNCLEAR":
                final_verdict = regex_verdict
                parse_method = "regex"
            else:
                final_verdict = "UNCLEAR"
                parse_method = "failed"

            detected = final_verdict == "INCORRECT"

            entry = {
                "idx": ti["idx"], "fold": ti["fold"],
                "label": ti["label"],
                "detected": detected,
                "final_verdict": final_verdict,
                "parse_method": parse_method,
                "direct_json_parsed": direct_parsed,
                "direct_verdict": direct_verdict if direct_parsed else None,
                "regex_verdict": regex_verdict,
                "qwen32b_verdict": qwen32b_parsed["verdict"],
                "qwen32b_error_type": qwen32b_parsed["error_type"],
                "qwen32b_wrong_claim": qwen32b_parsed.get("wrong_claim", ""),
                "qwen32b_notes_say": qwen32b_parsed.get("notes_say", ""),
                "qwen32b_json_parsed": qwen32b_parsed.get("json_parsed", False),
                "raw_output": raw_clean[:500],
                "qwen32b_raw": qwen32b_raw[:300],
            }
            results.append(entry)

            if (i + 1) % 10 == 0:
                w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
                c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
                wt = sum(1 for r in results if r["label"] == "wrong")
                ct = sum(1 for r in results if r["label"] == "correct")
                print(f"  [{i+1}/{len(test_items)}] wrong={w_det}/{wt} correct={c_det}/{ct}")

        all_results[pkey] = results

        # Summary
        w_det = sum(1 for r in results if r["label"] == "wrong" and r["detected"])
        c_det = sum(1 for r in results if r["label"] == "correct" and r["detected"])
        wt = sum(1 for r in results if r["label"] == "wrong")
        ct = sum(1 for r in results if r["label"] == "correct")
        sel = (w_det/max(wt,1)) / max(c_det/max(ct,1), 0.01)

        # Parse stats
        parse_methods = Counter(r["parse_method"] for r in results)
        print(f"\n  Parse method distribution:")
        for pm, count in parse_methods.most_common():
            print(f"    {pm}: {count}/{len(results)}")

        # Compare methods where both produced a verdict
        if qwen32b_available:
            both_have = [r for r in results if r["regex_verdict"] != "UNCLEAR"
                         and r["qwen32b_verdict"] != "UNCLEAR"]
            if both_have:
                agree = sum(1 for r in both_have if r["regex_verdict"] == r["qwen32b_verdict"])
                print(f"    Regex vs Qwen32B (where both non-UNCLEAR): {agree}/{len(both_have)} agree")
            qwen_json_ok = sum(1 for r in results if r.get("qwen32b_json_parsed"))
            print(f"    Qwen32B JSON parse success: {qwen_json_ok}/{len(results)}")

        print(f"\n  Detection (Qwen32B-parsed):")
        print(f"    Wrong: {w_det}/{wt} ({100*w_det/wt:.0f}%)")
        print(f"    Correct: {c_det}/{ct} ({100*c_det/ct:.0f}%)")
        print(f"    Selectivity: {sel:.1f}x")

        # Error type distribution (from Qwen32B)
        if qwen32b_available:
            wrong_detected = [r for r in results if r["label"] == "wrong" and r["detected"]]
            if wrong_detected:
                et_counts = Counter(r.get("qwen32b_error_type", "NONE") for r in wrong_detected)
                print(f"    Error types detected: {dict(et_counts)}")

    # Save ALL results with raw outputs
    out_file = OUTPUT_DIR / f"detection_v2_{args.model}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": args.model,
            "config": {k: v for k, v in cfg.items() if k != "stop"},
            "n_wrong": n_wrong, "n_correct": n_correct,
            "prompts": args.prompts,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Final comparison table
    if len(args.prompts) > 1:
        print(f"\n{'='*60}")
        print(f"PROMPT COMPARISON — {args.model}")
        print(f"{'='*60}")
        print(f"  {'Prompt':<15} {'Wrong':>10} {'Correct':>10} {'Select':>10}")
        for pkey in args.prompts:
            r = all_results[pkey]
            w = sum(1 for x in r if x["label"] == "wrong" and x["detected"])
            c = sum(1 for x in r if x["label"] == "correct" and x["detected"])
            sel = (w/max(n_wrong,1)) / max(c/max(n_correct,1), 0.01)
            print(f"  {pkey:<15} {w}/{n_wrong:>5} ({100*w/n_wrong:.0f}%) {c}/{n_correct:>5} ({100*c/n_correct:.0f}%) {sel:>8.1f}x")


if __name__ == "__main__":
    main()
