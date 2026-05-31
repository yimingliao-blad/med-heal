#!/usr/bin/env python3
"""
D2d full-scale test: all wrong + 50 correct on current vLLM model.
Free-form Qwen2.5/other model + Qwen3-32B extraction.
Saves all raw outputs + extracted JSON.
Progress saving every 10 items.

Usage:
    python run_d2d_fullscale.py --model qwen25 --port 8003
    python run_d2d_fullscale.py --model qwen3_nothink --port 8003
    python run_d2d_fullscale.py --model llama3 --port 8003
    python run_d2d_fullscale.py --model deepseek --port 8003
"""
import json, random, re, sys, argparse, time
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

MODEL_MAP = {
    "qwen25": {"dir": "qwen2.5-7b-instruct", "template": "chatml", "stop": ["<|im_end|>", "<|endoftext|>"], "max_tokens": 2048},
    "qwen3_nothink": {"dir": "qwen3-8b", "template": "qwen3_nothink", "stop": ["<|im_end|>", "<|endoftext|>"], "max_tokens": 2048},
    "llama3": {"dir": "llama-3.1-8b-instruct", "template": "llama3", "stop": ["<|eot_id|>", "<|end_of_text|>"], "max_tokens": 2048},
    "deepseek": {"dir": "deepseek-r1-distill-llama-8b", "template": "llama3", "stop": ["<|eot_id|>", "<｜end▁of▁sentence｜>"], "max_tokens": 4096},
}

D2D_PROMPT = """Discharge summary:
{{note}}

Question: {{question}}

Answer: {{answer}}

Extract ALL factual claims from this answer (aim for 5 or more). For each claim, find the EXACT evidence in the discharge notes.

Claim 1: <claim>
Notes: <exact quote or "NOT FOUND">
Verdict: SUPPORTED / CONTRADICTED / NOT IN NOTES

Claim 2: <claim>
Notes: <exact quote or "NOT FOUND">
Verdict: SUPPORTED / CONTRADICTED / NOT IN NOTES

[continue for all claims]

Also check: is there critical information in the notes relevant to the question that the answer does NOT mention?

Final assessment: is this answer correct or does it have errors?"""

EXTRACT_PROMPT = """/nothink
Read this self-critique from a medical AI checking its own answer.

SELF-CRITIQUE:
{raw_output}

Extract as JSON:
{{"verdict": "CORRECT" or "INCORRECT", "error_type": "MISREADING" or "FABRICATION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the wrong/missing claim as one sentence", "correct_statement": "what notes say as one sentence", "explanation": "brief"}}"""


def build_prompt(template, system, user):
    if template == "chatml":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n")
    elif template == "qwen3_nothink":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n/nothink\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n")
    elif template == "llama3":
        return ("<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return f"{system}\n\n{user}\n\nAssistant:"


def vllm_generate(port, prompt, stop_tokens, max_tokens=2048):
    try:
        model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
        resp = requests.post(f"http://localhost:{port}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": 0.0, "stop": stop_tokens},
            timeout=180)
        raw = resp.json()["choices"][0]["text"].strip()
        # Strip thinking tags
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"</think>", "", raw).strip()
        return raw
    except Exception as e:
        return f"ERROR: {e}"


def qwen32b_extract(raw_output):
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Extract structured info. Output ONLY valid JSON."},
                {"role": "user", "content": EXTRACT_PROMPT.format(raw_output=raw_output)},
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
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--n-correct", type=int, default=50)
    args = parser.parse_args()

    cfg = MODEL_MAP[args.model]
    notes = load_notes()

    # Load all folds
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / cfg["dir"] / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    wrong_df = all_df[all_df["binary_correct"] == 0]
    correct_df = all_df[all_df["binary_correct"] == 1]

    random.seed(42)
    correct_sample = correct_df.sample(n=min(args.n_correct, len(correct_df)), random_state=42)

    test_items = []
    for _, row in wrong_df.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "wrong", "row": row})
    for _, row in correct_sample.iterrows():
        test_items.append({"idx": int(row["idx"]), "fold": int(row["fold"]), "label": "correct", "row": row})

    n_w = sum(1 for t in test_items if t["label"] == "wrong")
    n_c = sum(1 for t in test_items if t["label"] == "correct")

    # Progress file
    progress_file = OUTPUT_DIR / f"d2d_fullscale_{args.model}_progress.json"
    results = []
    done_keys = set()
    if progress_file.exists():
        results = json.load(open(progress_file))
        done_keys = {(r["fold"], r["idx"]) for r in results}
        print(f"Resuming: {len(done_keys)} done")

    print(f"D2d Full-scale: {args.model} ({n_w} wrong + {n_c} correct)")
    print(f"Template: {cfg['template']}, max_tokens: {cfg['max_tokens']}")
    print("=" * 70)

    system = "You are a strict medical expert verifying clinical answers against discharge notes."
    d2d_template = D2D_PROMPT.replace("{{", "{").replace("}}", "}")

    for i, ti in enumerate(test_items):
        if (ti["fold"], ti["idx"]) in done_keys:
            continue

        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note:
            continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        msg = d2d_template.format(note=note, question=row["question"], answer=answer[:800])
        prompt = build_prompt(cfg["template"], system, msg)
        raw = vllm_generate(args.port, prompt, cfg["stop"], cfg["max_tokens"])

        obj = qwen32b_extract(raw)
        if obj:
            verdict = str(obj.get("verdict", "UNCLEAR")).upper()
            error_type = str(obj.get("error_type", "NONE")).upper()
            error_stmt = str(obj.get("error_statement", ""))[:250]
            correct_stmt = str(obj.get("correct_statement", ""))[:250]
            explanation = str(obj.get("explanation", ""))[:250]
            parse_ok = True
        else:
            verdict = "PARSE_FAIL"
            error_type = "NONE"
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

        if (len(results)) % 10 == 0:
            with open(progress_file, "w") as f:
                json.dump(results, f)
            w = sum(1 for r in results if r["label"] == "wrong" and r["verdict"] == "INCORRECT")
            c = sum(1 for r in results if r["label"] == "correct" and r["verdict"] == "INCORRECT")
            wt = sum(1 for r in results if r["label"] == "wrong")
            ct = sum(1 for r in results if r["label"] == "correct")
            pf = sum(1 for r in results if not r["parse_ok"])
            print(f"  [{len(results)}/{len(test_items)}] wrong={w}/{wt} correct={c}/{ct} pfail={pf}")

    # Final save
    with open(progress_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    w = sum(1 for r in results if r["label"] == "wrong" and r["verdict"] == "INCORRECT")
    c = sum(1 for r in results if r["label"] == "correct" and r["verdict"] == "INCORRECT")
    pf = sum(1 for r in results if not r["parse_ok"])
    sel = (w / max(n_w, 1)) / max(c / max(n_c, 1), 0.01)

    print(f"\n{'='*70}")
    print(f"D2d FULL-SCALE: {args.model}")
    print(f"  Wrong detected: {w}/{n_w} ({100*w/n_w:.0f}%)")
    print(f"  Correct detected (FP): {c}/{n_c} ({100*c/n_c:.0f}%)")
    print(f"  Selectivity: {sel:.1f}x")
    print(f"  Parse fail: {pf}/{len(results)}")

    # Error type distribution
    et_dist = Counter(r["error_type"] for r in results if r["label"] == "wrong" and r["verdict"] == "INCORRECT")
    if et_dist:
        print(f"  Error types (TP): {dict(et_dist)}")

    # Per-fold breakdown
    for fold in range(5):
        fw = [r for r in results if r["fold"] == fold and r["label"] == "wrong"]
        fc = [r for r in results if r["fold"] == fold and r["label"] == "correct"]
        fw_det = sum(1 for r in fw if r["verdict"] == "INCORRECT")
        fc_det = sum(1 for r in fc if r["verdict"] == "INCORRECT")
        if fw or fc:
            print(f"  Fold {fold}: wrong={fw_det}/{len(fw)} correct={fc_det}/{len(fc)}")

    print(f"\nSaved to {progress_file}")


if __name__ == "__main__":
    main()
