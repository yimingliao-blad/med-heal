#!/usr/bin/env python3
"""
Error Taxonomy Analysis for Qwen2.5-7B Zeroshot Answers.

Phase 1: Qwen3-32B analyzes ALL 109 wrong + 50 sampled correct answers (free)
Phase 2: GPT-4o confirms a subset of ~30 to validate Qwen3-32B classifications (paid)
Phase 3: Build taxonomy, check discriminability, document findings

User's proposed error categories:
  a. Hedging/lack of confidence — multiple conclusions in different directions
  b. Fabrication — hallucinated content not in notes, causing contradictions
  c. Misreading — factual error reading notes, misunderstanding/misinterpretation
  d. Omission — ignoring key details, leading to wrong conclusion
  e. Question misalignment — answers with irrelevant info, misses key point

Usage:
    python analyze_qwen25_errors.py --phase 1              # Qwen3-32B analysis
    python analyze_qwen25_errors.py --phase 2 --n-confirm 30  # GPT-4o confirmation
    python analyze_qwen25_errors.py --phase 3              # Summary + taxonomy
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

# Qwen3-32B on Mac Studio
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"
QWEN32B_MODEL = "Qwen/Qwen3-32B-MLX-bf16"

# OpenAI (for Phase 2 confirmation only)
api_key = None
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for line in open(env_file):
        if line.startswith("OPENAI_API_KEY="):
            api_key = line.strip().split("=", 1)[1]
            break

spending = {"qwen32b_calls": 0, "gpt4o_calls": 0, "gpt4o_cost": 0.0}


# ============================================================
# ANALYSIS PROMPTS
# ============================================================

WRONG_ANALYSIS_PROMPT = """You are a medical expert analyzing why an AI model's answer to a clinical question is incorrect.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER (Incorrect):
{model_answer}

Analyze the model's answer step by step:

1. WHAT THE QUESTION ASKS: What specific information is the question looking for?

2. WHAT THE NOTES SAY: What does the discharge summary say about this topic? Quote relevant passages.

3. WHAT THE MODEL SAID: Summarize the model's key claims.

4. WHERE IT WENT WRONG: Identify the specific error(s). For each error:
   - What claim is wrong?
   - What do the notes actually say?
   - Why might the model have made this mistake?

5. ERROR CLASSIFICATION: Classify the PRIMARY error as exactly ONE of:
   a. HEDGING — The model provides multiple possible answers or hedges instead of committing to what the notes clearly state. The correct information may even be mentioned but is diluted by alternatives.
   b. FABRICATION — The model states something that is NOT in the discharge notes at all. It invented or hallucinated a clinical detail (medication, procedure, diagnosis, date) with no basis in the notes.
   c. MISREADING — The model misread, misinterpreted, or confused information that IS in the notes. The source material exists but was understood incorrectly (e.g., wrong dosage, confused two medications, mixed up two visits).
   d. OMISSION — The model failed to mention critical information that IS in the notes and is needed to answer the question. The answer is incomplete in a way that changes the conclusion.
   e. QUESTION_MISALIGNMENT — The model answered a different question than what was asked, or focused on irrelevant aspects of the notes while missing the actual question's focus.
   f. OTHER — If none of the above fit, describe the error pattern.

6. CONFIDENCE ASSESSMENT: How confident does the model sound?
   - DEFINITIVE: States answer as fact
   - HEDGED: Uses qualifiers ("may", "possibly", "likely")
   - MULTI_ANSWER: Provides multiple possible answers

Respond in this exact format:
QUESTION_FOCUS: <what the question is really asking>
KEY_NOTES_EVIDENCE: <relevant quotes from notes>
MODEL_CLAIMS: <model's key claims>
ERROR_DESCRIPTION: <specific description of what went wrong>
PRIMARY_ERROR: <one of: HEDGING, FABRICATION, MISREADING, OMISSION, QUESTION_MISALIGNMENT, OTHER>
SECONDARY_ERRORS: <comma-separated list of any additional error types, or NONE>
CONFIDENCE: <DEFINITIVE, HEDGED, or MULTI_ANSWER>
SEVERITY: <CRITICAL (answer is completely wrong) or PARTIAL (answer has the right idea but misses key details)>"""


CORRECT_ANALYSIS_PROMPT = """You are a medical expert analyzing an AI model's answer to a clinical question. This answer was judged CORRECT, but analyze its characteristics.

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER (Correct):
{model_answer}

Analyze whether this correct answer shows any of these patterns:

1. Does the answer contain hedging or multiple possible conclusions? (The answer might still be correct but use uncertain language)
2. Does the answer include any details NOT explicitly in the discharge notes?
3. Does the answer miss any information that IS in the notes and relevant to the question?
4. Does the answer directly address what the question asks?
5. How confident does the model sound?

Respond in this exact format:
HAS_HEDGING: YES or NO — <brief description>
HAS_FABRICATION: YES or NO — <brief description>
HAS_OMISSION: YES or NO — <brief description>
QUESTION_ALIGNED: YES or NO — <brief description>
CONFIDENCE: <DEFINITIVE, HEDGED, or MULTI_ANSWER>
OVERALL_QUALITY: <CLEAN (no issues) / MINOR_ISSUES (small imperfections) / BORDERLINE (almost wrong)>"""


# ============================================================
# LLM CALL FUNCTIONS
# ============================================================

def qwen32b_call(system_msg, user_msg):
    """Call Qwen3-32B on Mac Studio."""
    try:
        resp = requests.post(
            QWEN32B_URL,
            json={
                "model": QWEN32B_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return ""
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip thinking tags if present
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        spending["qwen32b_calls"] += 1
        return text
    except Exception as e:
        print(f"  Qwen3-32B error: {e}")
        return ""


def gpt4o_call(system_msg, user_msg, max_tokens=800):
    """Call GPT-4o for confirmation."""
    if not api_key:
        print("No OPENAI_API_KEY")
        return ""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    time.sleep(1.5)
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        text = r.choices[0].message.content.strip()
        cost = r.usage.prompt_tokens * 2.5 / 1e6 + r.usage.completion_tokens * 10.0 / 1e6
        spending["gpt4o_calls"] += 1
        spending["gpt4o_cost"] += cost
        return text
    except Exception as e:
        print(f"  GPT-4o error: {e}")
        time.sleep(5)
        return ""


# ============================================================
# PARSE FUNCTIONS
# ============================================================

def parse_wrong_analysis(text):
    """Parse structured analysis of wrong answer."""
    result = {}
    for field in ["QUESTION_FOCUS", "KEY_NOTES_EVIDENCE", "MODEL_CLAIMS",
                   "ERROR_DESCRIPTION", "PRIMARY_ERROR", "SECONDARY_ERRORS",
                   "CONFIDENCE", "SEVERITY"]:
        m = re.search(rf"{field}:\s*(.+?)(?:\n[A-Z_]+:|$)", text, re.DOTALL)
        result[field] = m.group(1).strip() if m else ""

    pe = result.get("PRIMARY_ERROR", "").upper()
    valid = {"HEDGING", "FABRICATION", "MISREADING", "OMISSION", "QUESTION_MISALIGNMENT", "OTHER"}
    if pe not in valid:
        for v in valid:
            if v in pe:
                pe = v
                break
        else:
            pe = "OTHER"
    result["PRIMARY_ERROR"] = pe

    conf = result.get("CONFIDENCE", "").upper()
    if "MULTI" in conf: conf = "MULTI_ANSWER"
    elif "HEDGE" in conf: conf = "HEDGED"
    elif "DEFINIT" in conf: conf = "DEFINITIVE"
    else: conf = "UNKNOWN"
    result["CONFIDENCE"] = conf

    sev = result.get("SEVERITY", "").upper()
    result["SEVERITY"] = "CRITICAL" if "CRITICAL" in sev else "PARTIAL" if "PARTIAL" in sev else "UNKNOWN"

    return result


def parse_correct_analysis(text):
    """Parse structured analysis of correct answer."""
    result = {}
    for field in ["HAS_HEDGING", "HAS_FABRICATION", "HAS_OMISSION",
                   "QUESTION_ALIGNED", "CONFIDENCE", "OVERALL_QUALITY"]:
        m = re.search(rf"{field}:\s*(.+?)(?:\n[A-Z_]+:|$)", text, re.DOTALL)
        result[field] = m.group(1).strip() if m else ""

    for field in ["HAS_HEDGING", "HAS_FABRICATION", "HAS_OMISSION"]:
        result[field + "_BOOL"] = "YES" in result.get(field, "").upper()[:10]
    result["QUESTION_ALIGNED_BOOL"] = "YES" in result.get("QUESTION_ALIGNED", "").upper()[:10]

    conf = result.get("CONFIDENCE", "").upper()
    if "MULTI" in conf: conf = "MULTI_ANSWER"
    elif "HEDGE" in conf: conf = "HEDGED"
    elif "DEFINIT" in conf: conf = "DEFINITIVE"
    else: conf = "UNKNOWN"
    result["CONFIDENCE"] = conf

    qual = result.get("OVERALL_QUALITY", "").upper()
    if "CLEAN" in qual: qual = "CLEAN"
    elif "BORDER" in qual: qual = "BORDERLINE"
    elif "MINOR" in qual: qual = "MINOR_ISSUES"
    else: qual = "UNKNOWN"
    result["OVERALL_QUALITY"] = qual

    return result


# ============================================================
# DATA LOADING
# ============================================================

def load_qwen25_zeroshot():
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fold
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


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
# PHASE 1: GPT-4o analysis
# ============================================================

def run_phase1(all_df, notes, n_correct=50, max_wrong=None, resume=False):
    """GPT-4o analyzes all wrong answers + sampled correct answers."""
    wrong_df = all_df[all_df["binary_correct"] == 0]
    correct_df = all_df[all_df["binary_correct"] == 1]

    # Progress files
    wrong_progress = OUTPUT_DIR / "phase1_wrong_gpt4o.json"
    correct_progress = OUTPUT_DIR / "phase1_correct_gpt4o.json"

    # --- Wrong answers ---
    wrong_results = []
    done_wrong = set()
    if resume and wrong_progress.exists():
        wrong_results = json.load(open(wrong_progress))
        done_wrong = {(r["fold"], r["idx"]) for r in wrong_results}
        print(f"Resuming wrong: {len(done_wrong)} done")

    items = list(wrong_df.iterrows())
    if max_wrong:
        items = items[:max_wrong]

    est_cost = len(items) * 0.015
    print(f"\n--- Analyzing {len(items)} WRONG answers with GPT-4o ---")
    print(f"Estimated cost: ~${est_cost:.2f}")
    for i, (_, row) in enumerate(items):
        fold, idx = int(row["fold"]), int(row["idx"])
        if (fold, idx) in done_wrong:
            continue

        note = notes.get(str(row["patient_id"]), "")
        if not note:
            continue

        answer = str(row.get("openended_answer", row.get("model_answer", "")))
        raw = gpt4o_call(
            "You are a medical expert analyzing errors in clinical AI answers.",
            WRONG_ANALYSIS_PROMPT.format(
                note=note, question=row["question"],
                ground_truth=row["ground_truth"],
                model_answer=answer[:1000],
            ),
        )

        parsed = parse_wrong_analysis(raw)
        entry = {
            "idx": idx, "fold": fold,
            "patient_id": int(row["patient_id"]),
            "question": row["question"][:200],
            "ground_truth": row["ground_truth"][:200],
            "model_answer": answer[:300],
            "label": "wrong",
            **{k: v for k, v in parsed.items() if k != "KEY_NOTES_EVIDENCE"},
            "raw_analysis": raw[:600],
        }
        wrong_results.append(entry)
        print(f"  [{i+1}/{len(items)}] fold={fold} idx={idx} → {parsed['PRIMARY_ERROR']} ({parsed['CONFIDENCE']}, {parsed['SEVERITY']}) ${spending['gpt4o_cost']:.2f}")

        if len(wrong_results) % 10 == 0:
            with open(wrong_progress, "w") as f:
                json.dump(wrong_results, f, indent=2)

    with open(wrong_progress, "w") as f:
        json.dump(wrong_results, f, indent=2)

    # --- Correct answers (sample) ---
    correct_results = []
    done_correct = set()
    if resume and correct_progress.exists():
        correct_results = json.load(open(correct_progress))
        done_correct = {(r["fold"], r["idx"]) for r in correct_results}
        print(f"Resuming correct: {len(done_correct)} done")

    random.seed(42)
    correct_sample = correct_df.sample(n=min(n_correct, len(correct_df)), random_state=42)

    est_cost2 = len(correct_sample) * 0.012
    print(f"\n--- Analyzing {len(correct_sample)} CORRECT answers with GPT-4o ---")
    print(f"Estimated cost: ~${est_cost2:.2f}")
    for i, (_, row) in enumerate(correct_sample.iterrows()):
        fold, idx = int(row["fold"]), int(row["idx"])
        if (fold, idx) in done_correct:
            continue

        note = notes.get(str(row["patient_id"]), "")
        if not note:
            continue

        answer = str(row.get("openended_answer", row.get("model_answer", "")))
        raw = gpt4o_call(
            "You are a medical expert analyzing AI clinical answers.",
            CORRECT_ANALYSIS_PROMPT.format(
                note=note, question=row["question"],
                ground_truth=row["ground_truth"],
                model_answer=answer[:1000],
            ),
            max_tokens=500,
        )

        parsed = parse_correct_analysis(raw)
        entry = {
            "idx": idx, "fold": fold,
            "patient_id": int(row["patient_id"]),
            "question": row["question"][:200],
            "model_answer": answer[:300],
            "label": "correct",
            **parsed,
            "raw_analysis": raw[:600],
        }
        correct_results.append(entry)
        print(f"  [{i+1}/{len(correct_sample)}] fold={fold} idx={idx} → {parsed['CONFIDENCE']} {parsed['OVERALL_QUALITY']} ${spending['gpt4o_cost']:.2f}")

        if len(correct_results) % 10 == 0:
            with open(correct_progress, "w") as f:
                json.dump(correct_results, f, indent=2)

    with open(correct_progress, "w") as f:
        json.dump(correct_results, f, indent=2)

    print(f"\nGPT-4o: {spending['gpt4o_calls']} calls, ${spending['gpt4o_cost']:.3f}")
    return wrong_results, correct_results


# ============================================================
# PHASE 2: GPT-4o confirmation on subset
# ============================================================

def run_phase2(wrong_results, n_confirm=30):
    """GPT-4o confirms Qwen3-32B classifications on a stratified sample."""
    confirm_progress = OUTPUT_DIR / "phase2_gpt4o_confirm.json"

    # Stratified sample: proportional to error type distribution
    by_type = defaultdict(list)
    for r in wrong_results:
        by_type[r["PRIMARY_ERROR"]].append(r)

    sample = []
    per_type = max(3, n_confirm // len(by_type))
    for etype, items in by_type.items():
        sample.extend(random.sample(items, min(per_type, len(items))))
    random.shuffle(sample)
    sample = sample[:n_confirm]

    notes = load_notes()
    all_df = load_qwen25_zeroshot()

    print(f"\n--- GPT-4o confirming {len(sample)} cases ---")
    print(f"Estimated cost: ~${len(sample) * 0.02:.2f}")

    confirm_results = []
    for i, r in enumerate(sample):
        row = all_df[(all_df["fold"] == r["fold"]) & (all_df["idx"] == r["idx"])]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        note = notes.get(str(row["patient_id"]), "")
        answer = str(row.get("openended_answer", row.get("model_answer", "")))

        gpt_raw = gpt4o_call(
            "You are a medical expert analyzing errors in clinical AI answers.",
            WRONG_ANALYSIS_PROMPT.format(
                note=note, question=row["question"],
                ground_truth=row["ground_truth"],
                model_answer=answer[:1000],
            ),
        )

        gpt_parsed = parse_wrong_analysis(gpt_raw)
        agree = gpt_parsed["PRIMARY_ERROR"] == r["PRIMARY_ERROR"]

        confirm_results.append({
            "idx": r["idx"], "fold": r["fold"],
            "qwen32b_primary": r["PRIMARY_ERROR"],
            "gpt4o_primary": gpt_parsed["PRIMARY_ERROR"],
            "agree": agree,
            "qwen32b_confidence": r["CONFIDENCE"],
            "gpt4o_confidence": gpt_parsed["CONFIDENCE"],
        })

        symbol = "✓" if agree else "✗"
        print(f"  [{i+1}/{len(sample)}] idx={r['idx']} Q32B={r['PRIMARY_ERROR']:20s} GPT4o={gpt_parsed['PRIMARY_ERROR']:20s} {symbol}")

    with open(confirm_progress, "w") as f:
        json.dump(confirm_results, f, indent=2)

    n_agree = sum(1 for r in confirm_results if r["agree"])
    print(f"\nAgreement: {n_agree}/{len(confirm_results)} = {100*n_agree/max(len(confirm_results),1):.0f}%")
    print(f"GPT-4o: {spending['gpt4o_calls']} calls, ${spending['gpt4o_cost']:.3f}")
    return confirm_results


# ============================================================
# PHASE 3: Summary and taxonomy
# ============================================================

def run_phase3():
    """Summarize all findings, compare wrong vs correct patterns."""
    wrong_file = OUTPUT_DIR / "phase1_wrong_gpt4o.json"
    correct_file = OUTPUT_DIR / "phase1_correct_gpt4o.json"
    confirm_file = OUTPUT_DIR / "phase2_gpt4o_confirm.json"

    if not wrong_file.exists():
        print("Phase 1 not complete. Run --phase 1 first.")
        return

    wrong_results = json.load(open(wrong_file))
    correct_results = json.load(open(correct_file)) if correct_file.exists() else []
    confirm_results = json.load(open(confirm_file)) if confirm_file.exists() else []

    print(f"{'='*60}")
    print(f"ERROR TAXONOMY — Qwen2.5-7B Zeroshot")
    print(f"{'='*60}")
    print(f"Wrong answers analyzed: {len(wrong_results)}")
    print(f"Correct answers analyzed: {len(correct_results)}")
    if confirm_results:
        n_agree = sum(1 for r in confirm_results if r["agree"])
        print(f"GPT-4o confirmation: {n_agree}/{len(confirm_results)} agree ({100*n_agree/max(len(confirm_results),1):.0f}%)")

    # --- Wrong answer analysis ---
    print(f"\n{'='*60}")
    print("WRONG ANSWER ERROR TYPES")
    print(f"{'='*60}")

    pe_counts = Counter(r["PRIMARY_ERROR"] for r in wrong_results)
    print(f"\nPrimary Error Distribution:")
    for pe, count in pe_counts.most_common():
        pct = 100 * count / len(wrong_results)
        examples = [r for r in wrong_results if r["PRIMARY_ERROR"] == pe][:2]
        print(f"\n  {pe}: {count} ({pct:.0f}%)")
        for ex in examples:
            print(f"    ex: Q={ex['question'][:80]}...")
            print(f"        {ex.get('ERROR_DESCRIPTION', '')[:100]}")

    conf_counts = Counter(r["CONFIDENCE"] for r in wrong_results)
    print(f"\nConfidence Distribution (Wrong Answers):")
    for c, count in conf_counts.most_common():
        print(f"  {c:15s}: {count:3d} ({100*count/len(wrong_results):.0f}%)")

    sev_counts = Counter(r["SEVERITY"] for r in wrong_results)
    print(f"\nSeverity Distribution:")
    for s, count in sev_counts.most_common():
        print(f"  {s:15s}: {count:3d} ({100*count/len(wrong_results):.0f}%)")

    # Cross-tab: error × confidence
    print(f"\nError Type × Confidence:")
    conf_types = ["DEFINITIVE", "HEDGED", "MULTI_ANSWER", "UNKNOWN"]
    print(f"  {'':25s}", end="")
    for c in conf_types:
        print(f" {c:>12s}", end="")
    print()
    for pe in pe_counts:
        print(f"  {pe:25s}", end="")
        for c in conf_types:
            n = sum(1 for r in wrong_results if r["PRIMARY_ERROR"] == pe and r["CONFIDENCE"] == c)
            print(f" {n:>12d}", end="")
        print()

    # --- Correct answer analysis ---
    if correct_results:
        print(f"\n{'='*60}")
        print("CORRECT ANSWER PATTERNS (for comparison)")
        print(f"{'='*60}")

        corr_conf = Counter(r["CONFIDENCE"] for r in correct_results)
        print(f"\nConfidence Distribution (Correct Answers):")
        for c, count in corr_conf.most_common():
            print(f"  {c:15s}: {count:3d} ({100*count/len(correct_results):.0f}%)")

        corr_qual = Counter(r["OVERALL_QUALITY"] for r in correct_results)
        print(f"\nOverall Quality:")
        for q, count in corr_qual.most_common():
            print(f"  {q:15s}: {count:3d} ({100*count/len(correct_results):.0f}%)")

        # Pattern prevalence in correct answers
        for field, label in [("HAS_HEDGING_BOOL", "Has hedging"),
                             ("HAS_FABRICATION_BOOL", "Has fabrication"),
                             ("HAS_OMISSION_BOOL", "Has omission"),
                             ("QUESTION_ALIGNED_BOOL", "Question aligned")]:
            yes = sum(1 for r in correct_results if r.get(field, False))
            print(f"  {label:25s}: {yes}/{len(correct_results)} ({100*yes/len(correct_results):.0f}%)")

        # --- Discriminability analysis ---
        print(f"\n{'='*60}")
        print("DISCRIMINABILITY: Can these patterns separate wrong from correct?")
        print(f"{'='*60}")

        # Confidence as discriminator
        wrong_hedged = sum(1 for r in wrong_results if r["CONFIDENCE"] in ("HEDGED", "MULTI_ANSWER"))
        correct_hedged = sum(1 for r in correct_results if r["CONFIDENCE"] in ("HEDGED", "MULTI_ANSWER"))
        print(f"\n  Hedged/Multi-answer:")
        print(f"    Wrong:   {wrong_hedged}/{len(wrong_results)} ({100*wrong_hedged/len(wrong_results):.0f}%)")
        print(f"    Correct: {correct_hedged}/{len(correct_results)} ({100*correct_hedged/len(correct_results):.0f}%)")
        if wrong_hedged / max(len(wrong_results), 1) > correct_hedged / max(len(correct_results), 1) * 1.5:
            print(f"    → DISCRIMINATIVE: hedging is {wrong_hedged/max(len(wrong_results),1) / max(correct_hedged/max(len(correct_results),1), 0.01):.1f}x more common in wrong answers")
        else:
            print(f"    → NOT DISCRIMINATIVE")

        # Fabrication in correct answers
        corr_fab = sum(1 for r in correct_results if r.get("HAS_FABRICATION_BOOL", False))
        print(f"\n  Fabrication in correct answers: {corr_fab}/{len(correct_results)} ({100*corr_fab/len(correct_results):.0f}%)")
        if corr_fab / max(len(correct_results), 1) < 0.1:
            print(f"    → DISCRIMINATIVE: fabrication rare in correct answers")
        else:
            print(f"    → NOT DISCRIMINATIVE: fabrication also appears in correct answers")

    # Save full summary
    summary = {
        "n_wrong": len(wrong_results),
        "n_correct": len(correct_results),
        "primary_error_dist": dict(pe_counts),
        "wrong_confidence_dist": dict(conf_counts),
        "wrong_severity_dist": dict(sev_counts),
    }
    if correct_results:
        summary["correct_confidence_dist"] = dict(Counter(r["CONFIDENCE"] for r in correct_results))
        summary["correct_quality_dist"] = dict(Counter(r["OVERALL_QUALITY"] for r in correct_results))
        summary["correct_patterns"] = {
            "has_hedging": sum(1 for r in correct_results if r.get("HAS_HEDGING_BOOL", False)),
            "has_fabrication": sum(1 for r in correct_results if r.get("HAS_FABRICATION_BOOL", False)),
            "has_omission": sum(1 for r in correct_results if r.get("HAS_OMISSION_BOOL", False)),
        }
    if confirm_results:
        summary["gpt4o_agreement"] = sum(1 for r in confirm_results if r["agree"]) / len(confirm_results)

    with open(OUTPUT_DIR / "error_taxonomy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'error_taxonomy_summary.json'}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--max-wrong", type=int, default=None)
    parser.add_argument("--n-correct", type=int, default=50)
    parser.add_argument("--n-confirm", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.phase == 1:
        all_df = load_qwen25_zeroshot()
        notes = load_notes()
        print(f"Loaded {len(all_df)} items, {(all_df['binary_correct']==0).sum()} wrong")
        run_phase1(all_df, notes, n_correct=args.n_correct, max_wrong=args.max_wrong, resume=args.resume)

    elif args.phase == 2:
        wrong_file = OUTPUT_DIR / "phase1_wrong_qwen32b.json"
        if not wrong_file.exists():
            print("Run phase 1 first")
            return
        wrong_results = json.load(open(wrong_file))
        run_phase2(wrong_results, n_confirm=args.n_confirm)

    elif args.phase == 3:
        run_phase3()


if __name__ == "__main__":
    main()
