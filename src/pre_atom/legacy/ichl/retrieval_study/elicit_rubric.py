"""Step 2a: meta-prompt GPT-4o for the criteria of "useful in-context example for correction".

Asks GPT-4o (with concrete grounded examples from our pool) to enumerate the dimensions
on which a candidate case becomes useful for correcting a model's wrong answer to the
test case. We then turn these dimensions into separate per-pair scores in Step 2b.

Approach:
  1. Show 3 worked example sets to GPT-4o (each = test case where target was wrong + a
     hand-mixed candidate set: clearly useful / borderline / clearly useless).
  2. Ask GPT-4o to articulate WHAT MAKES the useful ones useful, as a 3-5 dim rubric.
  3. Save the rubric for use in Step 2b.

Cost: ~3 GPT-4o calls × ~3000 tokens each = ~$0.05.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
ERR_FILE = ROOT / "output" / "step8" / "error_classification" / "all_errors_by_patient.json"
OUT = ROOT / "output" / "ichl" / "retrieval_study" / "rubric_elicitation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are a senior clinical AI researcher designing a retrieval system. The system "
    "needs to find the most useful in-context examples to help an LLM correct its wrong "
    "answers to clinical questions about EHR notes. We need you to articulate the "
    "criteria for what makes a candidate case useful — not vague principles, but concrete "
    "dimensions that can be scored on each (test, candidate) pair."
)

USER_TMPL = """We are designing a retrieval system. Below is a TEST CASE where the AI model
gave a wrong answer, plus three CANDIDATE EXAMPLES grouped by usefulness for correcting
the model.

# TEST CASE (model was wrong)
QUESTION: {test_q}
GROUND TRUTH: {test_gt}
MODEL'S WRONG ANSWER: {test_wrong}
PRIMARY ERROR TYPE (per our taxonomy): {test_err}
NOTE EXCERPT: {test_note}

# CANDIDATE GROUP A — clearly USEFUL (would help correct the model)
{group_a}

# CANDIDATE GROUP B — borderline (some elements useful, some not)
{group_b}

# CANDIDATE GROUP C — clearly NOT useful
{group_c}

# YOUR TASK
Articulate the criteria. List 3-5 specific DIMENSIONS that distinguish the useful
candidates from the not-useful ones. For each dimension, give:
  - a short name (one or two words)
  - a one-sentence definition
  - a 0-3 rating scale (what each level means in concrete terms)

Format your answer as a JSON array of dimension objects, like:
[
  {{
    "name": "question_type_match",
    "definition": "Does the candidate ask the same kind of question as the test...",
    "scale": {{
      "0": "different question type",
      "1": "loosely related question type",
      "2": "similar question type",
      "3": "same question type"
    }}
  }},
  ...
]

Return ONLY the JSON array, no prose. Aim for 3-5 dimensions total."""


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        pid = int(r["patient_id"])
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        parts = []
        for j in [1, 2, 3]:
            v = r.get(f"note_{j}")
            if v and str(v).strip() and str(v).lower() != "nan":
                parts.append(str(v))
        note = ("\n\n".join(parts))[:600]
        out.append({"row_id": i, "patient_id": pid, "category": r.get("category", ""),
                    "question": str(r["question"]), "ground_truth": gt, "note": note})
    return out


def load_errors() -> dict[int, dict]:
    arr = json.loads(ERR_FILE.read_text())
    return {int(r["patient_id"]): r for r in arr}


def fmt_candidate(c: dict, max_note: int = 400) -> str:
    return (f"  Q: {c['question']}\n"
            f"  GT: {c['ground_truth']}\n"
            f"  Note excerpt: {c['note'][:max_note]}\n")


def main():
    rng = random.Random(7)
    print("Loading pool + error labels...")
    pool = load_pool()
    errors = load_errors()
    pool_by_pid = {p["patient_id"]: p for p in pool}

    # Pick 3 test cases: each must be a BM-error in our pool, with diverse error types
    err_in_pool = [r for r in errors.values() if int(r["patient_id"]) in pool_by_pid]
    by_type: dict[str, list] = {}
    for r in err_in_pool:
        by_type.setdefault(r["primary_error"], []).append(r)
    test_types = ["hallucination", "omission", "reasoning_failure"]
    tests = []
    for t in test_types:
        rng.shuffle(by_type[t])
        tests.append(by_type[t][0])

    # For each test case: build groups A/B/C from the pool
    # A (useful): same primary_error type, prefer same category
    # B (borderline): different primary_error but similar question style (use category match)
    # C (useless): pull random from pool with totally different category
    candidates_pool = [p for p in pool if p["patient_id"] not in {t["patient_id"] for t in tests}]

    examples_for_meta_prompt = []
    for tst in tests:
        test_pid = int(tst["patient_id"])
        test_pool_rec = pool_by_pid[test_pid]
        test_err = tst["primary_error"]
        # A: same error type
        same_err_pids = [r["patient_id"] for r in by_type.get(test_err, []) if int(r["patient_id"]) != test_pid]
        rng.shuffle(same_err_pids)
        a_recs = [pool_by_pid[int(p)] for p in same_err_pids[:3] if int(p) in pool_by_pid][:2]
        # B: different error type, same category
        diff_err = [p for p in candidates_pool
                    if p["category"] == test_pool_rec["category"]
                    and p["patient_id"] in {int(r["patient_id"]) for r in err_in_pool if r["primary_error"] != test_err}]
        rng.shuffle(diff_err)
        b_recs = diff_err[:2]
        # C: different category, BM-correct (no error)
        bm_correct_pids = {p["patient_id"] for p in pool} - {int(r["patient_id"]) for r in err_in_pool}
        c_pool = [p for p in candidates_pool
                  if p["patient_id"] in bm_correct_pids
                  and p["category"] != test_pool_rec["category"]]
        rng.shuffle(c_pool)
        c_recs = c_pool[:2]
        examples_for_meta_prompt.append({
            "test": tst, "test_pool_rec": test_pool_rec,
            "a": a_recs, "b": b_recs, "c": c_recs
        })

    # === Build the 3 meta-prompts ===
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_path = ROOT / ".env"
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    rubrics = []
    total_in = 0
    total_out = 0
    for i, ex in enumerate(examples_for_meta_prompt, 1):
        tst = ex["test"]
        test_rec = ex["test_pool_rec"]
        user = USER_TMPL.format(
            test_q=tst["question"],
            test_gt=tst["ground_truth"],
            test_wrong=tst["openended_answer"],
            test_err=tst["primary_error"],
            test_note=test_rec["note"],
            group_a="\n".join([f"[A{i+1}]\n{fmt_candidate(c)}" for i, c in enumerate(ex["a"])]) or "  (none available)",
            group_b="\n".join([f"[B{i+1}]\n{fmt_candidate(c)}" for i, c in enumerate(ex["b"])]) or "  (none available)",
            group_c="\n".join([f"[C{i+1}]\n{fmt_candidate(c)}" for i, c in enumerate(ex["c"])]) or "  (none available)",
        )
        print(f"\n[meta-prompt {i}/3]  test_pid={tst['patient_id']}  err={tst['primary_error']}  prompt_len_chars={len(user)}")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=1500,
        )
        usage = resp.usage
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens
        raw = resp.choices[0].message.content or ""
        # Try to parse JSON from the response
        try:
            j = raw.strip()
            if j.startswith("```"):
                j = j.split("```")[1].lstrip("json").strip()
            parsed = json.loads(j)
        except Exception as e:
            parsed = {"_parse_error": str(e), "raw": raw[:500]}
        rubrics.append({
            "iter": i, "test_pid": tst["patient_id"], "test_err": tst["primary_error"],
            "rubric": parsed, "raw_response": raw,
            "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens
        })
        print(f"  parsed dimensions: {len(parsed) if isinstance(parsed, list) else 'parse_error'}")
        if isinstance(parsed, list):
            for d in parsed:
                if isinstance(d, dict):
                    print(f"    - {d.get('name', '?')}: {d.get('definition', '?')[:80]}")

    cost = total_in * 5e-6 + total_out * 1.5e-5
    out_data = {"rubrics": rubrics, "total_prompt_tokens": total_in,
                "total_completion_tokens": total_out, "cost_usd": round(cost, 4)}
    OUT.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"\nDONE  cost ≈ ${cost:.3f}  saved: {OUT}")


if __name__ == "__main__":
    main()
