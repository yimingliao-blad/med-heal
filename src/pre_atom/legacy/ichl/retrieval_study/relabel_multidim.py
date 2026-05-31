"""Step 2b: re-label the 600 (anchor, candidate) pairs with 5-dimensional scores.

Uses the rubric elicited in elicit_rubric.py. Each pair gets a JSON object with
5 scores 0-3, in a single GPT-4o call.

Anchor side now also shows the BM wrong answer + primary error type when available
(so the judge can score "would this candidate help correct THIS specific error").

Output: output/ichl/retrieval_study/gold_pairs_multidim.jsonl
Cost estimate: 600 pairs × ~1500 prompt tok × $5/1M + 600 × ~80 output × $15/1M ≈ $5.20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
ERR_FILE = ROOT / "output" / "step8" / "error_classification" / "all_errors_by_patient.json"
PAIRS_DESIGN = ROOT / "output" / "ichl" / "retrieval_study" / "gold_pairs.jsonl"  # has anchor/candidate row_ids
OUT = ROOT / "output" / "ichl" / "retrieval_study" / "gold_pairs_multidim.jsonl"

DIMENSIONS = [
    {"name": "question_type_match",
     "definition": "Does the candidate ask the same kind of question (extraction vs reasoning, same focus area)?",
     "scale": "0=different question type, 1=loosely related, 2=similar type, 3=same type"},
    {"name": "error_type_match",
     "definition": "Would the candidate help correct the SAME kind of error the model made on the test (e.g. omission, hallucination, reasoning failure, context confusion)?",
     "scale": "0=different error type / not applicable, 1=tangentially related error, 2=similar error pattern, 3=same error type"},
    {"name": "clinical_context_similarity",
     "definition": "How similar is the clinical context (specialty, condition, setting, patient population)?",
     "scale": "0=different specialty/context, 1=same broad domain, 2=similar context, 3=very similar context"},
    {"name": "ground_truth_alignment",
     "definition": "Does the candidate's ground-truth answer contain information patterns that would help address the test's error?",
     "scale": "0=ground truth irrelevant, 1=loosely useful, 2=informative, 3=directly addresses the test's error pattern"},
    {"name": "critical_detail_overlap",
     "definition": "Do the candidate and test share specific clinical details (medications, doses, procedures, lab values, timing) that the test question hinges on?",
     "scale": "0=no shared specifics, 1=one or two superficial overlaps, 2=meaningful overlap, 3=many shared critical details"},
]

SYSTEM = (
    "You are a senior clinician scoring whether one clinical case (CANDIDATE) would help "
    "an AI model correct a wrong answer it gave on a different clinical case (TEST). "
    "You will score the pair on 5 independent dimensions, each 0-3. Higher = more useful "
    "for correction on that dimension."
)

USER_TMPL = """# TEST CASE (the AI got this wrong)
QUESTION: {test_q}
GROUND TRUTH: {test_gt}
{test_wrong_block}{test_err_block}NOTE EXCERPT: {test_note}

# CANDIDATE EXAMPLE (potential in-context aid)
QUESTION: {pool_q}
GROUND TRUTH: {pool_gt}
NOTE EXCERPT: {pool_note}

# DIMENSIONS TO SCORE (each 0-3)
{rubric_block}

# YOUR TASK
Score the pair on each dimension. Respond with ONLY a JSON object, no prose:
{{
  "question_type_match": 0|1|2|3,
  "error_type_match": 0|1|2|3,
  "clinical_context_similarity": 0|1|2|3,
  "ground_truth_alignment": 0|1|2|3,
  "critical_detail_overlap": 0|1|2|3
}}"""


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        pid = int(r["patient_id"])
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        # FULL note in step8 [Note i] format. NEVER truncate per [Workflow] No Silent
        # Truncation. Earlier `[:900]` cap was a calibration bug (gold pair scoring on
        # excerpts produced biased gold; fixed 2026-04-27).
        note = "\n\n".join(f"[Note {j}]\n{str(r.get(f'note_{j}','')).strip()}"
                           for j in [1, 2, 3]
                           if r.get(f"note_{j}") and str(r.get(f"note_{j}")).strip()
                           and str(r.get(f"note_{j}")).lower() != "nan")
        out.append({"row_id": i, "patient_id": pid, "category": r.get("category", ""),
                    "question": str(r["question"]), "ground_truth": gt, "note_excerpt": note})
    return out


def load_errors() -> dict[int, dict]:
    arr = json.loads(ERR_FILE.read_text())
    return {int(r["patient_id"]): r for r in arr}


def build_rubric_block() -> str:
    lines = []
    for i, d in enumerate(DIMENSIONS, 1):
        lines.append(f"{i}. {d['name']}")
        lines.append(f"   {d['definition']}")
        lines.append(f"   Scale: {d['scale']}")
    return "\n".join(lines)


def gpt4o_relabel(client, anchor: dict, candidate: dict, anchor_err: dict | None,
                  rubric_block: str, max_retries: int = 3) -> dict:
    test_wrong_block = ""
    test_err_block = ""
    if anchor_err is not None:
        test_wrong_block = f"MODEL'S WRONG ANSWER: {anchor_err['openended_answer']}\n"
        test_err_block = f"PRIMARY ERROR TYPE (taxonomy): {anchor_err['primary_error']}\n"
    user = USER_TMPL.format(
        test_q=anchor["question"], test_gt=anchor["ground_truth"],
        test_wrong_block=test_wrong_block, test_err_block=test_err_block,
        test_note=anchor["note_excerpt"],
        pool_q=candidate["question"], pool_gt=candidate["ground_truth"],
        pool_note=candidate["note_excerpt"],
        rubric_block=rubric_block,
    )
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=120,
            )
            lat = time.monotonic() - t0
            txt = resp.choices[0].message.content or ""
            # Strip code fences if any
            j = txt.strip()
            if j.startswith("```"):
                j = j.split("```")[1].lstrip("json").strip()
            try:
                parsed = json.loads(j)
            except Exception:
                # Try to extract JSON object substring
                m = re.search(r"\{[^{}]*\}", j, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {"_parse_error": True, "raw": j[:200]}
            usage = resp.usage
            return {
                "scores": parsed,
                "raw": txt[:200],
                "latency_s": round(lat, 2),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "anchor_had_error_label": anchor_err is not None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 + attempt * 3)
            else:
                return {"_error": str(e)[:200]}
    return {"_error": "max_retries"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 600 pairs")
    args = ap.parse_args()

    print("Loading pool, errors, prior pair design...")
    pool = load_pool()
    errors = load_errors()
    pairs = [json.loads(l) for l in PAIRS_DESIGN.open() if l.strip()]
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f"  pool={len(pool)}  err_labels={len(errors)}  pairs={len(pairs)}")

    rubric_block = build_rubric_block()
    print(f"  rubric block size: {len(rubric_block)} chars")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_path = ROOT / ".env"
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip(); break
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def label_one(pair: dict) -> dict:
        a = pool[pair["anchor_row_id"]]
        c = pool[pair["candidate_row_id"]]
        anchor_err = errors.get(a["patient_id"])
        result = gpt4o_relabel(client, a, c, anchor_err, rubric_block)
        return {**pair, **result}

    print(f"\nLabeling {len(pairs)} pairs with 5-dim rubric, {args.workers} workers...")
    t0 = time.monotonic()
    n_done = 0; n_errors = 0; total_in = 0; total_out = 0
    with OUT.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(label_one, pairs), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            n_done += 1
            if "_error" in r or (isinstance(r.get("scores"), dict) and r["scores"].get("_parse_error")):
                n_errors += 1
            total_in += r.get("prompt_tokens", 0) or 0
            total_out += r.get("completion_tokens", 0) or 0
            if i % 50 == 0:
                dt = time.monotonic() - t0
                eta = dt * (len(pairs) - i) / i
                cost = total_in * 5e-6 + total_out * 1.5e-5
                print(f"  {i}/{len(pairs)}  elapsed={dt:.0f}s  eta={eta:.0f}s  errors={n_errors}  cost~${cost:.2f}")
    elapsed = time.monotonic() - t0
    cost = total_in * 5e-6 + total_out * 1.5e-5
    print(f"\nDONE in {elapsed:.0f}s  errors={n_errors}/{n_done}  cost=${cost:.3f}")

    # Quick distribution
    rows = [json.loads(l) for l in OUT.open() if l.strip()]
    from collections import Counter
    print("\nPer-dimension score distribution:")
    for d in DIMENSIONS:
        dn = d["name"]
        scores = [r.get("scores", {}).get(dn) for r in rows if isinstance(r.get("scores"), dict)]
        scores = [s for s in scores if s is not None]
        c = Counter(scores)
        print(f"  {dn:35s}  {dict(sorted(c.items()))}")
    n_with_err_anchor = sum(1 for r in rows if r.get("anchor_had_error_label"))
    print(f"\nAnchors with error labels (BM-error subset): {n_with_err_anchor}/{len(rows)}")


if __name__ == "__main__":
    main()
