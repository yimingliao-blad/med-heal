#!/usr/bin/env python3
"""Two decompose arms run IN PARALLEL (not merged) so each signal stays separable.

Per user: two ways to improve Stage-1 decompose --
  ARM A: make it better at its OWN purpose (clean, complete extraction of the evidence the
         answer relies on; answer-anchored).
  ARM B: ENHANCE with the downstream requirement (add question-driven OPEN look-ups so
         omissions / wrong-frame answers become items the blind note-walk can resolve).
Run both separately and OBSERVE; merging initially could hide which change carries the signal.

Usage: python scripts/expN3_decompose_arms.py --n-wrong 40 --n-correct 20 --show 6
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa
from llm_audit import set_ledger  # noqa

# ARM A — better on its own purpose: clean, complete extraction of the answer's evidence look-ups
A_SYS = "You identify what information in a discharge note an answer relies on, so it can be looked up."
A_USER = """Question:
{question}

Answer that was given:
{answer}

Identify the evidence in the note that the answer relies on. For each distinct thing the answer states (each value, date, name, dose, count, finding), give one short look-up item naming what to find in the note and what it is about.
Output only a clean numbered list, one look-up item per line. Do not judge whether the answer is right."""

# ARM B — enhanced for downstream: add question-driven OPEN look-ups to catch omission / wrong-frame
B_SYS = "You identify what information in a discharge note one must look up to fully answer a question, so it can be checked."
B_USER = """Question:
{question}

Answer that was given:
{answer}

List what to look up in the note to FULLY answer the question. Cover BOTH:
- what the question asks for in its own right, written as OPEN look-ups (for example "all infections the patient was diagnosed with", "all treatments given during the stay"), so nothing the question requires is missed;
- the specific things the answer states.
Output only a clean numbered list, one short look-up phrase per line. Do not judge whether the answer is right."""


def process_one(row, port):
    out = {k: row[k] for k in ["fold", "idx", "stored_label", "question", "ground_truth"]}
    out["answer"] = row["original_answer"]
    out["armA"] = P2.vllm_chat(A_SYS, A_USER.format(question=row["question"], answer=row["original_answer"][:1500]), port, 700, 0.0, tag="decompose.A")
    out["armB"] = P2.vllm_chat(B_SYS, B_USER.format(question=row["question"], answer=row["original_answer"][:1500]), port, 700, 0.0, tag="decompose.B")
    return out


def nitems(s):
    return sum(1 for ln in s.splitlines() if ln.strip() and (ln.strip()[0].isdigit() or ln.strip()[0] in "-*•"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wrong", type=int, default=40)
    ap.add_argument("--n-correct", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = PROJECT_ROOT / "runs" / "expN3_decompose_arms" / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expN3_decompose_arms", served=P2.served_model_id(args.port))
    print(f"NOTE GUARD OK: {len(sample)} notes", flush=True)
    recs = []
    f = (out_dir / "records.jsonl").open("w")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); recs.append(r)
            f.write(json.dumps(r, default=str) + "\n"); f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)
    f.close()
    W = [r for r in recs if r["stored_label"] == 0]
    print(f"\n##### OBSERVE: Arm A (own purpose) vs Arm B (question-driven) on {args.show} wrong cases #####")
    for r in W[: args.show]:
        print("=" * 100)
        print("Q   :", r["question"][:170])
        print("ANS :", r["answer"][:150])
        print("GOLD:", r["ground_truth"][:170])
        print(f"--- ARM A ({nitems(r['armA'])} items) ---")
        print(r["armA"][:900])
        print(f"--- ARM B ({nitems(r['armB'])} items) ---")
        print(r["armB"][:900])
    print(f"\nsaved: {out_dir/'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
