#!/usr/bin/env python3
"""Stage 1 (corrected prompt) — run decompose and OBSERVE the raw look-up list.

Corrected per user: no 'reconstruct your reasoning' (LLM has no memory); just ask where the
evidence supporting the answer comes from -> a plain list of things to look up in the note.
Goal here is to EYEBALL whether the output surfaces all the useful look-up items (incl. the
ones needed to catch the known omission/value errors), then decide how to organize them
(externally vs in-prompt). No metric — just observation.

Usage: python scripts/expN2_decompose_observe.py --n-wrong 40 --n-correct 20 --show 8
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

DECOMP_SYS = "You identify what information in a discharge note an answer relies on, so it can be looked up."
DECOMP_USER = """Question:
{question}

Answer that was given:
{answer}

Think about where the evidence in the discharge note that would support this answer comes from: what would need to be stated in the note, and what each piece of evidence is about (which topic, event, or value).
Give a plain list of the things to look up in the note. Do not judge whether the answer is right."""


def process_one(row, port):
    out = {k: row[k] for k in ["fold", "idx", "stored_label", "question", "ground_truth"]}
    out["answer"] = row["original_answer"]
    out["lookup"] = P2.vllm_chat(DECOMP_SYS, DECOMP_USER.format(question=row["question"], answer=row["original_answer"][:1500]), port, 700, 0.0, tag="decompose2")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wrong", type=int, default=40)
    ap.add_argument("--n-correct", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = PROJECT_ROOT / "runs" / "expN2_decompose" / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expN2_decompose", served=P2.served_model_id(args.port))
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
    # OBSERVE: print full decompose output for the first --show WRONG cases
    W = [r for r in recs if r["stored_label"] == 0]
    print(f"\n##### OBSERVE: {args.show} wrong-case decompositions (does the lookup list cover the gold?) #####")
    for r in W[: args.show]:
        n_items = sum(1 for ln in r["lookup"].splitlines() if ln.strip() and (ln.strip()[0].isdigit() or ln.strip()[0] in "-*•"))
        print("=" * 95)
        print("Q   :", r["question"][:200])
        print("ANS :", r["answer"][:200])
        print("GOLD:", r["ground_truth"][:200])
        print(f"--- LOOKUP LIST ({n_items} items) ---")
        print(r["lookup"][:1400])
    print(f"\nsaved: {out_dir/'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
