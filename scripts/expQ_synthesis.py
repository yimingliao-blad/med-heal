#!/usr/bin/env python3
"""SYNTHESIS — compose a correction guideline from retrieved evidence, benchmarked vs the oracle.

Pipeline: decompose look-ups -> GTR retrieve real note spans -> SYNTHESIZE a correction guideline
(what's wrong/missing, correct fact, decisive evidence) -> rate it against the oracle hint.

Two arms (user: test both):
  ONE  : one prompt (Q + ZS + look-ups + spans -> guideline)
  TWO  : two prompts (resolve each look-up from the spans, THEN compose the guideline from the resolved facts)

Benchmark (judge, existing): rate_diagnosis(guideline vs oracle_error_description) -> AGREE/PARTIAL/WRONG.
AGREE rate = does our synthesis identify the SAME error as the oracle hint (which gave ~60% correction).
Sanity: feeding the oracle itself should score AGREE.

Reuses decompose look-ups from runs/expO_decompose_locate/.
Usage: python scripts/expQ_synthesis.py --k 8 --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import topk_spans, get_embedder  # noqa
from llm_audit import set_ledger  # noqa

A_SYS = "You write a precise correction guideline for an answer, using ONLY the retrieved note evidence."
A_USER = """Question:
{question}

Answer given:
{answer}

Things that were checked:
{lookups}

Retrieved note evidence (real sentences from the discharge note):
{spans}

Using ONLY the retrieved evidence, write a SHORT correction guideline: state specifically what in the answer is WRONG or what required fact is MISSING, what the note-supported correct fact is, and quote the decisive evidence sentence. If the evidence shows the answer is already correct, write exactly: NO CORRECTION NEEDED."""

R_SYS = "You answer look-up questions using ONLY the provided note sentences."
R_USER = """Note sentences (real, from the discharge note):
{spans}

For each look-up below, state what the note says (quote the sentence), or "not stated":
{lookups}"""

B_SYS = A_SYS
B_USER = """Question:
{question}

Answer given:
{answer}

What the note actually says for each checked item:
{resolved}

Using ONLY this, write a SHORT correction guideline: what in the answer is WRONG or MISSING, the note-supported correct fact, and the decisive evidence. If the answer is already correct, write exactly: NO CORRECTION NEEDED."""


def render(spans):
    return "\n".join(f"[{i+1}] {s['sentence']}" for i, s in enumerate(spans)) if spans else "(none)"


def process_one(row, lookups_raw, k, port):
    out = {k2: row[k2] for k2 in ["fold", "idx", "stored_label"]}
    q, a = row["question"], row["original_answer"][:1500]
    queries = [ln.strip(" -*0123456789.").strip() for ln in lookups_raw.splitlines() if ln.strip()]
    queries = [x for x in queries if len(x) > 4][:15]
    spans = topk_spans(row["note"], queries, k=k)
    sp = render(spans)
    lk = "\n".join(f"- {x}" for x in queries)
    # ARM ONE
    one = P2.vllm_chat(A_SYS, A_USER.format(question=q, answer=a, lookups=lk, spans=sp), port, 500, 0.0, tag="syn.one")
    # ARM TWO
    resolved = P2.vllm_chat(R_SYS, R_USER.format(spans=sp, lookups=lk), port, 700, 0.0, tag="syn.resolve")
    two = P2.vllm_chat(B_SYS, B_USER.format(question=q, answer=a, resolved=resolved[:3000]), port, 500, 0.0, tag="syn.two")
    out["rate_one"] = P2.rate_diagnosis(row, one)
    out["rate_two"] = P2.rate_diagnosis(row, two)
    out["one"], out["two"] = one, two
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expO", default=str(PROJECT_ROOT / "runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))
    args = ap.parse_args()
    looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open(args.expO))}
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    wrong = [k for k in looked if rows[k]["stored_label"] == 0]
    get_embedder()  # pre-load GTR in main thread (avoid concurrent lazy-init meta-tensor race)
    out_dir = PROJECT_ROOT / "runs" / "expQ_synthesis" / "qwen25_nw40_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expQ_synthesis", served=P2.served_model_id(args.port))
    # sanity: oracle rated against itself
    s = rows[wrong[0]]
    print("SANITY rate_diagnosis(oracle vs oracle):", P2.rate_diagnosis(s, s["oracle_error_description"]), flush=True)
    print(f"synthesis on {len(wrong)} wrong cases, k={args.k}", flush=True)
    recs = []
    f = (out_dir / "records.jsonl").open("w")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[key], looked[key], args.k, args.port) for key in wrong]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); recs.append(r)
            f.write(json.dumps(r, default=str) + "\n"); f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)
    f.close()
    for arm in ["rate_one", "rate_two"]:
        c = Counter(r[arm] for r in recs)
        agree = c.get("AGREE", 0); partial = c.get("PARTIAL", 0); n = len(recs)
        print(f"\n{arm:9} AGREE {agree}/{n}={agree/n*100:.0f}%  PARTIAL {partial}  WRONG {c.get('WRONG',0)}  "
              f"(AGREE+PARTIAL {(agree+partial)/n*100:.0f}%)")
    print(f"\nsaved: {out_dir/'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
