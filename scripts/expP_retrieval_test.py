#!/usr/bin/env python3
"""Build the retrieval and benchmark it against the ORACLE hint's critical info.

User direction: GTR embedding retrieval from notes, queries = the decompose look-ups; also
compare a string-based retriever; and test whether our retrieval surfaces the same CRITICAL
information the oracle hint provides (the oracle hint gave ~60% correction). Build retrieval first.

Three retrievers per wrong case (real note sentences, no LLM hallucination):
  GTR_ours    : topk_spans(note, queries = decompose look-ups)        [deployable]
  STR_ours    : keyword-overlap retrieval, same queries               [deployable, machinery cmp]
  GTR_oracle  : topk_spans(note, queries = [gold answer, oracle hint]) [CEILING — uses gold]

Metric (mechanical, no judge): GTR-T5 cosine of the GOLD answer to the retrieved span set
(= "is the critical evidence in here?"). Compare ours vs the oracle ceiling, and span overlap.
If GTR_ours coverage ~ GTR_oracle, our look-up-driven retrieval reaches the oracle's critical info.

Reuses decompose look-ups already saved in runs/expO_decompose_locate/.../records.jsonl.
Usage: python scripts/expP_retrieval_test.py --k 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import topk_spans, split_sentences, get_embedder  # noqa

STOP = set("the a an of to in for and or is was were be been being with on at by during this that what how "
           "all any each both her his their its patient note notes look up lookup find which were".split())


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - STOP


def string_retrieve(note, queries, k):
    sents = split_sentences(note)
    qw = set()
    for q in queries:
        qw |= words(q)
    scored = [(len(words(s) & qw), s) for s in sents]
    scored.sort(key=lambda x: -x[0])
    return [s for sc, s in scored[:k] if sc > 0]


def coverage(gold, spans, emb):
    """max GTR cosine of the gold answer to the retrieved span set (is the critical info present?)."""
    if not spans:
        return 0.0
    g = emb.encode([gold], normalize_embeddings=True, show_progress_bar=False)
    sp = emb.encode(spans, normalize_embeddings=True, show_progress_bar=False)
    return float((sp @ g.T).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--expO", default=str(PROJECT_ROOT / "runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))
    args = ap.parse_args()
    looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open(args.expO))}
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    emb = get_embedder()
    wrong = [k for k in looked if rows[k]["stored_label"] == 0]
    print(f"retrieval benchmark on {len(wrong)} wrong cases, k={args.k}")
    agg = {"GTR_ours": [], "STR_ours": [], "GTR_oracle": [], "overlap": []}
    for key in wrong:
        row = rows[key]
        note, gold, oracle = row["note"], row["ground_truth"], row.get("oracle_error_description", "")
        queries = [ln.strip(" -*0123456789.").strip() for ln in looked[key].splitlines() if ln.strip()]
        queries = [q for q in queries if len(q) > 4][:15]
        g_ours = topk_spans(note, queries, k=args.k)
        s_ours = string_retrieve(note, queries, args.k)
        g_orac = topk_spans(note, [q for q in [gold, oracle] if q], k=args.k)
        go = {s["sentence"] for s in g_ours}
        oo = {s["sentence"] for s in g_orac}
        agg["GTR_ours"].append(coverage(gold, [s["sentence"] for s in g_ours], emb))
        agg["STR_ours"].append(coverage(gold, s_ours, emb))
        agg["GTR_oracle"].append(coverage(gold, [s["sentence"] for s in g_orac], emb))
        agg["overlap"].append(len(go & oo) / max(1, len(oo)))
    def m(x):
        return sum(x) / max(1, len(x))
    print(f"\n=== critical-info coverage (GTR cosine of GOLD to retrieved spans; higher = evidence present) ===")
    print(f"  GTR_oracle (gold-query, CEILING): {m(agg['GTR_oracle']):.3f}")
    print(f"  GTR_ours   (decompose look-ups) : {m(agg['GTR_ours']):.3f}   ({m(agg['GTR_ours'])/m(agg['GTR_oracle'])*100:.0f}% of ceiling)")
    print(f"  STR_ours   (keyword overlap)    : {m(agg['STR_ours']):.3f}")
    print(f"\n  span overlap GTR_ours ∩ GTR_oracle: {m(agg['overlap'])*100:.0f}% of the oracle's critical spans also retrieved by ours")
    # how many cases reach near-ceiling
    near = sum(1 for a, b in zip(agg["GTR_ours"], agg["GTR_oracle"]) if a >= b - 0.05)
    print(f"  cases where GTR_ours within 0.05 of ceiling: {near}/{len(wrong)} = {near/len(wrong)*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
