#!/usr/bin/env python3
"""Verify the EXTRACT stage only: does the normalized fact list CONTAIN the needed information?

Per user: the extract is NOT an omit stage — it constructs a normalized list, and we check that
list (before any QA) for whether it covers the critical information. Two checks:
  1. EMBEDDING similarity (mechanical): GTR cosine of the normalized list to the gold answer and to
     the oracle hint — do the semantics match / is it covered?
  2. READ check (gpt-4o-mini): does the list CONTAIN all the information needed to produce the gold
     answer? YES / PARTIAL / NO.
Plus print a few lists next to gold+oracle so we can read them ourselves.

Reads the normalized list ('facts') from runs/expR_extract_qa/.
Usage: python scripts/expR_coverage.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import get_embedder  # noqa

REC = PROJECT_ROOT / "runs/expR_extract_qa/qwen25_k10_seed42/records.jsonl"

READ_SYS = "You check whether a list of facts contains the information needed to answer a question correctly."
READ_USER = """Question:
{q}

The correct answer:
{gold}

A list of facts extracted from the note:
{facts}

Does this fact list CONTAIN the information needed to produce the correct answer (even if worded differently)? Reply one word: YES (all needed info present), PARTIAL (some but not all), or NO (the key info is missing)."""


def main():
    recs = [json.loads(l) for l in open(REC)]
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    emb = get_embedder()
    W = [r for r in recs if r["stored_label"] == 0]
    import numpy as np
    sims_gold, sims_oracle = [], []
    reads = Counter()
    examples = []
    for r in W:
        s = rows[(r["fold"], r["idx"])]
        facts = r["facts"]
        gold = s["ground_truth"]
        oracle = s.get("oracle_error_description", "")
        fv = emb.encode([facts], normalize_embeddings=True, show_progress_bar=False)
        gv = emb.encode([gold], normalize_embeddings=True, show_progress_bar=False)
        sims_gold.append(float((fv @ gv.T)[0, 0]))
        if oracle:
            ov = emb.encode([oracle], normalize_embeddings=True, show_progress_bar=False)
            sims_oracle.append(float((fv @ ov.T)[0, 0]))
        rd = P2.gpt("gpt-4o-mini", READ_SYS, READ_USER.format(q=s["question"][:300], gold=gold[:400], facts=facts[:3000]), 6, 0.0, False, "cover.read").upper()
        tag = "YES" if "YES" in rd else ("PARTIAL" if "PARTIAL" in rd else ("NO" if "NO" in rd else "?"))
        reads[tag] += 1
        if len(examples) < 4:
            examples.append((s["question"][:90], gold[:90], facts[:400], tag))
    n = len(W)
    print(f"=== EXTRACT list coverage on {n} wrong cases (list checked BEFORE QA) ===")
    print(f"embedding cosine: list vs GOLD   mean {sum(sims_gold)/n:.3f}")
    print(f"embedding cosine: list vs ORACLE mean {sum(sims_oracle)/max(1,len(sims_oracle)):.3f}")
    print(f"READ check (contains needed info): YES {reads['YES']}/{n}={reads['YES']/n*100:.0f}%  "
          f"PARTIAL {reads['PARTIAL']}  NO {reads['NO']}  (YES+PARTIAL {(reads['YES']+reads['PARTIAL'])/n*100:.0f}%)")
    print("\n--- examples (read them) ---")
    for q, g, f, t in examples:
        print(f"\n[{t}] Q: {q}\n  GOLD: {g}\n  LIST: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
