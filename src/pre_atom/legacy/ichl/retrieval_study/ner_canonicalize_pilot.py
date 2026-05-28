"""Pilot: canonicalize GPT-4o-mini extracted entity strings to UMLS CUIs.

Goal: test whether mapping strings -> CUIs via SciSpacy's UMLS linker
yields meaningfully better agreement (Jaccard) with Q3-235B than raw-string
Jaccard does. Only worth building a full 962-item canonical index if the
pilot shows a clear lift.

Method:
  1. For the 30 row_ids in `ner_qwen3_targeted30.jsonl`:
     - Get GPT-4o-mini entity strings (per category)
     - Get Q3-235B entity strings (per category)
  2. For each string, run scispacy CandidateGenerator and take the top-1
     CUI when similarity > threshold (default 0.85). Otherwise drop.
  3. Compute Jaccard on raw normalized strings vs Jaccard on CUI sets.

Run from project root with the dedicated scispacy venv:
    .venv-scispacy/bin/python src/ichl/retrieval_study/ner_canonicalize_pilot.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
GPT_FILE = RS / "pool_index" / "ner_gpt4o_mini_962.jsonl"
Q3_FILE = RS / "ner_qwen3_targeted30.jsonl"
OUT = RS / "ner_canonicalize_pilot.json"

CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]
SIM_THRESHOLD = 0.85  # min similarity to accept a CUI mapping


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip()).strip(".,;:")


def strip_for_lookup(s: str) -> str:
    """For 'doses' / 'lab_values' format like 'metformin — 500 mg — PO — BID'
    or 'creatinine = 2.1', take the head term (drug or test name)."""
    s = s.strip()
    head = re.split(r"[—\-=:|/]", s, maxsplit=1)[0].strip()
    return head if head else s


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def get_strings(rec: dict) -> dict[str, list[str]]:
    """Return {cat: [normalized strings]}. Uses head term for doses/labs."""
    e = rec.get("entities", {})
    if not isinstance(e, dict) or "_err" in e:
        return {c: [] for c in CATS}
    out = {c: [] for c in CATS}
    for c in CATS:
        v = e.get(c, [])
        if not isinstance(v, list):
            continue
        for x in v:
            if not isinstance(x, str):
                continue
            head = strip_for_lookup(x) if c in ("doses", "lab_values") else x
            n = normalize(head)
            if n:
                out[c].append(n)
    return out


def main():
    print("Loading SciSpacy CandidateGenerator (UMLS)...")
    t0 = time.monotonic()
    from scispacy.candidate_generation import CandidateGenerator
    cg = CandidateGenerator(name="umls")
    print(f"  loaded in {time.monotonic()-t0:.1f}s")

    print(f"\nLoading GPT-4o-mini ({GPT_FILE.name}) and Q3-235B ({Q3_FILE.name})...")
    gpt = {int(json.loads(l)["row_id"]): json.loads(l) for l in GPT_FILE.open() if l.strip()}
    q3 = {int(json.loads(l)["row_id"]): json.loads(l) for l in Q3_FILE.open() if l.strip()}
    overlap = sorted(set(gpt) & set(q3))
    print(f"  {len(overlap)} row_ids in both files")

    # Build the universe of strings to canonicalize across both extractors.
    # CandidateGenerator does the embedding+ANN lookup in batch; ~much faster.
    all_strs = set()
    for rid in overlap:
        for c, lst in get_strings(gpt[rid]).items():
            all_strs.update(lst)
        for c, lst in get_strings(q3[rid]).items():
            all_strs.update(lst)
    all_strs_list = sorted(all_strs)
    print(f"\nCanonicalizing {len(all_strs_list)} unique strings -> CUIs...")
    t0 = time.monotonic()
    # CandidateGenerator.__call__(mention_texts: List[str], k: int) -> List[List[MentionCandidate]]
    candidates = cg(all_strs_list, 1)  # top-1 each
    str2cui: dict[str, str | None] = {}
    n_mapped = 0
    for s, cands in zip(all_strs_list, candidates):
        if cands and cands[0].similarities and cands[0].similarities[0] >= SIM_THRESHOLD:
            str2cui[s] = cands[0].concept_id
            n_mapped += 1
        else:
            str2cui[s] = None
    print(f"  done in {time.monotonic()-t0:.1f}s  mapped={n_mapped}/{len(all_strs_list)} "
          f"({100*n_mapped/max(len(all_strs_list),1):.1f}% at sim>={SIM_THRESHOLD})")

    # Per-item Jaccard: strings vs CUIs
    rows = []
    for rid in overlap:
        gpt_strs = get_strings(gpt[rid])
        q3_strs = get_strings(q3[rid])
        # union across categories
        gpt_set = set()
        for c in CATS:
            gpt_set.update(gpt_strs[c])
        q3_set = set()
        for c in CATS:
            q3_set.update(q3_strs[c])

        # CUI versions: drop strings without CUI; otherwise replace with CUI
        gpt_cui = {str2cui[s] for s in gpt_set if str2cui.get(s)}
        q3_cui = {str2cui[s] for s in q3_set if str2cui.get(s)}

        rows.append({
            "row_id": rid,
            "n_gpt_str": len(gpt_set), "n_q3_str": len(q3_set),
            "n_gpt_cui": len(gpt_cui), "n_q3_cui": len(q3_cui),
            "j_string": round(jaccard(gpt_set, q3_set), 4),
            "j_cui": round(jaccard(gpt_cui, q3_cui), 4),
        })

    js_str = [r["j_string"] for r in rows]
    js_cui = [r["j_cui"] for r in rows]
    delta = [c - s for s, c in zip(js_str, js_cui)]

    print("\n=== 30-item Jaccard: GPT-4o-mini vs Q3-235B ===")
    print(f"  raw strings : mean={np.mean(js_str):.3f}  median={np.median(js_str):.3f}  std={np.std(js_str):.3f}")
    print(f"  UMLS CUIs   : mean={np.mean(js_cui):.3f}  median={np.median(js_cui):.3f}  std={np.std(js_cui):.3f}")
    print(f"  delta (CUI - str) : mean={np.mean(delta):+.3f}  median={np.median(delta):+.3f}")
    n_better = sum(1 for d in delta if d > 0.001)
    n_worse = sum(1 for d in delta if d < -0.001)
    n_tie = len(delta) - n_better - n_worse
    print(f"  per-item: CUI better in {n_better}/30, worse in {n_worse}/30, tied in {n_tie}/30")

    OUT.write_text(json.dumps({
        "n": len(rows),
        "sim_threshold": SIM_THRESHOLD,
        "n_unique_strings": len(all_strs_list),
        "n_mapped_to_cui": n_mapped,
        "mean_jaccard_string": float(np.mean(js_str)),
        "mean_jaccard_cui": float(np.mean(js_cui)),
        "mean_delta": float(np.mean(delta)),
        "n_better": n_better, "n_worse": n_worse, "n_tie": n_tie,
        "per_item": rows,
    }, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
