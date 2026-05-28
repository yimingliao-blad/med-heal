"""Diagnose: which strings in the 30-item pilot fail to map to CUIs?
Helps decide whether to lower threshold, add fallback rules, or just accept the drop.
"""
from __future__ import annotations
import json, re, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
GPT_FILE = RS / "pool_index" / "ner_gpt4o_mini_962.jsonl"
Q3_FILE = RS / "ner_qwen3_targeted30.jsonl"

CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]

def normalize(s):
    return re.sub(r"\s+", " ", s.lower().strip()).strip(".,;:")

def strip_for_lookup(s, c):
    if c not in ("doses", "lab_values"):
        return s
    return re.split(r"[—\-=:|/]", s.strip(), maxsplit=1)[0].strip() or s

def main():
    print("Loading CandidateGenerator...")
    t0 = time.monotonic()
    from scispacy.candidate_generation import CandidateGenerator
    cg = CandidateGenerator(name="umls")
    print(f"  {time.monotonic()-t0:.1f}s")

    gpt = {int(json.loads(l)["row_id"]): json.loads(l) for l in GPT_FILE.open() if l.strip()}
    q3  = {int(json.loads(l)["row_id"]): json.loads(l) for l in Q3_FILE.open() if l.strip()}
    overlap = sorted(set(gpt) & set(q3))

    # Tag each string with its category for diagnostic
    tagged = []  # (cat, original, head_for_lookup)
    seen = set()
    for rid in overlap:
        for src in (gpt[rid], q3[rid]):
            e = src.get("entities", {})
            if not isinstance(e, dict) or "_err" in e: continue
            for c in CATS:
                v = e.get(c, [])
                if not isinstance(v, list): continue
                for x in v:
                    if not isinstance(x, str): continue
                    head = strip_for_lookup(x, c)
                    n = normalize(head)
                    if n and n not in seen:
                        seen.add(n)
                        tagged.append((c, x, n))

    strs = [t[2] for t in tagged]
    print(f"\nLooking up {len(strs)} strings...")
    cands = cg(strs, 1)
    rows = []
    for (cat, orig, lookup), cs in zip(tagged, cands):
        sim = cs[0].similarities[0] if cs and cs[0].similarities else 0.0
        cui = cs[0].concept_id if cs else None
        rows.append({"cat": cat, "orig": orig, "lookup": lookup, "sim": sim, "cui": cui})

    by_cat_total = Counter(r["cat"] for r in rows)
    by_cat_mapped = Counter(r["cat"] for r in rows if r["sim"] >= 0.85)
    by_cat_low = Counter(r["cat"] for r in rows if r["sim"] < 0.85)

    print("\n=== Mapping rate by category (sim >= 0.85) ===")
    for c in CATS:
        m, t = by_cat_mapped[c], by_cat_total[c]
        print(f"  {c:14s}  {m:>3d}/{t:<3d}  ({100*m/max(t,1):.1f}%)")

    # Sample unmapped strings per category
    print("\n=== Sample UNMAPPED strings (sim < 0.85) per category ===")
    for c in CATS:
        unmapped = [r for r in rows if r["cat"] == c and r["sim"] < 0.85]
        if not unmapped: continue
        print(f"\n  {c} ({len(unmapped)} unmapped):")
        for r in sorted(unmapped, key=lambda x: -x["sim"])[:6]:
            print(f"    sim={r['sim']:.3f}  '{r['lookup'][:60]}'  (orig: '{r['orig'][:60]}')")

    # Threshold sweep
    print("\n=== Mapping rate by threshold ===")
    for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        n = sum(1 for r in rows if r["sim"] >= t)
        print(f"  sim >= {t:.2f}  ->  {n}/{len(rows)} = {100*n/max(len(rows),1):.1f}%")


if __name__ == "__main__":
    main()
