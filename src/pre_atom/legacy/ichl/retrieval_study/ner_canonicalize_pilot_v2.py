"""Pilot v2: hybrid CUI-or-string Jaccard with threshold sweep.

Each entity becomes a token in one of two namespaces:
  - "cui:C0011860"  if linker maps with sim >= threshold
  - "str:metformin" otherwise (normalized raw string)

Both extractors (GPT-4o-mini, Q3-235B) get the same treatment, so a string
that fails to link in BOTH still matches via the str: namespace; one that
succeeds in both matches via cui: namespace. This avoids the 5-of-30
regression we saw when CUI dropped means falling out of intersection.
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
GPT_FILE = RS / "pool_index" / "ner_gpt4o_mini_962.jsonl"
Q3_FILE = RS / "ner_qwen3_targeted30.jsonl"
OUT = RS / "ner_canonicalize_pilot_v2.json"

CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]


def normalize(s):
    return re.sub(r"\s+", " ", s.lower().strip()).strip(".,;:")


def strip_for_lookup(s, c):
    if c not in ("doses", "lab_values"):
        return s
    return re.split(r"[—\-=:|/]", s.strip(), maxsplit=1)[0].strip() or s


def get_strings(rec):
    e = rec.get("entities", {})
    if not isinstance(e, dict) or "_err" in e:
        return []
    out = []
    for c in CATS:
        v = e.get(c, [])
        if not isinstance(v, list): continue
        for x in v:
            if not isinstance(x, str): continue
            head = strip_for_lookup(x, c)
            n = normalize(head)
            if n:
                out.append(n)
    return out


def jaccard(a, b):
    if not a and not b: return 0.0
    return len(a & b) / max(len(a | b), 1)


def main():
    print("Loading CandidateGenerator...")
    t0 = time.monotonic()
    from scispacy.candidate_generation import CandidateGenerator
    cg = CandidateGenerator(name="umls")
    print(f"  {time.monotonic()-t0:.1f}s")

    gpt = {int(json.loads(l)["row_id"]): json.loads(l) for l in GPT_FILE.open() if l.strip()}
    q3  = {int(json.loads(l)["row_id"]): json.loads(l) for l in Q3_FILE.open() if l.strip()}
    overlap = sorted(set(gpt) & set(q3))
    print(f"  {len(overlap)} overlapping row_ids")

    universe = set()
    for rid in overlap:
        universe.update(get_strings(gpt[rid]))
        universe.update(get_strings(q3[rid]))
    universe_list = sorted(universe)
    print(f"\nLooking up {len(universe_list)} unique strings...")
    cands = cg(universe_list, 1)
    sim_by_str = {}
    cui_by_str = {}
    for s, cs in zip(universe_list, cands):
        if cs and cs[0].similarities:
            sim_by_str[s] = cs[0].similarities[0]
            cui_by_str[s] = cs[0].concept_id
        else:
            sim_by_str[s] = 0.0
            cui_by_str[s] = None

    def to_tokens(strs, threshold):
        toks = set()
        for s in strs:
            if cui_by_str.get(s) and sim_by_str.get(s, 0) >= threshold:
                toks.add(f"cui:{cui_by_str[s]}")
            else:
                toks.add(f"str:{s}")
        return toks

    print("\n=== Hybrid (CUI-or-string) Jaccard, threshold sweep ===")
    print(f"  {'threshold':>10s}  {'str-only':>10s}  {'hybrid':>10s}  {'delta':>8s}  {'better':>7s}  {'worse':>7s}")
    results = {}
    # baseline: pure string
    js_str = []
    for rid in overlap:
        a = set(get_strings(gpt[rid]))
        b = set(get_strings(q3[rid]))
        js_str.append(jaccard(a, b))
    base_mean = float(np.mean(js_str))

    for thr in [0.70, 0.75, 0.80, 0.85, 0.90]:
        js_h = []
        better = worse = 0
        for rid, base in zip(overlap, js_str):
            a = to_tokens(get_strings(gpt[rid]), thr)
            b = to_tokens(get_strings(q3[rid]), thr)
            j = jaccard(a, b)
            js_h.append(j)
            if j > base + 0.001: better += 1
            elif j < base - 0.001: worse += 1
        mean_h = float(np.mean(js_h))
        results[thr] = {"mean_hybrid": mean_h, "delta": mean_h - base_mean,
                         "better": better, "worse": worse}
        print(f"  {thr:>10.2f}  {base_mean:>10.3f}  {mean_h:>10.3f}  {mean_h-base_mean:>+8.3f}  "
              f"{better:>7d}  {worse:>7d}")

    # Best threshold details
    best_thr = max(results, key=lambda t: results[t]["mean_hybrid"])
    print(f"\nBest threshold: {best_thr:.2f}  hybrid_mean={results[best_thr]['mean_hybrid']:.3f}  "
          f"delta vs str-only={results[best_thr]['delta']:+.3f}")

    OUT.write_text(json.dumps({
        "n": len(overlap),
        "n_unique_strings": len(universe_list),
        "baseline_string_mean": base_mean,
        "by_threshold": {f"{k:.2f}": v for k, v in results.items()},
        "best_threshold": best_thr,
    }, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
