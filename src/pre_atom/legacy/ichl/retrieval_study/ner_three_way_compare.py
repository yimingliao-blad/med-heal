"""Three-way NER comparison: Magistral vs Qwen3.6 vs Qwen3-235B (gold-ref).

For each pair:
  1. Phantom rate (entities not findable in source note)
  2. Distribution: items per category, zero counts, p95/max
  3. Jaccard agreement on overlap

Anchor question: does Qwen3.6 agree with Q3-235B more closely than Magistral does?
If yes, Qwen3.6 should be the new primary NER extractor.

Output: output/ichl/retrieval_study/ner_three_way_report.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
POOL_DIR = RS / "pool_index"
ITEMS = POOL_DIR / "items.jsonl"
MAG = POOL_DIR / "ner_magistral_962.jsonl"
QWEN36 = POOL_DIR / "ner_qwen36_962.jsonl"
Q3_50 = RS / "ner_qwen3-235b_sample50.jsonl"
Q3_30 = RS / "ner_qwen3_targeted30.jsonl"
OUT = RS / "ner_three_way_report.json"

CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip()).strip(".,;:")


def get_set(rec: dict, cat: str = None) -> set:
    e = rec.get("entities", {})
    if not isinstance(e, dict) or "_err" in e: return set()
    out = set()
    cats = [cat] if cat else CATS
    for c in cats:
        v = e.get(c, [])
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str): out.add(normalize(x))
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(len(a | b), 1)


def phantom_check(note: str, entity: str) -> bool:
    note_lower = note.lower()
    e = entity.lower().strip()
    if not e: return True
    head = re.split(r"[—\-=:|/]", e, maxsplit=1)[0].strip() or e
    if head in note_lower: return False
    fw = head.split()[0] if head.split() else head
    if len(fw) >= 4 and fw in note_lower: return False
    if "=" in entity:
        ln = entity.split("=")[0].strip().lower()
        if ln and len(ln) >= 3 and ln in note_lower: return False
    return True


def phantom_stats(rows: list[dict], items_by_rid: dict) -> dict:
    per_cat_phantoms = {c: [] for c in CATS}
    per_cat_totals = {c: [] for c in CATS}
    for r in rows:
        e = r.get("entities", {})
        if not isinstance(e, dict) or "_err" in e: continue
        rid = int(r["row_id"])
        if rid not in items_by_rid: continue
        note = items_by_rid[rid]["note_text_truncated"]
        for c in CATS:
            v = e.get(c, [])
            if not isinstance(v, list): continue
            cph = sum(1 for x in v if isinstance(x, str) and phantom_check(note, x))
            per_cat_phantoms[c].append(cph)
            per_cat_totals[c].append(len(v))
    out = {}
    overall_p, overall_t = 0, 0
    for c in CATS:
        t = sum(per_cat_totals[c]); p = sum(per_cat_phantoms[c])
        out[c] = {"total": t, "phantoms": p, "rate": round(p/max(t,1), 4),
                  "mean_per_item": round(np.mean(per_cat_totals[c]), 2) if per_cat_totals[c] else 0}
        overall_p += p; overall_t += t
    out["OVERALL"] = {"total": overall_t, "phantoms": overall_p, "rate": round(overall_p/max(overall_t,1), 4)}
    return out


def pairwise_jaccard(rows_a: list[dict], rows_b: list[dict]) -> dict:
    a_by_rid = {int(r["row_id"]): r for r in rows_a if isinstance(r.get("entities"), dict) and "_err" not in r["entities"]}
    b_by_rid = {int(r["row_id"]): r for r in rows_b if isinstance(r.get("entities"), dict) and "_err" not in r["entities"]}
    overlap = sorted(set(a_by_rid.keys()) & set(b_by_rid.keys()))
    js = []; per_cat = {c: [] for c in CATS}
    for rid in overlap:
        ra = a_by_rid[rid]; rb = b_by_rid[rid]
        sa = get_set(ra); sb = get_set(rb)
        if sa or sb: js.append(jaccard(sa, sb))
        for c in CATS:
            sca = get_set(ra, c); scb = get_set(rb, c)
            if sca or scb:
                per_cat[c].append(jaccard(sca, scb))
    return {
        "n_overlap": len(overlap),
        "overall_mean": float(np.mean(js)) if js else 0,
        "overall_median": float(np.median(js)) if js else 0,
        "overall_std": float(np.std(js)) if js else 0,
        "per_cat_mean": {c: float(np.mean(per_cat[c])) if per_cat[c] else 0 for c in CATS},
    }


def main():
    print("Loading items + 3 NER sets...")
    items = {int(json.loads(l)["row_id"]): json.loads(l) for l in ITEMS.open()}
    mag = [json.loads(l) for l in MAG.open() if l.strip()]
    qwen36 = [json.loads(l) for l in QWEN36.open() if l.strip()] if QWEN36.exists() else []
    q3_50 = [json.loads(l) for l in Q3_50.open() if l.strip()] if Q3_50.exists() else []
    q3_30 = [json.loads(l) for l in Q3_30.open() if l.strip()] if Q3_30.exists() else []
    # Combine Q3 samples (note: Q3_50 was indexed by old fold_0/train row_ids — need to remap by patient_id)
    # Q3_30 uses global row_ids (post-Phase 1)
    # For Q3_50, look up the row_id in current items by patient_id
    q3_50_remapped = []
    item_pid_to_rid = {int(items[rid]["patient_id"]): rid for rid in items}
    for r in q3_50:
        pid = int(r["patient_id"])
        if pid in item_pid_to_rid:
            new_r = dict(r)
            new_r["row_id"] = item_pid_to_rid[pid]
            q3_50_remapped.append(new_r)
    q3_combined = q3_50_remapped + q3_30
    # Dedup by row_id (q3_30 takes precedence if duplicate)
    seen = set()
    q3_dedup = []
    for r in reversed(q3_combined):  # q3_30 first
        rid = int(r["row_id"])
        if rid in seen: continue
        seen.add(rid)
        q3_dedup.append(r)
    print(f"  Magistral: {len(mag)}  Qwen3.6: {len(qwen36)}  Q3-235B (combined): {len(q3_dedup)}")

    # === Phantom rates ===
    print("\n=== Phantom rate per extractor ===")
    print(f"  {'extractor':12s}  {'OVERALL':>12s}  " + "  ".join(f"{c:>10s}" for c in CATS))
    for label, rows in [("Magistral", mag), ("Qwen3.6", qwen36), ("Q3-235B", q3_dedup)]:
        if not rows: continue
        ps = phantom_stats(rows, items)
        cells = [f"{ps['OVERALL']['rate']*100:>10.2f}%"] + [f"{ps[c]['rate']*100:>9.2f}%" for c in CATS]
        print(f"  {label:12s}  " + "  ".join(cells))

    # === Pairwise Jaccard agreement ===
    print("\n=== Pairwise Jaccard agreement ===")
    pairs = [
        ("Magistral", mag, "Qwen3.6", qwen36),
        ("Magistral", mag, "Q3-235B", q3_dedup),
        ("Qwen3.6", qwen36, "Q3-235B", q3_dedup),
    ]
    pairwise_results = {}
    for la, ra, lb, rb in pairs:
        if not ra or not rb: continue
        res = pairwise_jaccard(ra, rb)
        pairwise_results[f"{la} vs {lb}"] = res
        print(f"\n  {la} vs {lb}: n={res['n_overlap']}  mean={res['overall_mean']:.3f}  med={res['overall_median']:.3f}  std={res['overall_std']:.3f}")
        print(f"    per-cat: " + "  ".join(f"{c[:4]}={res['per_cat_mean'][c]:.3f}" for c in CATS))

    # === Distribution per extractor ===
    print("\n=== Distribution per extractor (mean entities per category) ===")
    print(f"  {'extractor':12s}  " + "  ".join(f"{c:>10s}" for c in CATS))
    distros = {}
    for label, rows in [("Magistral", mag), ("Qwen3.6", qwen36), ("Q3-235B", q3_dedup)]:
        if not rows: continue
        per_cat = {c: [] for c in CATS}
        for r in rows:
            e = r.get("entities", {})
            if not isinstance(e, dict) or "_err" in e: continue
            for c in CATS:
                v = e.get(c, [])
                if isinstance(v, list): per_cat[c].append(len(v))
        cells = [f"{np.mean(per_cat[c]):>9.2f}" for c in CATS]
        distros[label] = {c: {"mean": float(np.mean(per_cat[c])), "median": float(np.median(per_cat[c])),
                              "max": int(max(per_cat[c])) if per_cat[c] else 0,
                              "zero_count": int(sum(1 for x in per_cat[c] if x == 0))} for c in CATS}
        print(f"  {label:12s}  " + "  ".join(cells))

    # === Save ===
    report = {
        "n": {"Magistral": len(mag), "Qwen3.6": len(qwen36), "Q3-235B_combined": len(q3_dedup)},
        "phantom_rates": {
            label: phantom_stats(rows, items)
            for label, rows in [("Magistral", mag), ("Qwen3.6", qwen36), ("Q3-235B", q3_dedup)]
            if rows
        },
        "pairwise_jaccard": pairwise_results,
        "distributions": distros,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {OUT}")

    # === Decision indicator ===
    print("\n=== Decision indicator ===")
    if "Qwen3.6 vs Q3-235B" in pairwise_results and "Magistral vs Q3-235B" in pairwise_results:
        m_vs_q3 = pairwise_results["Magistral vs Q3-235B"]["overall_mean"]
        q36_vs_q3 = pairwise_results["Qwen3.6 vs Q3-235B"]["overall_mean"]
        print(f"  Magistral vs Q3-235B Jaccard: {m_vs_q3:.3f}")
        print(f"  Qwen3.6  vs Q3-235B Jaccard: {q36_vs_q3:.3f}")
        if q36_vs_q3 > m_vs_q3 + 0.03:
            print(f"  → Qwen3.6 is better aligned with Q3-235B (+{q36_vs_q3-m_vs_q3:.3f}). Recommend swap.")
        elif q36_vs_q3 < m_vs_q3 - 0.03:
            print(f"  → Magistral is better aligned with Q3-235B ({-(q36_vs_q3-m_vs_q3):.3f}). Keep Magistral.")
        else:
            print(f"  → Roughly tied (Δ={q36_vs_q3-m_vs_q3:+.3f}). Pick by phantom rate or distribution.")


if __name__ == "__main__":
    main()
