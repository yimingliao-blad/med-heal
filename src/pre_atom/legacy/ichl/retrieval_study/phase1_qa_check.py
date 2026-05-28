"""Quality-check Magistral's NER extractions:
  1. Phantom rate — fraction of extracted entities not found in source note (case-insensitive substring)
  2. Distribution check — flag items with extreme entity counts (over- or under-extraction)
  3. Cross-extractor agreement — if Qwen3-235B has a sample, compare Jaccard on the overlap

Output: output/ichl/retrieval_study/ner_qa_report.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = ROOT / "output" / "ichl" / "retrieval_study" / "pool_index"
ITEMS = POOL_DIR / "items.jsonl"
NER_FILE = POOL_DIR / "ner_magistral_962.jsonl"
QWEN_SAMPLE = ROOT / "output" / "ichl" / "retrieval_study" / "ner_qwen3-235b_sample50.jsonl"
OUT = ROOT / "output" / "ichl" / "retrieval_study" / "ner_qa_report.json"

CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip()).strip(".,;:")


def phantom_check(note: str, entity: str) -> bool:
    """Return True if entity 'looks like' it appears in the note (lenient)."""
    note_lower = note.lower()
    e = entity.lower().strip()
    if not e:
        return True  # empty entity is not a phantom
    # If entity has multiple parts (e.g. "metformin — 500 mg — PO — BID"), take the head term
    head = re.split(r"[—\-=:|/]", e, maxsplit=1)[0].strip()
    if not head: head = e
    # Try the head substring
    if head in note_lower:
        return False  # found
    # Try first word (often the drug name)
    first_word = head.split()[0] if head.split() else head
    if len(first_word) >= 4 and first_word in note_lower:
        return False  # found via first word
    # For lab_values like "creatinine = 2.1": try just "creatinine"
    if "=" in entity:
        lab_name = entity.split("=")[0].strip().lower()
        if lab_name and len(lab_name) >= 3 and lab_name in note_lower:
            return False
    return True  # phantom (not found by any partial match)


def main():
    print("Loading items + NER...")
    items = {int(json.loads(l)["row_id"]): json.loads(l) for l in ITEMS.open()}
    ner = [json.loads(l) for l in NER_FILE.open() if l.strip()]
    print(f"  {len(items)} items, {len(ner)} NER rows")

    # === 1. Phantom rate per category, per item ===
    per_cat_phantoms = {c: [] for c in CATS}
    per_cat_totals = {c: [] for c in CATS}
    item_phantom_counts = []
    item_total_counts = []

    for r in ner:
        rid = int(r["row_id"])
        e = r.get("entities", {})
        if not isinstance(e, dict) or "_err" in e:
            continue
        note = items[rid]["note_text_truncated"]
        item_phantoms = 0
        item_total = 0
        for c in CATS:
            v = e.get(c, [])
            if not isinstance(v, list):
                continue
            cat_phantoms = 0
            for entity in v:
                if isinstance(entity, str):
                    is_phantom = phantom_check(note, entity)
                    if is_phantom:
                        cat_phantoms += 1
                        item_phantoms += 1
                    item_total += 1
            per_cat_phantoms[c].append(cat_phantoms)
            per_cat_totals[c].append(len(v))
        item_phantom_counts.append(item_phantoms)
        item_total_counts.append(item_total)

    # === 2. Stats ===
    print("\n=== Phantom check (entities not findable in note) ===")
    print(f"  {'category':14s}  {'mean_extracted':>15s}  {'mean_phantoms':>14s}  {'phantom_rate':>13s}  {'max_phantoms':>13s}")
    cat_summary = {}
    for c in CATS:
        totals = per_cat_totals[c]
        phantoms = per_cat_phantoms[c]
        if not totals:
            continue
        mean_total = np.mean(totals)
        mean_phantom = np.mean(phantoms)
        rate = sum(phantoms) / max(sum(totals), 1)
        max_phantom = max(phantoms) if phantoms else 0
        cat_summary[c] = {
            "mean_extracted": round(mean_total, 2),
            "mean_phantoms": round(mean_phantom, 2),
            "phantom_rate": round(rate, 3),
            "max_phantoms_per_item": max_phantom,
            "total_extracted": sum(totals),
            "total_phantoms": sum(phantoms),
        }
        print(f"  {c:14s}  {mean_total:>15.2f}  {mean_phantom:>14.2f}  {rate*100:>12.1f}%  {max_phantom:>13d}")

    overall_total = sum(item_total_counts)
    overall_phantom = sum(item_phantom_counts)
    overall_rate = overall_phantom / max(overall_total, 1)
    print(f"\n  OVERALL: {overall_phantom}/{overall_total} = {overall_rate*100:.1f}% phantom rate")

    # Items with high phantom rate
    high_phantom_items = []
    for r in ner:
        rid = int(r["row_id"])
        e = r.get("entities", {})
        if not isinstance(e, dict) or "_err" in e: continue
        note = items[rid]["note_text_truncated"]
        ph = 0; tot = 0
        for c in CATS:
            v = e.get(c, [])
            if isinstance(v, list):
                for ent in v:
                    if isinstance(ent, str):
                        if phantom_check(note, ent): ph += 1
                        tot += 1
        if tot >= 10 and ph / tot > 0.3:  # >30% phantoms with at least 10 entities
            high_phantom_items.append({
                "row_id": rid, "patient_id": r["patient_id"],
                "phantom": ph, "total": tot, "rate": round(ph/tot, 3)
            })
    high_phantom_items.sort(key=lambda x: -x["rate"])
    print(f"\n  Items with >30% phantom rate (n={len(high_phantom_items)}):")
    for x in high_phantom_items[:10]:
        print(f"    row {x['row_id']}: {x['phantom']}/{x['total']} = {x['rate']*100:.1f}%")

    # === 3. Distribution: zero-extraction items per category ===
    print("\n=== Distribution check ===")
    print(f"  {'category':14s}  {'zero_count':>11s}  {'very_high (>20)':>16s}  {'mean':>6s}  {'p95':>6s}  {'max':>5s}")
    dist_summary = {}
    for c in CATS:
        totals = per_cat_totals[c]
        if not totals: continue
        zeros = sum(1 for t in totals if t == 0)
        very_high = sum(1 for t in totals if t > 20)
        p95 = int(np.percentile(totals, 95))
        dist_summary[c] = {"zeros": zeros, "very_high": very_high,
                           "mean": round(np.mean(totals), 2),
                           "p95": p95, "max": max(totals)}
        print(f"  {c:14s}  {zeros:>11d}  {very_high:>16d}  {np.mean(totals):>5.2f}  {p95:>5d}  {max(totals):>5d}")

    # === 4. Magistral vs Qwen3-235B agreement on Phase 1 sample ===
    print("\n=== Cross-extractor agreement (Magistral vs Qwen3-235B on 50-sample) ===")
    cross_agreement = None
    if QWEN_SAMPLE.exists():
        q3 = [json.loads(l) for l in QWEN_SAMPLE.open() if l.strip()]
        q3_by_pid = {r["patient_id"]: r for r in q3}
        mag_by_pid = {r["patient_id"]: r for r in ner}
        overlap = sorted(set(q3_by_pid.keys()) & set(mag_by_pid.keys()))
        if overlap:
            jaccards = []
            for pid in overlap:
                m_e = mag_by_pid[pid].get("entities", {})
                q_e = q3_by_pid[pid].get("entities", {})
                if not isinstance(m_e, dict) or not isinstance(q_e, dict): continue
                if "_err" in m_e or "_err" in q_e: continue
                m_set = set()
                q_set = set()
                for c in CATS:
                    for x in m_e.get(c, []):
                        if isinstance(x, str): m_set.add(normalize(x))
                    for x in q_e.get(c, []):
                        if isinstance(x, str): q_set.add(normalize(x))
                if m_set or q_set:
                    j = len(m_set & q_set) / max(len(m_set | q_set), 1)
                    jaccards.append(j)
            if jaccards:
                cross_agreement = {"n": len(jaccards), "mean_jaccard": round(np.mean(jaccards), 3),
                                    "median_jaccard": round(np.median(jaccards), 3),
                                    "std": round(np.std(jaccards), 3)}
                print(f"  n={len(jaccards)}  mean_jaccard={np.mean(jaccards):.3f}  median={np.median(jaccards):.3f}  std={np.std(jaccards):.3f}")

    # === 5. Save report ===
    report = {
        "n_items_checked": len([r for r in ner if isinstance(r.get("entities"), dict) and "_err" not in r["entities"]]),
        "overall_phantom_rate": round(overall_rate, 3),
        "overall_phantom_total": f"{overall_phantom}/{overall_total}",
        "per_category": cat_summary,
        "distribution_per_category": dist_summary,
        "items_with_high_phantom_rate": high_phantom_items,
        "cross_extractor_agreement": cross_agreement,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved: {OUT}")


if __name__ == "__main__":
    main()
