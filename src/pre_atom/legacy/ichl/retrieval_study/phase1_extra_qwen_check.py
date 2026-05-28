"""Run Qwen3-235B NER on a targeted sample to validate Magistral's extractions.

Sample: 20 high-phantom items + 10 random unseen-by-Q3 items = 30 items total.
Compares per-item Jaccard, focusing on whether Q3 confirms Magistral's "phantoms".

Output: output/ichl/retrieval_study/ner_qwen3_targeted30.jsonl
        output/ichl/retrieval_study/ner_validation_report.json
"""
from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = ROOT / "output" / "ichl" / "retrieval_study" / "pool_index"
ITEMS_FILE = POOL_DIR / "items.jsonl"
NER_FILE = POOL_DIR / "ner_magistral_962.jsonl"
QA_REPORT = ROOT / "output" / "ichl" / "retrieval_study" / "ner_qa_report.json"
QWEN_EXISTING = ROOT / "output" / "ichl" / "retrieval_study" / "ner_qwen3-235b_sample50.jsonl"
OUT_NER = ROOT / "output" / "ichl" / "retrieval_study" / "ner_qwen3_targeted30.jsonl"
OUT_REPORT = ROOT / "output" / "ichl" / "retrieval_study" / "ner_validation_report.json"

QWEN3_URL = "http://192.168.68.107:8800/v1"
QWEN3_MODEL = "/Users/madblade/.lmstudio/models/lmstudio-community/Qwen3-235B-A22B-Instruct-2507-MLX-4bit"

CATS = ["medications", "doses", "procedures", "lab_values", "diagnoses"]

NER_USER_TMPL = """Extract clinical entities from this clinical note. Return a JSON object with EXACTLY these 5 keys:
- "medications": list of distinct drug/medication names (just names, no doses, no duplicates)
- "doses": list of "drug NAME — DOSE — ROUTE — FREQ" strings (e.g., "metformin — 500 mg — PO — BID"); use "?" for missing fields
- "procedures": list of distinct surgical, diagnostic, or therapeutic procedures
- "lab_values": list of "TEST = VALUE" strings (e.g., "creatinine = 2.1", "Hgb = 8.5")
- "diagnoses": list of distinct clinical conditions / diagnoses

Normalize entities to lowercase. De-duplicate. Keep entries terse and canonical (use generic drug names, not brand). If a category has no entities, return [] for it.

NOTE:
{note}

Respond with ONLY a JSON object, no prose."""


def parse_json(text: str) -> dict:
    if not text: return {"_err": "empty"}
    s = text.strip()
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m: s = m.group(1).strip()
    try: return json.loads(s)
    except: pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try: return json.loads(m.group(0))
        except Exception as e: return {"_err": f"parse_fail: {str(e)[:80]}", "_raw": s[:300]}
    return {"_err": "no_json", "_raw": s[:300]}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip()).strip(".,;:")


def get_set(rec: dict, cat: str = None) -> set:
    """Get normalized entity set; if cat is None, union across all categories."""
    e = rec.get("entities", {})
    if not isinstance(e, dict) or "_err" in e: return set()
    out = set()
    cats = [cat] if cat else CATS
    for c in cats:
        v = e.get(c, [])
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    out.add(normalize(x))
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(len(a | b), 1)


def main():
    print("Loading items + existing extractions...")
    items = {int(json.loads(l)["row_id"]): json.loads(l) for l in ITEMS_FILE.open()}
    mag_ner = {int(json.loads(l)["row_id"]): json.loads(l) for l in NER_FILE.open()}

    # Existing Q3 50-sample (use to avoid duplicates)
    existing_q3_pids = set()
    if QWEN_EXISTING.exists():
        for l in QWEN_EXISTING.open():
            r = json.loads(l)
            existing_q3_pids.add(int(r["patient_id"]))
    print(f"  existing Q3 sample covers {len(existing_q3_pids)} pids")

    # Read QA report for high-phantom items
    qa = json.loads(QA_REPORT.read_text())
    high_phantom = qa.get("items_with_high_phantom_rate", [])
    print(f"  high-phantom items in QA report: {len(high_phantom)}")

    # Target sample: 20 high-phantom + 10 random unseen
    sample_rids = set()
    for x in high_phantom[:20]:
        sample_rids.add(int(x["row_id"]))
    rng = random.Random(7)
    all_rids = list(items.keys())
    rng.shuffle(all_rids)
    for rid in all_rids:
        if len(sample_rids) >= 30: break
        # Avoid duplicates with existing Q3 sample (compare by patient_id)
        pid = int(items[rid]["patient_id"])
        if rid in sample_rids: continue
        if pid in existing_q3_pids: continue
        sample_rids.add(rid)
    sample_rids = sorted(sample_rids)
    print(f"  targeted sample: {len(sample_rids)} row_ids ({len([x for x in high_phantom[:20] if int(x['row_id']) in sample_rids])} from high-phantom)")

    # Run Qwen3-235B
    from openai import OpenAI
    client = OpenAI(base_url=QWEN3_URL, api_key="not-needed", timeout=600)

    def extract_one(rid: int) -> dict:
        item = items[rid]
        user = NER_USER_TMPL.format(note=item["note_text_truncated"])
        try:
            r = client.chat.completions.create(
                model=QWEN3_MODEL,
                messages=[{"role": "system", "content": "You are a clinical information extractor."},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=1500,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = r.choices[0].message.content or ""
            return {"row_id": rid, "patient_id": item["patient_id"],
                    "entities": parse_json(content), "raw": content[:300],
                    "comp_tok": r.usage.completion_tokens if r.usage else None,
                    "prompt_tok": r.usage.prompt_tokens if r.usage else None}
        except Exception as e:
            return {"row_id": rid, "patient_id": item["patient_id"], "_err": str(e)[:200]}

    print(f"\nRunning Qwen3-235B-MLX on {len(sample_rids)} items (no_thinking, workers=1)...")
    t0 = time.monotonic()
    out_rows = []
    with OUT_NER.open("w") as f:
        for i, rid in enumerate(sample_rids, 1):
            r = extract_one(rid)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            out_rows.append(r)
            success = isinstance(r.get("entities"), dict) and "_err" not in r["entities"]
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(sample_rids) - i) / i
            print(f"  [{i}/{len(sample_rids)}] row {rid}: {'ok' if success else 'ERR'}  "
                  f"comp_tok={r.get('comp_tok')}  elapsed={elapsed:.0f}s eta={eta:.0f}s")

    # === Comparison ===
    print(f"\n=== Magistral vs Qwen3-235B comparison on {len(out_rows)} items ===")
    detail_rows = []
    for r in out_rows:
        rid = r["row_id"]
        if "_err" in r or (isinstance(r.get("entities"), dict) and "_err" in r["entities"]):
            continue
        m = mag_ner.get(rid, {})
        m_total = get_set(m)
        q_total = get_set(r)
        j_overall = jaccard(m_total, q_total)
        # Per-category jaccard
        per_cat = {c: jaccard(get_set(m, c), get_set(r, c)) for c in CATS}
        # How many of Magistral's "phantoms" does Q3 also NOT include?
        # An entity is a "phantom not confirmed by Q3" if it's only in Magistral's set, not Q3's
        only_mag = m_total - q_total
        only_q3 = q_total - m_total
        is_high_phantom = any(int(x["row_id"]) == rid for x in qa.get("items_with_high_phantom_rate", [])[:20])
        detail_rows.append({
            "row_id": rid, "patient_id": r["patient_id"],
            "is_high_phantom_item": is_high_phantom,
            "n_mag": len(m_total), "n_q3": len(q_total),
            "n_overlap": len(m_total & q_total),
            "n_only_mag": len(only_mag), "n_only_q3": len(only_q3),
            "jaccard_overall": round(j_overall, 3),
            "jaccard_per_cat": {c: round(per_cat[c], 3) for c in CATS},
        })

    # Stats
    succ = [d for d in detail_rows]
    if succ:
        print(f"  successfully compared: {len(succ)}/{len(out_rows)}")
        j_overall = [d["jaccard_overall"] for d in succ]
        print(f"  overall jaccard: mean={np.mean(j_overall):.3f}  median={np.median(j_overall):.3f}  std={np.std(j_overall):.3f}")
        for c in CATS:
            jc = [d["jaccard_per_cat"][c] for d in succ]
            print(f"  {c:14s} jaccard: mean={np.mean(jc):.3f}  median={np.median(jc):.3f}")
        # High-phantom subset
        hp = [d for d in succ if d["is_high_phantom_item"]]
        nhp = [d for d in succ if not d["is_high_phantom_item"]]
        print(f"\n  HIGH-PHANTOM items (n={len(hp)}): mean jaccard={np.mean([d['jaccard_overall'] for d in hp]):.3f}")
        print(f"  RANDOM items   (n={len(nhp)}): mean jaccard={np.mean([d['jaccard_overall'] for d in nhp]):.3f}")
        # Show worst-disagreement items
        print(f"\n  Worst-disagreement items (lowest jaccard):")
        for d in sorted(succ, key=lambda x: x["jaccard_overall"])[:5]:
            print(f"    row {d['row_id']}: jacc={d['jaccard_overall']}  mag={d['n_mag']}  q3={d['n_q3']}  overlap={d['n_overlap']}  only_mag={d['n_only_mag']}  only_q3={d['n_only_q3']}  hp={d['is_high_phantom_item']}")

    # Save report
    report = {
        "n_compared": len(succ),
        "overall_mean_jaccard": float(np.mean(j_overall)) if succ else None,
        "overall_median_jaccard": float(np.median(j_overall)) if succ else None,
        "per_category_mean_jaccard": {c: float(np.mean([d["jaccard_per_cat"][c] for d in succ])) for c in CATS} if succ else None,
        "high_phantom_subset_mean_jaccard": float(np.mean([d["jaccard_overall"] for d in hp])) if hp else None,
        "random_subset_mean_jaccard": float(np.mean([d["jaccard_overall"] for d in nhp])) if nhp else None,
        "n_high_phantom_in_sample": len(hp),
        "n_random_in_sample": len(nhp),
        "details": detail_rows,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
