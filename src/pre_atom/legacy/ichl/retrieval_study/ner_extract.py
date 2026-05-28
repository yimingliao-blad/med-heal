"""NER extraction for the retrieval pool — 4th component of the multi-component scorer.

For each pool item, extract clinical entities in 5 categories via an LLM:
  medications, doses, procedures, lab_values, diagnoses

Then per-pair Jaccard overlap on the union of these sets is the "critical_detail_overlap"
component for retrieval scoring.

Two backends:
  - magistral (default, fast):  local vLLM at port 8003 — primary workhorse for full pool
  - qwen3-235b (sanity-check): MLX server at port 8800 — sample subset for direction check

Output: output/ichl/retrieval_study/ner_<backend>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "retrieval_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAGISTRAL_URL = "http://localhost:8003/v1"
MAGISTRAL_MODEL = "Magistral-Small-2509-AWQ"
QWEN3_URL = "http://192.168.68.107:8800/v1"
QWEN3_MODEL = "/Users/madblade/.lmstudio/models/lmstudio-community/Qwen3-235B-A22B-Instruct-2507-MLX-4bit"

SYSTEM = "You are a clinical information extractor."

USER_TMPL = """Extract clinical entities from this clinical note. Return a JSON object with EXACTLY these 5 keys:
- "medications": list of distinct drug/medication names (just names, no doses, no duplicates)
- "doses": list of "drug NAME — DOSE — ROUTE — FREQ" strings (e.g., "metformin — 500 mg — PO — BID"); use "?" for missing fields
- "procedures": list of distinct surgical, diagnostic, or therapeutic procedures
- "lab_values": list of "TEST = VALUE" strings (e.g., "creatinine = 2.1", "Hgb = 8.5")
- "diagnoses": list of distinct clinical conditions / diagnoses

Normalize entities to lowercase. De-duplicate. Keep entries terse and canonical (use generic drug names, not brand). If a category has no entities, return [] for it.

NOTE:
{note}

Respond with ONLY a JSON object, no prose."""


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        pid = int(r["patient_id"])
        parts = []
        for j in [1, 2, 3]:
            v = r.get(f"note_{j}")
            if v and str(v).strip() and str(v).lower() != "nan":
                parts.append(str(v))
        note = "\n\n".join(parts)[:3000]  # bigger window for entity richness
        out.append({"row_id": i, "patient_id": pid, "note": note})
    return out


def parse_json_response(text: str) -> dict:
    """Robustly extract a JSON object from a possibly noisy response."""
    if not text:
        return {"_err": "empty"}
    s = text.strip()
    # Strip code fences
    if s.startswith("```"):
        # Get content between fences
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m: s = m.group(1).strip()
    # Try direct parse
    try:
        d = json.loads(s)
        if isinstance(d, dict): return d
    except Exception:
        pass
    # Try to find a {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception as e:
            return {"_err": f"parse_fail: {str(e)[:80]}", "_raw": s[:200]}
    return {"_err": "no_json_found", "_raw": s[:200]}


def extract_one(args):
    client, model, item = args
    user = USER_TMPL.format(note=item["note"])
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0.0, max_tokens=800,
        )
        lat = time.monotonic() - t0
        content = resp.choices[0].message.content or ""
        parsed = parse_json_response(content)
        usage = resp.usage
        return {
            "row_id": item["row_id"], "patient_id": item["patient_id"],
            "entities": parsed, "raw": content[:500],
            "comp_tok": usage.completion_tokens if usage else None,
            "prompt_tok": usage.prompt_tokens if usage else None,
            "latency_s": round(lat, 2),
        }
    except Exception as e:
        return {"row_id": item["row_id"], "patient_id": item["patient_id"],
                "_err": str(e)[:200], "latency_s": round(time.monotonic() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["magistral", "qwen3-235b"], default="magistral")
    ap.add_argument("--limit", type=int, default=0, help="0 = full pool (769)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--row-ids", type=int, nargs="*", default=None,
                    help="Specific row_ids to extract (for sanity-check sample)")
    args = ap.parse_args()

    if args.backend == "magistral":
        base_url = MAGISTRAL_URL; model = MAGISTRAL_MODEL
    else:
        base_url = QWEN3_URL; model = QWEN3_MODEL

    print(f"Loading pool...")
    pool = load_pool()
    if args.row_ids:
        pool = [p for p in pool if p["row_id"] in args.row_ids]
        print(f"  filtered to {len(pool)} explicit row_ids")
    elif args.limit > 0:
        pool = pool[:args.limit]
    print(f"  extracting on n={len(pool)} items via {args.backend}")

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=300)

    out_path = OUT_DIR / f"ner_{args.backend}.jsonl"
    if args.row_ids or args.limit:
        # Don't overwrite full extraction with a sample
        out_path = OUT_DIR / f"ner_{args.backend}_sample{len(pool)}.jsonl"

    print(f"Writing to {out_path}")
    t0 = time.monotonic()
    n_done = 0; n_err = 0
    tasks = [(client, model, it) for it in pool]
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(extract_one, tasks), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            n_done += 1
            if "_err" in r or (isinstance(r.get("entities"), dict) and "_err" in r["entities"]):
                n_err += 1
            if i % 50 == 0 or (len(pool) <= 50 and i % 10 == 0):
                dt = time.monotonic() - t0
                eta = dt * (len(pool) - i) / i
                print(f"  {i}/{len(pool)}  elapsed={dt:.0f}s  eta={eta:.0f}s  errors={n_err}")
    elapsed = time.monotonic() - t0
    print(f"\nDONE in {elapsed:.0f}s  errors={n_err}/{n_done}")

    # Quick stats
    rows = [json.loads(l) for l in out_path.open() if l.strip()]
    cat_sizes = {"medications": [], "doses": [], "procedures": [], "lab_values": [], "diagnoses": []}
    for r in rows:
        e = r.get("entities", {})
        if isinstance(e, dict):
            for c in cat_sizes:
                v = e.get(c, [])
                if isinstance(v, list):
                    cat_sizes[c].append(len(v))
    import statistics
    print(f"\nMean entities per category:")
    for c, sizes in cat_sizes.items():
        if sizes:
            print(f"  {c:14s}  mean={statistics.mean(sizes):.1f}  med={statistics.median(sizes):.0f}  max={max(sizes)}  zero_cnt={sum(1 for s in sizes if s==0)}")


if __name__ == "__main__":
    main()
