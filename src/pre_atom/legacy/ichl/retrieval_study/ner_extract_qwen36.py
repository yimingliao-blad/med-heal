"""NER extraction using Qwen3.6-27B-AWQ on vLLM 0.19 (port 8003).

Forward-looking primary NER extractor (replaces Magistral). Same prompt as Magistral
extractor; compares results to existing Magistral output and Q3-235B sample.

Output: output/ichl/retrieval_study/pool_index/ner_qwen36_962.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = ROOT / "output" / "ichl" / "retrieval_study" / "pool_index"
ITEMS_FILE = POOL_DIR / "items.jsonl"
OUT_FILE = POOL_DIR / "ner_qwen36_962.jsonl"

QWEN36_URL = "http://localhost:8003/v1"
QWEN36_MODEL = "Qwen3.6-27B-AWQ"

NER_SYSTEM = "You are a clinical information extractor."
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4000)  # generous to avoid truncation
    args = ap.parse_args()

    items = [json.loads(l) for l in ITEMS_FILE.open()]
    if args.limit > 0:
        items = items[:args.limit]
    print(f"Loaded {len(items)} items")

    from openai import OpenAI
    client = OpenAI(base_url=QWEN36_URL, api_key="not-needed", timeout=300)

    def extract_one(it):
        user = NER_USER_TMPL.format(note=it["note_text_truncated"])
        try:
            r = client.chat.completions.create(
                model=QWEN36_MODEL,
                messages=[{"role": "system", "content": NER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = r.choices[0].message.content or ""
            return {"row_id": int(it["row_id"]), "patient_id": int(it["patient_id"]),
                    "entities": parse_json(content), "raw": content[:300],
                    "comp_tok": r.usage.completion_tokens if r.usage else None,
                    "prompt_tok": r.usage.prompt_tokens if r.usage else None}
        except Exception as e:
            return {"row_id": int(it["row_id"]), "patient_id": int(it["patient_id"]),
                    "_err": str(e)[:200]}

    print(f"\nRunning Qwen3.6 NER on {len(items)} items, {args.workers} workers...")
    t0 = time.monotonic()
    n_done = 0; n_err = 0
    with OUT_FILE.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(extract_one, items), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            n_done += 1
            if "_err" in r or (isinstance(r.get("entities"), dict) and "_err" in r["entities"]):
                n_err += 1
            if i % 50 == 0:
                dt = time.monotonic() - t0
                eta = dt * (len(items) - i) / i
                print(f"  {i}/{len(items)}  elapsed={dt:.0f}s  eta={eta:.0f}s  errors={n_err}")
    elapsed = time.monotonic() - t0
    print(f"\nDONE in {elapsed:.0f}s  errors={n_err}/{n_done}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
