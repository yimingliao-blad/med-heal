"""NER extraction via GPT-4o-mini API. ONE-TIME pool pre-compute.

This is a CALIBRATION pre-compute (not a production runtime path) — like our
retrieval-rubric gold labels. The production retrieval looks up the pre-computed
entity sets; no OpenAI runtime dependency.

Cost: ~$0.50 for the full 962 items (~700K input + ~480K output tokens at
$0.15/1M input, $0.60/1M output).

Output: output/ichl/retrieval_study/pool_index/ner_gpt4o_mini_962.jsonl
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
POOL_DIR = ROOT / "output" / "ichl" / "retrieval_study" / "pool_index"
ITEMS_FILE = POOL_DIR / "items.jsonl"
OUT_FILE = POOL_DIR / "ner_gpt4o_mini_962.jsonl"

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
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_path = ROOT / ".env"
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip(); break
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set and not in .env")

    items = [json.loads(l) for l in ITEMS_FILE.open()]
    if args.limit > 0:
        items = items[:args.limit]
    print(f"Loaded {len(items)} items, model={args.model}, workers={args.workers}, max_tokens={args.max_tokens}")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def extract_one(it):
        user = NER_USER_TMPL.format(note=it["note_text_truncated"])
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": NER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=args.max_tokens,
                response_format={"type": "json_object"},  # forces valid JSON
            )
            content = r.choices[0].message.content or ""
            usage = r.usage
            return {"row_id": int(it["row_id"]), "patient_id": int(it["patient_id"]),
                    "entities": parse_json(content), "raw": content[:300],
                    "comp_tok": usage.completion_tokens if usage else None,
                    "prompt_tok": usage.prompt_tokens if usage else None,
                    "finish_reason": r.choices[0].finish_reason}
        except Exception as e:
            return {"row_id": int(it["row_id"]), "patient_id": int(it["patient_id"]),
                    "_err": str(e)[:200]}

    print(f"\nRunning {args.model} NER on {len(items)} items, {args.workers} workers...")
    t0 = time.monotonic()
    n_done = 0; n_err = 0; total_in = 0; total_out = 0
    with OUT_FILE.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(extract_one, items), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            n_done += 1
            if "_err" in r or (isinstance(r.get("entities"), dict) and "_err" in r["entities"]):
                n_err += 1
            else:
                total_in += r.get("prompt_tok", 0) or 0
                total_out += r.get("comp_tok", 0) or 0
            if i % 50 == 0:
                dt = time.monotonic() - t0
                eta = dt * (len(items) - i) / i
                cost = total_in * 0.15e-6 + total_out * 0.60e-6  # GPT-4o-mini
                print(f"  {i}/{len(items)}  elapsed={dt:.0f}s  eta={eta:.0f}s  errors={n_err}  cost~${cost:.3f}")
    elapsed = time.monotonic() - t0
    final_cost = total_in * 0.15e-6 + total_out * 0.60e-6
    print(f"\nDONE in {elapsed:.0f}s  errors={n_err}/{n_done}")
    print(f"  prompt_tokens: {total_in}  completion_tokens: {total_out}  cost: ${final_cost:.3f}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
