"""Recover the 17 truncated NER entries by re-extracting with max_tokens=2000.

Reads existing ner_magistral_962.jsonl, identifies items where entities is an _err dict,
re-runs NER with bigger output budget, merges back. Writes the file in place.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = ROOT / "output" / "ichl" / "retrieval_study" / "pool_index"
NER_FILE = POOL_DIR / "ner_magistral_962.jsonl"
ITEMS_FILE = POOL_DIR / "items.jsonl"

MAGISTRAL_URL = "http://localhost:8003/v1"
MAGISTRAL_MODEL = "Magistral-Small-2509-AWQ"

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
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try: return json.loads(m.group(0))
            except Exception as e: return {"_err": f"parse_fail: {str(e)[:80]}", "_raw": s[:300]}
    return {"_err": "no_json", "_raw": s[:300]}


def magistral_alive() -> bool:
    try:
        from openai import OpenAI
        c = OpenAI(base_url=MAGISTRAL_URL, api_key="not-needed", timeout=2)
        ms = c.models.list().data
        return any("Magistral" in m.id for m in ms)
    except Exception:
        return False


def boot_magistral():
    cmd = [
        ".venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "cyankiwi/Magistral-Small-2509-AWQ-4bit",
        "--served-model-name", "Magistral-Small-2509-AWQ",
        "--tokenizer-mode", "mistral",
        "--max-model-len", "14336",
        "--gpu-memory-utilization", "0.93",
        "--port", "8003",
        "--dtype", "auto",
    ]
    log = "/tmp/vllm_magistral_recover.log"
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    print(f"  vLLM PID: {p.pid}, log: {log}")
    for i in range(60):
        if magistral_alive():
            print(f"  Magistral READY in {i*5} s")
            return
        time.sleep(5)
    raise RuntimeError("Magistral failed to come up")


def main():
    print("Loading existing NER + items...")
    ner_rows = [json.loads(l) for l in NER_FILE.open() if l.strip()]
    items = {int(json.loads(l)["row_id"]): json.loads(l)
             for l in ITEMS_FILE.open() if l.strip()}
    err_rows = [r for r in ner_rows
                if isinstance(r.get("entities"), dict)
                and ("_err" in r["entities"] or "_parse_error" in r["entities"])]
    print(f"  total NER rows: {len(ner_rows)}")
    print(f"  parse-error rows: {len(err_rows)}")

    if not err_rows:
        print("  no errors to fix; exiting")
        return

    if not magistral_alive():
        print("Booting Magistral...")
        boot_magistral()

    from openai import OpenAI
    client = OpenAI(base_url=MAGISTRAL_URL, api_key="not-needed", timeout=300)

    def extract_one(row):
        item = items[int(row["row_id"])]
        note = item["note_text_truncated"]
        user = NER_USER_TMPL.format(note=note)
        try:
            r = client.chat.completions.create(
                model=MAGISTRAL_MODEL,
                messages=[{"role": "system", "content": NER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=2000,  # increased budget
            )
            content = r.choices[0].message.content or ""
            parsed = parse_json(content)
            return {"row_id": row["row_id"], "patient_id": row["patient_id"],
                    "entities": parsed, "raw": content[:300],
                    "comp_tok": r.usage.completion_tokens if r.usage else None,
                    "prompt_tok": r.usage.prompt_tokens if r.usage else None,
                    "_recovered": True}
        except Exception as e:
            return {"row_id": row["row_id"], "patient_id": row["patient_id"],
                    "_err": str(e)[:200]}

    print(f"\nRe-extracting NER on {len(err_rows)} truncated items with max_tokens=2000...")
    t0 = time.monotonic()
    new_results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(extract_one, err_rows), 1):
            new_results.append(r)
            success = isinstance(r.get("entities"), dict) and "_err" not in r["entities"] and "_parse_error" not in r["entities"]
            tag = "ok" if success else "still ERR"
            print(f"  [{i}/{len(err_rows)}] row {r['row_id']}: {tag}  comp_tok={r.get('comp_tok')}")
    print(f"\n  done in {time.monotonic()-t0:.0f}s")

    # Merge back
    new_by_row = {r["row_id"]: r for r in new_results}
    merged = []
    n_recovered = 0
    for row in ner_rows:
        rid = int(row["row_id"])
        if rid in new_by_row and isinstance(new_by_row[rid].get("entities"), dict) and "_err" not in new_by_row[rid]["entities"]:
            merged.append(new_by_row[rid])
            n_recovered += 1
        else:
            merged.append(row)

    with NER_FILE.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\n  recovered: {n_recovered}/{len(err_rows)}")
    print(f"  merged file: {NER_FILE}")

    # Verify final coverage
    final_err = sum(1 for r in merged
                    if isinstance(r.get("entities"), dict)
                    and ("_err" in r["entities"] or "_parse_error" in r["entities"]))
    print(f"  remaining errors: {final_err}/{len(merged)}")


if __name__ == "__main__":
    main()
