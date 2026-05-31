#!/usr/bin/env python3
"""Tag bm_contrast_pool entries with T2 (correction operation) labels.

User-approved 2026-05-29 default: B-1 overlap with T2 as the analogy's primary
taxonomy. This is a one-time classification pass that does not modify the pool
JSON files; output is a sidecar file per fold.

T2 vocabulary (mirrors the natural-pipeline correction_operation field):
  - REPLACE_VALUE
  - ADD_MISSING_SLOT
  - REMOVE_UNSUPPORTED_CLAIM
  - REFOCUS_TIME_OR_VISIT

KEEP_ORIGINAL is excluded — pool entries are wrong-answer cases by construction,
so KEEP_ORIGINAL never applies. UNCLEAR is allowed as a tiebreaker output for
entries the classifier can't confidently slot.

Cost: ~1600 entries x gpt-4o-mini at ~$0.0002/call = ~$0.32. Sequential calls.

Outputs:
  bm_contrast_pool/fold_X_t2_tags.json
  with: [{pool_index, operation, confidence, rationale}]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", PROJECT_ROOT.parent / "llm-ehr-hallucination"))
POOL_DIR = SOURCE_REPO / "workspace" / "self_critique" / "data" / "bm_contrast_pool"


def load_api_key() -> str:
    for env in (PROJECT_ROOT / ".env", SOURCE_REPO / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("OPENAI_API_KEY not found")


_openai_client: OpenAI | None = None


def openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=load_api_key())
    return _openai_client


SYSTEM = (
    "You are classifying a single past wrong answer into the smallest edit "
    "operation that would have fixed it. Use exactly one of the five labels."
)

USER_TEMPLATE = """Question:
{question}

Wrong answer:
{wrong_answer}

What was wrong (audited):
{what_was_wrong}

Ground-truth answer:
{ground_truth}

Pick ONE operation that best describes the edit needed to turn the wrong answer into the ground-truth answer:

- REPLACE_VALUE: a specific value/date/medication/fact in the wrong answer is wrong and should be replaced with a different value from the same answer slot.
- ADD_MISSING_SLOT: the wrong answer omits a required fact, list item, or central piece that the question asks for; the edit is to add it.
- REMOVE_UNSUPPORTED_CLAIM: the wrong answer contains an extra/unsupported claim that should be removed; the rest is fine.
- REFOCUS_TIME_OR_VISIT: the wrong answer answers about the wrong visit/time/aspect; the edit is to rewrite to answer the correct focus.
- UNCLEAR: none of the above clearly applies, or the wrong answer needs multiple distinct edits.

Reply ONLY in this format:
OPERATION: <one of REPLACE_VALUE|ADD_MISSING_SLOT|REMOVE_UNSUPPORTED_CLAIM|REFOCUS_TIME_OR_VISIT|UNCLEAR>
CONFIDENCE: <HIGH|MEDIUM|LOW>
RATIONALE: <one short sentence>
"""


VALID_OPS = {"REPLACE_VALUE", "ADD_MISSING_SLOT", "REMOVE_UNSUPPORTED_CLAIM", "REFOCUS_TIME_OR_VISIT", "UNCLEAR"}


def parse_reply(text: str) -> dict[str, str]:
    out = {"operation": "UNCLEAR", "confidence": "LOW", "rationale": ""}
    for line in (text or "").splitlines():
        line = line.strip()
        m = re.match(r"^OPERATION\s*:\s*([A-Z_]+)", line, re.I)
        if m:
            op = m.group(1).upper()
            if op in VALID_OPS:
                out["operation"] = op
            continue
        m = re.match(r"^CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", line, re.I)
        if m:
            out["confidence"] = m.group(1).upper()
            continue
        m = re.match(r"^RATIONALE\s*:\s*(.+)$", line, re.I)
        if m:
            out["rationale"] = m.group(1).strip()
    return out


def classify_one(entry: dict[str, Any]) -> dict[str, Any]:
    user = USER_TEMPLATE.format(
        question=str(entry.get("question", ""))[:1000],
        wrong_answer=str(entry.get("wrong_answer", ""))[:1000],
        what_was_wrong=str(entry.get("what_was_wrong", ""))[:800],
        ground_truth=str(entry.get("ground_truth", ""))[:800],
    )
    for attempt in range(4):
        try:
            r = openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=120,
                temperature=0.0,
            )
            raw = (r.choices[0].message.content or "").strip()
            parsed = parse_reply(raw)
            return {"raw": raw, **parsed}
        except Exception as e:
            if attempt == 3:
                return {"raw": "", "operation": "UNCLEAR", "confidence": "LOW", "rationale": "", "error": str(e)}
            time.sleep(1 + attempt)
    return {"raw": "", "operation": "UNCLEAR", "confidence": "LOW", "rationale": ""}


def tag_fold(fold: int, *, resume: bool = True) -> dict[str, Any]:
    pool = json.loads((POOL_DIR / f"fold_{fold}_pool.json").read_text())
    out_path = POOL_DIR / f"fold_{fold}_t2_tags.json"
    existing: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        seen_indices = {int(t["pool_index"]) for t in existing if "pool_index" in t}
    tags = list(existing)
    print(f"fold {fold}: {len(pool)} entries, resume from {len(seen_indices)} cached", flush=True)
    for i, entry in enumerate(pool):
        if i in seen_indices:
            continue
        cls = classify_one(entry)
        rec = {
            "pool_index": i,
            "fold": int(entry.get("fold", -1)),
            "idx": int(entry.get("idx", -1)),
            "patient_id": int(entry.get("patient_id", -1)),
            **cls,
        }
        tags.append(rec)
        if (i + 1) % 25 == 0 or i == len(pool) - 1:
            out_path.write_text(json.dumps(tags, indent=2, ensure_ascii=False))
            print(f"  fold {fold}: tagged {i + 1}/{len(pool)} (checkpoint flushed)", flush=True)
    out_path.write_text(json.dumps(tags, indent=2, ensure_ascii=False))
    return summarize(tags)


def summarize(tags: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    ops = Counter(t.get("operation") for t in tags)
    conf = Counter(t.get("confidence") for t in tags)
    return {
        "n": len(tags),
        "operations": dict(ops),
        "confidence": dict(conf),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    fold_summaries: dict[int, dict[str, Any]] = {}
    for fold in args.folds:
        fold_summaries[fold] = tag_fold(fold, resume=not args.no_resume)
    print(json.dumps({"per_fold": fold_summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
