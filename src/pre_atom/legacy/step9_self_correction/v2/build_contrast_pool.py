#!/usr/bin/env python3
"""
Build BM contrast pool (Step C).

For each BioMistral wrong item from step8, ask GPT-4o (build-time only,
not runtime) to curate:
  - what_was_wrong : one-line tag of the error type/content
  - evidence_from_notes : 1-3 verbatim quoted sentences from the note

Then embed the question text with GTR-T5 and save per-fold disjoint pool
files at:
  workspace/self_critique/data/bm_contrast_pool/fold_{N}_pool.json
  workspace/self_critique/data/bm_contrast_pool/fold_{N}_question_embeddings.npy

For pipeline fold N at runtime, retrieval uses the pool from folds {0..4}\\{N}.
This is the same fold-disjoint convention as the v1 atom pool.

Build-time GPT-4o usage is offline curation, exactly like the human curators
behind EHRNoteQA. It does NOT run inside the pipeline at runtime.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from judge import client, _load_notes_lookup

POOL_DIR = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_contrast_pool"
POOL_DIR.mkdir(parents=True, exist_ok=True)


CURATE_SYS = (
    "You are a senior medical informatician auditing wrong answers from a "
    "medical AI to build training contrast examples. You quote evidence verbatim."
)

CURATE_USER_TMPL = """QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

WRONG MODEL ANSWER:
{wrong_answer}

DISCHARGE NOTE:
{note}

A small medical AI produced the WRONG MODEL ANSWER above to the question. The
GROUND TRUTH ANSWER is what it should have produced.

In the structured format below, give:
1. WHAT_WAS_WRONG — a single short sentence labelling the type / content of
   the error. Examples: "fabricated a medication that wasn't prescribed",
   "missed the second admission entirely", "answered about the wrong visit",
   "confused two patients' lab values".
2. EVIDENCE_FROM_NOTES — 1 to 3 verbatim sentences from the discharge note
   above that the correct answer relies on. Each sentence MUST be a verbatim
   quote from the note (no paraphrasing).

Format your reply as:
WHAT_WAS_WRONG: <one short sentence>
EVIDENCE_FROM_NOTES:
- "<verbatim sentence 1>"
- "<verbatim sentence 2>"   (or omit)
- "<verbatim sentence 3>"   (or omit)"""


_RE_BULLET = re.compile(r'^\s*-\s*"([^"]+)"', re.MULTILINE)


def _parse_curation(text: str) -> dict:
    out = {"what_was_wrong": "", "evidence_from_notes": []}
    if not text:
        return out
    for line in text.splitlines():
        if line.strip().upper().startswith("WHAT_WAS_WRONG:"):
            out["what_was_wrong"] = line.split(":", 1)[1].strip()
            break
    out["evidence_from_notes"] = [m.group(1).strip()
                                  for m in _RE_BULLET.finditer(text)]
    return out


def _curate_one(question: str, ground_truth: str, wrong_answer: str,
                note: str, *, retries: int = 3) -> dict:
    user = CURATE_USER_TMPL.format(
        question=question[:600],
        ground_truth=ground_truth[:400],
        wrong_answer=wrong_answer[:600],
        note=note[:8000],
    )
    for attempt in range(retries):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": CURATE_SYS},
                    {"role": "user", "content": user},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            parsed = _parse_curation(text)
            parsed["raw"] = text
            return parsed
        except Exception as e:
            print(f"  curate retry {attempt+1}/{retries}: {e}", flush=True)
            time.sleep(5)
    return {"what_was_wrong": "", "evidence_from_notes": [], "raw": ""}


def _verify_quotes(evidence: list[str], note: str) -> list[bool]:
    """Check whether each quoted sentence is actually present in the note
    (case-insensitive, whitespace-tolerant). Returns a list of booleans."""
    norm_note = re.sub(r"\s+", " ", note).lower()
    out = []
    for q in evidence:
        norm_q = re.sub(r"\s+", " ", q).lower().strip()
        out.append(norm_q in norm_note)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="cap total wrong items (for smoke testing)")
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()

    print("Loading BioMistral step8 zeroshot rows...", flush=True)
    parts = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "biomistral-7b" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            df = pd.read_csv(f); df["fold"] = fold
            parts.append(df)
    bm = pd.concat(parts, ignore_index=True)
    wrong = bm[bm["binary_correct"] == 0].reset_index(drop=True)
    print(f"BM wrong items: {len(wrong)}", flush=True)
    notes = _load_notes_lookup()

    raw_path = POOL_DIR / "all_curated_raw.jsonl"
    # Resume support
    done_keys: set[tuple[int, int]] = set()
    if raw_path.exists():
        for line in open(raw_path):
            try:
                obj = json.loads(line)
                done_keys.add((int(obj["fold"]), int(obj["idx"])))
            except Exception:
                continue
        print(f"Resuming: {len(done_keys)} curations already done", flush=True)

    target = wrong if args.limit is None else wrong.iloc[args.start:args.start + args.limit]
    f_out = open(raw_path, "a")
    n_kept = 0
    n_failed = 0
    n_unverified = 0

    for i, row in target.iterrows():
        key = (int(row["fold"]), int(row["idx"]))
        if key in done_keys:
            continue
        note = notes.get(str(row["patient_id"]), "")
        if not note:
            continue
        cur = _curate_one(row["question"], row["ground_truth"],
                          str(row["model_answer"]), note)
        verified = _verify_quotes(cur["evidence_from_notes"], note)
        cur["evidence_verified"] = verified
        n_verified = sum(1 for v in verified if v)
        if not cur["what_was_wrong"] or not cur["evidence_from_notes"]:
            n_failed += 1
        elif n_verified == 0:
            n_unverified += 1
        else:
            n_kept += 1

        entry = {
            "fold": int(row["fold"]),
            "idx": int(row["idx"]),
            "patient_id": int(row["patient_id"]),
            "question": str(row["question"]),
            "ground_truth": str(row["ground_truth"]),
            "wrong_answer": str(row["model_answer"]),
            "what_was_wrong": cur["what_was_wrong"],
            "evidence_from_notes": cur["evidence_from_notes"],
            "evidence_verified": verified,
            "n_verified_quotes": n_verified,
            "raw": cur["raw"],
        }
        f_out.write(json.dumps(entry) + "\n")
        f_out.flush()
        if (n_kept + n_failed + n_unverified) % 10 == 0:
            print(f"  ...{n_kept + n_failed + n_unverified}: kept={n_kept} failed={n_failed} unverified={n_unverified}", flush=True)
        time.sleep(0.5)

    f_out.close()
    print(f"\nDone. kept={n_kept} failed={n_failed} unverified={n_unverified}", flush=True)

    # Build per-fold pools
    print("\nBuilding per-fold pools and embeddings...", flush=True)
    all_entries = [json.loads(l) for l in open(raw_path)]
    # Keep only entries with at least one verified quote
    all_entries = [e for e in all_entries if e.get("n_verified_quotes", 0) > 0
                   and e.get("what_was_wrong", "")]
    print(f"  Total usable contrast entries: {len(all_entries)}", flush=True)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")

    for hold_out_fold in range(5):
        train_entries = [e for e in all_entries if e["fold"] != hold_out_fold]
        if not train_entries:
            continue
        questions = [e["question"] for e in train_entries]
        embs = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=False)
        out_pool = POOL_DIR / f"fold_{hold_out_fold}_pool.json"
        out_emb = POOL_DIR / f"fold_{hold_out_fold}_question_embeddings.npy"
        out_pool.write_text(json.dumps(train_entries, indent=2))
        np.save(out_emb, embs)
        print(f"  fold {hold_out_fold}: {len(train_entries)} entries → {out_pool.name}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
