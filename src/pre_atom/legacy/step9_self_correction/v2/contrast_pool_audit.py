#!/usr/bin/env python3
"""
Audit the rebuilt BM contrast pool (Step C-audit).

Mirrors the v1 BM atom pool audit (Module 3a) — we sample 50 entries from
the rebuilt contrast pool and ask GPT-4o whether each one is a coherent
"this is the wrong answer / what was wrong / verbatim evidence" demonstration.

The plan gate: if ≥80% PAIR_COHERENT, the contrast pool is considered usable
and the correction prompt slot is enabled. (v1 atom pool was at 42% and
got dropped — our target here is significantly higher.)

Output: output/step9_v2/contrast_pool_audit.json + .md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from judge import client

POOL_DIR = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_contrast_pool"
OUT_DIR = PROJECT_ROOT / "output" / "step9_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


AUDIT_SYS = (
    "You are auditing a contrast-example pool used for medical self-correction. "
    "Each entry is a (wrong-answer, error-label, evidence-quotes, correct-answer) "
    "tuple meant to teach a small medical AI how to fix similar errors."
)

AUDIT_USER_TMPL = """Here is one entry from the pool:

QUESTION:
{question}

WRONG ANSWER (the small AI's previous mistake):
{wrong_answer}

WHAT WAS WRONG (one-line label):
{what_was_wrong}

EVIDENCE FROM THE NOTE (verbatim quotes):
{evidence_block}

CORRECT ANSWER (the ground truth):
{ground_truth}

Evaluate the entry on three criteria:

WRONG_IS_PLAUSIBLE_ERROR: yes|no
  (Is "WRONG ANSWER" a coherent wrong response to the question, the kind a
   small medical AI might really produce?)

LABEL_IS_ACCURATE: yes|no
  (Does "WHAT WAS WRONG" correctly characterize the actual difference between
   the wrong answer and the correct answer?)

PAIR_COHERENT: yes|no
  (Taken together, is this entry a coherent demonstration of the form
   "wrong claim → evidence in the notes → corrected claim"? Could a learner
   use it as a worked example of how to fix that kind of error?)

Reply with exactly three lines in that format, then one short sentence
explaining the PAIR_COHERENT verdict.
"""


def parse_yesno(text: str, key: str) -> str | None:
    for line in text.splitlines():
        if line.upper().startswith(key.upper()):
            after = line.split(":", 1)[1].strip().lower()
            if after.startswith("yes"):
                return "yes"
            if after.startswith("no"):
                return "no"
    return None


def grade_one(entry: dict) -> tuple[dict, str]:
    evidence_block = "\n".join(f"  - \"{q}\"" for q in entry.get("evidence_from_notes", []))
    if not evidence_block:
        evidence_block = "  (no evidence quoted)"
    user = AUDIT_USER_TMPL.format(
        question=entry.get("question", "")[:500],
        wrong_answer=entry.get("wrong_answer", "")[:500],
        what_was_wrong=entry.get("what_was_wrong", ""),
        evidence_block=evidence_block,
        ground_truth=entry.get("ground_truth", "")[:500],
    )
    for attempt in range(3):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": AUDIT_SYS},
                          {"role": "user", "content": user}],
                max_tokens=200,
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            return {
                "wrong_is_plausible_error": parse_yesno(text, "WRONG_IS_PLAUSIBLE_ERROR"),
                "label_is_accurate": parse_yesno(text, "LABEL_IS_ACCURATE"),
                "pair_coherent": parse_yesno(text, "PAIR_COHERENT"),
            }, text
        except Exception as e:
            print(f"  audit retry {attempt+1}/3: {e}", flush=True)
            time.sleep(5)
    return {"wrong_is_plausible_error": None, "label_is_accurate": None,
            "pair_coherent": None}, ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool", type=Path,
                   default=POOL_DIR / "all_curated_raw.jsonl")
    args = p.parse_args()

    if not args.pool.exists():
        print(f"!! pool not found: {args.pool}")
        return 1

    pool = []
    for line in open(args.pool):
        try:
            obj = json.loads(line)
            if obj.get("n_verified_quotes", 0) > 0 and obj.get("what_was_wrong"):
                pool.append(obj)
        except Exception:
            continue
    print(f"Loaded {len(pool)} usable curated entries from {args.pool.name}", flush=True)

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    print(f"Auditing {len(sample)} sampled entries...", flush=True)

    results = []
    for k, entry in enumerate(sample, 1):
        verdict, raw = grade_one(entry)
        results.append({
            "fold": entry.get("fold"),
            "idx": entry.get("idx"),
            "question": entry.get("question", "")[:200],
            "what_was_wrong": entry.get("what_was_wrong", ""),
            "n_verified_quotes": entry.get("n_verified_quotes"),
            "verdict": verdict,
            "raw": raw,
        })
        if k % 10 == 0:
            print(f"  audited {k}/{len(sample)}", flush=True)
        time.sleep(0.5)

    plausible = Counter(r["verdict"]["wrong_is_plausible_error"] for r in results)
    label = Counter(r["verdict"]["label_is_accurate"] for r in results)
    coherent = Counter(r["verdict"]["pair_coherent"] for r in results)

    pct_plaus = 100 * plausible.get("yes", 0) / max(1, sum(plausible.values()))
    pct_label = 100 * label.get("yes", 0) / max(1, sum(label.values()))
    pct_coh = 100 * coherent.get("yes", 0) / max(1, sum(coherent.values()))

    print()
    print("=" * 60)
    print(f"BM CONTRAST POOL AUDIT  (N={len(results)})")
    print("=" * 60)
    print(f"  WRONG_IS_PLAUSIBLE_ERROR=yes: {pct_plaus:.0f}% ({plausible})")
    print(f"  LABEL_IS_ACCURATE=yes:        {pct_label:.0f}% ({label})")
    print(f"  PAIR_COHERENT=yes:            {pct_coh:.0f}% ({coherent})")
    print()
    print(f"  Plan gate: ≥80% PAIR_COHERENT to enable the pool slot in correction")
    print(f"  Verdict   : {'KEEP' if pct_coh >= 80 else ('MARGINAL' if pct_coh >= 50 else 'DROP')}")
    print(f"  v1 atom pool was 42% (DROP).")

    json_path = OUT_DIR / "contrast_pool_audit.json"
    json_path.write_text(json.dumps({
        "n": len(results),
        "pct_plausible": pct_plaus,
        "pct_label": pct_label,
        "pct_pair_coherent": pct_coh,
        "verdict": "KEEP" if pct_coh >= 80 else ("MARGINAL" if pct_coh >= 50 else "DROP"),
        "items": results,
    }, indent=2))
    md_path = OUT_DIR / "contrast_pool_audit.md"
    md_path.write_text(
        "# BM Contrast Pool Audit (Step C-audit)\n\n"
        f"Sampled {len(results)} entries (random, seed={args.seed}) from the rebuilt pool.\n\n"
        "## Aggregate metrics (GPT-4o, temp=0)\n\n"
        "| Metric | Yes rate |\n|---|---:|\n"
        f"| WRONG_IS_PLAUSIBLE_ERROR | {pct_plaus:.0f}% |\n"
        f"| LABEL_IS_ACCURATE | {pct_label:.0f}% |\n"
        f"| **PAIR_COHERENT** | **{pct_coh:.0f}%** |\n\n"
        f"**Decision**: {'KEEP' if pct_coh >= 80 else ('MARGINAL' if pct_coh >= 50 else 'DROP')} "
        f"(plan gate ≥80%; v1 atom pool was 42% and was DROPPED).\n"
    )
    print(f"\n  Wrote {json_path}\n  Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
