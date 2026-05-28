#!/usr/bin/env python3
"""
Module 3a — BM atom pool coherence audit.

Sample 50 atoms from the BioMistral atomic error pool and ask GPT-4o whether
each (text_raw → gt_atom_raw) pair is actually a coherent
"this was wrong → this is the fix" demonstration.

The plan's correction module uses these pairs as few-shot worked examples for
the model to learn the *form* of an atomic correction. If the pairs are not
coherent (e.g. answer says X about meds, GT atom is about an aneurysm), then
the few-shot slot is harmful or noise. This audit decides whether the
correction module includes the BM pool slot at all.

Output: output/step9_v2/bm_pool_audit.md + bm_pool_audit.json (per-atom raw).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from judge import client, _load_api_key  # noqa: F401  (loads .env)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POOL = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool" / "fold_0_atoms.json"
OUT_DIR = PROJECT_ROOT / "output" / "step9_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_SYSTEM = (
    "You are auditing an atomic error pool used for medical self-correction. "
    "Each entry should be a coherent (wrong claim, corrected claim) pair where the "
    "correction directly addresses the same fact as the wrong claim."
)

AUDIT_USER_TMPL = """For one atomic error pool entry, judge whether it is a coherent
"wrong claim → corrected claim" pair that could plausibly serve as a worked example
of how to fix that specific error.

QUESTION (the original question both atoms relate to):
{question}

WRONG CLAIM (from the model's answer):
{text_raw}

PROPOSED CORRECTED CLAIM (from the ground truth):
{gt_atom_raw}

Evaluate the pair on three criteria. Reply in this exact format:

WRONG_IS_WRONG: yes|no   (is the wrong claim actually a plausible wrong statement that the model might have produced?)
GT_IS_VALID:    yes|no   (is the corrected claim a clear, factual statement?)
PAIR_COHERENT:  yes|no   (does the corrected claim directly address the SAME topic / fact as the wrong claim?  no = the two atoms are about unrelated facts)

Then a single short sentence explaining the PAIR_COHERENT verdict.
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


def call_gpt4o(question: str, text_raw: str, gt_atom_raw: str) -> tuple[dict, str]:
    user = AUDIT_USER_TMPL.format(question=question[:500], text_raw=text_raw[:500],
                                  gt_atom_raw=gt_atom_raw[:500])
    for attempt in range(3):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": AUDIT_SYSTEM},
                          {"role": "user", "content": user}],
                max_tokens=200,
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            return {
                "wrong_is_wrong": parse_yesno(text, "WRONG_IS_WRONG"),
                "gt_is_valid": parse_yesno(text, "GT_IS_VALID"),
                "pair_coherent": parse_yesno(text, "PAIR_COHERENT"),
            }, text
        except Exception as e:
            print(f"  retry {attempt+1}/3: {e}", flush=True)
            time.sleep(5)
    return {"wrong_is_wrong": None, "gt_is_valid": None, "pair_coherent": None}, ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=50, help="number of atoms to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool", type=Path, default=POOL)
    args = p.parse_args()

    if not args.pool.exists():
        print(f"!! pool file missing: {args.pool}")
        return 1

    pool = json.loads(args.pool.read_text())
    print(f"Loaded {len(pool)} pool atoms from {args.pool.name}", flush=True)

    # Only consider atoms that actually have a paired gt_atom_raw — that's the
    # subset retrieve_pool() in run_fullscale.py uses.
    eligible = [i for i, a in enumerate(pool) if (a.get("gt_atom_raw") or "").strip()]
    print(f"Eligible (gt_atom_raw non-empty): {len(eligible)}", flush=True)

    rng = random.Random(args.seed)
    sample_idx = rng.sample(eligible, min(args.n, len(eligible)))

    results = []
    for k, idx in enumerate(sample_idx, 1):
        atom = pool[idx]
        verdict, raw = call_gpt4o(
            atom.get("question", ""),
            atom.get("text_raw", ""),
            atom.get("gt_atom_raw", ""),
        )
        results.append({
            "pool_index": idx,
            "main_error_type": atom.get("main_error_type"),
            "sub_error_type": atom.get("sub_error_type"),
            "question": atom.get("question", "")[:200],
            "text_raw": atom.get("text_raw", "")[:300],
            "gt_atom_raw": atom.get("gt_atom_raw", "")[:300],
            "verdict": verdict,
            "raw": raw,
        })
        if k % 5 == 0:
            print(f"  audited {k}/{len(sample_idx)}", flush=True)
        time.sleep(0.5)

    # Tally
    coh = Counter(r["verdict"]["pair_coherent"] for r in results)
    wrong_ok = Counter(r["verdict"]["wrong_is_wrong"] for r in results)
    gt_ok = Counter(r["verdict"]["gt_is_valid"] for r in results)

    pct_coh = 100.0 * coh.get("yes", 0) / max(1, sum(coh.values()))
    pct_wrong = 100.0 * wrong_ok.get("yes", 0) / max(1, sum(wrong_ok.values()))
    pct_gt = 100.0 * gt_ok.get("yes", 0) / max(1, sum(gt_ok.values()))

    print()
    print("=" * 60)
    print("BM POOL AUDIT  (fold 0)")
    print("=" * 60)
    print(f"  N audited           : {len(results)}")
    print(f"  WRONG_IS_WRONG=yes  : {pct_wrong:.0f}% ({wrong_ok})")
    print(f"  GT_IS_VALID=yes     : {pct_gt:.0f}% ({gt_ok})")
    print(f"  PAIR_COHERENT=yes   : {pct_coh:.0f}% ({coh})")
    print()
    print(f"  Plan gate: ≥50% PAIR_COHERENT to keep BM pool slot in correction")
    print(f"  Verdict   : {'KEEP' if pct_coh >= 50 else 'DROP'}")

    json_path = OUT_DIR / "bm_pool_audit.json"
    json_path.write_text(json.dumps({
        "n": len(results),
        "pct_pair_coherent": pct_coh,
        "pct_wrong_is_wrong": pct_wrong,
        "pct_gt_is_valid": pct_gt,
        "verdict": "KEEP" if pct_coh >= 50 else "DROP",
        "items": results,
    }, indent=2))
    md_path = OUT_DIR / "bm_pool_audit.md"
    md_lines = [
        "# BM Atom Pool Audit (Module 3a)",
        "",
        f"Sampled **{len(results)}** atoms (random, seed={args.seed}) from "
        f"`workspace/self_critique/data/bm_atomic_pool/fold_0_atoms.json`",
        f"({len(eligible)} eligible atoms with non-empty `gt_atom_raw`).",
        "",
        "## Aggregate metrics (GPT-4o judgements, temp=0)",
        "",
        f"| Metric | Yes rate |",
        f"|---|---:|",
        f"| WRONG_IS_WRONG (wrong claim is a plausible wrong statement) | **{pct_wrong:.0f}%** |",
        f"| GT_IS_VALID (gt_atom_raw is a clear factual statement) | **{pct_gt:.0f}%** |",
        f"| **PAIR_COHERENT (the two atoms address the SAME fact)** | **{pct_coh:.0f}%** |",
        "",
        f"**Decision**: PAIR_COHERENT yes-rate is {pct_coh:.0f}%. "
        f"Plan gate is ≥50% — verdict: **{'KEEP' if pct_coh >= 50 else 'DROP'}** the "
        f"BM pool slot in the correction module.",
        "",
        "## Sample of incoherent pairs (PAIR_COHERENT=no)",
        "",
    ]
    bad = [r for r in results if r["verdict"]["pair_coherent"] == "no"][:6]
    for r in bad:
        md_lines.append(f"- **Q**: {r['question']}")
        md_lines.append(f"  - wrong: «{r['text_raw']}»")
        md_lines.append(f"  - gt:    «{r['gt_atom_raw']}»")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines))
    print(f"\n  Wrote {json_path}\n  Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
