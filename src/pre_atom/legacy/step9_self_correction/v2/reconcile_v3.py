#!/usr/bin/env python3
"""
Retroactively apply the disagreement-keep-original rule to V3 audit logs.

The V3 pipeline patched in a "regex/q3 disagreement → keep original" rule
midway through the multi-model run. Some pilots ran with the old rule (where
regex was used as primary and q3 as fallback). This script reads each pilot's
audit log and re-derives the outcome under the new rule, since both regex_winner
and q3_winner are persisted on every item.

The new rule mirrors run_verdict_v3:
  1. tie                            → keep original
  2. regex/q3 disagree (both A/B)   → keep original
  3. regex non-tie winner           → use it
  4. regex unparseable, q3 winner   → use it
  5. neither                        → keep original

After reconciliation, items that were previously "corrected" but now "kept_original"
are flipped — outcome.action = kept_original, outcome.delta = 0, judge_corrected
preserved for traceability but ignored in the new outcome.

Output: writes a `regen_v3_reconciled.jsonl` next to each `regen_v3_audit.jsonl`,
plus a comparison table.
"""
from __future__ import annotations

import os
import json
import sys
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUT_DIR = RUN_ROOT / "output" / "step9_v2" / "multi_model"


def reconcile_one(rec: dict) -> dict:
    """Recompute the outcome under the disagreement-keep-original rule.
    Returns a new record (does not mutate the original)."""
    new = json.loads(json.dumps(rec, default=str))
    v = new.get("verdict") or {}
    if not v:
        return new

    regex_parsed = v.get("regex_parsed") or {}
    regex_winner = v.get("regex_winner")
    q3_winner = v.get("q3_winner")
    is_tie = bool(regex_parsed.get("is_tie"))
    corrected_slot = v.get("corrected_slot", "B")

    # Apply the new rule
    new_accept = False
    new_source = "none"
    if is_tie:
        new_source = "regex_tie"
    elif (regex_winner in ("A", "B") and q3_winner in ("A", "B")
          and regex_winner != q3_winner):
        new_source = "disagreement_keep_original"
    elif regex_winner in ("A", "B"):
        new_source = "regex"
        new_accept = (regex_winner == corrected_slot)
    elif q3_winner in ("A", "B"):
        new_source = "q32_fallback"
        new_accept = (q3_winner == corrected_slot)
    else:
        new_source = "none"

    v["reconciled_accept"] = new_accept
    v["reconciled_source"] = new_source

    # Recompute outcome
    old_outcome = new.get("outcome") or {}
    j_orig = (new.get("judge_orig") or {}).get("label")
    j_cor = (new.get("judge_corrected") or {}).get("label")

    if not new_accept:
        new["outcome"] = {
            "action": "kept_original",
            "delta": 0,
            "final_eval": j_orig if j_orig is not None else 0,
            "reconciled_from": old_outcome.get("action"),
        }
    elif j_cor is not None and j_orig is not None:
        # Was accepted under new rule and we have the corrected judge
        delta = (1 if j_cor == 1 and j_orig == 0
                 else (-1 if j_cor == 0 and j_orig == 1 else 0))
        new["outcome"] = {
            "action": "corrected",
            "delta": delta,
            "final_eval": j_cor,
        }
    else:
        # Accepted but no corrected judge available — must default to keep_original
        new["outcome"] = {
            "action": "kept_original",
            "delta": 0,
            "final_eval": j_orig if j_orig is not None else 0,
            "reconciled_note": "would-accept-but-no-judge",
        }
    return new


def summarize(recs: list[dict], label: str) -> dict:
    fixes = sum(1 for r in recs if (r.get("outcome") or {}).get("delta") == 1)
    brks = sum(1 for r in recs if (r.get("outcome") or {}).get("delta") == -1)
    actions = Counter((r.get("outcome") or {}).get("action", "?") for r in recs)
    return {
        "label": label,
        "n": len(recs),
        "fixes": fixes,
        "breaks": brks,
        "actions": dict(actions),
    }


def main() -> int:
    models = [
        ("qwen2.5", "qwen2.5-7b-instruct"),
        ("llama3", "llama-3.1-8b-instruct"),
        ("deepseek", "deepseek-r1-distill-llama-8b"),
        ("qwen3", "qwen3-8b"),
    ]
    print(f"{'Model':<12} {'phase':<12} {'fix':>5} {'brk':>5} {'corr':>5} {'kept':>5}")
    print("-" * 50)
    summaries = []
    for alias, dirname in models:
        log_path = OUT_DIR / dirname / "regen_v3_audit.jsonl"
        if not log_path.exists():
            print(f"{alias}: NO V3 LOG")
            continue
        recs = [json.loads(l) for l in open(log_path)]
        s_old = summarize(recs, "before")
        recs_new = [reconcile_one(r) for r in recs]
        s_new = summarize(recs_new, "after")

        out_path = OUT_DIR / dirname / "regen_v3_reconciled.jsonl"
        with open(out_path, "w") as f:
            for r in recs_new:
                f.write(json.dumps(r, default=str) + "\n")

        for s in (s_old, s_new):
            a = s["actions"]
            print(f"{alias:<12} {s['label']:<12} {s['fixes']:>5} {s['breaks']:>5} "
                  f"{a.get('corrected', 0):>5} {a.get('kept_original', 0):>5}")
        print("-" * 50)
        summaries.append({"alias": alias, "before": s_old, "after": s_new})

    # Save aggregate
    agg_path = OUT_DIR / "v3_reconciled_summary.json"
    agg_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
