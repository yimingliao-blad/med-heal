#!/usr/bin/env python3
"""
Llama bottleneck analysis: is the failure in correction quality or verdict quality?

For each item in the Llama V3 phase 2 audit log, judge the *proposed* (corrected)
answer with GPT-4o (T=0) regardless of whether the verdict accepted it. Then
compare to the verdict's accept/reject decision to build a 2x2:

                    | regen actually correct  | regen actually wrong   |
  verdict accept    | TRUE FIX                | BREAK / NO-OP          |
  verdict reject    | MISSED FIX              | TRUE REJECT            |

The four quadrants tell us:
  - TRUE FIX: regen worked AND verdict accepted (good)
  - MISSED FIX: regen worked but verdict rejected (verdict bottleneck)
  - BREAK / NO-OP: regen failed but verdict accepted (correction bottleneck)
  - TRUE REJECT: regen failed AND verdict rejected (good)

Then:
  - If MISSED FIX is large → the regen quality is fine, the verdict is the bottleneck
  - If BREAK / NO-OP is large → the verdict can't tell good from bad, OR the regens
    are mostly bad, OR both
  - We split BREAK / NO-OP further: BREAK (judge_orig=1, judge_proposed=0) vs
    NO-OP (judge_orig=0, judge_proposed=0)

Cost: ~100 GPT-4o calls × $0.005 = $0.50.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from judge import judge as judge_call

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path,
                   default=OUT_DIR / "llama-3.1-8b-instruct" / "regen_p2_v3.jsonl")
    args = p.parse_args()

    if not args.log.exists():
        print(f"!! log not found: {args.log}")
        return 1
    recs = [json.loads(l) for l in open(args.log)]
    print(f"Llama V3 phase 2 items: {len(recs)}")

    # Judge the proposed answer for every item (where a proposed exists)
    out = []
    for i, r in enumerate(recs, 1):
        item = r["item"]
        cor = r.get("correction") or {}
        proposed = cor.get("proposed", "")
        if not proposed:
            continue
        j_orig = (r.get("judge_orig") or {}).get("label")
        j_cor_existing = (r.get("judge_corrected") or {}).get("label")
        accept = (r.get("verdict") or {}).get("accept_correction")

        # Re-judge the proposed (oracle, not pipeline) — even if the verdict
        # rejected it, we want to know what the oracle would say.
        if j_cor_existing is not None:
            j_proposed = j_cor_existing  # already have it
            from_existing = True
        else:
            res = judge_call(item["note"], item["question"], item["ground_truth"],
                             proposed, n=1, temperature=0.0)
            j_proposed = res["label"]
            from_existing = False
            time.sleep(0.5)

        out.append({
            "fold": r["fold"],
            "idx": r["idx"],
            "j_orig": j_orig,
            "j_proposed": j_proposed,
            "verdict_accept": accept,
            "from_existing": from_existing,
            "outcome_action": (r.get("outcome") or {}).get("action"),
            "outcome_delta": (r.get("outcome") or {}).get("delta"),
        })
        if i % 10 == 0:
            print(f"  judged {i}/{len(recs)}", flush=True)

    # Build the 2x2
    quadrants = Counter()
    for o in out:
        if o["j_proposed"] is None or o["j_orig"] is None:
            quadrants["unknown"] += 1
            continue
        proposed_correct = (o["j_proposed"] == 1)
        accepted = bool(o["verdict_accept"])
        if accepted and proposed_correct:
            quadrants["TRUE_FIX_or_keep_correct"] += 1
        elif accepted and not proposed_correct:
            quadrants["BREAK_or_NOOP"] += 1
        elif (not accepted) and proposed_correct:
            quadrants["MISSED_FIX_or_kept_correct"] += 1
        else:
            quadrants["TRUE_REJECT"] += 1

    # Split with judge_orig context
    # We care more about the cases where judge_orig=0 (wrong original)
    fine = Counter()
    for o in out:
        if o["j_proposed"] is None or o["j_orig"] is None:
            continue
        proposed_correct = (o["j_proposed"] == 1)
        accepted = bool(o["verdict_accept"])
        orig_correct = (o["j_orig"] == 1)
        # 8 cases — but many degenerate
        if not orig_correct:  # original was WRONG
            if proposed_correct and accepted:
                fine["W_REGEN_FIXED_VERDICT_ACCEPT"] += 1   # ideal fix
            elif proposed_correct and not accepted:
                fine["W_REGEN_FIXED_VERDICT_REJECT"] += 1   # MISSED FIX (verdict bottleneck)
            elif not proposed_correct and accepted:
                fine["W_REGEN_BAD_VERDICT_ACCEPT"] += 1     # no-op (correction bottleneck, wasted accept)
            else:
                fine["W_REGEN_BAD_VERDICT_REJECT"] += 1     # safe no-op
        else:  # original was CORRECT
            if proposed_correct and accepted:
                fine["C_REGEN_OK_VERDICT_ACCEPT"] += 1      # safe accept
            elif proposed_correct and not accepted:
                fine["C_REGEN_OK_VERDICT_REJECT"] += 1      # safe reject
            elif not proposed_correct and accepted:
                fine["C_REGEN_BAD_VERDICT_ACCEPT"] += 1     # BREAK (correction broke it, verdict failed to catch)
            else:
                fine["C_REGEN_BAD_VERDICT_REJECT"] += 1     # SAVED by verdict (correction would have broken, verdict caught)

    print()
    print("=" * 70)
    print(f"LLAMA V3 PHASE 2 BOTTLENECK ANALYSIS  (N={len(out)})")
    print("=" * 70)
    print()
    print("Coarse 2x2 (no orig label distinction):")
    for k in ("TRUE_FIX_or_keep_correct", "MISSED_FIX_or_kept_correct",
              "BREAK_or_NOOP", "TRUE_REJECT", "unknown"):
        print(f"  {k:<35} {quadrants.get(k, 0):>4}")
    print()
    print("Fine breakdown by orig label (W=originally wrong, C=originally correct):")
    keys_order = [
        "W_REGEN_FIXED_VERDICT_ACCEPT",   # real fix
        "W_REGEN_FIXED_VERDICT_REJECT",   # MISSED FIX (verdict bottleneck)
        "W_REGEN_BAD_VERDICT_ACCEPT",     # corrupted accept (correction bottleneck)
        "W_REGEN_BAD_VERDICT_REJECT",     # safe (both worked)
        "C_REGEN_OK_VERDICT_ACCEPT",      # safe (both worked, regen happens to be right too)
        "C_REGEN_OK_VERDICT_REJECT",      # safe (verdict kept original)
        "C_REGEN_BAD_VERDICT_ACCEPT",     # BREAK (correction broke, verdict failed)
        "C_REGEN_BAD_VERDICT_REJECT",     # SAVED BY VERDICT
    ]
    for k in keys_order:
        v = fine.get(k, 0)
        print(f"  {k:<40} {v:>4}")

    n_wrong = sum(fine.get(k, 0) for k in (
        "W_REGEN_FIXED_VERDICT_ACCEPT", "W_REGEN_FIXED_VERDICT_REJECT",
        "W_REGEN_BAD_VERDICT_ACCEPT", "W_REGEN_BAD_VERDICT_REJECT"))
    n_correct = sum(fine.get(k, 0) for k in (
        "C_REGEN_OK_VERDICT_ACCEPT", "C_REGEN_OK_VERDICT_REJECT",
        "C_REGEN_BAD_VERDICT_ACCEPT", "C_REGEN_BAD_VERDICT_REJECT"))
    n_fix = fine.get("W_REGEN_FIXED_VERDICT_ACCEPT", 0)
    n_missed = fine.get("W_REGEN_FIXED_VERDICT_REJECT", 0)
    n_break = fine.get("C_REGEN_BAD_VERDICT_ACCEPT", 0)
    n_saved = fine.get("C_REGEN_BAD_VERDICT_REJECT", 0)
    n_regen_fixed_total = fine.get("W_REGEN_FIXED_VERDICT_ACCEPT", 0) + fine.get("W_REGEN_FIXED_VERDICT_REJECT", 0)
    n_regen_broke_total = fine.get("C_REGEN_BAD_VERDICT_ACCEPT", 0) + fine.get("C_REGEN_BAD_VERDICT_REJECT", 0)

    print()
    print("Diagnostic ratios:")
    print(f"  Wrong items (judge_orig=0):  {n_wrong}")
    print(f"  Correct items (judge_orig=1): {n_correct}")
    print()
    print(f"  REGEN QUALITY (capability ceiling):")
    print(f"    Fixable wrong items the regen actually fixed:  {n_regen_fixed_total}/{n_wrong} = {100*n_regen_fixed_total/max(1,n_wrong):.0f}%")
    print(f"    Correct items the regen broke:                 {n_regen_broke_total}/{n_correct} = {100*n_regen_broke_total/max(1,n_correct):.0f}%")
    print()
    print(f"  VERDICT QUALITY (gating):")
    print(f"    Fixes captured (verdict accepted a good regen): {n_fix}/{n_regen_fixed_total} = {100*n_fix/max(1,n_regen_fixed_total):.0f}% recall on good regens")
    print(f"    Breaks prevented (verdict rejected a bad regen): {n_saved}/{n_regen_broke_total} = {100*n_saved/max(1,n_regen_broke_total):.0f}% recall on bad regens")
    print()
    print(f"  PIPELINE OUTCOME:")
    print(f"    True fixes: {n_fix}")
    print(f"    Missed fixes (verdict rejected a good regen): {n_missed}  ← VERDICT BOTTLENECK")
    print(f"    Breaks (verdict accepted a bad regen): {n_break}  ← VERDICT FAILS TO CATCH BAD REGEN")
    print()
    print("Bottleneck verdict:")
    if n_regen_fixed_total <= n_fix + n_missed and n_regen_fixed_total > 0:
        if n_missed > n_break:
            print(f"  → VERDICT BOTTLENECK: regen produces {n_regen_fixed_total} fixable answers but verdict only captures {n_fix}.")
            print(f"    Lost {n_missed} fixes the regen actually achieved.")
        else:
            print(f"  → MIXED: regen captures {n_regen_fixed_total} fixes, verdict accepts {n_fix} ({100*n_fix/max(1,n_regen_fixed_total):.0f}%)")
    if n_regen_broke_total > 0:
        print(f"  → CORRECTION QUALITY: regen breaks {n_regen_broke_total}/{n_correct} correct items.")
        print(f"    Verdict catches {n_saved}/{n_regen_broke_total} of these = {100*n_saved/max(1,n_regen_broke_total):.0f}%.")

    out_path = OUT_DIR / "llama_bottleneck.json"
    out_path.write_text(json.dumps({
        "n": len(out),
        "quadrants": quadrants,
        "fine": fine,
        "items": out,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
