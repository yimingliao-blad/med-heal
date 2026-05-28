#!/usr/bin/env python3
"""
Module 4a — Verdict quality evaluation.

Instead of building a synthetic labeled test set up-front, we analyze the
actual (original, corrected) pairs that the pipeline pilot already produced
in `pilot_audit_log.jsonl`. For each pair we have:

  - judge_orig.label_T0           (GPT-4o T=0 on the original answer)
  - judge_corrected.label_T0      (GPT-4o T=0 on the corrected answer; only
                                   if the verdict accepted the correction)
  - verdict.majority_pick         (the verdict's choice: A, B, TIE, UNCLEAR)
  - verdict.accept_correction     (True if the verdict accepted)

Definition of "true better answer":
  if judge_corrected != judge_orig:
      true_better = "corrected" if judge_corrected==1 else "original"
  else:
      true_better = "tie" (both correct or both wrong)

For accepted-correction items only (since unaccepted ones never see the
corrected GPT-4o judge), we can measure verdict precision:

  precision = #(accepted AND truly improved) / #(accepted)
  break rate = #(accepted AND truly worsened) / #(accepted)

We also re-run the verdict step on the SAME (orig, corrected) pairs with
alternative variants (v1j) to measure their precision against the same
ground truth, without paying the cost of regenerating corrections.

Usage:
    python verdict_quality_eval.py --port 8003 --log pilot_audit_log.jsonl
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

from audit_log import AuditLog
from verdict import run_verdict, VARIANTS

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUT_DIR = RUN_ROOT / "output" / "step9_v2"


def analyze(log: AuditLog) -> dict:
    """Compute verdict precision/break-rate for whatever variant produced
    this log (typically v1f initially)."""
    accepted = []
    for r in log.all():
        v = r.get("verdict") or {}
        if not v.get("accept_correction"):
            continue
        jc = r.get("judge_corrected") or {}
        jo = r.get("judge_orig") or {}
        if jc.get("label") is None or jo.get("label") is None:
            continue
        accepted.append({
            "fold": r["fold"], "idx": r["idx"],
            "orig_label": jo["label"],
            "cor_label": jc["label"],
            "improved": jc["label"] == 1 and jo["label"] == 0,
            "broken":   jc["label"] == 0 and jo["label"] == 1,
            "neutral":  jc["label"] == jo["label"],
        })
    n = len(accepted)
    fixes = sum(1 for a in accepted if a["improved"])
    breaks = sum(1 for a in accepted if a["broken"])
    return {
        "n_accepted": n,
        "fixes": fixes,
        "breaks": breaks,
        "neutral": n - fixes - breaks,
        "precision_fix_among_accepted": (fixes / n) if n else 0.0,
        "break_rate_among_accepted":   (breaks / n) if n else 0.0,
        "items": accepted,
    }


def re_run_verdict(log: AuditLog, variant: str, port: int, k: int) -> dict:
    """Re-run an alternative verdict variant on every (orig, corrected) pair
    that the pilot generated and produce the same precision/break stats."""
    out_items = []
    fixes = breaks = neutral = 0
    n_accepted = 0
    for r in log.all():
        cor = r.get("correction") or {}
        if cor.get("skipped_reason") is not None:
            continue
        proposed = cor.get("proposed")
        if not proposed:
            continue
        item = r.get("item") or {}
        v = run_verdict(variant, r["fold"], r["idx"],
                        item.get("note", ""),
                        item.get("question", ""),
                        item.get("original_answer", ""),
                        proposed,
                        port=port, k=k)
        if not v["accept_correction"]:
            continue
        n_accepted += 1
        # We need the corrected GPT-4o label; the audit log has it from the
        # original v1f run if v1f also accepted. Otherwise we'd need a fresh
        # GPT-4o call (small overhead, opt-in below).
        jc = r.get("judge_corrected") or {}
        if jc.get("label") is None:
            # need a fresh judge call
            from judge import judge as judge_call
            res = judge_call(item.get("note", ""), item.get("question", ""),
                             item.get("ground_truth", ""), proposed,
                             n=1, temperature=0.0)
            cor_label = res["label"]
        else:
            cor_label = jc["label"]
        jo = r.get("judge_orig") or {}
        orig_label = jo.get("label")
        if cor_label is None or orig_label is None:
            continue
        improved = cor_label == 1 and orig_label == 0
        broken = cor_label == 0 and orig_label == 1
        if improved: fixes += 1
        elif broken: breaks += 1
        else: neutral += 1
        out_items.append({
            "fold": r["fold"], "idx": r["idx"],
            "orig_label": orig_label, "cor_label": cor_label,
            "improved": improved, "broken": broken,
            "v_majority": v["majority_pick"],
            "v_unanimity": v["unanimity"],
        })
    return {
        "variant": variant,
        "n_accepted": n_accepted,
        "fixes": fixes,
        "breaks": breaks,
        "neutral": neutral,
        "precision_fix_among_accepted": (fixes / n_accepted) if n_accepted else 0.0,
        "break_rate_among_accepted":   (breaks / n_accepted) if n_accepted else 0.0,
        "items": out_items,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=OUT_DIR / "pilot_audit_log.jsonl")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--also-rerun", default="v1j",
                   help="comma-separated alternative variants to re-run on the same pairs")
    args = p.parse_args()

    if not args.log.exists():
        print(f"!! audit log not found: {args.log}")
        return 1
    log = AuditLog(args.log)

    print("=== Quality of the variant that produced the log (in-vivo) ===")
    base = analyze(log)
    print(f"  accepted: {base['n_accepted']}, fixes: {base['fixes']}, breaks: {base['breaks']}, "
          f"neutral: {base['neutral']}")
    print(f"  fix-precision among accepted:  {100*base['precision_fix_among_accepted']:.1f}%")
    print(f"  break-rate among accepted:     {100*base['break_rate_among_accepted']:.1f}%")

    out: dict = {"baseline": base, "alternatives": {}}
    for v in args.also_rerun.split(","):
        v = v.strip()
        if v not in VARIANTS:
            continue
        print(f"\n=== Re-running variant {v} on the same (orig, corrected) pairs ===")
        alt = re_run_verdict(log, v, args.port, args.k)
        print(f"  accepted: {alt['n_accepted']}, fixes: {alt['fixes']}, breaks: {alt['breaks']}, "
              f"neutral: {alt['neutral']}")
        print(f"  fix-precision among accepted:  {100*alt['precision_fix_among_accepted']:.1f}%")
        print(f"  break-rate among accepted:     {100*alt['break_rate_among_accepted']:.1f}%")
        out["alternatives"][v] = alt

    out_path = OUT_DIR / "verdict_quality.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
