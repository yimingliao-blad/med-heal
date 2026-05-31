#!/usr/bin/env python3
"""Project every (gate x diagnoser x correction-arm x verdict) pipeline from the cascade
records onto the TRUE base rate, reporting net lift over zeroshot.

The cascade collected, per case: gate flags, diagnoser flags, a correction per (diagnoser,arm),
each correction's judge label, and each correction's verdict (accept/reject) per verdict variant.
Any pipeline is therefore a post-hoc projection on the SAME cases — no re-running.

Decision rule for a combo (gate g, diagnoser d, arm a, verdict v):
  act = gates[g] AND diagnosers[d].flagged          # correct only if BOTH gate and diagnoser flag
  if act and correction (d,a) exists:
      accept = verdict_v(correction)                 # verdict shuts off bad fixes
      final = corrected if accept else original
  else:
      final = original
All correctness measured by the SAME judge (judge_original vs judge_corrected) for a fair lift.

BASE-RATE WEIGHTING: sample = all 109 ZS-wrong (stored_label=0) + 200 of 853 ZS-correct
(stored_label=1). Weight wrong x1.0, correct x(853/200) so the projection reflects the real
962-case mix (109 wrong / 853 correct). Net = sum(weight * (final_correct - zs_correct)),
i.e. projected change in #correct over 962. Lift>0 at the true base rate is the bar.

Verdict variants projected: the collected C3_cot, C3_strict, plus reference bounds
accept_all (no verdict gate) and oracle (accept iff it actually improves) = the verdict ceiling.

Usage: python scripts/expK_project.py [runs/expK_cascade/<dir>/records.jsonl]
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = PROJECT_ROOT / "runs" / "expK_cascade" / "qwen25_nw-1_nc200_seed42" / "records.jsonl"

GATES = ["none", "union", "majority", "all", "plain_confirm", "positive_confirm"]
DIAGS = ["blind_plain", "blind_cot", "blind_cot_clean"]
ARMS = ["source_led", "raicl"]
VERDICTS = ["C3_cot", "C3_strict", "accept_all", "oracle"]

N_WRONG_POP = 109
N_CORRECT_POP = 853


def jlabel(j):
    return int(j.get("label", 0)) if isinstance(j, dict) else 0


def final_label(rec, g, d, a, v):
    """Return (final_correct, zs_correct) ints for this combo on this record."""
    zs = jlabel(rec.get("judge_original"))
    gates = rec.get("gates", {})
    diags = rec.get("diagnosers", {})
    act = bool(gates.get(g)) and bool(diags.get(d, {}).get("flagged"))
    if not act:
        return zs, zs
    corr = rec.get("corrections", {}).get(f"{d}|{a}")
    if not corr:
        return zs, zs
    cl = jlabel(corr.get("judge_corrected"))
    if v == "accept_all":
        accept = True
    elif v == "oracle":
        accept = cl > zs  # accept only if it genuinely improves (verdict ceiling)
    else:
        accept = bool(corr.get("verdict", {}).get(v))
    return (cl if accept else zs), zs


def main(path):
    recs = [json.loads(l) for l in open(path)]
    nw = sum(1 for r in recs if r.get("stored_label") == 0)
    nc = sum(1 for r in recs if r.get("stored_label") == 1)
    w_wrong = 1.0
    w_correct = (N_CORRECT_POP / nc) if nc else 0.0
    W = lambda r: w_wrong if r.get("stored_label") == 0 else w_correct

    # zeroshot baseline accuracy at base rate
    tot_w = sum(W(r) for r in recs)
    zs_correct_w = sum(W(r) * jlabel(r.get("judge_original")) for r in recs)
    zs_acc = zs_correct_w / tot_w if tot_w else 0

    print(f"=== cascade projection: {path} ===")
    print(f"records: {len(recs)}  (ZS-wrong={nw}, ZS-correct={nc})   correct-weight x{w_correct:.3f}")
    print(f"zeroshot accuracy @ base rate: {zs_acc*100:.2f}%  (weighted total = {tot_w:.0f} ~ 962)")
    print(f"projected #correct at zeroshot: {zs_correct_w:.0f} / {tot_w:.0f}\n")

    rows = []
    for g, d, a, v in product(GATES, DIAGS, ARMS, VERDICTS):
        fixes = breaks_w = net_w = 0.0
        n_fix = n_break = 0
        for r in recs:
            fin, zs = final_label(r, g, d, a, v)
            d_delta = fin - zs
            net_w += W(r) * d_delta
            if d_delta > 0:
                fixes += W(r) * d_delta; n_fix += 1
            elif d_delta < 0:
                breaks_w += W(r) * (-d_delta); n_break += 1
        # n_acted = cases where correction was applied AND accepted (final != original-from-no-action)
        # (approx via fix+break + accepted-no-change; recompute precisely)
        n_applied = 0
        for r in recs:
            gates = r.get("gates", {}); diags = r.get("diagnosers", {})
            if bool(gates.get(g)) and bool(diags.get(d, {}).get("flagged")) and r.get("corrections", {}).get(f"{d}|{a}"):
                corr = r["corrections"][f"{d}|{a}"]
                zs = jlabel(r.get("judge_original")); cl = jlabel(corr.get("judge_corrected"))
                acc = True if v == "accept_all" else (cl > zs if v == "oracle" else bool(corr.get("verdict", {}).get(v)))
                if acc:
                    n_applied += 1
        rows.append((net_w, g, d, a, v, fixes, breaks_w, n_fix, n_break, n_applied))

    rows.sort(reverse=True)
    print(f"{'net@962':>8} {'acc_lift':>8} | {'gate':14} {'diagnoser':16} {'arm':10} {'verdict':10} | {'fix#':>5} {'brk#':>5} {'fixΔw':>7} {'brkΔw':>7} {'applied':>7}")
    print("-" * 120)
    for net_w, g, d, a, v, fixes, breaks_w, n_fix, n_break, n_applied in rows[:30]:
        lift = net_w / tot_w * 100
        print(f"{net_w:>+8.1f} {lift:>+7.2f}% | {g:14} {d:16} {a:10} {v:10} | {n_fix:>5} {n_break:>5} {fixes:>7.1f} {breaks_w:>7.1f} {n_applied:>7}")
    print(f"\n(showing top 30 of {len(rows)} combos by projected net #correct over 962; zeroshot = 0 by definition)")
    print("net@962 = projected change in #correct vs do-nothing at the true base rate; acc_lift = same in accuracy points.")
    print("fixΔw/brkΔw are base-rate-weighted; breaks count x" + f"{w_correct:.2f} because each sampled correct case stands for that many.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
