#!/usr/bin/env python3
"""
Cross-detection pilot: Qwen3 as auxiliary critic for other models' answers.

Tests whether a stronger model (Qwen3) can detect errors in weaker models'
zero-shot answers better than those models detect their own errors.

For each of the 4 other models' 100 pilot items:
  1. Qwen3 regens the answer (its own zero-shot)
  2. Qwen3 runs count-compare (original model answer vs Qwen3 regen)
  3. Qwen3-32B parses the verdict
  4. Compare Qwen3's detection with oracle (binary_correct) and self-detection

No GPT-4o calls needed — we only measure detection accuracy, not correction quality.
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
sys.path.insert(0, str(Path(__file__).parent))
from detection_format_bakeoff import served_model_id, vllm_chat, set_default_chat_template_kwargs
from regen_pilot import regen_zeroshot, count_compare, qwen3_parse_decision

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--models", default="qwen2.5,llama3,deepseek,biomistral",
                   help="models whose answers Qwen3 will critique")
    args = p.parse_args()

    served = served_model_id(args.port)
    print(f"vLLM serving (auxiliary critic): {served}")
    if "qwen3" not in served.lower():
        print(f"!! Expected Qwen3 on vLLM, got {served}")
        return 1

    # Disable thinking for speed
    set_default_chat_template_kwargs({"enable_thinking": False})

    model_dirs = {
        "qwen2.5": "qwen2.5-7b-instruct",
        "llama3": "llama-3.1-8b-instruct",
        "deepseek": "deepseek-r1-distill-llama-8b",
        "biomistral": "biomistral-7b",
    }
    selected = [m.strip() for m in args.models.split(",") if m.strip() in model_dirs]

    # Also load self-detection results for comparison
    self_detection = {}  # (model, fold, idx) -> accept_correction
    for alias, mdir in model_dirs.items():
        fn = OUT_DIR / mdir / "regen_p1.jsonl"
        if not fn.exists():
            continue
        for line in open(fn):
            r = json.loads(line)
            v = r.get("verdict") or {}
            key = (alias, r["fold"], r["idx"])
            self_detection[key] = v.get("accept_correction", False)

    all_results = {}

    for alias in selected:
        mdir = model_dirs[alias]
        fn = OUT_DIR / mdir / "regen_p1.jsonl"
        if not fn.exists():
            print(f"!! {fn} not found, skipping {alias}")
            continue

        recs = [json.loads(l) for l in open(fn)]
        print(f"\n{'='*60}")
        print(f"CROSS-DETECTION: Qwen3 critiques {alias} ({len(recs)} items)")
        print(f"{'='*60}")

        tp = fp = fn_ = tn = 0

        # Stream results to file as we go (don't cache in memory)
        out_path = OUT_DIR / mdir / "cross_detection_qwen3.jsonl"
        fh = open(out_path, "w")

        import random

        for i, r in enumerate(recs, 1):
            item = r.get("item", {})
            note = item.get("note", "")
            question = item.get("question", "")
            orig_answer = item.get("original_answer", "")
            jo = (r.get("judge_orig") or {}).get("label")

            if not note or not question or jo is None:
                continue

            # Qwen3 regens the answer
            try:
                qwen3_regen = regen_zeroshot(note, question, args.port)
            except Exception as e:
                print(f"  [{i}] regen err: {e}", flush=True)
                continue

            # Qwen3 count-compares (original model answer vs Qwen3 regen)
            rng = random.Random(42 + (r["fold"] << 16) + r["idx"])
            orig_in_a = rng.random() > 0.5
            ans_a = orig_answer if orig_in_a else qwen3_regen
            ans_b = qwen3_regen if orig_in_a else orig_answer

            try:
                cc_raw = count_compare(note, question, ans_a, ans_b, args.port)
            except Exception as e:
                print(f"  [{i}] cc err: {e}", flush=True)
                continue

            pick, reason = qwen3_parse_decision(cc_raw) if cc_raw else ("A", "empty")
            # "accept correction" = Qwen3 prefers its own regen over the original
            accept = (pick == "B") if orig_in_a else (pick == "A")

            # Compare with oracle
            if jo == 0:  # wrong item
                if accept: tp += 1
                else: fn_ += 1
            else:  # correct item
                if accept: fp += 1
                else: tn += 1

            rec = {
                "fold": r["fold"], "idx": r["idx"],
                "judge_orig": jo,
                "qwen3_accept": accept,
                "qwen3_pick": pick,
                "qwen3_reason": reason[:200],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            if i % 5 == 0:
                tpr = tp / max(1, tp + fn_)
                fpr = fp / max(1, fp + tn)
                print(f"  [{i}/{len(recs)}] TP={tp} FP={fp} FN={fn_} TN={tn} TPR={tpr:.1%} FPR={fpr:.1%}", flush=True)

        fh.close()

        n = tp + fp + fn_ + tn
        w = tp + fn_
        c = fp + tn
        tpr = tp / max(1, w)
        fpr = fp / max(1, c)
        prec = tp / max(1, tp + fp)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn_)

        all_results[alias] = {
            "n": n, "w": w, "c": c,
            "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
            "tpr": tpr, "fpr": fpr, "prec": prec, "f1": f1,
        }

        print(f"  Wrote {out_path}")

    # Summary comparison
    print(f"\n{'='*80}")
    print(f"CROSS-DETECTION SUMMARY: Qwen3 as auxiliary critic")
    print(f"{'='*80}")
    print(f"{'Model':<14} {'N':>4} {'W':>3} {'C':>3} | {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} | {'TPR':>6} {'FPR':>6} {'Prec':>6} {'F1':>6}")
    print("-" * 80)
    for alias in selected:
        r = all_results.get(alias)
        if not r:
            continue
        print(f"{alias:<14} {r['n']:>4} {r['w']:>3} {r['c']:>3} | {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} {r['tn']:>3} | {r['tpr']:>6.1%} {r['fpr']:>6.1%} {r['prec']:>6.1%} {r['f1']:>6.1%}")

    # Compare with self-detection
    print(f"\n{'='*80}")
    print(f"SELF vs CROSS DETECTION COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<14} | {'Self-detection':^24} | {'Qwen3 cross-det':^24} | {'Delta':^12}")
    print(f"{'':14} | {'TPR':>6} {'FPR':>6} {'F1':>6}   | {'TPR':>6} {'FPR':>6} {'F1':>6}   | {'dTPR':>6} {'dF1':>6}")
    print("-" * 80)

    # Recompute self-detection from regen_p1.jsonl
    for alias in selected:
        mdir = model_dirs[alias]
        fn = OUT_DIR / mdir / "regen_p1.jsonl"
        if not fn.exists():
            continue
        recs = [json.loads(l) for l in open(fn)]
        s_tp = s_fp = s_fn = s_tn = 0
        for r in recs:
            jo = (r.get("judge_orig") or {}).get("label")
            v = r.get("verdict") or {}
            accept = v.get("accept_correction", False)
            if jo == 0:
                if accept: s_tp += 1
                else: s_fn += 1
            elif jo == 1:
                if accept: s_fp += 1
                else: s_tn += 1

        s_tpr = s_tp / max(1, s_tp + s_fn)
        s_fpr = s_fp / max(1, s_fp + s_tn)
        s_f1 = 2 * s_tp / max(1, 2 * s_tp + s_fp + s_fn)

        xr = all_results.get(alias, {})
        x_tpr = xr.get("tpr", 0)
        x_fpr = xr.get("fpr", 0)
        x_f1 = xr.get("f1", 0)

        d_tpr = x_tpr - s_tpr
        d_f1 = x_f1 - s_f1
        print(f"{alias:<14} | {s_tpr:>6.1%} {s_fpr:>6.1%} {s_f1:>6.1%}   | {x_tpr:>6.1%} {x_fpr:>6.1%} {x_f1:>6.1%}   | {d_tpr:>+6.1%} {d_f1:>+6.1%}")

    out_path = OUT_DIR / "cross_detection_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
