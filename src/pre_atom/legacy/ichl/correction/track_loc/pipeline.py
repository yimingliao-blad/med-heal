"""Stage IV Track-Loc — corrector pipeline.

Steps (selected via --step):
  probe        : Step 0 — token-budget probe (5 items, max_tokens=16384)
  smoke        : Step 1 — 3-item format smoke
  iterate      : Step 2 (Phase A) — corrector iteration with GOLD narrative input
  baseline     : T0.a regen control (bare regen, no locator info)
  lockdown_a   : Step 3 (Phase A) — winning corrector × 5 folds with GOLD input
  lockdown_b   : Step 4 (Phase B) — winning corrector × 5 folds with v4 LOCATOR input
  judge        : GPT-4o Stage-1 binary judge on corrected answers
  report       : Step 7 — pool + write summary

All LLM calls inherit `vllm_call` (with retry + truncation gate) from
`src/ichl/error_location/pipeline.py` per Implementation Discipline Rule 1.

Plan: https://app.notion.com/p/3516be46cf3c818495a7f3ed974c78d1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Inherit existing wrappers
from ichl.error_location.pipeline import (
    load_wrong_zs,
    vllm_call,
    gpt4o_call,
    _openai_client,
)
from ichl.correction.track_loc.prompts import CORRECTOR_VERSIONS


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "output" / "ichl" / "correction" / "track_loc"
ERROR_LOC_GOLD = ROOT / "output" / "ichl" / "error_location" / "gold"
ERROR_LOC_V4 = ROOT / "output" / "ichl" / "error_location" / "locator" / "v4"


# ============================================================
# Data loading — gold narratives + v4 locator outputs
# ============================================================
def load_gold_narratives(folds: list[int], target: str = "qwen2.5-7b-instruct") -> dict[int, dict]:
    """patient_id -> {gold_narrative_text, gold_claim, gold_contradiction, gold_section}.
    Qwen2.5 reads from gold/fold_N/ (back-compat); other targets from gold/<target>/fold_N/."""
    base = ERROR_LOC_GOLD if target == "qwen2.5-7b-instruct" else ERROR_LOC_GOLD / target
    out = {}
    for fold in folds:
        p = base / f"fold_{fold}" / "gold_narratives.jsonl"
        if not p.exists():
            print(f"  [warn] missing {p}")
            continue
        for line in p.open():
            r = json.loads(line)
            out[r["patient_id"]] = r
    return out


def load_v4_locator(folds: list[int]) -> dict[int, dict]:
    """patient_id -> v4 locator record (model_narrative_text has 3 candidates)"""
    out = {}
    for fold in folds:
        p = ERROR_LOC_V4 / f"fold_{fold}" / "qwen25_v4.jsonl"
        if not p.exists(): continue
        for line in p.open():
            r = json.loads(line)
            out[r["patient_id"]] = r
    return out


# ============================================================
# Step 0 — Token-budget probe
# ============================================================
def step_probe(args):
    """Probe corrector v1 with gold-narrative input on 5 fold_0 items."""
    out_dir = OUT / "step0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_wrong_zs(folds=[0])[:5]
    gold = load_gold_narratives([0])

    print(f"[Step 0] Probe corrector v1 with GOLD input, {len(items)} fold_0 items, max_tokens={args.probe_max_tokens}")

    from ichl.common import vllm_manager
    vllm_manager.stop()
    vllm_manager.ensure_model(args.target, log_dir=out_dir / "vllm_logs")
    from openai import OpenAI
    vllm = OpenAI(base_url=args.vllm_url, api_key="not-needed", timeout=600)

    sys_p, tmpl = CORRECTOR_VERSIONS["v1"]
    out_file = out_dir / "probe_outputs.jsonl"
    comp_toks = []
    with out_file.open("w") as fp:
        for it in items:
            pid = it["patient_id"]
            g = gold.get(pid)
            if not g:
                print(f"  [skip] pid={pid} no gold narrative")
                continue
            user = tmpl.format(
                note=it["note"], question=it["question"],
                zs_answer=it["zs_answer"],
                contradiction_narrative=g["gold_narrative_text"],
            )
            r = vllm_call(vllm, args.vllm_model, sys_p, user,
                          max_tokens=args.probe_max_tokens, temperature=0.0,
                          target=args.target, enable_thinking=(False if "qwen3" in args.target else None),
                          max_model_len=32768, max_retries=0)
            row = {**it, **r, "step": "probe"}
            fp.write(json.dumps(row) + "\n"); fp.flush()
            ct = r.get("completion_tokens")
            print(f"  pid={pid}  comp_tok={ct}  finish={r.get('finish_reason')}  trunc_certain={r['truncation_report']['is_truncated_certain']}  text_head={(r.get('text') or '')[:120]!r}")
            if ct: comp_toks.append(ct)

    if not comp_toks:
        raise SystemExit("[Step 0] No completion_tokens recorded")

    # Probe-itself-truncation hard fail
    n_trunc = sum(1 for r in (json.loads(l) for l in out_file.open())
                  if r.get("truncation_report", {}).get("is_truncated_certain"))
    if n_trunc > 0:
        print(f"\n[Step 0] FAIL: {n_trunc}/{len(comp_toks)} probe items truncated at max_tokens={args.probe_max_tokens}.")
        raise SystemExit("[Step 0] Probe was truncated; bump --probe-max-tokens.")

    arr = np.array(comp_toks)
    p95 = float(np.percentile(arr, 95))
    max_obs = int(arr.max())
    production = max(int(2 * p95), int(max_obs * 1.2))
    summary = {
        "n_probed": len(comp_toks),
        "n_probe_truncated": n_trunc,
        "completion_tokens": {"min": int(arr.min()),
                               "p50": int(np.percentile(arr, 50)),
                               "p95": int(p95),
                               "p99": int(np.percentile(arr, 99)),
                               "max": max_obs},
        "production_max_tokens": production,
        "formula": "max(2 * p95, max * 1.2)",
        "principle": "Truncation Detection on Every LLM Output § Step 0 probe pattern",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[Step 0] DONE.")
    print(f"  comp_tok: min={summary['completion_tokens']['min']}  p50={summary['completion_tokens']['p50']}  p95={int(p95)}  p99={summary['completion_tokens']['p99']}  max={max_obs}")
    print(f"  → production max_tokens = {production}")
    print(f"  Saved: {out_dir / 'summary.json'}")


# ============================================================
# Generic correct step — handles iterate / baseline / lockdown_a / lockdown_b
# ============================================================
def step_correct(args):
    """Run corrector (or baseline) on items, save per-item records.

    Routes:
      step=iterate    : corrector_version on fold_0 with GOLD input  (Phase A iteration)
      step=baseline   : T0.a regen on folds (no contradiction info)
      step=lockdown_a : corrector_version on all 5 folds with GOLD input
      step=lockdown_b : corrector_version on all 5 folds with v4 LOCATOR input
                        (sub-variants: --candidates top1 | all3)
    """
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0]
    items = load_wrong_zs(target=args.target, folds=folds)
    if args.limit > 0:
        items = items[: args.limit]

    # Decide input source per step
    use_gold = args.step in ("iterate", "lockdown_a")
    use_locator = args.step == "lockdown_b"
    use_baseline = args.step == "baseline"

    if use_gold:
        contrad_lookup = load_gold_narratives(folds, target=args.target)
        input_label = "gold"
    elif use_locator:
        contrad_lookup = load_v4_locator(folds)
        input_label = f"v4_{args.candidates}"
    else:
        contrad_lookup = {}
        input_label = "baseline"

    sys_p, tmpl = CORRECTOR_VERSIONS[args.corrector_version if not use_baseline else "t0a"]
    sub = args.corrector_version if not use_baseline else "t0a"
    out_dir = OUT / args.step / sub
    print(f"[Correct] step={args.step} sub={sub} input={input_label} folds={folds} n={len(items)}")

    from ichl.common import vllm_manager
    vllm_manager.stop()
    vllm_manager.ensure_model(args.target, log_dir=out_dir / "vllm_logs")
    from openai import OpenAI
    vllm = OpenAI(base_url=args.vllm_url, api_key="not-needed", timeout=600)

    by_fold: dict[int, list[dict]] = {f: [] for f in folds}
    n_done = n_err = n_trunc = n_no_contrad = 0
    for it in items:
        pid = it["patient_id"]
        if use_baseline:
            user = tmpl.format(note=it["note"], question=it["question"])
        else:
            g = contrad_lookup.get(pid)
            if not g:
                n_no_contrad += 1
                continue
            if use_gold:
                contradiction = g["gold_narrative_text"]
            else:
                # locator case: TOP-1 candidate or all 3
                full = g.get("model_narrative_text", "")
                if args.candidates == "top1":
                    # extract CANDIDATE 1 block
                    parts = re.split(r"CANDIDATE\s*[2-9]:", full, maxsplit=1, flags=re.IGNORECASE)
                    contradiction = parts[0].strip()
                else:
                    contradiction = full
            # Some corrector versions use parsed structured fields (claim/truth/section);
            # others use a single narrative blob. Detect by template placeholders.
            if "{contradiction_claim}" in tmpl:
                if use_gold:
                    user = tmpl.format(
                        note=it["note"], question=it["question"],
                        zs_answer=it["zs_answer"],
                        contradiction_claim=g.get("gold_claim") or "",
                        contradiction_truth=g.get("gold_contradiction") or "",
                        contradiction_section=g.get("gold_section") or "",
                    )
                else:
                    # locator path: parse CANDIDATE 1's CLAIM/CONTRADICTION/SECTION
                    full = g.get("model_narrative_text", "")
                    cm = re.search(r"CLAIM:\s*(.+?)(?:\n|$)", full, re.IGNORECASE)
                    cn = re.search(r"CONTRADICTION:\s*(.+?)(?:\n|$)", full, re.IGNORECASE)
                    sc = re.search(r"SECTION:\s*(.+?)(?:\n|$)", full, re.IGNORECASE)
                    user = tmpl.format(
                        note=it["note"], question=it["question"],
                        zs_answer=it["zs_answer"],
                        contradiction_claim=(cm.group(1).strip() if cm else ""),
                        contradiction_truth=(cn.group(1).strip() if cn else ""),
                        contradiction_section=(sc.group(1).strip() if sc else ""),
                    )
            else:
                user = tmpl.format(
                    note=it["note"], question=it["question"],
                    zs_answer=it["zs_answer"],
                    contradiction_narrative=contradiction,
                )
        r = vllm_call(vllm, args.vllm_model, sys_p, user,
                      max_tokens=args.max_gen_tokens, temperature=0.0,
                      target=args.target, enable_thinking=(False if "qwen3" in args.target else None),
                      max_model_len=32768, max_retries=2)
        n_done += 1
        if r["_err"]: n_err += 1
        elif r["truncation_report"]["is_truncated_certain"]: n_trunc += 1
        rec = {
            "patient_id": pid, "fold": it["fold"],
            "step": args.step, "corrector_version": sub,
            "input_label": input_label,
            "zs_answer": it["zs_answer"],
            "ground_truth": it["ground_truth"],
            "corrected_answer": r["text"],
            "truncation_report": r["truncation_report"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "latency_s": r["latency_s"],
            "_err": r["_err"],
            "retry_history": r.get("retry_history", []),
        }
        by_fold[it["fold"]].append(rec)
        if n_done % 25 == 0:
            print(f"  done {n_done}/{len(items)}  err={n_err} trunc={n_trunc} no_contrad={n_no_contrad}")

    trunc_rate = n_trunc / max(n_done, 1)
    err_rate = n_err / max(n_done, 1)
    print(f"\n[Correct] post-batch trunc={100*trunc_rate:.1f}% err={100*err_rate:.1f}%")
    if trunc_rate > 0.10:
        print(f"  RUN REJECTED: trunc>10% per Truncation Detection.")
        raise SystemExit(2)
    if err_rate > 0.05:
        print(f"  WARN: err rate >5%; flag in Finding")

    for f, rows in by_fold.items():
        fold_dir = out_dir / f"fold_{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / f"corrected_{input_label}.jsonl").open("w") as fp:
            for r_ in rows:
                fp.write(json.dumps(r_) + "\n")
    print(f"  Saved: {out_dir}/fold_*/corrected_{input_label}.jsonl")


# ============================================================
# Judge — GPT-4o Stage-1 binary on corrected answers vs GT
# ============================================================
GPT4O_JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."
GPT4O_JUDGE_USER_TMPL = """DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect"""

def step_judge(args):
    """Re-judge corrected answers via GPT-4o Stage-1 binary."""
    from ichl.error_location.pipeline import load_full_notes
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0]

    # Find input dir per step's output structure
    sub = args.corrector_version
    input_label = args.input_label  # "gold" / "v4_top1" / "v4_all3" / "baseline"
    src_step = args.step_dir or "iterate"
    src_dir = OUT / src_step / sub

    notes_by_pid = load_full_notes()
    client = _openai_client()

    n_done = n_correct = n_err = 0
    by_fold: dict[int, list[dict]] = {f: [] for f in folds}
    for fold in folds:
        in_file = src_dir / f"fold_{fold}" / f"corrected_{input_label}.jsonl"
        if not in_file.exists():
            print(f"  [skip] missing {in_file}")
            continue
        for line in in_file.open():
            r = json.loads(line)
            pid = r["patient_id"]
            note = notes_by_pid[pid]["note"] if pid in notes_by_pid else ""
            user = GPT4O_JUDGE_USER_TMPL.format(
                note=note, question=notes_by_pid[pid]["question"] if pid in notes_by_pid else "",
                ground_truth=r["ground_truth"], model_answer=r["corrected_answer"],
            )
            jr = gpt4o_call(client, GPT4O_JUDGE_SYSTEM, user, max_tokens=10, temperature=0.0,
                            sub_variant="judge_corrected")
            txt = (jr["text"] or "").strip()
            m = re.search(r"[01]", txt)
            score = int(m.group(0)) if m else None
            if score == 1: n_correct += 1
            if jr["_err"]: n_err += 1
            n_done += 1
            by_fold[fold].append({
                "patient_id": pid, "fold": fold, "input_label": input_label,
                "corrector_version": sub, "step_dir": src_step,
                "corrected_correct": score, "judge_raw": txt,
                "_err": jr["_err"],
                "truncation_report": jr["truncation_report"],
            })
            if n_done % 25 == 0:
                print(f"  judged {n_done}  correct={n_correct} err={n_err}")

    out_dir = OUT / "judge" / sub
    for f, rows in by_fold.items():
        fold_dir = out_dir / f"fold_{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / f"gpt4o_{src_step}_{input_label}.jsonl").open("w") as fp:
            for r_ in rows:
                fp.write(json.dumps(r_) + "\n")

    print(f"\n[Judge] DONE. n={n_done} correct={n_correct} err={n_err}")
    print(f"  Saved: {out_dir}/fold_*/gpt4o_{src_step}_{input_label}.jsonl")


# ============================================================
# Report — pool + Wilson CI
# ============================================================
def step_report(args):
    from scipy.stats import beta
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0]
    sub = args.corrector_version
    input_label = args.input_label
    src_step = args.step_dir or "iterate"

    by_fold = {}
    pooled = {"n_total": 0, "n_judged": 0, "n_correct": 0, "n_err": 0}
    for fold in folds:
        p = OUT / "judge" / sub / f"fold_{fold}" / f"gpt4o_{src_step}_{input_label}.jsonl"
        if not p.exists(): continue
        rows = [json.loads(l) for l in p.open()]
        n = len(rows)
        nj = sum(1 for r in rows if r.get("corrected_correct") in (0, 1))
        nc = sum(1 for r in rows if r.get("corrected_correct") == 1)
        ne = sum(1 for r in rows if r.get("_err"))
        if nj > 0:
            pp = nc / nj
            lo = float(beta.ppf(0.025, nc, nj - nc + 1)) if nc > 0 else 0.0
            hi = float(beta.ppf(0.975, nc + 1, nj - nc)) if nc < nj else 1.0
        else:
            pp, lo, hi = 0.0, 0.0, 1.0
        by_fold[fold] = {"n_total": n, "n_judged": nj, "n_correct": nc, "n_err": ne,
                         "lift_pp": round(pp, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        for k_ in ["n_total", "n_judged", "n_correct", "n_err"]:
            pooled[k_] += by_fold[fold][k_]

    nj = pooled["n_judged"]; nc = pooled["n_correct"]
    if nj > 0:
        pp = nc / nj
        lo = float(beta.ppf(0.025, nc, nj - nc + 1)) if nc > 0 else 0.0
        hi = float(beta.ppf(0.975, nc + 1, nj - nc)) if nc < nj else 1.0
    else:
        pp, lo, hi = 0.0, 0.0, 1.0
    pooled["lift_pp"] = round(pp, 4)
    pooled["ci95"] = [round(lo, 4), round(hi, 4)]

    summary = {
        "corrector_version": sub, "input_label": input_label, "step_dir": src_step,
        "by_fold": {str(f): v for f, v in by_fold.items()},
        "pooled": pooled,
        "judge": "gpt4o-stage1-binary",
        "scope": {
            "target": "qwen2.5-7b-instruct (zs answer producer); qwen3-8b (corrector)",
            "wrong_filter": "binary_correct=0 from step8 GPT-4o judge",
            "temperature": 0.0,
        },
    }
    out_path = OUT / f"summary_{src_step}_{sub}_{input_label}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Report] Saved {out_path}")
    print(f"  Pooled: {nc}/{nj} = {pp:.3f}  CI95={pooled['ci95']}")
    for f, v in by_fold.items():
        print(f"  fold_{f}: {v['n_correct']}/{v['n_judged']} = {v['lift_pp']:.3f}")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["probe", "smoke", "iterate", "baseline",
                             "lockdown_a", "lockdown_b", "judge", "report"])
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--target", default="qwen3-8b",
                    help="vllm_manager TARGETS key (qwen2.5-7b-instruct / qwen3-8b / etc).")
    ap.add_argument("--probe-max-tokens", type=int, default=16384,
                    help="Step 0 probe budget (per Truncation Detection § Step 0: 'very generous, e.g., 16384 or model max').")
    ap.add_argument("--max-gen-tokens", type=int, default=2048,
                    help="Production gen budget (set from Step 0 result).")
    ap.add_argument("--corrector-version", default="v1", choices=list(CORRECTOR_VERSIONS.keys()))
    ap.add_argument("--folds", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--candidates", default="top1", choices=["top1", "all3"],
                    help="Phase B only: feed locator's TOP-1 only or all 3.")
    ap.add_argument("--input-label", default="gold",
                    help="For judge/report: 'gold' / 'v4_top1' / 'v4_all3' / 'baseline'.")
    ap.add_argument("--step-dir", default="iterate",
                    help="For judge/report: which corrector dir to read (iterate/baseline/lockdown_a/lockdown_b).")
    args = ap.parse_args()

    if args.step == "probe":
        step_probe(args)
    elif args.step in ("iterate", "baseline", "lockdown_a", "lockdown_b"):
        step_correct(args)
    elif args.step == "judge":
        step_judge(args)
    elif args.step == "report":
        step_report(args)
    else:
        raise SystemExit(f"unknown step={args.step}")


if __name__ == "__main__":
    main()
