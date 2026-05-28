"""BioMistral-7B self-detection at full-scale, single cell, no GPT-4o judge calls.

Companion / sibling to `full_scale_unified.py`, which deliberately skipped BM.
This script fills the BM row in the Stage I locked-cells table by inheriting
all the calibrated pieces from the existing run dir:

  - step0 token budget per `step0_token_budget.json`  (BM chosen_max_tokens = 256)
  - Q3 top-2 template (the Q3-source rank-2 candidate, used for DS in the existing run)
  - BM parser config (regex + LLM parser)
  - BM step8 binary-judged ground truth (no fresh GPT-4o calls)
  - vllm_manager / make_client / evaluate_cell — all imported, NOT reimplemented

Per `[Workflow] Implementation Discipline` Rule 1: inherit calibrated wrappers,
do not reimplement. Per `[Workflow] Execution Discipline`: pre-flight token
budget audit, in-flight per-50 checkpoint, no judge (the cached binary_correct
in pilot_loader's output is sufficient for accuracy/precision/recall).

Usage:
  .venv/bin/python -m ichl.prompt_engineering.scripts.bm_detection_run \
    --run-dir output/ichl/detection/runs/20260422_0015_verdict_only_pilot
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_fullscale
from ichl.prompt_engineering.evaluator import evaluate_cell
from ichl.prompt_engineering.parsers import LLMParser, RegexParser

# Inherit helpers from full_scale_unified.py (don't reimplement)
from ichl.prompt_engineering.scripts.full_scale_unified import (
    _load_top_candidate,
    _load_parser_config,
    _summary,
)


TARGET = "biomistral-7b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Existing detection run dir (provides step0 budget + parser configs + Q3 templates)")
    ap.add_argument("--source-target", default="qwen3-8b",
                    help="Source target whose top-N candidate template to run on BM (default: qwen3-8b)")
    ap.add_argument("--rank", type=int, default=2,
                    help="Top-N rank to load from source target (default: 2 = Q3 top-2)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    sub_pilot_dir = run_dir / "sub_pilot"
    out_dir = run_dir / "fullscale_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    # Pre-flight: token budget audit (state explicitly per Token Budget Pre-Flight Audit)
    step0 = json.loads((run_dir / "step0_token_budget.json").read_text())
    if TARGET not in step0["per_target"]:
        raise SystemExit(f"No step0 budget for {TARGET}; can't audit token budget.")
    bm_budget = step0["per_target"][TARGET]
    chosen_max_tokens = int(bm_budget["chosen_max_tokens"])
    print("=" * 60)
    print(f"BM Detection Full-Scale — Pre-flight Token Budget Audit")
    print("=" * 60)
    print(f"  target              : {TARGET}")
    print(f"  source_template     : {args.source_target} top-{args.rank}")
    print(f"  max_model_len       : 8192 (BM default per MEMORY)")
    print(f"  chosen_max_tokens   : {chosen_max_tokens} (from step0 probe)")
    print(f"  observed_max_compl  : {bm_budget.get('observed_max_completion_tokens', '?')} tokens")
    print(f"  step0_any_truncated : {bm_budget.get('any_truncated')}")
    # Rough worst-case: long note ~7000 chars / 4 chars/tok = 1750 + template 200 + answer 200 + safety 200
    worst_case_prompt = 2350
    total = worst_case_prompt + chosen_max_tokens + 200
    print(f"  worst_case_estimate : {worst_case_prompt} prompt + {chosen_max_tokens} out + 200 safety = {total}")
    if total > 8192:
        raise SystemExit(f"Budget audit FAIL: {total} > 8192. Halt.")
    print(f"  AUDIT PASS          : {total} <= 8192 (headroom {8192 - total})")
    print("=" * 60)

    # Load Q3 top-2 template
    cand_name, template = _load_top_candidate(run_dir, args.source_target, args.rank)
    print(f"\nTemplate: {cand_name}")
    print(f"  preview: {template[:120].replace(chr(10), ' ')}...")

    # Load parser config for BM
    regex_pat, llm_tpl = _load_parser_config(sub_pilot_dir, TARGET)
    parsers = [RegexParser(pattern=regex_pat) if regex_pat else RegexParser()]
    parsers.append(LLMParser(user_template=llm_tpl) if llm_tpl else LLMParser())
    print(f"  regex_pat: {regex_pat[:60] if regex_pat else '(default)'}")

    # Load full-scale items (962, all 5 folds)
    print(f"\nLoading BM full-scale items...")
    pilot_data = load_detection_fullscale(TARGET)
    print(f"  {len(pilot_data)} items")

    # Per-cell output path (resume-safe per existing convention)
    safe = cand_name.replace("/", "_").replace(" ", "_")[:120]
    per_item_path = out_dir / f"candidate_{safe}_{TARGET}_results.jsonl"
    if per_item_path.exists():
        n_existing = sum(1 for _ in per_item_path.open())
        if n_existing >= len(pilot_data):
            print(f"\n[skip] {per_item_path.name} already complete ({n_existing} rows)")
            return
        print(f"\n[resume] {per_item_path.name} has {n_existing}/{len(pilot_data)} rows")

    # Spin up vLLM with BM
    print(f"\nEnsuring vLLM target = {TARGET}...")
    vllm_manager.ensure_model(TARGET, log_dir=out_dir / "logs")
    client = make_client(TARGET)

    # Run the cell
    # Signature inherited verbatim from full_scale_unified.py invocation pattern.
    print(f"\nRunning evaluate_cell on {TARGET} x {cand_name}...")
    t0 = time.monotonic()
    log_dir = out_dir / "logs" / f"{safe}_{TARGET}"
    log_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_cell(
        candidate_name=cand_name,
        prompt_template=template,
        pilot_data=pilot_data,
        target_client=client,
        parsers=parsers,
        max_tokens=chosen_max_tokens,
        log_dir=log_dir,
        per_item_path=per_item_path,
    )
    elapsed = time.monotonic() - t0
    print(f"\nDONE in {elapsed:.0f}s")
    print(f"  cell score (accuracy vs binary_correct): {result.score:.4f}")
    print(f"  saved per-item: {per_item_path}")

    # Per-cell summary
    rows = [json.loads(l) for l in per_item_path.open() if l.strip()]
    summary = _summary(rows, cand_name, TARGET)
    summary_path = out_dir / f"summary_{safe}_{TARGET}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved summary  : {summary_path}")

    # Truncation rate audit per principle
    n_certain_trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
    n_unknown = sum(1 for r in rows if r.get("chosen_verdict") == "UNKNOWN")
    print(f"\n  certain truncation: {n_certain_trunc}/{len(rows)} = {100*n_certain_trunc/max(len(rows),1):.2f}%")
    print(f"  UNKNOWN verdicts  : {n_unknown}/{len(rows)} = {100*n_unknown/max(len(rows),1):.2f}%")
    if n_certain_trunc > 0.05 * len(rows):
        print(f"  WARNING: certain truncation rate > 5%, results suspect per Truncation Detection principle")


if __name__ == "__main__":
    main()
