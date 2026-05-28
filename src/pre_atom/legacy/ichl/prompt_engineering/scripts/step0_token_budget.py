"""Step 0 — Token-budget calibration per target model.

Per Notion 'Claude: Plan: Detection — Pilot Runner Design' § Step 0:

  Procedure:
    1. Pick 3 items spanning short / medium / long notes.
    2. For each target model, send A1 with a generous starting cap:
         Qwen3-8B (think) = 2048, Qwen2.5-7B = 256, DeepSeek-R1-8B = 4096.
    3. Record `completion_tokens` and `finish_reason`. If any response has
       `finish_reason == "length"`, DOUBLE the cap and re-run that target
       until none truncate.
    4. Chosen budget: max(64, ⌈1.5 × max_observed_completion_tokens⌉).
    5. Save step0_token_budget.json to the run dir.

  Gate: no sub-pilot starts until `any_truncated == false` for every target.

Usage:
    python -m ichl.prompt_engineering.scripts.step0_token_budget \
        --run-dir output/ichl/detection/runs/<timestamp>_verdict_only_pilot/ \
        --targets qwen3-8b qwen2.5-7b-instruct

This script handles the vLLM model swap per target (authorised — see MEMORY).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common.pilot_loader import load_detection_pilot_sub
from ichl.common import vllm_manager
from ichl.prompt_engineering.evaluator import SYSTEM_MSG
from ichl.prompt_engineering.pool import load_seeds


# Starting caps per target — generous enough to avoid capping real outputs.
# If observed completion_tokens hit this cap (finish_reason=length), double
# and re-run per the Step 0 rule.
STARTING_CAPS: dict[str, int] = {
    "qwen3-8b": 2048,               # think mode → many tokens
    "qwen2.5-7b-instruct": 256,     # short answers, single word expected
    "biomistral-7b": 256,
    "llama-3.1-8b-instruct": 256,
    "deepseek-r1-distill-llama-8b": 4096,   # always thinks, larger budget
}


def _pick_items(target: str, n: int = 3) -> list[dict[str, Any]]:
    """Pick 3 items spanning short / medium / long notes. Deterministic."""
    # Load the full 40-item pilot to get a representative spread.
    from ichl.common.pilot_loader import load_detection_pilot
    all_items = load_detection_pilot(target)
    # Sort by note length, pick 5th-percentile / median / 95th-percentile.
    sorted_items = sorted(all_items, key=lambda i: len(i["note"]))
    n_items = len(sorted_items)
    if n_items < 3:
        return sorted_items
    idx_short = max(0, int(0.05 * n_items))
    idx_mid = n_items // 2
    idx_long = min(n_items - 1, int(0.95 * n_items))
    picked = [sorted_items[idx_short], sorted_items[idx_mid], sorted_items[idx_long]]
    return picked[:n]


def _run_target(
    target_key: str,
    template: str,
    starting_cap: int,
    max_doublings: int = 3,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Step 0 for one target: test 3 items, double on truncation."""
    spec = vllm_manager.ensure_model(target_key, log_dir=log_dir)
    client = make_client(target_key)
    items = _pick_items(target_key)
    print(f"\n── Step 0 / {target_key} ── using 3 items "
          f"(note lens: {[len(it['note']) for it in items]})")

    cap = starting_cap
    for attempt in range(max_doublings + 1):
        results = []
        any_truncated = False
        for item in items:
            prompt = template.format(
                note=item["note"], question=item["question"],
                model_answer=item["model_answer"],
                answer=item["model_answer"], choices="",
            )
            t0 = time.monotonic()
            resp = client.call(
                system=SYSTEM_MSG, user=prompt,
                temperature=0.0, max_tokens=cap,
            )
            lat = time.monotonic() - t0
            ct = resp.usage.get("completion_tokens") if resp.usage else None
            fr = resp.finish_reason
            truncated = (fr == "length")
            if truncated:
                any_truncated = True
            results.append({
                "pilot_item_id": item["pilot_item_id"],
                "note_chars": len(item["note"]),
                "completion_tokens": ct,
                "finish_reason": fr,
                "truncated": truncated,
                "latency_s": round(lat, 2),
                "success": resp.success,
                "error": resp.error,
            })
            print(f"  {item['pilot_item_id']}  ct={ct}  finish={fr}  "
                  f"lat={lat:.1f}s  {'[TRUNC]' if truncated else ''}")

        max_observed = max((r["completion_tokens"] or 0) for r in results)
        print(f"  attempt {attempt+1}: cap={cap}, max_observed={max_observed}, "
              f"any_truncated={any_truncated}")

        if not any_truncated:
            # Chosen budget: 1.5× safety margin, floor 64.
            chosen = max(64, math.ceil(1.5 * max_observed))
            return {
                "target_model": target_key,
                "served_name": spec.served_name,
                "starting_cap": starting_cap,
                "final_cap_tried": cap,
                "doublings": attempt,
                "observed_max_completion_tokens": max_observed,
                "chosen_max_tokens": chosen,
                "any_truncated": False,
                "per_item": results,
            }
        # At least one truncation: double the cap, retry.
        cap *= 2

    # Exhausted doublings.
    return {
        "target_model": target_key,
        "served_name": spec.served_name,
        "starting_cap": starting_cap,
        "final_cap_tried": cap,
        "doublings": max_doublings,
        "observed_max_completion_tokens": None,
        "chosen_max_tokens": None,
        "any_truncated": True,
        "per_item": results,
        "error": f"still truncated after {max_doublings} doublings",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="pilot run directory (output/ichl/detection/runs/<ts>_...)")
    ap.add_argument("--targets", nargs="+", required=True,
                    help="target-model internal keys (e.g. qwen3-8b qwen2.5-7b-instruct)")
    ap.add_argument("--candidate", default="A1_minimal_neutral",
                    help="which seed to use for the probe (default A1)")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    # Load the A1 prompt template.
    seeds_path = Path(__file__).resolve().parents[1] / "prompts" / "detection" / "seeds.yaml"
    seeds = load_seeds(seeds_path)
    seed = next((s for s in seeds if s.name == args.candidate), None)
    if seed is None:
        raise SystemExit(f"candidate '{args.candidate}' not in {seeds_path}")

    print(f"Step 0 run_dir: {run_dir}")
    print(f"Using candidate: {seed.name}")
    print(f"Targets: {args.targets}")

    out_path = run_dir / "step0_token_budget.json"
    # Merge with any existing result (so re-running for one target preserves the other).
    existing: dict[str, Any] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())

    all_results = existing.get("per_target", {})
    for target in args.targets:
        starting_cap = STARTING_CAPS.get(target, 2048)
        res = _run_target(
            target_key=target, template=seed.template,
            starting_cap=starting_cap,
            log_dir=run_dir / "logs",
        )
        all_results[target] = res

    all_ready = all(not r.get("any_truncated", True) for r in all_results.values())
    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": seed.name,
        "per_target": all_results,
        "gate_passed": all_ready,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Gate passed (no truncations): {all_ready}")
    for t, r in all_results.items():
        print(f"  {t}: chosen_max_tokens = {r.get('chosen_max_tokens')}, "
              f"observed_max = {r.get('observed_max_completion_tokens')}")


if __name__ == "__main__":
    main()
