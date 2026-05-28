"""Step 0 — token-budget probe for T0 correction.

For ONE target model, pick 5 detection-flagged items at note-length
percentiles 5/25/50/75/95, run each of the 5 sub-variants once, record
{completion_tokens, finish_reason, unclosed_think_block, latency_s, success}
and recommend chosen_max_tokens per sub-variant.

Rule:
    chosen_max_tokens = max(ceil(2 * max_observed_completion_tokens),
                            starting_cap // 2)
If any call truncated (finish_reason == 'length'), DOUBLE the cap and
re-run that sub-variant (up to 5 doublings).

Usage:
    python -m ichl.prompt_engineering.correction.step0_probe \\
        --target deepseek-r1-distill-llama-8b \\
        --detection-jsonl <path to Stage-I cell JSONL> \\
        --run-dir output/ichl/correction/runs/<ts>_t0_anchor \\
        [--starting-max-tokens 16384] [--temperature 1.0] \\
        [--sub-variants a b c d e]

Outputs under run-dir:
    step0_probe_<target_key>.json
    raw_outputs/<target_key>/<sub_variant>/<pilot_item_id>.json
    logs/<client_name>_calls.jsonl  (via the LLM client's per-call audit log)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.prompt_engineering.correction.data_loader import load_correction_items
from ichl.prompt_engineering.correction.runner import run_correction_one_item
from ichl.prompt_engineering.correction.sub_variants import SUB_VARIANTS


# Percentiles we probe (per the spec).
PERCENTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# Starting max_tokens caps per target (Step 0 doubles on truncation up to 5x).
STARTING_CAPS: dict[str, int] = {
    "deepseek-r1-distill-llama-8b": 16384,   # DS always thinks
    "qwen3-8b": 8192,                        # think-on
    "qwen2.5-7b-instruct": 2048,             # non-think
    "llama-3.1-8b-instruct": 2048,           # non-think
    "biomistral-7b": 2048,                   # non-think
}


def _pick_percentile_items(
    items: list[dict[str, Any]], percentiles: list[float],
) -> list[dict[str, Any]]:
    """Pick one item at each given percentile of note length (deterministic)."""
    if not items:
        return []
    sorted_items = sorted(items, key=lambda it: len(it["note"]))
    n = len(sorted_items)
    picked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for p in percentiles:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        it = sorted_items[idx]
        if it["pilot_item_id"] in seen_ids:
            # Scan forward, then backward, to avoid duplicates.
            j = idx
            while j < n and sorted_items[j]["pilot_item_id"] in seen_ids:
                j += 1
            if j == n:
                j = idx
                while j >= 0 and sorted_items[j]["pilot_item_id"] in seen_ids:
                    j -= 1
            if j < 0 or j >= n:
                continue
            it = sorted_items[j]
        picked.append(it)
        seen_ids.add(it["pilot_item_id"])
    return picked


def _chosen_from_observed(
    max_observed: int, starting_cap: int,
) -> int:
    """chosen = max(ceil(2 * max_observed), starting_cap // 2). Floor at 64."""
    if max_observed is None or max_observed <= 0:
        return max(64, starting_cap // 2)
    return max(64, math.ceil(2 * max_observed), starting_cap // 2)


def run_probe_for_sub_variant(
    client, target: str, sub_variant_id: str, picked_items: list[dict[str, Any]],
    starting_cap: int, max_doublings: int, raw_log_dir: Path,
) -> dict[str, Any]:
    """Run one sub-variant across the 5 picked items, doubling cap on truncation."""
    cap = starting_cap
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_doublings + 1):
        per_item_records: list[dict[str, Any]] = []
        per_item_summary: list[dict[str, Any]] = []
        any_truncated = False
        for it in picked_items:
            rec = run_correction_one_item(
                client=client,
                target=target,
                sub_variant_id=sub_variant_id,
                item=it,
                max_tokens=cap,
                temperature=1.0,
                raw_log_dir=raw_log_dir,
            )
            per_item_records.append(rec)
            if rec["truncated"]:
                any_truncated = True
            per_item_summary.append({
                "pilot_item_id": rec["pilot_item_id"],
                "note_chars": rec["note_chars"],
                "completion_tokens": rec["completion_tokens"],
                "prompt_tokens": rec["prompt_tokens"],
                "finish_reason": rec["finish_reason"],
                "truncated": rec["truncated"],
                "unclosed_think_block": rec["unclosed_think_block"],
                "latency_s": rec["latency_s"],
                "success": rec["success"],
                "error": rec.get("error"),
            })
            flag = ""
            if rec["truncated"]:
                flag += " [TRUNC]"
            if rec["unclosed_think_block"]:
                flag += " [UNCLOSED_THINK]"
            if not rec["success"]:
                flag += f" [FAIL:{rec.get('error')}]"
            print(f"    {sub_variant_id}/{rec['pilot_item_id']}  "
                  f"ct={rec['completion_tokens']}  fr={rec['finish_reason']}  "
                  f"lat={rec['latency_s']}s{flag}")
        max_observed = max((r["completion_tokens"] or 0) for r in per_item_summary)
        attempts.append({
            "attempt": attempt + 1,
            "cap": cap,
            "max_observed_completion_tokens": max_observed,
            "any_truncated": any_truncated,
            "per_item": per_item_summary,
        })
        print(f"  [sub-variant {sub_variant_id}] attempt {attempt+1}: cap={cap} "
              f"max_observed={max_observed} any_truncated={any_truncated}")
        if not any_truncated:
            chosen = _chosen_from_observed(max_observed, starting_cap)
            return {
                "sub_variant_id": sub_variant_id,
                "sub_variant_name": SUB_VARIANTS[sub_variant_id].name,
                "starting_cap": starting_cap,
                "final_cap_tried": cap,
                "doublings": attempt,
                "observed_max_completion_tokens": max_observed,
                "chosen_max_tokens": chosen,
                "any_truncated": False,
                "attempts": attempts,
                "finish_reasons": _finish_reason_distribution(per_item_summary),
                "unclosed_think_count": sum(
                    1 for r in per_item_summary if r["unclosed_think_block"]
                ),
            }
        cap *= 2
    # Exhausted doublings.
    return {
        "sub_variant_id": sub_variant_id,
        "sub_variant_name": SUB_VARIANTS[sub_variant_id].name,
        "starting_cap": starting_cap,
        "final_cap_tried": cap,
        "doublings": max_doublings,
        "observed_max_completion_tokens": max_observed,
        "chosen_max_tokens": None,
        "any_truncated": True,
        "attempts": attempts,
        "finish_reasons": _finish_reason_distribution(attempts[-1]["per_item"]),
        "unclosed_think_count": sum(
            1 for r in attempts[-1]["per_item"] if r["unclosed_think_block"]
        ),
        "error": f"still truncated after {max_doublings} doublings",
    }


def _finish_reason_distribution(per_item: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in per_item:
        fr = r.get("finish_reason") or "<none>"
        dist[fr] = dist.get(fr, 0) + 1
    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target model internal key")
    ap.add_argument("--detection-jsonl", required=True, type=Path,
                    help="Stage-I detection fullscale result JSONL for this target")
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="run directory (will be created)")
    ap.add_argument("--sub-variants", nargs="+", default=list(SUB_VARIANTS.keys()),
                    help="sub-variant IDs to probe (default: all 5)")
    ap.add_argument("--starting-max-tokens", type=int, default=None,
                    help="override starting cap (default from STARTING_CAPS)")
    ap.add_argument("--max-doublings", type=int, default=5)
    ap.add_argument("--percentiles", nargs="+", type=float, default=PERCENTILES)
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    raw_base = run_dir / "raw_outputs" / args.target
    raw_base.mkdir(parents=True, exist_ok=True)

    starting_cap = args.starting_max_tokens or STARTING_CAPS.get(args.target, 4096)
    print("\n=== Step 0 Probe ===")
    print(f"target:        {args.target}")
    print(f"detection:     {args.detection_jsonl}")
    print(f"run_dir:       {run_dir}")
    print(f"sub_variants:  {args.sub_variants}")
    print(f"starting_cap:  {starting_cap}")
    print(f"percentiles:   {args.percentiles}")

    # 1) Load flagged items for this target.
    t_load = time.monotonic()
    all_flagged = load_correction_items(
        target_model=args.target,
        detection_jsonl_path=args.detection_jsonl,
    )
    print(f"flagged items: {len(all_flagged)} "
          f"(loaded in {time.monotonic() - t_load:.1f}s)")
    if not all_flagged:
        raise SystemExit("No flagged items found. Check detection JSONL path/target.")

    picked = _pick_percentile_items(all_flagged, args.percentiles)
    if len(picked) < len(args.percentiles):
        print(f"  WARN: only {len(picked)} unique percentile items "
              f"({len(args.percentiles)} requested).")
    print(f"picked items ({len(picked)}): " +
          ", ".join(f"{p:.0%}={it['pilot_item_id']}(n={len(it['note'])})"
                    for p, it in zip(args.percentiles, picked)))

    # 2) Build client once.
    client = make_client(args.target)

    # 3) Run each sub-variant.
    probe_start = time.monotonic()
    per_sub: dict[str, Any] = {}
    for sv_id in args.sub_variants:
        if sv_id not in SUB_VARIANTS:
            print(f"  SKIP unknown sub-variant '{sv_id}'")
            continue
        print(f"\n-- sub-variant {sv_id} ({SUB_VARIANTS[sv_id].name}) --")
        raw_dir = raw_base / sv_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        res = run_probe_for_sub_variant(
            client=client,
            target=args.target,
            sub_variant_id=sv_id,
            picked_items=picked,
            starting_cap=starting_cap,
            max_doublings=args.max_doublings,
            raw_log_dir=raw_dir,
        )
        per_sub[sv_id] = res
    total_wall_s = time.monotonic() - probe_start

    # 4) Summary.
    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target": args.target,
        "detection_jsonl": str(args.detection_jsonl),
        "starting_cap": starting_cap,
        "percentiles": args.percentiles,
        "picked_items": [
            {"pilot_item_id": it["pilot_item_id"],
             "patient_id": it["patient_id"],
             "fold": it["fold"],
             "note_chars": len(it["note"]),
             "A0_binary_correct": it.get("A0_binary_correct")}
            for it in picked
        ],
        "per_sub_variant": per_sub,
        "total_wall_s": round(total_wall_s, 2),
        "num_flagged_items_total": len(all_flagged),
    }
    out_path = run_dir / f"step0_probe_{_short_target(args.target)}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Total wall-clock: {total_wall_s:.1f}s")
    print("Per-sub-variant chosen max_tokens:")
    for sv_id, r in per_sub.items():
        print(f"  {sv_id} ({r['sub_variant_name']}): "
              f"observed_max={r['observed_max_completion_tokens']} "
              f"chosen_max_tokens={r['chosen_max_tokens']} "
              f"any_truncated={r['any_truncated']} "
              f"unclosed_think={r.get('unclosed_think_count', 0)}")


def _short_target(target: str) -> str:
    # deepseek-r1-distill-llama-8b -> deepseek
    # qwen2.5-7b-instruct          -> qwen2.5
    # qwen3-8b                     -> qwen3
    # llama-3.1-8b-instruct        -> llama
    t = target.lower()
    if t.startswith("deepseek"):
        return "deepseek"
    if t.startswith("qwen2.5"):
        return "qwen2.5"
    if t.startswith("qwen3"):
        return "qwen3"
    if t.startswith("llama"):
        return "llama"
    if t.startswith("biomistral"):
        return "biomistral"
    return t


if __name__ == "__main__":
    main()
