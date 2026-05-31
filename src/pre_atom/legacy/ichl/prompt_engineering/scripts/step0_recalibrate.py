"""Re-run Step 0 using a representative COMPLEX iteration prompt (not A1).

This addresses the lookback-audit finding: max_tokens calibrated on A1 was too
small for complex iteration variants with thinking-mode models. We recalibrate
using the Qwen3 top-1 variant (longest reasoning-inducing prompt) as the probe.

Per `Claude: Principle: Experiment Audit Guidelines § Lookback audit`: budget
must be recalibrated when the prompt family changes.

Usage:
    python -m ichl.prompt_engineering.scripts.step0_recalibrate \\
        --run-dir <run_dir> --targets qwen2.5-7b-instruct qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_pilot
from ichl.prompt_engineering.evaluator import SYSTEM_MSG


# Much more generous starting caps — we now know the old ones are too small.
STARTING_CAPS: dict[str, int] = {
    "qwen3-8b": 8192,
    "qwen2.5-7b-instruct": 1024,
    "biomistral-7b": 2048,
    "llama-3.1-8b-instruct": 2048,
    "deepseek-r1-distill-llama-8b": 16384,
}


def _load_probe_template(run_dir: Path, target: str) -> tuple[str, str]:
    """Use the top-1 iteration winner as the probe (worst-case prompt complexity)."""
    top1_path = run_dir / "iteration" / target / "top_candidates" / "candidate_01.txt"
    if not top1_path.exists():
        # Fallback: use A1 if no iteration result yet.
        seeds_path = Path(__file__).resolve().parents[1] / "prompts" / "detection" / "seeds.yaml"
        from ichl.prompt_engineering.pool import load_seeds
        seed = load_seeds(seeds_path)[0]
        return seed.name, seed.template

    text = top1_path.read_text()
    lines = text.splitlines()
    header_end = 0
    for idx, ln in enumerate(lines):
        if not ln.startswith("#") and ln.strip():
            header_end = idx
            break
    name = next((ln.replace("# name:", "").strip() for ln in lines[:header_end] if ln.startswith("# name:")), "top1")
    return name, "\n".join(lines[header_end:]).strip()


def _pick_items(target: str, n: int = 5) -> list[dict[str, Any]]:
    """5 items instead of 3 now — better coverage of note-length variance."""
    all_items = load_detection_pilot(target)
    sorted_items = sorted(all_items, key=lambda i: len(i["note"]))
    n_items = len(sorted_items)
    if n_items < n:
        return sorted_items
    idx = [int(p * (n_items - 1)) for p in [0.05, 0.25, 0.50, 0.75, 0.95]]
    return [sorted_items[i] for i in idx]


def _run(target: str, probe_name: str, template: str, starting_cap: int,
         max_doublings: int = 4, log_dir: Path | None = None) -> dict[str, Any]:
    vllm_manager.ensure_model(target, log_dir=log_dir)
    client = make_client(target)
    items = _pick_items(target, n=5)
    print(f"\n── recalibrate / {target} ── probe={probe_name[:80]}")
    print(f"   5 items, note lens: {[len(it['note']) for it in items]}")

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
            text_clean = resp.text or ""
            unclosed_think = "<think>" in text_clean and "</think>" not in text_clean
            if truncated or unclosed_think:
                any_truncated = True
            results.append({
                "pilot_item_id": item["pilot_item_id"],
                "note_chars": len(item["note"]),
                "completion_tokens": ct,
                "finish_reason": fr,
                "truncated": truncated,
                "unclosed_think_in_clean": unclosed_think,
                "latency_s": round(lat, 2),
                "success": resp.success,
            })
            print(f"   {item['pilot_item_id']}  ct={ct}  finish={fr}  lat={lat:.1f}s"
                  f"  {'[TRUNC]' if truncated else ''}"
                  f"{'  [UNCLOSED-THINK]' if unclosed_think else ''}")

        max_observed = max((r["completion_tokens"] or 0) for r in results)
        print(f"   attempt {attempt+1}: cap={cap}  max_observed={max_observed}  "
              f"any_truncated={any_truncated}")

        if not any_truncated:
            chosen = max(256, math.ceil(1.5 * max_observed))
            return {
                "target_model": target, "probe": probe_name,
                "starting_cap": starting_cap, "final_cap_tried": cap,
                "doublings": attempt,
                "observed_max_completion_tokens": max_observed,
                "chosen_max_tokens": chosen,
                "any_truncated": False, "per_item": results,
            }
        cap = min(cap * 2, 32768)

    return {
        "target_model": target, "probe": probe_name,
        "starting_cap": starting_cap, "final_cap_tried": cap,
        "doublings": max_doublings,
        "observed_max_completion_tokens": None,
        "chosen_max_tokens": None, "any_truncated": True,
        "per_item": results,
        "error": f"still truncated after {max_doublings} doublings",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_path = run_dir / "step0_token_budget.json"

    existing: dict[str, Any] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())

    per_target = existing.get("per_target", {})
    # Archive the old entries for audit.
    if per_target:
        archive_path = run_dir / "step0_token_budget_recalibrated_archive.json"
        archive_path.write_text(json.dumps(existing, indent=2, default=str))
        print(f"archived prior step0 → {archive_path.name}")

    for target in args.targets:
        probe_name, template = _load_probe_template(run_dir, target)
        starting_cap = STARTING_CAPS.get(target, 4096)
        res = _run(target, probe_name, template, starting_cap, log_dir=run_dir / "logs")
        per_target[target] = res

    all_ready = all(not r.get("any_truncated", True) for r in per_target.values())
    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Recalibrated using iteration top-1 probes (post-lookback-audit).",
        "per_target": per_target,
        "gate_passed": all_ready,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Gate passed (no truncations): {all_ready}")
    for t, r in per_target.items():
        print(f"  {t}: chosen_max_tokens = {r.get('chosen_max_tokens')}  "
              f"observed_max = {r.get('observed_max_completion_tokens')}")


if __name__ == "__main__":
    main()
