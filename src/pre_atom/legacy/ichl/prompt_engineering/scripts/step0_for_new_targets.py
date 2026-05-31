"""Step 0 + parser sub-pilot for BM / Llama / DeepSeek.

Uses Qwen2.5 top-1 (complex iterated prompt) as the probe to set a worst-case
max_tokens budget. Then runs a 5-item sub-pilot with BOTH parsers to validate
regex reliability on each new target's output shape.

Per `Claude: Principle: Regex Parser Unreliability § Parsers are per-model`:
every new target needs its own sub-pilot before we trust any parser on it.

Usage:
    python -m ichl.prompt_engineering.scripts.step0_for_new_targets \\
        --run-dir <run_dir> --targets biomistral-7b llama-3.1-8b-instruct deepseek-r1-distill-llama-8b
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
from ichl.common.pilot_loader import load_detection_pilot_sub, load_detection_pilot
from ichl.prompt_engineering.evaluator import SYSTEM_MSG
from ichl.prompt_engineering.parsers import LLMParser, RegexParser


STARTING_CAPS: dict[str, int] = {
    "biomistral-7b": 2048,
    "llama-3.1-8b-instruct": 2048,
    "deepseek-r1-distill-llama-8b": 16384,
}


def _probe_template(run_dir: Path) -> tuple[str, str]:
    """Use Qwen2.5 top-1 iteration winner as a worst-case-complexity probe."""
    path = run_dir / "iteration" / "qwen2.5-7b-instruct" / "top_candidates_corrected" / "candidate_01.txt"
    if not path.exists():
        path = run_dir / "iteration" / "qwen2.5-7b-instruct" / "top_candidates" / "candidate_01.txt"
    text = path.read_text()
    lines = text.splitlines()
    header_end = 0
    for idx, ln in enumerate(lines):
        if not ln.startswith("#") and ln.strip():
            header_end = idx; break
    name = next((ln.replace("# name:", "").strip() for ln in lines[:header_end] if ln.startswith("# name:")), "probe")
    return name, "\n".join(lines[header_end:]).strip()


def _pick_items(target: str, n: int) -> list[dict[str, Any]]:
    all_items = load_detection_pilot(target)
    sorted_items = sorted(all_items, key=lambda i: len(i["note"]))
    if len(sorted_items) < n:
        return sorted_items
    idx = [int(p * (len(sorted_items) - 1)) for p in [0.05, 0.25, 0.5, 0.75, 0.95]]
    return [sorted_items[i] for i in idx[:n]]


def _step0(target: str, probe_name: str, template: str, starting_cap: int, log_dir: Path) -> dict[str, Any]:
    """Find a max_tokens that doesn't truncate the probe."""
    vllm_manager.ensure_model(target, log_dir=log_dir)
    client = make_client(target)
    items = _pick_items(target, 5)
    print(f"\n── step0 / {target} ── probe={probe_name[:60]}  5 items (note lens: {[len(it['note']) for it in items]})")

    cap = starting_cap
    for attempt in range(5):
        results = []
        any_trunc = False
        for item in items:
            prompt = template.format(
                note=item["note"], question=item["question"],
                model_answer=item["model_answer"], answer=item["model_answer"], choices="",
            )
            t0 = time.monotonic()
            resp = client.call(system=SYSTEM_MSG, user=prompt, temperature=0.0, max_tokens=cap)
            lat = time.monotonic() - t0
            ct = resp.usage.get("completion_tokens") if resp.usage else None
            fr = resp.finish_reason
            unclosed = "<think>" in (resp.text or "") and "</think>" not in (resp.text or "")
            trunc = (fr == "length") or unclosed
            if trunc: any_trunc = True
            results.append({
                "pilot_item_id": item["pilot_item_id"], "note_chars": len(item["note"]),
                "completion_tokens": ct, "finish_reason": fr, "truncated": trunc,
                "unclosed_think": unclosed, "latency_s": round(lat, 2), "success": resp.success,
            })
            print(f"   {item['pilot_item_id']}  ct={ct}  finish={fr}  lat={lat:.1f}s  {'[TRUNC]' if trunc else ''}")

        max_obs = max((r["completion_tokens"] or 0) for r in results)
        print(f"   attempt {attempt+1}: cap={cap} max_obs={max_obs} any_trunc={any_trunc}")
        if not any_trunc:
            chosen = max(256, math.ceil(2.0 * max_obs))  # 2x safety for untested models
            return {
                "target_model": target, "probe": probe_name,
                "starting_cap": starting_cap, "final_cap_tried": cap, "doublings": attempt,
                "observed_max_completion_tokens": max_obs, "chosen_max_tokens": chosen,
                "any_truncated": False, "per_item": results,
            }
        cap = min(cap * 2, 32768)
    return {
        "target_model": target, "probe": probe_name,
        "chosen_max_tokens": None, "any_truncated": True, "per_item": results,
        "error": "still truncated after 5 doublings",
    }


def _sub_pilot(target: str, template_name: str, template: str, max_tokens: int, out_dir: Path) -> dict[str, Any]:
    """5-item parser sub-pilot with dual-parser agreement check."""
    client = make_client(target)
    regex_parser = RegexParser()
    llm_parser = LLMParser()
    items = load_detection_pilot_sub(target, n_total=5, n_incorrect=2)
    print(f"\n── sub-pilot / {target} ── 5 items with {template_name[:60]}")

    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        prompt = template.format(
            note=item["note"], question=item["question"],
            model_answer=item["model_answer"], answer=item["model_answer"], choices="",
        )
        resp = client.call(system=SYSTEM_MSG, user=prompt, temperature=0.0, max_tokens=max_tokens)
        rx = regex_parser.parse(resp.text)
        lx = llm_parser.parse(resp.text)
        row = {
            "pilot_item_id": item["pilot_item_id"],
            "gt": "CORRECT" if item["binary_correct"] == 1 else "INCORRECT",
            "text_clean_preview": (resp.text or "")[:300],
            "finish_reason": resp.finish_reason,
            "regex_verdict": rx.verdict, "llm_verdict": lx.verdict,
            "agree": rx.verdict == lx.verdict,
            "completion_tokens": resp.usage.get("completion_tokens") if resp.usage else None,
        }
        rows.append(row)
        print(f"   {item['pilot_item_id']}  gt={row['gt']}  regex={rx.verdict}  llm={lx.verdict}  "
              f"agree={row['agree']}  finish={resp.finish_reason}")

    (out_dir / f"{target}_sub_pilot.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in rows) + "\n"
    )
    n_agree = sum(1 for r in rows if r["agree"])
    print(f"   agreement: {n_agree}/5 = {100*n_agree/5:.0f}%")
    return {
        "target_model": target, "agreement_pct": 100 * n_agree / 5,
        "n_rows": len(rows), "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    probe_name, template = _probe_template(run_dir)
    print(f"Using probe: {probe_name[:80]}")

    step0_path = run_dir / "step0_token_budget.json"
    existing = json.loads(step0_path.read_text()) if step0_path.exists() else {"per_target": {}}

    sub_pilot_dir = run_dir / "sub_pilot" / "cross_model"
    sub_pilot_results = []

    for target in args.targets:
        # Step 0
        res = _step0(target, probe_name, template, STARTING_CAPS.get(target, 4096),
                     log_dir=run_dir / "logs")
        existing["per_target"][target] = res
        if res.get("chosen_max_tokens") is None:
            print(f"  [warn] {target}: step0 failed; skipping sub-pilot")
            continue
        mt = int(res["chosen_max_tokens"])
        sp = _sub_pilot(target, probe_name, template, mt, sub_pilot_dir)
        sub_pilot_results.append(sp)

    step0_path.write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": existing.get("note", "") + " | Extended with new targets 2026-04-22 using Qwen2.5 top-1 probe.",
        "per_target": existing["per_target"],
        "gate_passed": all(not e.get("any_truncated", True) for e in existing["per_target"].values()),
    }, indent=2, default=str))

    sp_summary = sub_pilot_dir / "cross_model_sub_pilot_summary.json"
    sp_summary.write_text(json.dumps(sub_pilot_results, indent=2, default=str))
    print(f"\nStep 0 + sub-pilot complete.")
    print(f"  Step 0 budgets: {step0_path}")
    print(f"  Sub-pilot summary: {sp_summary}")
    for sp in sub_pilot_results:
        print(f"    {sp['target_model']}: {sp['agreement_pct']:.0f}% parser agreement")


if __name__ == "__main__":
    main()
