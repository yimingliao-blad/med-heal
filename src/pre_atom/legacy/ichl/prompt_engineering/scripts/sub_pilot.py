"""Step 1 — Parser-design sub-pilot (5 items × 3 candidates × 2 targets = 30 calls).

Per Notion 'Claude: Plan: Detection — Pilot Runner Design' § Step 1:

  1. Loader call: load_detection_pilot_sub(n_total=5, n_incorrect=2) → 5 items.
  2. For each of A1 / A2 / A3, call the target client with temperature=0 and
     the per-target max_tokens from Step 0; save raw response. → 3 × 2 × 5 = 30 raw outputs.
  3. Human-in-the-loop inspection (Claude reads the raw outputs): note where
     the verdict actually appears, any thinking-mode leaks, any unexpected
     phrasings.
  4. Draft regex + LLM-parser prompt; run both against the 30 outputs;
     iterate until agreement = 100% on the sub-pilot.
  5. Save: sub_pilot/raw_outputs/, sub_pilot/regex_pattern.txt,
     sub_pilot/llm_parser_prompt.txt, sub_pilot/notes.md.

This script handles phase A (generate + save raw outputs) and phase C
(run both parsers, compute agreement) of Step 1. Phase B (human
inspection) happens between: Claude reads the raw outputs, drafts parser
config, saves it, then re-runs phase C until agreement = 100%.

Usage:
    # Phase A: generate (call the currently-served vLLM target model)
    python -m ichl.prompt_engineering.scripts.sub_pilot generate \\
        --run-dir <run_dir> --targets qwen3-8b qwen2.5-7b-instruct

    # Phase C: parse + agreement (no LLM calls to target; just parsers)
    python -m ichl.prompt_engineering.scripts.sub_pilot parse \\
        --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_pilot_sub
from ichl.prompt_engineering.evaluator import SYSTEM_MSG
from ichl.prompt_engineering.parsers import LLMParser, RegexParser
from ichl.prompt_engineering.pool import load_seeds


SEEDS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "detection" / "seeds.yaml"


# ─────────────────────── phase A: generate ───────────────────────

def _generate_one_target(
    target_key: str,
    seeds: list,
    max_tokens: int,
    sub_pilot_dir: Path,
    n_total: int = 5,
    n_incorrect: int = 2,
) -> None:
    """Generate all candidate × item outputs for one target. vLLM must serve this target."""
    vllm_manager.ensure_model(target_key, log_dir=sub_pilot_dir / "logs")
    client = make_client(target_key)

    items = load_detection_pilot_sub(target_key, n_total=n_total, n_incorrect=n_incorrect)
    print(f"\n── sub-pilot / {target_key} ── {len(items)} items "
          f"(incorrect={sum(1 for i in items if i['binary_correct']==0)})")

    # Save the selected items so we can re-parse later.
    items_path = sub_pilot_dir / f"sub_pilot_items_{target_key}.jsonl"
    with open(items_path, "w") as f:
        for it in items:
            f.write(json.dumps(it, default=str) + "\n")

    for seed in seeds:
        out_dir = sub_pilot_dir / "raw_outputs" / seed.name / target_key
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            prompt = seed.template.format(
                note=item["note"], question=item["question"],
                model_answer=item["model_answer"],
                answer=item["model_answer"], choices="",
            )
            t0 = time.monotonic()
            resp = client.call(
                system=SYSTEM_MSG, user=prompt,
                temperature=0.0, max_tokens=max_tokens,
            )
            lat = time.monotonic() - t0
            out_path = out_dir / f"{item['pilot_item_id']}.json"
            out_path.write_text(json.dumps({
                "pilot_item_id": item["pilot_item_id"],
                "candidate": seed.name,
                "target_model": target_key,
                "binary_correct": item["binary_correct"],
                "user_prompt": prompt,
                "system_prompt": SYSTEM_MSG,
                "raw_response": resp.raw_text,
                "text_clean": resp.text,
                "latency_s": round(lat, 2),
                "finish_reason": resp.finish_reason,
                "usage": resp.usage,
                "success": resp.success,
                "error": resp.error,
            }, default=str, indent=2))
            print(f"  {seed.name} / {item['pilot_item_id']}  ct={resp.usage.get('completion_tokens')}  "
                  f"finish={resp.finish_reason}  lat={lat:.1f}s")


def cmd_generate(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    sub_pilot_dir = run_dir / "sub_pilot"
    sub_pilot_dir.mkdir(parents=True, exist_ok=True)
    (sub_pilot_dir / "logs").mkdir(exist_ok=True)

    # Load Step 0 budgets.
    step0_path = run_dir / "step0_token_budget.json"
    if not step0_path.exists():
        raise SystemExit(f"Step 0 budget file missing: {step0_path}. Run step0_token_budget first.")
    budgets = json.loads(step0_path.read_text())
    if not budgets.get("gate_passed"):
        raise SystemExit("Step 0 gate not passed (truncations present). Fix before sub-pilot.")

    seeds = load_seeds(SEEDS_PATH)
    print(f"Sub-pilot generate: {len(seeds)} candidates, targets={args.targets}")
    for target in args.targets:
        per = budgets.get("per_target", {}).get(target)
        if per is None:
            raise SystemExit(f"No Step 0 entry for target '{target}'")
        max_tokens = int(per["chosen_max_tokens"])
        _generate_one_target(target, seeds, max_tokens, sub_pilot_dir)
    print(f"\nDone. Raw outputs: {sub_pilot_dir / 'raw_outputs'}")


# ─────────────────────── phase C: parse + agreement ───────────────────────

def cmd_parse(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    sub_pilot_dir = run_dir / "sub_pilot"
    raw_dir = sub_pilot_dir / "raw_outputs"
    if not raw_dir.exists():
        raise SystemExit(f"raw_outputs missing: {raw_dir}. Run `generate` first.")

    # Parser config — search order (first hit wins):
    #   1. sub_pilot/parser_configs/<target_model>/{regex_pattern,llm_parser_prompt}.*  (per-target override)
    #   2. sub_pilot/parser_configs/shared_qwen_verdict_only/...                         (shared default for Qwen2.5 + Qwen3)
    #   3. legacy flat sub_pilot/{regex_pattern,llm_parser_prompt}.txt                  (deprecated; kept for back-compat)
    # Per `Claude: Principle: Regex Parser Unreliability` § Parsers are per-model:
    # per-target overrides are the correct long-term shape once we have more than
    # one target family. For the current Qwen-verdict pilot, the shared default is
    # used for every item.
    def _find_config(filename: str) -> Path | None:
        pc = sub_pilot_dir / "parser_configs"
        shared_dir = pc / "shared_qwen_verdict_only"
        for candidate in [shared_dir / filename, sub_pilot_dir / filename]:
            if candidate.exists():
                return candidate
        return None

    regex_file = _find_config("regex_pattern.txt")
    llm_file = _find_config("llm_parser_prompt.txt")
    regex_pattern = regex_file.read_text().strip() if regex_file else None
    llm_user_tpl = llm_file.read_text() if llm_file else None

    regex_parser = RegexParser(pattern=regex_pattern) if regex_pattern else RegexParser()
    llm_parser = LLMParser(user_template=llm_user_tpl) if llm_user_tpl else LLMParser()

    rows: list[dict[str, Any]] = []
    n_total = 0
    n_agree = 0
    for jf in sorted(raw_dir.rglob("*.json")):
        data = json.loads(jf.read_text())
        clean = data.get("text_clean", "")
        raw = data.get("raw_response", "")
        regex_res = regex_parser.parse(clean)
        llm_res = llm_parser.parse(clean)
        agree = regex_res.verdict == llm_res.verdict
        n_total += 1
        n_agree += int(agree)
        gt = data.get("binary_correct")
        expected = "CORRECT" if gt == 1 else "INCORRECT"
        rows.append({
            "candidate": data.get("candidate"),
            "target_model": data.get("target_model"),
            "pilot_item_id": data.get("pilot_item_id"),
            "binary_correct": gt,
            "expected_verdict": expected,
            "regex_verdict": regex_res.verdict,
            "llm_verdict": llm_res.verdict,
            "agree": agree,
            "regex_match_pos": regex_res.match_pos,
            "regex_match_text": regex_res.match_text,
            "llm_notes": llm_res.notes,
            "raw_text_len": len(raw),
            "clean_text_len": len(clean),
            "finish_reason": data.get("finish_reason"),
        })

    csv_path = sub_pilot_dir / "parser_agreement_sub_pilot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}  ({n_total} rows)")

    # Summary
    from collections import Counter
    per_cand = Counter()
    per_cand_agree = Counter()
    for r in rows:
        key = (r["candidate"], r["target_model"])
        per_cand[key] += 1
        per_cand_agree[key] += int(r["agree"])

    print(f"\n=== SUB-PILOT PARSER AGREEMENT ===")
    print(f"Overall: {n_agree}/{n_total} = {100*n_agree/n_total:.1f}%")
    print(f"\nPer cell (candidate × target):")
    for (c, t), tot in sorted(per_cand.items()):
        ag = per_cand_agree[(c, t)]
        print(f"  {c:28} {t:24}  {ag}/{tot} = {100*ag/tot:.0f}%")

    # Disagreements
    disagree = [r for r in rows if not r["agree"]]
    if disagree:
        print(f"\n=== DISAGREEMENTS ({len(disagree)}) ===")
        for d in disagree:
            print(f"  {d['candidate']} / {d['target_model']} / {d['pilot_item_id']}: "
                  f"regex={d['regex_verdict']}  llm={d['llm_verdict']}  "
                  f"gt={d['expected_verdict']}")

    # Ground-truth correctness summary (not the goal of sub-pilot, but useful).
    correct_regex = sum(1 for r in rows if r["regex_verdict"] == r["expected_verdict"])
    correct_llm = sum(1 for r in rows if r["llm_verdict"] == r["expected_verdict"])
    print(f"\nAccuracy vs GT (not the goal — just context):")
    print(f"  regex: {correct_regex}/{n_total} = {100*correct_regex/n_total:.0f}%")
    print(f"  llm:   {correct_llm}/{n_total} = {100*correct_llm/n_total:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate raw outputs for A1/A2/A3 × each target")
    g.add_argument("--run-dir", type=Path, required=True)
    g.add_argument("--targets", nargs="+", required=True)
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("parse", help="Run regex + LLM parsers on saved outputs; report agreement")
    p.add_argument("--run-dir", type=Path, required=True)
    p.set_defaults(func=cmd_parse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
