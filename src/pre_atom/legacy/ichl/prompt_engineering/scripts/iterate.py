"""Iteration orchestrator — runs the prompt-engineering optimizer per target.

Per Notion `Claude: Plan: Detection — Pilot Runner Design` and the follow-on
iteration discussion:
  1. For each target (Qwen2.5 → Qwen3): ensure vLLM serves it, then run the
     full optimizer loop (Round 0 baseline → Rounds 1..5 mutate+polish+prune).
  2. One vLLM client + one MLX parser client is built per target; reused
     across all variants.
  3. Per-target output dir: <run_dir>/iteration/<target>/

Usage:
    python -m ichl.prompt_engineering.scripts.iterate \\
        --run-dir <run_dir> --targets qwen2.5-7b-instruct qwen3-8b \\
        [--max-rounds 5] [--variants-per-candidate 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_pilot
from ichl.prompt_engineering import optimizer


SEEDS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "detection" / "seeds.yaml"
VARIATIONS_PATH = Path(__file__).resolve().parents[1] / "variations" / "detection.yaml"


def _load_parser_config(sub_pilot_dir: Path, target: str) -> tuple[str | None, str | None]:
    """Same search order as main_pilot.py."""
    pc = sub_pilot_dir / "parser_configs"
    search = [pc / target, pc / "shared_qwen_verdict_only", sub_pilot_dir]
    regex = None
    llm = None
    for d in search:
        rf = d / "regex_pattern.txt"
        if regex is None and rf.exists():
            regex = rf.read_text().strip()
        lf = d / "llm_parser_prompt.txt"
        if llm is None and lf.exists():
            llm = lf.read_text()
        if regex is not None and llm is not None:
            break
    return regex, llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--variants-per-candidate", type=int, default=3)
    ap.add_argument("--top-n-candidates", type=int, default=3)
    ap.add_argument("--tool-model", default="gpt-4o")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    sub_pilot_dir = run_dir / "sub_pilot"
    iteration_dir = run_dir / "iteration"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    # Step 0 budgets gate.
    step0_path = run_dir / "step0_token_budget.json"
    budgets = json.loads(step0_path.read_text())
    if not budgets.get("gate_passed"):
        raise SystemExit("Step 0 gate not passed.")

    all_results: dict[str, dict[str, Any]] = {}

    for target in args.targets:
        print(f"\n{'='*64}")
        print(f"== Iteration for target: {target}")
        print(f"{'='*64}")

        # Swap vLLM.
        vllm_manager.ensure_model(target, log_dir=run_dir / "logs")

        # Per-target config.
        regex_pat, llm_tpl = _load_parser_config(sub_pilot_dir, target)
        max_tokens = int(budgets["per_target"][target]["chosen_max_tokens"])
        pilot = load_detection_pilot(target)
        out = iteration_dir / target

        result = optimizer.run(
            step="detection",
            target_client_name=target,
            base_prompt_pool=SEEDS_PATH,
            variation_pool=VARIATIONS_PATH,
            pilot_data=pilot,
            max_tokens=max_tokens,
            out_dir=out,
            tool_model=args.tool_model,
            top_n_candidates=args.top_n_candidates,
            max_rounds=args.max_rounds,
            variants_per_candidate=args.variants_per_candidate,
            regex_pattern=regex_pat,
            llm_parser_user_tpl=llm_tpl,
        )
        print(f"\n── Final top-{args.top_n_candidates} for {target} ──")
        for i, c in enumerate(result.top_candidates, start=1):
            print(f"  #{i}  {c.name}  acc={c.score:.3f}  round_added={c.round_added}")

        all_results[target] = {
            "top_candidates": [
                {"rank": i, "name": c.name, "score": c.score,
                 "round_added": c.round_added, "provenance": c.provenance}
                for i, c in enumerate(result.top_candidates, start=1)
            ],
            "history": result.history,
            "run_dir": str(result.run_dir),
        }

    # Aggregate per-run summary.
    summary_path = iteration_dir / "iteration_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n✅ Iteration complete for all {len(args.targets)} targets.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
