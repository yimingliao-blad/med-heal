"""Full-scale for unified-prompt analysis across Qwen2.5 / Qwen3 / Llama / DS.

User-directive (2026-04-22): find ONE detection prompt that works across all
target models (not per-target optimization). BM is skipped (confirmed no prompt
beats its 53.8 % floor materially). DS uses Q3 top-2 (user preference for
recall over precision).

Cells specified explicitly below; resume-safe via per-cell JSONL files.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_fullscale
from ichl.prompt_engineering.evaluator import evaluate_cell
from ichl.prompt_engineering.parsers import LLMParser, RegexParser


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LEGACY_JSON = PROJECT_ROOT / "output" / "external_judge_benchmark" / "crossmodel_detection_results.json"
LEGACY_PROMPTS = ["P1_minimal", "P9_notes_first", "P12_self_verify"]


def _load_top_candidate(run_dir: Path, source_target: str, rank: int) -> tuple[str, str]:
    """Load a specific top-N candidate's template from iteration/corrected/."""
    for dirname in ("top_candidates_corrected", "top_candidates"):
        p = run_dir / "iteration" / source_target / dirname / f"candidate_{rank:02d}.txt"
        if p.exists():
            text = p.read_text()
            lines = text.splitlines()
            header_end = 0
            for i, ln in enumerate(lines):
                if not ln.startswith("#") and ln.strip():
                    header_end = i; break
            name = next((ln.replace("# name:", "").strip() for ln in lines[:header_end] if ln.startswith("# name:")), f"{source_target}_top{rank}")
            tpl = "\n".join(lines[header_end:]).strip()
            return name, tpl
    raise FileNotFoundError(f"No top-{rank} candidate for {source_target}")


def _load_parser_config(sub_pilot_dir: Path, target: str):
    pc = sub_pilot_dir / "parser_configs"
    search = [pc / target, pc / "shared_qwen_verdict_only", sub_pilot_dir]
    regex = None; llm = None
    for d in search:
        rf = d / "regex_pattern.txt"
        if regex is None and rf.exists(): regex = rf.read_text().strip()
        lf = d / "llm_parser_prompt.txt"
        if llm is None and lf.exists(): llm = lf.read_text()
        if regex and llm: break
    return regex, llm


def _pull_candidate_b(target: str, out_dir: Path) -> dict[str, int]:
    if not LEGACY_JSON.exists():
        return {}
    data = json.loads(LEGACY_JSON.read_text())
    out = {}
    for prompt in LEGACY_PROMPTS:
        key = f"{target}__{target}__{prompt}"
        bucket = data.get(key, [])
        out_path = out_dir / f"candidate_B_{prompt}_{target}_results.jsonl"
        with out_path.open("w") as f:
            for it in bucket:
                row = {
                    "candidate": f"B_{prompt}", "target_model": target,
                    "patient_id": it["patient_id"], "fold": it.get("fold"),
                    "binary_correct": it["binary_correct"],
                    "chosen_verdict": it.get("final_verdict", "UNKNOWN"),
                    "verdict_correct": (
                        1 if it.get("final_verdict") == ("CORRECT" if it["binary_correct"] == 1 else "INCORRECT") else 0
                    ),
                }
                f.write(json.dumps(row, default=str) + "\n")
        out[prompt] = len(bucket)
    return out


def _summary(rows: list[dict[str, Any]], cand: str, target: str) -> dict[str, Any]:
    n = len(rows)
    n_hit = sum(1 for r in rows if r.get("verdict_correct") == 1)
    n_unknown = sum(1 for r in rows if r.get("chosen_verdict") == "UNKNOWN")
    n_inc = sum(1 for r in rows if r.get("binary_correct") == 0)
    n_cor = sum(1 for r in rows if r.get("binary_correct") == 1)
    tp = sum(1 for r in rows if r.get("binary_correct") == 0 and r.get("chosen_verdict") == "INCORRECT")
    fp = sum(1 for r in rows if r.get("binary_correct") == 1 and r.get("chosen_verdict") == "INCORRECT")
    return {
        "candidate": cand, "target": target, "n": n,
        "accuracy": n_hit / n if n else 0,
        "inc_prec": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "inc_rec": tp / n_inc if n_inc > 0 else 0,
        "unknown": n_unknown,
        "n_incorrect_class": n_inc,
        "n_correct_class": n_cor,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    sub_pilot_dir = run_dir / "sub_pilot"
    out_dir = run_dir / "fullscale_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    # Load budgets
    step0 = json.loads((run_dir / "step0_token_budget.json").read_text())

    # Load templates by source target + rank
    # Qwen2.5 source: Q2.5 top-1/2/3
    q25_top1 = _load_top_candidate(run_dir, "qwen2.5-7b-instruct", 1)
    q25_top2 = _load_top_candidate(run_dir, "qwen2.5-7b-instruct", 2)
    q25_top3 = _load_top_candidate(run_dir, "qwen2.5-7b-instruct", 3)
    # Qwen3 source: Q3 top-1/2/3
    q3_top1 = _load_top_candidate(run_dir, "qwen3-8b", 1)
    q3_top2 = _load_top_candidate(run_dir, "qwen3-8b", 2)
    q3_top3 = _load_top_candidate(run_dir, "qwen3-8b", 3)

    # Cells to run (target, candidate_name, template). Resume-safe per-cell.
    cells = [
        # Qwen2.5 (fast; 3 native + 2 cross-test)
        ("qwen2.5-7b-instruct", *q25_top1),
        ("qwen2.5-7b-instruct", *q25_top2),
        ("qwen2.5-7b-instruct", *q25_top3),
        ("qwen2.5-7b-instruct", *q3_top2),   # cross-test: unified candidate
        ("qwen2.5-7b-instruct", *q3_top3),   # cross-test
        # Qwen3 (slow; 3 native + 1 cross-test)
        ("qwen3-8b", *q3_top1),
        ("qwen3-8b", *q3_top2),
        ("qwen3-8b", *q3_top3),
        ("qwen3-8b", *q25_top2),   # cross-test: unified candidate
        # Llama (2 variants — Q3 family transfers well)
        ("llama-3.1-8b-instruct", *q3_top2),
        ("llama-3.1-8b-instruct", *q3_top3),
        # DS (user pick: Q3 top-2, prioritize recall)
        ("deepseek-r1-distill-llama-8b", *q3_top2),
    ]

    # Group by target so we swap vLLM only once per target
    # (minimize swaps: qwen2.5 first, then qwen3, llama, ds)
    target_order = ["qwen2.5-7b-instruct", "qwen3-8b", "llama-3.1-8b-instruct", "deepseek-r1-distill-llama-8b"]
    all_summaries: list[dict[str, Any]] = []

    for target in target_order:
        target_cells = [c for c in cells if c[0] == target]
        if not target_cells:
            continue
        print(f"\n{'='*60}\n== {target} — {len(target_cells)} cells\n{'='*60}")
        vllm_manager.ensure_model(target, log_dir=out_dir / "logs")
        client = make_client(target)
        regex_pat, llm_tpl = _load_parser_config(sub_pilot_dir, target)
        parsers = [RegexParser(pattern=regex_pat) if regex_pat else RegexParser()]
        # Keep LLM parser too for the audit — per methodology: report disagreements
        parsers.append(LLMParser(user_template=llm_tpl) if llm_tpl else LLMParser())
        mt = int(step0["per_target"][target]["chosen_max_tokens"])
        print(f"  max_tokens={mt}")

        pilot_data = load_detection_fullscale(target)
        print(f"  {len(pilot_data)} items")

        for _, cand_name, template in target_cells:
            safe = cand_name.replace("/", "_").replace(" ", "_")[:120]
            per_item_path = out_dir / f"candidate_{safe}_{target}_results.jsonl"
            if per_item_path.exists():
                n_rows = sum(1 for _ in per_item_path.open())
                if n_rows >= len(pilot_data):
                    rows = [json.loads(l) for l in per_item_path.open() if l.strip()]
                    s = _summary(rows, cand_name, target)
                    all_summaries.append(s)
                    print(f"  [skip] {cand_name[:60]}: {n_rows} rows already")
                    continue
                else:
                    # Partial — delete and restart
                    print(f"  [restart partial] {cand_name[:60]}: had {n_rows}/{len(pilot_data)} rows")
                    per_item_path.unlink()

            log_dir = out_dir / "raw_outputs" / safe / target
            t0 = time.monotonic()
            result = evaluate_cell(
                candidate_name=cand_name, prompt_template=template,
                pilot_data=pilot_data, target_client=client, parsers=parsers,
                max_tokens=mt, log_dir=log_dir, per_item_path=per_item_path,
            )
            elapsed = time.monotonic() - t0
            rows = [json.loads(l) for l in per_item_path.open() if l.strip()]
            s = _summary(rows, cand_name, target)
            all_summaries.append(s)
            print(f"  {cand_name[:60]}: acc={100*s['accuracy']:.1f}%  prec={100*s['inc_prec']:.0f}%  "
                  f"rec={100*s['inc_rec']:.0f}%  unk={s['unknown']}  elapsed={elapsed/60:.1f}m")

        # Pull Candidate B legacy for this target (no compute)
        print(f"  pulling Candidate B legacy verdicts for {target}...")
        counts = _pull_candidate_b(target, out_dir)
        print(f"  {target} Candidate B: {counts}")
        for prompt in LEGACY_PROMPTS:
            jf = out_dir / f"candidate_B_{prompt}_{target}_results.jsonl"
            if jf.exists():
                rows = [json.loads(l) for l in jf.open() if l.strip()]
                if rows:
                    all_summaries.append(_summary(rows, f"B_{prompt}", target))

    # Write aggregate summary
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        if all_summaries:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            for s in all_summaries:
                w.writerow(s)
    (out_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2, default=str))

    # Markdown summary: sorted by target then accuracy
    lines = ["# Full-scale unified-prompt analysis — summary (962 items per cell)\n"]
    lines.append("| target | candidate | n | acc | inc-prec | inc-rec | unk | always-CORRECT floor |")
    lines.append("|---|---|---|---|---|---|---|---|")
    floors = {"qwen3-8b": 0.924, "llama-3.1-8b-instruct": 0.891,
              "qwen2.5-7b-instruct": 0.887, "deepseek-r1-distill-llama-8b": 0.769}
    for s in sorted(all_summaries, key=lambda x: (x["target"], -x["accuracy"])):
        floor = floors.get(s["target"], 0.0)
        gain = 100 * (s["accuracy"] - floor)
        lines.append(
            f"| {s['target']} | {s['candidate'][:60]} | {s['n']} | "
            f"{100*s['accuracy']:.1f}% | {100*s['inc_prec']:.0f}% | {100*s['inc_rec']:.0f}% | {s['unknown']} | "
            f"{100*floor:.1f}% (gain: {gain:+.1f}pp) |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\n✅ Full-scale unified complete. Wrote {csv_path}, summary.md, summary.json")


if __name__ == "__main__":
    main()
