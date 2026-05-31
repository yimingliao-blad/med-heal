"""Full-scale orchestration across all 5 targets.

Per-target plan:
  - qwen2.5-7b-instruct, qwen3-8b: use top-3 from iteration/<target>/top_candidates_corrected/
  - biomistral-7b, llama-3.1-8b-instruct, deepseek-r1-distill-llama-8b:
      use the best-1 winner from cross_model/best_per_target.json

Each (target, candidate) cell runs against all 962 items. Uses dual-parser
(regex + MLX LLM) for audit per the lookback rule. Pulls Candidate B legacy
verdicts for each target from crossmodel_detection_results.json without
re-parsing (B uses verdict+location format).

Resumable via per-cell JSONL files.

Usage:
    python -m ichl.prompt_engineering.scripts.full_scale_all --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_fullscale
from ichl.prompt_engineering.evaluator import evaluate_cell
from ichl.prompt_engineering.parsers import LLMParser, RegexParser


LEGACY_JSON = Path(__file__).resolve().parents[4] / "output" / "external_judge_benchmark" / "crossmodel_detection_results.json"
LEGACY_PROMPTS = ["P1_minimal", "P9_notes_first", "P12_self_verify"]
SOURCE_TARGETS = ["qwen2.5-7b-instruct", "qwen3-8b"]
NEW_TARGETS = ["biomistral-7b", "llama-3.1-8b-instruct", "deepseek-r1-distill-llama-8b"]


def _load_top_candidates(run_dir: Path, target: str, top_n: int) -> list[tuple[str, str]]:
    for dirname in ("top_candidates_corrected", "top_candidates"):
        d = run_dir / "iteration" / target / dirname
        if d.exists():
            out = []
            for i in range(1, top_n + 1):
                p = d / f"candidate_{i:02d}.txt"
                if not p.exists(): break
                text = p.read_text()
                lines = text.splitlines()
                header_end = 0
                for idx, ln in enumerate(lines):
                    if not ln.startswith("#") and ln.strip():
                        header_end = idx; break
                name = next((ln.replace("# name:", "").strip() for ln in lines[:header_end] if ln.startswith("# name:")), f"top{i}")
                tpl = "\n".join(lines[header_end:]).strip()
                out.append((name, tpl))
            if out: return out
    return []


def _load_cross_best(run_dir: Path, target: str) -> tuple[str, str] | None:
    best_path = run_dir / "cross_model" / "best_per_target.json"
    if not best_path.exists():
        return None
    best = json.loads(best_path.read_text())
    entry = best.get(target)
    if not entry: return None
    cand_name = entry["candidate"]
    # Template lives in the cross_model cell's per_item JSONL row's OR we need to look it up.
    # Better: the cross pilot evaluated templates fetched from iteration top-3; find that match.
    for src in SOURCE_TARGETS:
        tops = _load_top_candidates(run_dir, src, 3)
        for name, tpl in tops:
            full_name = f"{src}#{name[:60]}"
            if full_name == cand_name:
                return full_name, tpl
    return None


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


def _pull_candidate_b(target: str, fullscale_dir: Path) -> dict[str, int]:
    if not LEGACY_JSON.exists():
        return {}
    data = json.loads(LEGACY_JSON.read_text())
    out = {}
    for prompt in LEGACY_PROMPTS:
        key = f"{target}__{target}__{prompt}"
        bucket = data.get(key, [])
        out_path = fullscale_dir / f"candidate_B_{prompt}_{target}_results.jsonl"
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
    tp = sum(1 for r in rows if r.get("binary_correct") == 0 and r.get("chosen_verdict") == "INCORRECT")
    fp = sum(1 for r in rows if r.get("binary_correct") == 1 and r.get("chosen_verdict") == "INCORRECT")
    return {
        "candidate": cand, "target": target, "n": n,
        "accuracy": n_hit / n if n else 0,
        "inc_prec": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "inc_rec": tp / n_inc if n_inc > 0 else 0,
        "unknown": n_unknown,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--top-n-qwen", type=int, default=3)
    ap.add_argument("--with-llm-parser", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    fullscale_dir = run_dir / "fullscale_final"
    fullscale_dir.mkdir(exist_ok=True)
    (fullscale_dir / "logs").mkdir(exist_ok=True)
    sub_pilot_dir = run_dir / "sub_pilot"
    step0 = json.loads((run_dir / "step0_token_budget.json").read_text())

    # Build the full task list: target → [(cand_name, template), ...]
    tasks: dict[str, list[tuple[str, str]]] = {}
    for t in SOURCE_TARGETS:
        tops = _load_top_candidates(run_dir, t, args.top_n_qwen)
        if not tops:
            print(f"[warn] no top-{args.top_n_qwen} for {t}, skipping")
            continue
        tasks[t] = tops
    for t in NEW_TARGETS:
        best = _load_cross_best(run_dir, t)
        if best is None:
            print(f"[warn] no cross-model best for {t}, skipping")
            continue
        tasks[t] = [best]

    print("Full-scale task plan:")
    for t, lst in tasks.items():
        print(f"  {t}: {len(lst)} candidate(s)")

    all_summaries: list[dict[str, Any]] = []

    for target, candidates in tasks.items():
        print(f"\n{'='*60}\n== {target}\n{'='*60}")
        vllm_manager.ensure_model(target, log_dir=fullscale_dir / "logs")
        client = make_client(target)
        regex_pat, llm_tpl = _load_parser_config(sub_pilot_dir, target)
        parsers = [RegexParser(pattern=regex_pat) if regex_pat else RegexParser()]
        if args.with_llm_parser:
            parsers.append(LLMParser(user_template=llm_tpl) if llm_tpl else LLMParser())
        mt = int(step0["per_target"][target]["chosen_max_tokens"])
        print(f"  max_tokens={mt}  parsers={[p.name for p in parsers]}")

        pilot_data = load_detection_fullscale(target)
        print(f"  {len(pilot_data)} items")

        for cand_name, template in candidates:
            safe = cand_name.replace("/", "_").replace(" ", "_")[:120]
            per_item_path = fullscale_dir / f"candidate_{safe}_{target}_results.jsonl"
            if per_item_path.exists():
                n_rows = sum(1 for _ in per_item_path.open())
                if n_rows >= len(pilot_data):
                    rows = [json.loads(l) for l in per_item_path.open() if l.strip()]
                    all_summaries.append(_summary(rows, cand_name, target))
                    print(f"  [resume] {cand_name[:60]}: {n_rows} rows")
                    continue
                per_item_path.unlink()

            log_dir = fullscale_dir / "raw_outputs" / safe / target
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
            print(f"  {cand_name[:60]}: acc={100*s['accuracy']:.1f}%  "
                  f"inc_prec={100*s['inc_prec']:.0f}%  inc_rec={100*s['inc_rec']:.0f}%  "
                  f"unk={s['unknown']}  elapsed={elapsed/60:.1f}m")

        # Pull Candidate B for this target.
        counts = _pull_candidate_b(target, fullscale_dir)
        print(f"  Candidate B legacy: {counts}")
        for prompt in LEGACY_PROMPTS:
            jf = fullscale_dir / f"candidate_B_{prompt}_{target}_results.jsonl"
            if jf.exists():
                rows = [json.loads(l) for l in jf.open() if l.strip()]
                if rows:
                    all_summaries.append(_summary(rows, f"B_{prompt}", target))

    # Write aggregate summaries
    csv_path = fullscale_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        if all_summaries:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            for s in all_summaries:
                w.writerow(s)
    lines = ["# Full-scale final summary (all 5 targets × candidates × 962 items)\n"]
    lines.append("| target | candidate | n | acc | inc-prec | inc-rec | unknown |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in sorted(all_summaries, key=lambda x: (x["target"], -x["accuracy"])):
        lines.append(
            f"| {s['target']} | {s['candidate'][:60]} | {s['n']} | "
            f"{100*s['accuracy']:.1f}% | {100*s['inc_prec']:.0f}% | {100*s['inc_rec']:.0f}% | {s['unknown']} |"
        )
    (fullscale_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (fullscale_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2, default=str))
    print(f"\n✅ Full-scale complete. {csv_path}")


if __name__ == "__main__":
    main()
