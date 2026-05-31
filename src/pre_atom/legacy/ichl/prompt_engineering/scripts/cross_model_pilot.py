"""Cross-model pilot: top-3 prompts from Qwen2.5 + top-3 from Qwen3, evaluated on
new targets (BM, Llama, DS) to measure prompt transferability.

Reads `iteration/<source_target>/top_candidates_corrected/candidate_NN.txt` for
each source target; runs every unique template against each new target on 40
stratified pilot items. Dual-parser (regex + MLX) logged per item, per the
lookback audit rule.

Writes:
    <run_dir>/cross_model/<target>/
        pilot_data.jsonl
        per_item/<candidate>.jsonl
        summary.json
    <run_dir>/cross_model/cross_model_summary.csv
    <run_dir>/cross_model/cross_model_summary.md

Usage:
    python -m ichl.prompt_engineering.scripts.cross_model_pilot \\
        --run-dir <run_dir> --source-targets qwen2.5-7b-instruct qwen3-8b \\
        --eval-targets biomistral-7b llama-3.1-8b-instruct deepseek-r1-distill-llama-8b \\
        [--top-n 3]
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
from ichl.common.pilot_loader import load_detection_pilot
from ichl.prompt_engineering.evaluator import evaluate_cell
from ichl.prompt_engineering.parsers import LLMParser, RegexParser


def _load_top_candidates(run_dir: Path, source_target: str, top_n: int) -> list[tuple[str, str]]:
    """Prefer top_candidates_corrected/ over top_candidates/. Returns (name, template) tuples."""
    for dirname in ("top_candidates_corrected", "top_candidates"):
        d = run_dir / "iteration" / source_target / dirname
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
                name = next((ln.replace("# name:", "").strip() for ln in lines[:header_end] if ln.startswith("# name:")), f"{source_target}_top{i}")
                tpl = "\n".join(lines[header_end:]).strip()
                out.append((name, tpl))
            if out:
                return out
    return []


def _dedup(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop duplicate templates (by hash)."""
    seen = set()
    out = []
    for name, tpl in candidates:
        h = hashlib.sha1(tpl.encode()).hexdigest()[:12]
        if h in seen: continue
        seen.add(h)
        out.append((name, tpl))
    return out


def _load_parser_config(sub_pilot_dir: Path, target: str) -> tuple[str | None, str | None]:
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


def _summary_row(rows: list[dict[str, Any]], cand_name: str, target: str) -> dict[str, Any]:
    n = len(rows)
    n_hit = sum(1 for r in rows if r.get("verdict_correct") == 1)
    n_unknown = sum(1 for r in rows if r.get("chosen_verdict") == "UNKNOWN")
    n_disagree = sum(1 for r in rows if r.get("regex_verdict") != r.get("llm_verdict"))
    n_trunc = sum(1 for r in rows if r.get("finish_reason") == "length"
                  or ("<think>" in (r.get("text_clean","") or "") and "</think>" not in (r.get("text_clean","") or "")))
    n_inc = sum(1 for r in rows if r.get("binary_correct") == 0)
    tp = sum(1 for r in rows if r.get("binary_correct") == 0 and r.get("chosen_verdict") == "INCORRECT")
    fp = sum(1 for r in rows if r.get("binary_correct") == 1 and r.get("chosen_verdict") == "INCORRECT")
    return {
        "candidate": cand_name, "target": target, "n": n,
        "accuracy": n_hit / n if n else 0,
        "inc_prec": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "inc_rec": tp / n_inc if n_inc > 0 else 0,
        "unknown": n_unknown, "disagree": n_disagree, "truncated": n_trunc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--source-targets", nargs="+", required=True,
                    help="where to pull top-N templates from (e.g. qwen2.5-7b-instruct qwen3-8b)")
    ap.add_argument("--eval-targets", nargs="+", required=True,
                    help="targets to evaluate the templates on (BM/Llama/DS)")
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    # Collect + dedup all templates.
    all_templates: list[tuple[str, str]] = []
    # Issue B fix (Iteration Process Audit 2026-04-22): always include A1 and A3
    # as baseline controls so cross-model results are interpretable.
    from ichl.prompt_engineering.pool import load_seeds
    seeds_path = Path(__file__).resolve().parents[1] / "prompts" / "detection" / "seeds.yaml"
    for seed in load_seeds(seeds_path):
        if seed.name in ("A1_minimal_neutral", "A3_task_strict_output"):
            all_templates.append((f"baseline#{seed.name}", seed.template))
    for src in args.source_targets:
        tops = _load_top_candidates(run_dir, src, args.top_n)
        print(f"loaded {len(tops)} top-{args.top_n} from {src}")
        for name, tpl in tops:
            all_templates.append((f"{src}#{name[:60]}", tpl))
    all_templates = _dedup(all_templates)
    print(f"\ntotal unique templates (A1+A3 baselines + top-N per source): {len(all_templates)}")

    sub_pilot_dir = run_dir / "sub_pilot"
    step0 = json.loads((run_dir / "step0_token_budget.json").read_text())

    cross_dir = run_dir / "cross_model"
    cross_dir.mkdir(exist_ok=True)
    all_summaries: list[dict[str, Any]] = []

    for target in args.eval_targets:
        print(f"\n{'='*60}\n== Cross-model pilot / {target}\n{'='*60}")
        vllm_manager.ensure_model(target, log_dir=cross_dir / "logs")
        client = make_client(target)
        regex_pat, llm_tpl = _load_parser_config(sub_pilot_dir, target)
        parsers = [
            RegexParser(pattern=regex_pat) if regex_pat else RegexParser(),
            LLMParser(user_template=llm_tpl) if llm_tpl else LLMParser(),
        ]
        mt = int(step0["per_target"][target]["chosen_max_tokens"])
        print(f"  max_tokens={mt}  parsers={[p.name for p in parsers]}")

        pilot_data = load_detection_pilot(target)
        target_dir = cross_dir / target
        target_dir.mkdir(exist_ok=True)
        (target_dir / "pilot_data.jsonl").write_text(
            "\n".join(json.dumps(i, default=str) for i in pilot_data) + "\n"
        )
        per_item_dir = target_dir / "per_item"
        per_item_dir.mkdir(exist_ok=True)

        for cand_name, template in all_templates:
            # Short safe name for JSONL path
            safe = cand_name.replace("/", "_").replace(" ", "_")[:120]
            per_item_path = per_item_dir / f"{safe}.jsonl"
            if per_item_path.exists() and sum(1 for _ in per_item_path.open()) >= len(pilot_data):
                rows = [json.loads(l) for l in per_item_path.open() if l.strip()]
                all_summaries.append(_summary_row(rows, cand_name, target))
                print(f"  [resume] {cand_name[:60]}")
                continue
            if per_item_path.exists():
                per_item_path.unlink()
            log_dir = target_dir / "raw_outputs" / safe
            t0 = time.monotonic()
            result = evaluate_cell(
                candidate_name=cand_name, prompt_template=template,
                pilot_data=pilot_data, target_client=client, parsers=parsers,
                max_tokens=mt, log_dir=log_dir, per_item_path=per_item_path,
            )
            elapsed = time.monotonic() - t0
            rows = [json.loads(l) for l in per_item_path.open() if l.strip()]
            s = _summary_row(rows, cand_name, target)
            all_summaries.append(s)
            print(f"  {cand_name[:60]}: acc={100*s['accuracy']:.1f}%  "
                  f"inc_prec={100*s['inc_prec']:.0f}%  inc_rec={100*s['inc_rec']:.0f}%  "
                  f"unk={s['unknown']}  dis={s['disagree']}  trunc={s['truncated']}  "
                  f"elapsed={elapsed/60:.1f}m")

    # Aggregate outputs.
    csv_path = cross_dir / "cross_model_summary.csv"
    with csv_path.open("w", newline="") as f:
        if all_summaries:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            for r in all_summaries:
                w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Pick best-1 per target
    best_per_target: dict[str, dict[str, Any]] = {}
    for s in all_summaries:
        t = s["target"]
        if t not in best_per_target or s["accuracy"] > best_per_target[t]["accuracy"]:
            best_per_target[t] = s

    md = ["# Cross-model pilot summary\n"]
    md.append("| target | candidate | n | acc | inc-prec | inc-rec | unk | dis | trunc |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(all_summaries, key=lambda x: (x["target"], -x["accuracy"])):
        md.append(
            f"| {s['target']} | {s['candidate'][:60]} | {s['n']} | {100*s['accuracy']:.1f}% | "
            f"{100*s['inc_prec']:.0f}% | {100*s['inc_rec']:.0f}% | "
            f"{s['unknown']} | {s['disagree']} | {s['truncated']} |"
        )
    md.append("\n## Best-1 per target")
    md.append("| target | winner | acc |")
    md.append("|---|---|---|")
    for t, s in best_per_target.items():
        md.append(f"| {t} | {s['candidate'][:80]} | {100*s['accuracy']:.1f}% |")
    (cross_dir / "cross_model_summary.md").write_text("\n".join(md) + "\n")
    (cross_dir / "best_per_target.json").write_text(json.dumps(best_per_target, indent=2, default=str))

    print(f"\n✅ Cross-model pilot complete.")
    print("Best-1 per target:")
    for t, s in best_per_target.items():
        print(f"  {t}: {s['candidate'][:80]} (acc={100*s['accuracy']:.1f}%)")


if __name__ == "__main__":
    main()
