"""Full-scale evaluation of iteration top-N candidates on 962-item test set.

Reads the top-K `candidate_NN.txt` files from the iteration output, runs each
against all 962 items per target, and joins with legacy Candidate B verdicts
(from `crossmodel_detection_results.json`) for a head-to-head summary.

Uses REGEX-ONLY parser by default — the pilot-first audit (30 sub-pilot + 240
main pilot + 468 iteration variants) achieved 100 % regex==LLM agreement on
verdict-only output. The MLX LLM parser is kept as optional fallback on UNKNOWN.

Usage:
    python -m ichl.prompt_engineering.scripts.full_scale \\
        --run-dir <run_dir> --targets qwen2.5-7b-instruct qwen3-8b \\
        [--top-n 3] [--with-llm-parser]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_fullscale
from ichl.prompt_engineering.evaluator import evaluate_cell
from ichl.prompt_engineering.metrics.accuracy import summarize
from ichl.prompt_engineering.parsers import LLMParser, RegexParser


SEEDS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "detection" / "seeds.yaml"
LEGACY_JSON = (
    Path(__file__).resolve().parents[4]
    / "output" / "external_judge_benchmark" / "crossmodel_detection_results.json"
)
LEGACY_PROMPTS = ["P1_minimal", "P9_notes_first", "P12_self_verify"]


def _load_top_candidates(iter_dir: Path, target: str, top_n: int) -> list[tuple[str, str]]:
    """Load the top-N candidate templates. Returns list of (name, template)."""
    td = iter_dir / target / "top_candidates"
    if not td.exists():
        raise FileNotFoundError(f"Iteration top_candidates dir missing: {td}")
    out: list[tuple[str, str]] = []
    for i in range(1, top_n + 1):
        path = td / f"candidate_{i:02d}.txt"
        if not path.exists():
            break
        text = path.read_text()
        # Strip '# name: ...', '# score: ...', '# round_added: ...' header comments.
        lines = text.splitlines()
        header_end = 0
        for idx, ln in enumerate(lines):
            if not ln.startswith("#") and ln.strip():
                header_end = idx
                break
        # Extract human name from comment
        name = None
        for ln in lines[:header_end]:
            if ln.startswith("# name:"):
                name = ln.replace("# name:", "").strip()
                break
        name = name or f"top{i:02d}_{target}"
        template = "\n".join(lines[header_end:]).strip()
        out.append((name, template))
    return out


def _load_parser_config(sub_pilot_dir: Path, target: str) -> tuple[str | None, str | None]:
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


def _pull_candidate_b(target: str, all_pids: set[int], fullscale_dir: Path) -> dict[str, int]:
    """Pull legacy B verdicts for target × {P1, P9, P12} on every pid we have."""
    if not LEGACY_JSON.exists():
        raise FileNotFoundError(f"Legacy JSON missing: {LEGACY_JSON}")
    data = json.loads(LEGACY_JSON.read_text())
    out: dict[str, int] = {}
    for prompt in LEGACY_PROMPTS:
        key = f"{target}__{target}__{prompt}"
        bucket = data.get(key, [])
        matched = [it for it in bucket if int(it["patient_id"]) in all_pids]
        out_path = fullscale_dir / f"candidate_B_{prompt}_{target}_results.jsonl"
        with out_path.open("w") as f:
            for it in matched:
                row = {
                    "candidate": f"B_{prompt}",
                    "target_model": target,
                    "patient_id": it["patient_id"],
                    "fold": it.get("fold"),
                    "binary_correct": it["binary_correct"],
                    "chosen_verdict": it.get("final_verdict", "UNKNOWN"),
                    "verdict_correct": (
                        1 if it.get("final_verdict") ==
                        ("CORRECT" if it["binary_correct"] == 1 else "INCORRECT")
                        else 0
                    ),
                }
                f.write(json.dumps(row, default=str) + "\n")
        out[prompt] = len(matched)
    return out


def _summary_row(rows: list[dict[str, Any]], cand: str, target: str) -> dict[str, Any]:
    n = len(rows)
    n_inc = sum(1 for r in rows if r.get("binary_correct") == 0)
    n_cor = sum(1 for r in rows if r.get("binary_correct") == 1)
    n_hit = sum(1 for r in rows if r.get("verdict_correct") == 1)
    tp_inc = sum(1 for r in rows if r.get("binary_correct") == 0 and r.get("chosen_verdict") == "INCORRECT")
    fp_inc = sum(1 for r in rows if r.get("binary_correct") == 1 and r.get("chosen_verdict") == "INCORRECT")
    n_unknown = sum(1 for r in rows if r.get("chosen_verdict") == "UNKNOWN")
    prec_inc = tp_inc / (tp_inc + fp_inc) if (tp_inc + fp_inc) > 0 else 0
    rec_inc = tp_inc / n_inc if n_inc > 0 else 0
    return {
        "candidate": cand, "target_model": target, "n": n,
        "accuracy": n_hit / n if n else 0,
        "n_incorrect_class": n_inc, "n_correct_class": n_cor,
        "tp_incorrect": tp_inc, "fp_incorrect": fp_inc,
        "precision_incorrect": prec_inc, "recall_incorrect": rec_inc,
        "n_unknown": n_unknown,
    }


def _run_target(
    target: str, candidates: list[tuple[str, str]],
    regex_pat: str | None, llm_tpl: str | None, with_llm: bool,
    max_tokens: int, fullscale_dir: Path,
) -> list[dict[str, Any]]:
    vllm_manager.ensure_model(target, log_dir=fullscale_dir / "logs")
    client = make_client(target)
    parsers = [RegexParser(pattern=regex_pat) if regex_pat else RegexParser()]
    if with_llm:
        parsers.append(LLMParser(user_template=llm_tpl) if llm_tpl else LLMParser())
    pilot_data = load_detection_fullscale(target)
    print(f"\n── Full-scale / {target} ── {len(pilot_data)} items × {len(candidates)} candidates")

    summaries: list[dict[str, Any]] = []
    raw_root = fullscale_dir / "raw_outputs"
    for cand_name, template in candidates:
        # Tag the cell with a short safe name for the JSONL.
        safe = f"top_{cand_name[:80].replace('/', '_')}"
        per_item_path = fullscale_dir / f"candidate_{safe}_{target}_results.jsonl"
        if per_item_path.exists():
            n_rows = sum(1 for _ in per_item_path.open())
            if n_rows >= len(pilot_data):
                rows = [json.loads(l) for l in per_item_path.open()]
                summaries.append(_summary_row(rows, cand_name, target))
                print(f"  [resume] {safe} × {target}: {n_rows} rows already present")
                continue

        log_dir = raw_root / safe / target
        t0 = time.monotonic()
        result = evaluate_cell(
            candidate_name=cand_name,
            prompt_template=template,
            pilot_data=pilot_data,
            target_client=client,
            parsers=parsers,
            max_tokens=max_tokens,
            log_dir=log_dir,
            per_item_path=per_item_path,
        )
        elapsed = time.monotonic() - t0
        rows = [json.loads(l) for l in per_item_path.open()]
        summaries.append(_summary_row(rows, cand_name, target))
        print(f"  {safe} × {target}: acc={result.score:.3f}  "
              f"n_unknown={result.summary.n_unknown}  elapsed={elapsed/60:.1f}m")
    return summaries


def _build_summary_md(fullscale_dir: Path, all_summaries: list[dict[str, Any]]) -> Path:
    lines = ["# Full-scale — summary (962 items per cell)\n"]
    lines.append("## Candidate A (iteration top-N) and Candidate B (legacy)\n")
    lines.append("| candidate | target | n | accuracy | inc-prec | inc-recall | unknown |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in sorted(all_summaries, key=lambda x: (x["target_model"], -x["accuracy"])):
        lines.append(
            f"| {s['candidate']} | {s['target_model']} | {s['n']} | "
            f"{100*s['accuracy']:.1f}% | {100*s['precision_incorrect']:.0f}% | "
            f"{100*s['recall_incorrect']:.0f}% | {s['n_unknown']} |"
        )
    path = fullscale_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--with-llm-parser", action="store_true",
                    help="also run MLX LLM parser on every item (slower but extra audit)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    sub_pilot_dir = run_dir / "sub_pilot"
    iter_dir = run_dir / "iteration"
    fullscale_dir = run_dir / "fullscale"
    fullscale_dir.mkdir(parents=True, exist_ok=True)
    (fullscale_dir / "logs").mkdir(exist_ok=True)

    step0 = json.loads((run_dir / "step0_token_budget.json").read_text())

    all_summaries: list[dict[str, Any]] = []
    all_pids_per_target: dict[str, set[int]] = {}

    # ── Candidate A (iteration top-N per target) ──
    for target in args.targets:
        candidates = _load_top_candidates(iter_dir, target, args.top_n)
        if not candidates:
            print(f"[warn] no top candidates for {target}, skipping")
            continue
        regex_pat, llm_tpl = _load_parser_config(sub_pilot_dir, target)
        max_tokens = int(step0["per_target"][target]["chosen_max_tokens"])
        summaries = _run_target(
            target=target, candidates=candidates,
            regex_pat=regex_pat, llm_tpl=llm_tpl,
            with_llm=args.with_llm_parser, max_tokens=max_tokens,
            fullscale_dir=fullscale_dir,
        )
        all_summaries.extend(summaries)
        all_pids_per_target[target] = {int(s["patient_id"])
                                        for jf in fullscale_dir.glob(f"candidate_top_*_{target}_results.jsonl")
                                        for s in (json.loads(l) for l in jf.open())}

    # ── Candidate B (legacy, no re-run, no re-parse) ──
    print("\n=== Candidate B join (legacy MLX verdicts, no re-parse) ===")
    for target in args.targets:
        pids = all_pids_per_target.get(target)
        if not pids:
            continue
        counts = _pull_candidate_b(target, pids, fullscale_dir)
        print(f"  {target}: " + ", ".join(f"{p}={c}" for p, c in counts.items()))
        # Also build per-cell summaries.
        for prompt in LEGACY_PROMPTS:
            p = fullscale_dir / f"candidate_B_{prompt}_{target}_results.jsonl"
            rows = [json.loads(l) for l in p.open()]
            if rows:
                all_summaries.append(_summary_row(rows, f"B_{prompt}", target))

    # ── Aggregation ──
    summary_path = _build_summary_md(fullscale_dir, all_summaries)
    print(f"\nWrote {summary_path}")
    # Also save raw summaries JSON.
    (fullscale_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, default=str)
    )
    print(f"Wrote {fullscale_dir / 'summary.json'}")
    print("\n✅ Full-scale complete.")


if __name__ == "__main__":
    main()
