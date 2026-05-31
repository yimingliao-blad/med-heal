"""Step 2 — Main pilot.

Per Notion 'Claude: Plan: Detection — Pilot Runner Design' § Step 2:

  40 items × 3 candidates × 2 target models = 240 vLLM calls + 240 MLX parser calls.

Reads Step 0 `step0_token_budget.json` for per-target max_tokens.
Reads `sub_pilot/parser_configs/shared_qwen_verdict_only/` for the finalised
parser config. Runs every cell via `evaluate_cell`. Writes per-cell JSONL +
raw outputs + a cell summary.

After the 6 Candidate A cells run, pulls Candidate B verdicts from the legacy
`crossmodel_detection_results.json` (no re-parse — B uses verdict+location
format, A's parser does not apply) and filters to the main pilot's 40 pids
per target. Writes `candidate_B_<prompt>_<target>_results.jsonl`.

Finally builds:
  - `parser_agreement.csv`, `parser_agreement_summary.md`
  - `comparison.csv` (one row per (pid, target), gt + A1/A2/A3 + B-P1/P9/P12)
  - `summary.md` (accuracy table, head-to-head, rollup stats)

Usage:
    python -m ichl.prompt_engineering.scripts.main_pilot \\
        --run-dir <run_dir> --targets qwen3-8b qwen2.5-7b-instruct
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_pilot
from ichl.prompt_engineering.evaluator import evaluate_cell
from ichl.prompt_engineering.metrics.accuracy import summarize
from ichl.prompt_engineering.parsers import LLMParser, RegexParser
from ichl.prompt_engineering.pool import load_seeds


SEEDS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "detection" / "seeds.yaml"
LEGACY_JSON = (
    Path(__file__).resolve().parents[4]
    / "output" / "external_judge_benchmark" / "crossmodel_detection_results.json"
)
LEGACY_PROMPTS = ["P1_minimal", "P9_notes_first", "P12_self_verify"]


# ─────────────────────── parser config loading ───────────────────────

def _load_parser_config(sub_pilot_dir: Path, target: str) -> tuple[str | None, str | None]:
    """Same search order as sub_pilot.py::cmd_parse."""
    pc = sub_pilot_dir / "parser_configs"
    search = [
        pc / target,
        pc / "shared_qwen_verdict_only",
        sub_pilot_dir,
    ]
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


# ─────────────────────── Candidate A runs ───────────────────────

def _run_candidate_a_for_target(
    *, target: str, seeds: list, pilot_data: list[dict[str, Any]],
    parsers: list, max_tokens: int, run_dir: Path,
) -> list[dict[str, Any]]:
    """Run all candidates for one target. Writes per-cell JSONL + raw outputs."""
    cells: list[dict[str, Any]] = []
    client = make_client(target)
    raw_root = run_dir / "raw_outputs"
    for seed in seeds:
        per_item_path = run_dir / f"candidate_{seed.name}_{target}_results.jsonl"
        # Idempotency: if the per-item JSONL exists with 40 rows, skip.
        if per_item_path.exists():
            n_rows = sum(1 for _ in per_item_path.open())
            if n_rows == len(pilot_data):
                print(f"  [skip] {seed.name} × {target}: JSONL has {n_rows} rows")
                rows = [json.loads(line) for line in per_item_path.open()]
                summary = summarize(rows)
                cells.append({
                    "candidate": seed.name, "target_model": target,
                    "accuracy": summary.accuracy, "n": summary.n,
                    "summary": summary.as_dict(),
                })
                continue
            else:
                print(f"  [redo] {seed.name} × {target}: JSONL has {n_rows}/{len(pilot_data)}, rerunning")
                per_item_path.unlink()

        log_dir = raw_root / seed.name / target
        t0 = time.monotonic()
        result = evaluate_cell(
            candidate_name=seed.name,
            prompt_template=seed.template,
            pilot_data=pilot_data,
            target_client=client,
            parsers=parsers,
            max_tokens=max_tokens,
            log_dir=log_dir,
            per_item_path=per_item_path,
        )
        elapsed = time.monotonic() - t0
        print(f"  {seed.name} × {target}: acc={result.score:.3f}  "
              f"n={result.summary.n}  n_unknown={result.summary.n_unknown}  "
              f"n_disagree={result.summary.n_parser_disagree}  "
              f"elapsed={elapsed/60:.1f}m")
        cells.append({
            "candidate": seed.name, "target_model": target,
            "accuracy": result.score, "n": result.summary.n,
            "summary": result.summary.as_dict(),
        })
    return cells


# ─────────────────────── Candidate B join (no re-parse) ───────────────────────

def _pull_candidate_b(target: str, pids: set[int], run_dir: Path) -> dict[str, int]:
    """Pull legacy B verdicts for target × {P1, P9, P12} on the given pids.

    Writes one JSONL per (prompt × target). Returns dict {prompt: n_matched}.
    """
    if not LEGACY_JSON.exists():
        raise FileNotFoundError(f"Legacy JSON missing: {LEGACY_JSON}")
    data = json.loads(LEGACY_JSON.read_text())
    out: dict[str, int] = {}
    for prompt in LEGACY_PROMPTS:
        key = f"{target}__{target}__{prompt}"
        bucket = data.get(key, [])
        matched = [it for it in bucket if int(it["patient_id"]) in pids]
        out_path = run_dir / f"candidate_B_{prompt}_{target}_results.jsonl"
        with out_path.open("w") as f:
            for it in matched:
                # Reshape to mirror Candidate A per-item rows where it makes sense.
                row = {
                    "candidate": f"B_{prompt}",
                    "target_model": target,
                    "patient_id": it["patient_id"],
                    "fold": it.get("fold"),
                    "binary_correct": it["binary_correct"],
                    "regex_verdict": it.get("regex_verdict", "UNKNOWN"),
                    "llm_verdict": it.get("qwen35_verdict", "UNKNOWN"),
                    "chosen_verdict": it.get("final_verdict", "UNKNOWN"),
                    "chosen_parser": "qwen35_mlx",     # legacy final verdict is MLX
                    "parsers_agree": it.get("parsers_agree"),
                    "verdict_correct": (
                        1 if it.get("final_verdict") ==
                        ("CORRECT" if it["binary_correct"] == 1 else "INCORRECT")
                        else 0
                    ),
                }
                f.write(json.dumps(row, default=str) + "\n")
        out[prompt] = len(matched)
    return out


# ─────────────────────── aggregation ───────────────────────

def _build_parser_agreement_csv(run_dir: Path) -> Path:
    """Aggregate every Candidate A per-item row into one agreement CSV."""
    rows: list[dict[str, Any]] = []
    for jf in sorted(run_dir.glob("candidate_*_results.jsonl")):
        if jf.name.startswith("candidate_B_"):
            continue      # B rows come from legacy JSON, merged into comparison.csv
        for line in jf.open():
            r = json.loads(line)
            rows.append({
                "candidate": r.get("candidate"),
                "target_model": r.get("target_model"),
                "patient_id": r.get("patient_id"),
                "fold": r.get("fold"),
                "binary_correct": r.get("binary_correct"),
                "regex_verdict": r.get("regex_verdict"),
                "llm_verdict": r.get("llm_verdict"),
                "agree": r.get("agree"),
                "chosen_parser": r.get("chosen_parser"),
                "chosen_verdict": r.get("chosen_verdict"),
                "verdict_correct": r.get("verdict_correct"),
            })
    out_path = run_dir / "parser_agreement.csv"
    if rows:
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return out_path


def _build_parser_agreement_summary(run_dir: Path) -> Path:
    """Per-cell agreement, regex precision/recall, and 5 disagreement spot-checks."""
    lines: list[str] = ["# Parser-agreement summary (main pilot)\n"]
    per_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for jf in sorted(run_dir.glob("candidate_*_results.jsonl")):
        if jf.name.startswith("candidate_B_"):
            continue
        for line in jf.open():
            r = json.loads(line)
            per_cell[(r.get("candidate"), r.get("target_model"))].append(r)

    lines.append("| candidate | target | n | agree | regex_correct | llm_correct | n_unknown |")
    lines.append("|---|---|---|---|---|---|---|")
    disagreements: list[dict[str, Any]] = []
    for (cand, target), rows in sorted(per_cell.items()):
        n = len(rows)
        n_agree = sum(1 for r in rows if r.get("agree"))
        n_regex_correct = sum(
            1 for r in rows
            if r.get("regex_verdict") == ("CORRECT" if r["binary_correct"] == 1 else "INCORRECT")
        )
        n_llm_correct = sum(
            1 for r in rows
            if r.get("llm_verdict") == ("CORRECT" if r["binary_correct"] == 1 else "INCORRECT")
        )
        n_unknown = sum(1 for r in rows if r.get("chosen_verdict") == "UNKNOWN")
        lines.append(
            f"| {cand} | {target} | {n} | {n_agree}/{n}={100*n_agree/n:.0f}% | "
            f"{n_regex_correct}/{n}={100*n_regex_correct/n:.0f}% | "
            f"{n_llm_correct}/{n}={100*n_llm_correct/n:.0f}% | {n_unknown} |"
        )
        for r in rows:
            if not r.get("agree"):
                disagreements.append(r)

    lines.append(f"\n## Disagreement spot-check ({len(disagreements)} total)\n")
    for d in disagreements[:5]:
        lines.append(
            f"- `{d.get('candidate')}` × `{d.get('target_model')}` / pid={d.get('patient_id')}: "
            f"regex={d.get('regex_verdict')}  llm={d.get('llm_verdict')}  "
            f"gt={'CORRECT' if d['binary_correct']==1 else 'INCORRECT'}"
        )

    out = run_dir / "parser_agreement_summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _build_comparison_csv(run_dir: Path, targets: list[str]) -> Path:
    """One row per (patient_id, target) with gt + A1/A2/A3 + B-P1/P9/P12 verdicts."""
    # Collect rows keyed by (target, pid)
    combined: dict[tuple[str, int], dict[str, Any]] = {}
    for jf in sorted(run_dir.glob("candidate_*_results.jsonl")):
        for line in jf.open():
            r = json.loads(line)
            cand = r.get("candidate")
            target = r.get("target_model")
            pid = int(r.get("patient_id"))
            key = (target, pid)
            row = combined.setdefault(key, {
                "target_model": target, "patient_id": pid,
                "fold": r.get("fold"), "binary_correct": r.get("binary_correct"),
                "expected_verdict": "CORRECT" if r["binary_correct"] == 1 else "INCORRECT",
            })
            row[f"{cand}_verdict"] = r.get("chosen_verdict")
            row[f"{cand}_correct"] = r.get("verdict_correct")

    out_path = run_dir / "comparison.csv"
    if not combined:
        out_path.write_text("")
        return out_path
    fieldnames = sorted({k for row in combined.values() for k in row})
    # Put essential columns first.
    lead = ["target_model", "patient_id", "fold", "binary_correct", "expected_verdict"]
    other = [f for f in fieldnames if f not in lead]
    fieldnames = lead + sorted(other)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (_, _), row in sorted(combined.items()):
            w.writerow(row)
    return out_path


def _build_summary_md(run_dir: Path, targets: list[str]) -> Path:
    """Accuracy table + key numbers for the Finding page."""
    rows_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for jf in sorted(run_dir.glob("candidate_*_results.jsonl")):
        label_prefix = "A" if not jf.name.startswith("candidate_B_") else "B"
        for line in jf.open():
            r = json.loads(line)
            # Candidate A: 'candidate' field is A1/A2/A3; B: reshape has 'candidate' = 'B_<prompt>'.
            rows_by_cell[(r["candidate"], r["target_model"])].append(r)

    # Build accuracy table.
    lines = ["# Main pilot — summary\n"]
    lines.append("## Accuracy per (candidate × target)")
    lines.append("")
    lines.append("| candidate | target | n | accuracy | correct | incorrect-recall | unknown |")
    lines.append("|---|---|---|---|---|---|---|")
    for (cand, target), rows in sorted(rows_by_cell.items()):
        n = len(rows)
        n_hit = sum(1 for r in rows if r.get("verdict_correct") == 1)
        n_unknown = sum(1 for r in rows if r.get("chosen_verdict") == "UNKNOWN")
        # Recall on INCORRECT class = how often gt=0 → chosen=INCORRECT
        n_incorrect = sum(1 for r in rows if r.get("binary_correct") == 0)
        n_detected_incorrect = sum(
            1 for r in rows
            if r.get("binary_correct") == 0 and r.get("chosen_verdict") == "INCORRECT"
        )
        inc_recall = n_detected_incorrect / n_incorrect if n_incorrect > 0 else 0
        lines.append(
            f"| {cand} | {target} | {n} | {n_hit}/{n}={100*n_hit/n:.1f}% | "
            f"{n_hit} | {n_detected_incorrect}/{n_incorrect}={100*inc_recall:.0f}% | {n_unknown} |"
        )

    # Head-to-head: best A vs best B per target
    lines.append("\n## Head-to-head: best A vs best B, per target")
    lines.append("")
    lines.append("| target | A winner | A acc | B winner | B acc | Δ (A − B) |")
    lines.append("|---|---|---|---|---|---|")
    for target in targets:
        a_cells = [(c, rows) for (c, t), rows in rows_by_cell.items()
                   if t == target and not c.startswith("B_")]
        b_cells = [(c, rows) for (c, t), rows in rows_by_cell.items()
                   if t == target and c.startswith("B_")]
        def _acc(rows: list[dict[str, Any]]) -> float:
            if not rows: return 0.0
            return sum(1 for r in rows if r.get("verdict_correct") == 1) / len(rows)
        a_cells.sort(key=lambda x: _acc(x[1]), reverse=True)
        b_cells.sort(key=lambda x: _acc(x[1]), reverse=True)
        a_top = a_cells[0] if a_cells else (None, [])
        b_top = b_cells[0] if b_cells else (None, [])
        a_acc = _acc(a_top[1]) if a_top[0] else 0.0
        b_acc = _acc(b_top[1]) if b_top[0] else 0.0
        lines.append(
            f"| {target} | {a_top[0] or '—'} | {100*a_acc:.1f}% | "
            f"{b_top[0] or '—'} | {100*b_acc:.1f}% | {100*(a_acc-b_acc):+.1f}pp |"
        )

    out = run_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


# ─────────────────────── main ───────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    # Step 0 budgets (gate).
    step0_path = run_dir / "step0_token_budget.json"
    budgets = json.loads(step0_path.read_text())
    if not budgets.get("gate_passed"):
        raise SystemExit("Step 0 gate not passed.")

    seeds = load_seeds(SEEDS_PATH)
    sub_pilot_dir = run_dir / "sub_pilot"

    # Freeze and save pilot data per target (for reproducibility).
    pilot_by_target: dict[str, list[dict[str, Any]]] = {}
    for target in args.targets:
        items = load_detection_pilot(target)
        pilot_by_target[target] = items
        dst = run_dir / f"pilot_data_{target}.jsonl"
        with dst.open("w") as f:
            for it in items:
                f.write(json.dumps(it, default=str) + "\n")
        print(f"pilot {target}: {len(items)} items  → {dst.name}")

    # ── Candidate A — run each target in turn (swaps vLLM on demand).
    all_cells: list[dict[str, Any]] = []
    for target in args.targets:
        print(f"\n=== Candidate A / {target} ===")
        vllm_manager.ensure_model(target, log_dir=run_dir / "logs")
        regex_pat, llm_user_tpl = _load_parser_config(sub_pilot_dir, target)
        parsers = [
            RegexParser(pattern=regex_pat) if regex_pat else RegexParser(),
            LLMParser(user_template=llm_user_tpl) if llm_user_tpl else LLMParser(),
        ]
        per_entry = budgets["per_target"][target]
        max_tokens = int(per_entry["chosen_max_tokens"])
        cells = _run_candidate_a_for_target(
            target=target, seeds=seeds,
            pilot_data=pilot_by_target[target],
            parsers=parsers, max_tokens=max_tokens,
            run_dir=run_dir,
        )
        all_cells.extend(cells)

    # ── Candidate B — pull legacy verdicts (no vLLM, no MLX re-parse).
    print("\n=== Candidate B join (legacy MLX verdicts, no re-parse) ===")
    for target in args.targets:
        pids = {it["patient_id"] for it in pilot_by_target[target]}
        counts = _pull_candidate_b(target, pids, run_dir)
        print(f"  {target}: " + ", ".join(f"{p}={c}" for p, c in counts.items()))

    # ── Aggregations.
    print("\n=== Building aggregation artifacts ===")
    p1 = _build_parser_agreement_csv(run_dir)
    p2 = _build_parser_agreement_summary(run_dir)
    p3 = _build_comparison_csv(run_dir, args.targets)
    p4 = _build_summary_md(run_dir, args.targets)
    for p in [p1, p2, p3, p4]:
        print(f"  {p.relative_to(run_dir)}")

    print("\n✅ Main pilot complete. See summary.md for results.")


if __name__ == "__main__":
    main()
