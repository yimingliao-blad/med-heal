"""T0 Anchor FULL-SCALE — run all detection-flagged items per target.

Mirror of pilot_round0.py but:
  - NO stratified sampling — use ALL detection-flagged INCORRECT items.
  - Per-item streaming append (flush after each) so partial runs resume cleanly.
  - Outputs under run_dir/fullscale/ and run_dir/raw_outputs_fullscale/.

Resume: if fullscale/<target>_<sv>.jsonl exists, read pilot_item_ids already
written and skip them. Append new rows in order of remaining items.

Usage:
    .venv/bin/python -m ichl.prompt_engineering.correction.fullscale \
        --target deepseek-r1-distill-llama-8b [--sub-variants a b c d e]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from ichl.clients.factory import make_client  # noqa: E402
from ichl.prompt_engineering.correction.data_loader import load_correction_items  # noqa: E402
from ichl.prompt_engineering.correction.pilot_round0 import (  # noqa: E402
    DETECTION_JSONLS,
    DEFAULT_RUN_DIR,
    RHO0,
    N_FOLDS,
    get_sub_variant_runs,
    load_chosen_max_tokens,
    load_ground_truth_map,
    judge_binary,
    strip_think,
    run_correction_one,
    load_raw_record,
)


def _read_completed_ids(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Return (set of already-completed pilot_item_ids, existing row list)."""
    if not path.exists():
        return set(), []
    ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            pid = r.get("pilot_item_id")
            if pid:
                ids.add(str(pid))
                rows.append(r)
    return ids, rows


def run_fullscale_for_target(
    target: str,
    run_dir: Path,
    *,
    sub_variants_filter: list[str] | None = None,
    regen_temperature: float = 1.0,
    judge_model: str = "gpt-4o",
    gpt_client=None,
    reuse_existing_raw: bool = True,
) -> dict[str, Any]:
    t_target_start = time.monotonic()

    detection_jsonl = Path(__file__).resolve().parents[4] / DETECTION_JSONLS[target]
    items = load_correction_items(target, detection_jsonl, only_verdict="INCORRECT")
    # deterministic order
    items.sort(key=lambda it: (int(it["fold"]), int(it["patient_id"])))
    print(f"[{target}] detection-flagged INCORRECT items: {len(items)}")

    cmt_map = load_chosen_max_tokens(run_dir, target)
    gt_map = load_ground_truth_map(target)

    sv_runs = get_sub_variant_runs(target)
    if sub_variants_filter:
        sv_runs = [
            (run_id, base, thinking) for (run_id, base, thinking) in sv_runs
            if base in sub_variants_filter or run_id in sub_variants_filter
        ]
    print(f"[{target}] sub-variant-runs to execute: {[r[0] for r in sv_runs]}")

    vllm_client = None

    fullscale_dir = run_dir / "fullscale"
    fullscale_dir.mkdir(parents=True, exist_ok=True)
    raw_base = run_dir / "raw_outputs_fullscale" / target
    raw_base.mkdir(parents=True, exist_ok=True)

    per_target_summaries: list[dict[str, Any]] = []
    total_correction_calls = 0
    total_judge_calls = 0
    total_retries = 0
    total_certain_trunc = 0
    total_likely_trunc = 0

    for sv_run_id, sv_base, thinking_override in sv_runs:
        t_sv_start = time.monotonic()
        print(f"\n[{target}/{sv_run_id}] (base={sv_base}, think={thinking_override}) starting")
        max_tokens = cmt_map.get(sv_run_id, cmt_map.get(sv_base, 4096))
        print(f"[{target}/{sv_run_id}] chosen_max_tokens={max_tokens}")

        raw_dir = raw_base / sv_run_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        per_item_path = fullscale_dir / f"{target}_{sv_run_id}.jsonl"
        completed_ids, existing_rows = _read_completed_ids(per_item_path)
        if completed_ids:
            print(f"  [{target}/{sv_run_id}] resume: {len(completed_ids)}/{len(items)} already done")

        remaining = [it for it in items if it["pilot_item_id"] not in completed_ids]
        if not remaining:
            print(f"  [{target}/{sv_run_id}] fully complete — skipping")
            per_item_records = existing_rows
        else:
            per_item_records = list(existing_rows)
            fout = open(per_item_path, "a")
            try:
                for i, it in enumerate(remaining):
                    # correction
                    rec = None
                    if reuse_existing_raw:
                        rec = load_raw_record(raw_dir, it["pilot_item_id"])
                    if rec is None:
                        if vllm_client is None:
                            print(f"[{target}] building vLLM client...")
                            vllm_client = make_client(target)
                        rec = run_correction_one(
                            client=vllm_client, target=target,
                            sub_variant_id=sv_base, item=it,
                            max_tokens=max_tokens, temperature=regen_temperature,
                            raw_log_dir=raw_dir,
                            enable_thinking_override=thinking_override,
                        )
                        total_correction_calls += 1

                    if rec.get("retry_triggered"):
                        total_retries += 1
                    tr = rec.get("truncation_report") or {}
                    if tr.get("is_truncated_certain"):
                        total_certain_trunc += 1
                    if tr.get("is_truncated_likely"):
                        total_likely_trunc += 1

                    # judge
                    pid = int(rec["patient_id"])
                    fold = int(rec["fold"])
                    gt = gt_map.get((fold, pid), "")
                    raw_text = rec.get("raw_text") or rec.get("text") or ""
                    stripped = strip_think(raw_text)
                    note_for_judge = it["note"]
                    question_for_judge = it["question"]

                    label, pt, ct = judge_binary(
                        gpt_client, note_for_judge, question_for_judge, gt, stripped,
                        gpt_model=judge_model,
                    )
                    time.sleep(0.05)
                    total_judge_calls += 1

                    a0 = int(rec["A0_binary_correct"])
                    fix = int(a0 == 0 and label == 1)
                    brk = int(a0 == 1 and label == 0)
                    sr = int(a0 == 1 and label == 1)
                    sw = int(a0 == 0 and label == 0)
                    unk = int(label is None)

                    row = {
                        "pilot_item_id": rec["pilot_item_id"],
                        "patient_id": pid, "fold": fold,
                        "target_model": target,
                        "sub_variant_id": sv_run_id,
                        "sub_variant_name": rec.get("sub_variant_name"),
                        "enable_thinking": rec.get("enable_thinking"),
                        "note_chars": rec.get("note_chars"),
                        "question": question_for_judge,
                        "A0": rec.get("A0"),
                        "A0_binary_correct": a0,
                        "ground_truth": gt,
                        "A_corrected_raw": raw_text,
                        "A_corrected": stripped,
                        "judge_label": label,
                        "judge_prompt_tokens": pt,
                        "judge_completion_tokens": ct,
                        "fix": bool(fix),
                        "break": bool(brk),
                        "stay_right": bool(sr),
                        "stay_wrong": bool(sw),
                        "unknown": bool(unk),
                        "finish_reason": rec.get("finish_reason"),
                        "completion_tokens": rec.get("completion_tokens"),
                        "prompt_tokens": rec.get("prompt_tokens"),
                        "latency_ms": int(round(float(rec.get("latency_s") or 0) * 1000)),
                        "truncation_report": rec.get("truncation_report"),
                        "retry_attempts": rec.get("retry_attempts", 1),
                        "retry_history": rec.get("retry_history", []),
                    }
                    per_item_records.append(row)
                    fout.write(json.dumps(row, default=str) + "\n")
                    fout.flush()

                    done = len(per_item_records)
                    if done % 20 == 0:
                        print(f"  [{target}/{sv_run_id}] {done}/{len(items)} "
                              f"FIX={sum(1 for r in per_item_records if r.get('fix'))} "
                              f"BRK={sum(1 for r in per_item_records if r.get('break'))} "
                              f"UNK={sum(1 for r in per_item_records if r.get('unknown'))}")
            finally:
                fout.close()

        # summarize
        n = len(per_item_records)
        n_tp = sum(1 for r in per_item_records if int(r.get("A0_binary_correct", -1)) == 0)
        n_fp = sum(1 for r in per_item_records if int(r.get("A0_binary_correct", -1)) == 1)
        n_fix = sum(1 for r in per_item_records if r.get("fix"))
        n_break = sum(1 for r in per_item_records if r.get("break"))
        n_stay_right = sum(1 for r in per_item_records if r.get("stay_right"))
        n_stay_wrong = sum(1 for r in per_item_records if r.get("stay_wrong"))
        n_unk = sum(1 for r in per_item_records if r.get("unknown"))
        denom_tp = n_tp or 1
        denom_fp = n_fp or 1
        r_fix = n_fix / denom_tp
        r_break = n_break / denom_fp
        net_raw = n_fix - n_break
        rho0 = RHO0.get(target, 1.0)
        net_rho = r_fix - rho0 * r_break
        latencies = [float(r.get("latency_ms", 0)) / 1000.0 for r in per_item_records if r.get("latency_ms")]
        output_lens = [int(r.get("completion_tokens") or 0) for r in per_item_records if r.get("completion_tokens")]
        mean_out = (sum(output_lens) / len(output_lens)) if output_lens else 0.0
        mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
        n_cert_sv = sum(1 for r in per_item_records if (r.get("truncation_report") or {}).get("is_truncated_certain"))
        n_lik_sv = sum(1 for r in per_item_records if (r.get("truncation_report") or {}).get("is_truncated_likely"))
        n_retries_sv = sum(1 for r in per_item_records if (r.get("retry_attempts") or 1) > 1)

        sv_wall = time.monotonic() - t_sv_start
        sv_summary = {
            "target": target,
            "sub_variant_run": sv_run_id,
            "sub_variant_base": sv_base,
            "enable_thinking_override": thinking_override,
            "n": n, "n_tp": n_tp, "n_fp": n_fp,
            "n_fix": n_fix, "n_break": n_break,
            "n_stay_right": n_stay_right, "n_stay_wrong": n_stay_wrong,
            "n_unknown": n_unk,
            "r_fix": r_fix, "r_break": r_break,
            "net_raw": net_raw,
            "net_rho": net_rho, "rho0": rho0,
            "mean_output_tokens": mean_out,
            "mean_latency_s": mean_lat,
            "n_certain_trunc": n_cert_sv,
            "n_likely_trunc": n_lik_sv,
            "n_retries_triggered": n_retries_sv,
            "chosen_max_tokens": max_tokens,
            "wall_clock_s": round(sv_wall, 1),
        }
        per_target_summaries.append(sv_summary)
        print(
            f"  [{target}/{sv_run_id}] DONE  n={n} FIX={n_fix} BRK={n_break} "
            f"r_fix={r_fix:.2%} r_break={r_break:.2%} net_raw={net_raw} "
            f"net_rho={net_rho:.3f} UNK={n_unk} wall={sv_wall:.0f}s"
        )

    target_wall = time.monotonic() - t_target_start
    target_summary = {
        "target": target,
        "detection_jsonl": str(detection_jsonl),
        "n_flagged": len(items),
        "sub_variant_runs": [r[0] for r in sv_runs],
        "summaries": per_target_summaries,
        "total_correction_calls": total_correction_calls,
        "total_judge_calls": total_judge_calls,
        "total_retries_triggered": total_retries,
        "total_certain_truncation": total_certain_trunc,
        "total_likely_truncation": total_likely_trunc,
        "wall_clock_s": round(target_wall, 1),
        "rho0": RHO0.get(target, 1.0),
    }
    target_summary_path = fullscale_dir / f"summary_{target}.json"
    with open(target_summary_path, "w") as f:
        json.dump(target_summary, f, indent=2, default=str)
    print(f"[{target}] wrote {target_summary_path} (wall={target_wall:.0f}s)")
    return target_summary


def regen_summary_md(run_dir: Path) -> None:
    fullscale_dir = run_dir / "fullscale"
    lines = [
        "# T0 Anchor FULL-SCALE — Cross-target × Cross-SV Summary",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "| target | sv_run | n | TP | FP | FIX | BRK | r_fix | r_break | net_raw | net_ρ | UNK | mean_lat_s | cmt | retries | trunc_c |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(fullscale_dir.glob("summary_*.json")):
        d = json.load(open(p))
        for s in d["summaries"]:
            lines.append(
                f"| {s['target']} | {s['sub_variant_run']} | {s['n']} | "
                f"{s['n_tp']} | {s['n_fp']} | {s['n_fix']} | {s['n_break']} | "
                f"{s['r_fix']:.2%} | {s['r_break']:.2%} | {s['net_raw']} | "
                f"{s['net_rho']:.3f} | {s['n_unknown']} | {s['mean_latency_s']:.1f} | "
                f"{s['chosen_max_tokens']} | {s['n_retries_triggered']} | "
                f"{s['n_certain_trunc']} |"
            )
    (fullscale_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {fullscale_dir / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(DETECTION_JSONLS.keys()))
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--sub-variants", nargs="+", default=None)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--regen-temperature", type=float, default=1.0)
    ap.add_argument("--no-reuse-raw", action="store_true")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    from openai import OpenAI
    gpt_client = OpenAI(api_key=api_key)

    print(f"=== T0 Anchor FULL-SCALE :: target={args.target} ===")
    print(f"run_dir: {run_dir}")
    t0 = time.monotonic()
    summary = run_fullscale_for_target(
        target=args.target, run_dir=run_dir,
        sub_variants_filter=args.sub_variants,
        regen_temperature=args.regen_temperature,
        judge_model=args.judge_model,
        gpt_client=gpt_client,
        reuse_existing_raw=(not args.no_reuse_raw),
    )
    elapsed = time.monotonic() - t0
    print(f"\n=== {args.target} DONE :: wall={elapsed:.0f}s ===")
    print(f"  sub-variant-runs: {len(summary['sub_variant_runs'])}")
    print(f"  correction calls: {summary['total_correction_calls']}")
    print(f"  judge calls:      {summary['total_judge_calls']}")
    regen_summary_md(run_dir)


if __name__ == "__main__":
    main()
