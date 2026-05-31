"""T0 Anchor Round 0 — main pilot runner (simplified, parser-free).

Per `Design from basic purpose` principle: correction output IS the answer.
The Stage-1 binary GPT-4o judge reads the post-think-strip text directly.
No regex parser, no MLX parser, no dual-judge.

Per-item pipeline:
    (note, question, A0)
      → format prompt per sub-variant
      → vLLM correction call (temp=1.0, k=1, chosen_max_tokens)
      → runner truncation-retry (2× cap, up to 32768)
      → think-strip: text.split("</think>")[-1].strip() if "</think>" else text
      → Stage-1 binary GPT-4o judge → 0 / 1 / None
      → record JSONL line

Outputs under run_dir:
    pilot_sample/<target>.jsonl            — 40-item samples (deterministic)
    pilot_round0/<target>_<sv>.jsonl       — per-item records
    pilot_round0/summary_<target>.json     — per-sv aggregate
    pilot_round0/summary.md                — cross-target table
    raw_outputs_pilot/<target>/<sv>/<id>.json  — raw correction outputs

Usage:
    .venv/bin/python -m ichl.prompt_engineering.correction.pilot_round0 \\
        --target llama-3.1-8b-instruct [--sub-variants a b c d e] \\
        [--reuse-existing-raw] [--rho0 4.0]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from ichl.clients.factory import make_client  # noqa: E402
from ichl.prompt_engineering.correction.data_loader import load_correction_items  # noqa: E402
from ichl.prompt_engineering.correction.sub_variants import SUB_VARIANTS  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "output" / "ichl" / "correction" / "runs" / "20260423_1909_t0_anchor"
)
STEP8_DIR = PROJECT_ROOT / "output" / "step8"

DETECTION_JSONLS: dict[str, str] = {
    "qwen2.5-7b-instruct": (
        "output/ichl/detection/runs/20260422_0015_verdict_only_pilot/fullscale_final/"
        "candidate_r05__r02__r01__A3_task_strict_output__role__role_adversarial__00__polish_00__"
        "role__role_skeptical_auditor__00__structura_qwen2.5-7b-instruct_results.jsonl"
    ),
    "qwen3-8b": (
        "output/ichl/detection/runs/20260422_0015_verdict_only_pilot/fullscale_final/"
        "candidate_r03__r02__r01__A3_task_strict_output__role__role_adversarial__00__polish_00__"
        "role__role_skeptical_auditor__00__output_fo_qwen3-8b_results.jsonl"
    ),
    "llama-3.1-8b-instruct": (
        "output/ichl/detection/runs/20260422_0015_verdict_only_pilot/fullscale_final/"
        "candidate_B_P12_self_verify_llama-3.1-8b-instruct_results.jsonl"
    ),
    "deepseek-r1-distill-llama-8b": (
        "output/ichl/detection/runs/20260422_0015_verdict_only_pilot/fullscale_final/"
        "candidate_r02__r01__A3_task_strict_output__role__role_adversarial__00__polish_00__"
        "role__role_skeptical_auditor__00_deepseek-r1-distill-llama-8b_results.jsonl"
    ),
}

SEED = 42
TARGET_TP = 20
TARGET_FP = 20
N_FOLDS = 5

RHO0 = {
    "qwen2.5-7b-instruct": 2.52,
    "qwen3-8b": 3.35,
    "llama-3.1-8b-instruct": 4.0,
    "deepseek-r1-distill-llama-8b": 1.27,
}


# ───────────── sub-variant-run plan ─────────────

def get_sub_variant_runs(target: str) -> list[tuple[str, str, bool | None]]:
    if target == "qwen3-8b":
        runs: list[tuple[str, str, bool | None]] = []
        for sv_id in SUB_VARIANTS:
            runs.append((f"{sv_id}_think", sv_id, True))
            runs.append((f"{sv_id}_nothink", sv_id, False))
        return runs
    return [(sv_id, sv_id, None) for sv_id in SUB_VARIANTS]


# ───────────── stratified sampler ─────────────

def stratified_sample(
    items: list[dict[str, Any]], *, target_tp: int = TARGET_TP,
    target_fp: int = TARGET_FP, n_folds: int = N_FOLDS, seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_fold_class: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for it in items:
        k = (int(it["fold"]), int(it["A0_binary_correct"]))
        by_fold_class.setdefault(k, []).append(it)
    for v in by_fold_class.values():
        rng.shuffle(v)

    def sample_class(target_total: int, class_val: int):
        per_fold_quota: dict[int, int] = {f: target_total // n_folds for f in range(n_folds)}
        remainder = target_total - sum(per_fold_quota.values())
        for f in range(remainder):
            per_fold_quota[f] += 1
        picked = []
        taken_per_fold = dict.fromkeys(range(n_folds), 0)
        for fold in range(n_folds):
            pool = list(by_fold_class.get((fold, class_val), []))
            take = pool[: min(per_fold_quota[fold], len(pool))]
            picked.extend(take)
            taken_per_fold[fold] = len(take)
        still_needed = target_total - len(picked)
        if still_needed > 0:
            leftover = []
            for fold in range(n_folds):
                pool = list(by_fold_class.get((fold, class_val), []))
                leftover.extend(pool[taken_per_fold[fold]:])
            rng.shuffle(leftover)
            add = leftover[:still_needed]
            picked.extend(add)
            for x in add:
                taken_per_fold[int(x["fold"])] += 1
        return picked, taken_per_fold

    tp, tp_fold = sample_class(target_tp, class_val=0)
    fp, fp_fold = sample_class(target_fp, class_val=1)
    picked = tp + fp
    picked.sort(key=lambda it: (int(it["fold"]), int(it["A0_binary_correct"]), int(it["patient_id"])))
    diag = {
        "seed": seed,
        "target_tp": target_tp, "target_fp": target_fp,
        "n_tp_picked": len(tp), "n_fp_picked": len(fp),
        "tp_per_fold": tp_fold, "fp_per_fold": fp_fold,
        "n_total_picked": len(picked),
        "short_of_target": (len(tp) < target_tp) or (len(fp) < target_fp),
    }
    return picked, diag


def save_pilot_sample(path: Path, picked: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in picked:
            rec = {
                "pilot_item_id": it["pilot_item_id"],
                "patient_id": int(it["patient_id"]),
                "fold": int(it["fold"]),
                "binary_correct": int(it["A0_binary_correct"]),
                "note_chars": len(it["note"]),
            }
            f.write(json.dumps(rec) + "\n")


# ───────────── step0 chosen_max_tokens ─────────────

def _short_target(t: str) -> str:
    t = t.lower()
    if t.startswith("deepseek"):
        return "deepseek"
    if t.startswith("qwen2.5"):
        return "qwen2.5"
    if t.startswith("qwen3"):
        return "qwen3"
    if t.startswith("llama"):
        return "llama"
    if t.startswith("biomistral"):
        return "biomistral"
    return t


def load_chosen_max_tokens(run_dir: Path, target: str) -> dict[str, int]:
    p = run_dir / f"step0_probe_{_short_target(target)}.json"
    if not p.exists():
        raise FileNotFoundError(f"Step-0 probe file missing: {p}")
    with open(p) as f:
        data = json.load(f)
    out: dict[str, int] = {}
    for sv_run_id, rec in data["per_sub_variant"].items():
        cmt = rec.get("chosen_max_tokens") or rec.get("starting_cap", 8192)
        out[sv_run_id] = int(cmt)
    return out


# ───────────── ground-truth map ─────────────

def load_ground_truth_map(target: str) -> dict[tuple[int, int], str]:
    import pandas as pd
    gt: dict[tuple[int, int], str] = {}
    for fold in range(N_FOLDS):
        f = STEP8_DIR / target / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if not f.exists():
            print(f"[WARN] GT file missing: {f}")
            continue
        df = pd.read_csv(f)
        fold_col = "fold_id" if "fold_id" in df.columns else "fold"
        for _, r in df.iterrows():
            try:
                pid = int(r["patient_id"])
                fd = int(r[fold_col])
            except Exception:
                continue
            gt[(fd, pid)] = str(r.get("ground_truth", "") or "")
    return gt


# ───────────── Stage-1 binary GPT-4o judge ─────────────

_JUDGE_SYSTEM = (
    "You are a medical expert evaluating an AI model's answer to a clinical question."
)


def judge_binary(
    client, note: str, question: str, ground_truth: str, model_answer: str,
    *, gpt_model: str = "gpt-4o",
) -> tuple[int | None, int, int]:
    """Return (label, prompt_tokens, completion_tokens). Label is 1/0/None."""
    user_content = (
        f"DISCHARGE SUMMARY:\n{note}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
        f"MODEL'S ANSWER:\n{model_answer}\n\n"
        f"Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        f"Respond with ONLY a single digit:\n"
        f"1 = Correct\n"
        f"0 = Incorrect"
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                max_tokens=10,
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            usage = resp.usage
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            if "1" in content and "0" not in content:
                return 1, pt, ct
            if "0" in content:
                return 0, pt, ct
            return None, pt, ct
        except Exception as e:  # noqa: BLE001
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"    judge ERROR: {e}")
                return None, 0, 0
    return None, 0, 0


# ───────────── think-strip ─────────────

def strip_think(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


# ───────────── correction runner (with think override) ─────────────

def run_correction_one(
    *, client, target: str, sub_variant_id: str, item: dict[str, Any],
    max_tokens: int, temperature: float, raw_log_dir: Path,
    enable_thinking_override: bool | None,
) -> dict[str, Any]:
    """Mirror runner.run_correction_one_item but with explicit enable_thinking override."""
    from ichl.prompt_engineering.correction.sub_variants import (
        SYSTEM_MSG, format_prompt, get_enable_thinking as _get_et,
        get_sub_variant as _get_sv,
    )
    from ichl.prompt_engineering.correction.truncation_detector import detect_truncation
    from ichl.prompt_engineering.correction.runner import _call_one

    sv = _get_sv(sub_variant_id)
    user_prompt = format_prompt(
        sub_variant_id=sub_variant_id,
        note=item["note"], question=item["question"], a0=item.get("A0", ""),
    )
    if enable_thinking_override is not None:
        enable_thinking = enable_thinking_override
    else:
        enable_thinking = _get_et(target, sub_variant_id)

    attempts: list[dict[str, Any]] = []
    current_cap = max_tokens
    TRUNC_RETRIES = 2
    TRUNC_MAX_CAP = 32768
    for attempt_idx in range(TRUNC_RETRIES + 1):
        call_out = _call_one(
            client=client, system=SYSTEM_MSG, user=user_prompt,
            temperature=temperature, max_tokens=current_cap,
            enable_thinking=enable_thinking,
        )
        report = detect_truncation(
            raw_response=call_out["raw_text"], text_clean=call_out["text"],
            finish_reason=call_out["finish_reason"],
            usage={
                "completion_tokens": call_out["completion_tokens"],
                "prompt_tokens": call_out["prompt_tokens"],
            },
            max_tokens=current_cap, target=target, sub_variant=sub_variant_id,
        )
        call_out["truncation_report"] = report.as_dict()
        attempts.append(call_out)
        if not report.is_truncated_certain:
            break
        next_cap = current_cap * 2
        if next_cap > TRUNC_MAX_CAP or attempt_idx + 1 > TRUNC_RETRIES:
            break
        current_cap = next_cap

    final = dict(attempts[-1])
    final["retry_attempts"] = len(attempts)
    final["retry_triggered"] = len(attempts) > 1
    if len(attempts) > 1:
        final["retry_history"] = [
            {
                "attempt_idx": i, "max_tokens_cap": a["max_tokens_cap"],
                "finish_reason": a["finish_reason"],
                "completion_tokens": a["completion_tokens"],
                "truncation_report": a["truncation_report"],
            }
            for i, a in enumerate(attempts[:-1])
        ]

    record = {
        "pilot_item_id": item["pilot_item_id"],
        "patient_id": item["patient_id"], "fold": item["fold"],
        "target_model": target, "sub_variant_id": sub_variant_id,
        "sub_variant_name": sv.name, "note_chars": len(item["note"]),
        "A0": item.get("A0", ""),
        "A0_binary_correct": item.get("A0_binary_correct"),
        "system_prompt": SYSTEM_MSG, "user_prompt": user_prompt,
        **final,
    }
    if raw_log_dir is not None:
        raw_log_dir = Path(raw_log_dir)
        raw_log_dir.mkdir(parents=True, exist_ok=True)
        (raw_log_dir / f"{record['pilot_item_id']}.json").write_text(
            json.dumps(record, indent=2, default=str))
    return record


def load_raw_record(raw_dir: Path, pilot_item_id: str) -> dict[str, Any] | None:
    p = raw_dir / f"{pilot_item_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ───────────── per-target driver ─────────────

def run_pilot_for_target(
    target: str, run_dir: Path, *, sub_variants_filter: list[str] | None = None,
    regen_temperature: float = 1.0, judge_model: str = "gpt-4o",
    gpt_client=None, reuse_existing_raw: bool = False,
) -> dict[str, Any]:
    t_target_start = time.monotonic()

    detection_jsonl = PROJECT_ROOT / DETECTION_JSONLS[target]
    items = load_correction_items(target, detection_jsonl, only_verdict="INCORRECT")
    print(f"[{target}] detection-flagged INCORRECT items: {len(items)}")

    picked, diag = stratified_sample(items)
    print(f"[{target}] stratified sample: n_total={diag['n_total_picked']} "
          f"TP={diag['n_tp_picked']} FP={diag['n_fp_picked']}")
    if diag["short_of_target"]:
        print(f"[{target}] WARN short: {diag}")

    sample_path = run_dir / "pilot_sample" / f"{target}.jsonl"
    save_pilot_sample(sample_path, picked)
    print(f"[{target}] wrote sample to {sample_path}")

    cmt_map = load_chosen_max_tokens(run_dir, target)
    gt_map = load_ground_truth_map(target)

    sv_runs = get_sub_variant_runs(target)
    if sub_variants_filter:
        sv_runs = [
            (run_id, base, thinking) for (run_id, base, thinking) in sv_runs
            if base in sub_variants_filter or run_id in sub_variants_filter
        ]
    print(f"[{target}] sub-variant-runs to execute: {[r[0] for r in sv_runs]}")

    # Build vLLM client (skip if only judging existing raw)
    vllm_client = None

    per_target_summaries: list[dict[str, Any]] = []
    raw_base = run_dir / "raw_outputs_pilot" / target
    per_item_base = run_dir / "pilot_round0"
    per_item_base.mkdir(parents=True, exist_ok=True)

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

        # --- correction calls (sequential; reuse raw if available+flag) ---
        records: list[dict[str, Any]] = []
        n_retried = 0
        n_cert = 0
        n_lik = 0
        n_reused = 0
        for i, it in enumerate(picked):
            rec = None
            if reuse_existing_raw:
                rec = load_raw_record(raw_dir, it["pilot_item_id"])
                if rec is not None:
                    n_reused += 1
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
            records.append(rec)
            if rec.get("retry_triggered"):
                n_retried += 1
            tr = rec.get("truncation_report") or {}
            if tr.get("is_truncated_certain"):
                n_cert += 1
            if tr.get("is_truncated_likely"):
                n_lik += 1
            if (i + 1) % 10 == 0:
                print(f"  [{target}/{sv_run_id}] {i+1}/{len(picked)} "
                      f"(reused={n_reused} ct={rec.get('completion_tokens')} "
                      f"fr={rec.get('finish_reason')} trunc_c={n_cert})")

        total_retries += n_retried
        total_certain_trunc += n_cert
        total_likely_trunc += n_lik
        n_items = len(records)
        cert_rate = (n_cert / n_items) if n_items else 0.0
        if cert_rate > 0.05:
            print(f"  [{target}/{sv_run_id}] WARN TRUNC ALERT: certain-trunc {100*cert_rate:.1f}%")

        # --- judge sequentially ---
        per_item_records: list[dict[str, Any]] = []
        n_fix = 0
        n_break = 0
        n_stay_right = 0
        n_stay_wrong = 0
        n_unk = 0
        latencies: list[float] = []
        output_lens: list[int] = []

        for rec in records:
            pid = int(rec["patient_id"])
            fold = int(rec["fold"])
            gt = gt_map.get((fold, pid), "")
            raw_text = rec.get("raw_text") or rec.get("text") or ""
            stripped = strip_think(raw_text)
            # source for note+question
            src = next((x for x in picked if x["pilot_item_id"] == rec["pilot_item_id"]), None)
            note_for_judge = src["note"] if src else ""
            question_for_judge = src["question"] if src else ""

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
            n_fix += fix
            n_break += brk
            n_stay_right += sr
            n_stay_wrong += sw
            n_unk += unk

            if rec.get("latency_s") is not None:
                latencies.append(float(rec["latency_s"]))
            ol = rec.get("completion_tokens")
            if ol:
                output_lens.append(int(ol))

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

        per_item_path = per_item_base / f"{target}_{sv_run_id}.jsonl"
        with open(per_item_path, "w") as f:
            for row in per_item_records:
                f.write(json.dumps(row, default=str) + "\n")
        print(f"  [{target}/{sv_run_id}] wrote {per_item_path}")

        n = len(per_item_records)
        n_tp = sum(1 for r in per_item_records if r["A0_binary_correct"] == 0)
        n_fp = sum(1 for r in per_item_records if r["A0_binary_correct"] == 1)
        denom_tp = n_tp or 1
        denom_fp = n_fp or 1
        r_fix = n_fix / denom_tp
        r_break = n_break / denom_fp
        net_raw = n_fix - n_break
        rho0 = RHO0.get(target, 1.0)
        net_rho = r_fix - rho0 * r_break
        mean_out = (sum(output_lens) / len(output_lens)) if output_lens else 0.0
        mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
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
            "n_certain_trunc": n_cert, "n_likely_trunc": n_lik,
            "n_retries_triggered": n_retried,
            "n_reused_raw": n_reused,
            "chosen_max_tokens": max_tokens,
            "wall_clock_s": round(sv_wall, 1),
        }
        per_target_summaries.append(sv_summary)
        print(
            f"  [{target}/{sv_run_id}] DONE  n={n} FIX={n_fix} BRK={n_break} "
            f"r_fix={r_fix:.2%} r_break={r_break:.2%} net_raw={net_raw} "
            f"net_rho={net_rho:.3f} UNK={n_unk} meanLat={mean_lat:.1f}s wall={sv_wall:.0f}s"
        )

    target_wall = time.monotonic() - t_target_start
    target_summary = {
        "target": target,
        "detection_jsonl": str(detection_jsonl),
        "sample_size": diag["n_total_picked"],
        "sample_diag": diag,
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
    target_summary_path = per_item_base / f"summary_{target}.json"
    with open(target_summary_path, "w") as f:
        json.dump(target_summary, f, indent=2, default=str)
    print(f"[{target}] wrote {target_summary_path} (wall={target_wall:.0f}s)")
    return target_summary


# ───────────── cross-target summary.md ─────────────

def regen_summary_md(run_dir: Path) -> None:
    per_item_base = run_dir / "pilot_round0"
    lines = [
        "# T0 Anchor Round 0 — Main Pilot Summary (simplified, parser-free)",
        "",
        f"Run dir: `{run_dir}`",
        f"Seed: {SEED}",
        "",
        "| target | sv_run | n | TP | FP | FIX | BRK | r_fix | r_break | net_raw | net_ρ | UNK | mean_lat_s | cmt | retries | trunc_c |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(per_item_base.glob("summary_*.json")):
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
    (per_item_base / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {per_item_base / 'summary.md'}")


# ───────────── CLI ─────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(DETECTION_JSONLS.keys()))
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--sub-variants", nargs="+", default=None)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--regen-temperature", type=float, default=1.0)
    ap.add_argument("--reuse-existing-raw", action="store_true",
                    help="If set, reuse any raw_outputs_pilot/<target>/<sv>/<id>.json files.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    from openai import OpenAI
    gpt_client = OpenAI(api_key=api_key)

    print(f"=== T0 Anchor Round 0 (simplified) :: target={args.target} ===")
    print(f"run_dir: {run_dir}")
    t0 = time.monotonic()
    summary = run_pilot_for_target(
        target=args.target, run_dir=run_dir,
        sub_variants_filter=args.sub_variants,
        regen_temperature=args.regen_temperature,
        judge_model=args.judge_model,
        gpt_client=gpt_client,
        reuse_existing_raw=args.reuse_existing_raw,
    )
    elapsed = time.monotonic() - t0
    print(f"\n=== {args.target} DONE :: wall={elapsed:.0f}s ===")
    print(f"  sub-variant-runs: {len(summary['sub_variant_runs'])}")
    print(f"  correction calls: {summary['total_correction_calls']}")
    print(f"  judge calls:      {summary['total_judge_calls']}")
    regen_summary_md(run_dir)


if __name__ == "__main__":
    main()
