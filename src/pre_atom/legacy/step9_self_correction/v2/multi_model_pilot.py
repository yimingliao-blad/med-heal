#!/usr/bin/env python3
"""
Multi-model self-correction pilot.

Runs the D2 + correction (union retrieval) + V2 verdict pipeline on a SAMPLE
of each target model's own zeroshot answers. Each target model self-corrects
its own answers — vLLM serves the target model, and the same model is used
as the helper for D2 detection, correction generation, V2 verdict, AND R3
LLM-based span retrieval.

The script does NOT swap vLLM for you. Before running each model section,
the user is prompted to (1) restart vLLM with that model's weights and
(2) press Enter when ready.

Sampling per model:
  - 10 wrong items (judged 0 by the legacy temp=0.1 binary judge stored in
    output/step8/{model}/fold_*/zeroshot_evaluated_binary.csv)
  - 30 correct items (judged 1)
  - drawn evenly across the 5 folds where possible
  - seeded for reproducibility

For each item the pipeline runs:
  judge_orig (GPT-4o T=0)         oracle label
   → D2 detect (target model)     atomic yes/no, K=5
   → correction (target model)    union(R3,R2) retrieval + EVIDENCE/ANSWER prompt
   → V2 verdict (target model)    pairwise A/B, K=5
   → judge_corrected (GPT-4o T=0) oracle label
   → outcome

Audit log per model: output/step9_v2/multi_model/{model_dir}/audit.jsonl

Per-model differences handled:
  - Chat template applied automatically by vLLM via the loaded model's tokenizer
  - <think> blocks emitted by Qwen3 / DeepSeek-R1 are stripped in vllm_chat
  - For Qwen3-8B: prepend "/no_think" to system message to disable thinking

Usage:
    python multi_model_pilot.py --models qwen2.5,llama3,deepseek,qwen3 \\
        --n-wrong 10 --n-correct 30 --port 8003

    # Or run one model at a time:
    python multi_model_pilot.py --models qwen2.5 --port 8003
    # ... swap vLLM ...
    python multi_model_pilot.py --models llama3 --port 8003
"""
from __future__ import annotations

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
sys.path.insert(0, str(Path(__file__).parent))
from audit_log import AuditLog, make_record
from correction import build_correction_prompt, generate_corrections
from correction_prompt_v2 import parse_premises_output, verify_evidence_quotes
from detection_d2 import detect_d2
from detection_format_bakeoff import served_model_id
from judge import _load_notes_lookup, judge as judge_call
from verdict_v2 import run_verdict_v2

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------

# Map a short --models alias → (step8 directory name, expected vLLM
# model_id substring, optional system-prompt prefix). The expected_id_substring
# is used to verify that the user has actually loaded the right model into
# vLLM before running the pipeline.

MODELS = {
    "biomistral": {
        "step8_dir": "biomistral-7b",
        "expected_id_substring": "biomistral",
        # BioMistral has very weak instruction-following on complex prompts.
        # Use only the simplest regen prompt; do not trust V3 6-count format.
    },
    "qwen2.5": {
        "step8_dir": "qwen2.5-7b-instruct",
        "expected_id_substring": "qwen2.5",
    },
    "llama3": {
        "step8_dir": "llama-3.1-8b-instruct",
        "expected_id_substring": "llama-3.1",
    },
    "deepseek": {
        "step8_dir": "deepseek-r1-distill-llama-8b",
        "expected_id_substring": "deepseek",
    },
    "qwen3": {
        "step8_dir": "qwen3-8b",
        "expected_id_substring": "qwen3",
        # Qwen3 has a hard non-thinking mode toggled via the chat template.
        # vLLM forwards chat_template_kwargs to the model's Jinja template,
        # and Qwen3's template exposes an enable_thinking variable. Setting
        # this to False stops the model from emitting <think>...</think>
        # blocks (and the soft "/no_think" directive becomes unnecessary).
        # Source: Qwen3-8B model card + Qwen vLLM deployment docs (2025).
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


# ---------------------------------------------------------------------------
# Test set sampling
# ---------------------------------------------------------------------------

def sample_random_per_fold(step8_dir: str, n_per_fold: int = 20,
                           seed: int = 42) -> list[dict]:
    """Sample n_per_fold items uniformly at random from each fold.

    Unlike sample_test_items (which stratifies by binary_correct), this
    samples from the full fold without regard to W/C label, preserving the
    natural class distribution (~12% wrong). Per the multi-model pilot
    spec: 20 per fold × 5 folds = 100 items per model.
    """
    parts = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / step8_dir / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["fold"] = fold
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No step8 zeroshot CSVs for {step8_dir}")
    base = pd.concat(parts, ignore_index=True)

    rng = random.Random(seed)
    sampled = []
    for fold in range(5):
        f_df = base[base["fold"] == fold]
        if len(f_df) < n_per_fold:
            n = len(f_df)
        else:
            n = n_per_fold
        idxs = rng.sample(list(f_df.index), n)
        sampled.extend(f_df.loc[idxs].to_dict("records"))

    items = []
    for r in sampled:
        items.append({
            "fold": int(r["fold"]),
            "idx": int(r["idx"]),
            "patient_id": int(r["patient_id"]),
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "model_answer": str(r["model_answer"]),
            "label": int(r["binary_correct"]),
        })
    return items


def sample_test_items(step8_dir: str, n_wrong: int, n_correct: int,
                      seed: int = 42) -> list[dict]:
    """Sample n_wrong + n_correct items evenly across the 5 folds.

    Uses the legacy temp=0.1 binary_correct labels for sampling. The pipeline
    will re-judge each item at temp=0 via judge_orig and use that as the
    oracle truth for fix/break measurement.
    """
    parts = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / step8_dir / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["fold"] = fold
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No step8 zeroshot CSVs for {step8_dir}")
    base = pd.concat(parts, ignore_index=True)

    # Even split across folds
    rng = random.Random(seed)
    per_fold_w = max(1, n_wrong // 5)
    per_fold_c = max(1, n_correct // 5)
    extra_w = n_wrong - per_fold_w * 5
    extra_c = n_correct - per_fold_c * 5

    sampled: list[dict] = []
    for fold in range(5):
        f_df = base[base["fold"] == fold]
        wrong = f_df[f_df["binary_correct"] == 0]
        correct = f_df[f_df["binary_correct"] == 1]
        n_w = per_fold_w + (1 if fold < extra_w else 0)
        n_c = per_fold_c + (1 if fold < extra_c else 0)
        if len(wrong) < n_w:
            n_w = len(wrong)
        if len(correct) < n_c:
            n_c = len(correct)
        if n_w > 0:
            wrong_idx = rng.sample(list(wrong.index), n_w)
            sampled.extend(wrong.loc[wrong_idx].to_dict("records"))
        if n_c > 0:
            correct_idx = rng.sample(list(correct.index), n_c)
            sampled.extend(correct.loc[correct_idx].to_dict("records"))

    items = []
    for r in sampled:
        items.append({
            "fold": int(r["fold"]),
            "idx": int(r["idx"]),
            "patient_id": int(r["patient_id"]),
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "model_answer": str(r["model_answer"]),
            "label": int(r["binary_correct"]),
        })
    return items


# ---------------------------------------------------------------------------
# Per-item pipeline runner (mirrors run_pipeline.py:run_one)
# ---------------------------------------------------------------------------

def run_one_item(item: dict, notes: dict, *, port: int, model_cfg: dict,
                 args, log: AuditLog) -> None:
    fold, idx = item["fold"], item["idx"]
    if log.has(fold, idx) and not args.force:
        return
    note = notes.get(str(item["patient_id"]), "")
    if not note:
        return

    rec = make_record(fold, idx, item={
        "question": item["question"],
        "patient_id": item["patient_id"],
        "ground_truth": item["ground_truth"],
        "note": note,
        "original_answer": item["model_answer"],
        "label": item["label"],
    })

    # ---- judge_orig (GPT-4o T=0 oracle) ----
    j_orig = judge_call(note, item["question"], item["ground_truth"],
                        item["model_answer"], n=1, temperature=0.0)
    rec["judge_orig"] = {
        "label": j_orig["label"],
        "raw": j_orig["raws"][0] if j_orig["raws"] else "",
        "label_T01_legacy": int(item["label"]),
    }
    eval_orig = j_orig["label"] if j_orig["label"] is not None else int(item["label"])

    # ---- detection (D2) ----
    det = detect_d2(note, item["question"], item["model_answer"], port,
                    k=args.k_det, severity_threshold=args.severity_thresh)
    rec["detection"] = {
        "variant": "D2",
        "k": args.k_det,
        "severity_threshold": args.severity_thresh,
        "contradiction": det["contradiction"],
        "qmis": det["qmis"],
        "fired": det["fired"],
        "fired_reason": det["fired_reason"],
        "error_type": det["error_type"],
        "error_statement": det["error_statement"],
    }

    if not det["fired"]:
        rec["correction"] = {"skipped_reason": "no_detection_or_weak_signal"}
        rec["verdict"] = None
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "keep", "delta": 0, "final_eval": eval_orig}
        log.write(rec)
        return

    # ---- correction (union retrieval + factored CoVe + EVIDENCE/ANSWER) ----
    detection_final_for_correction = {
        "verdict": "INCORRECT",
        "error_type": det["error_type"],
        "error_statement": det["error_statement"],
        "correct_statement": "",
        "contradiction": det["contradiction"],
        "qmis": det["qmis"],
    }
    plan = build_correction_prompt(detection_final_for_correction, note,
                                   item["question"], item["model_answer"], fold,
                                   similarity_threshold=args.span_thresh,
                                   port=port, retriever="union")
    if plan["skipped_reason"]:
        rec["correction"] = {
            "skipped_reason": plan["skipped_reason"],
            "best_sim": plan.get("best_sim", 0.0),
            "spans": plan["spans"],
            "queries": plan.get("queries", []),
            "retriever_used": plan.get("retriever_used"),
            "error_type": plan["error_type"],
        }
        rec["verdict"] = None
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "kept_original_low_evidence", "delta": 0,
                          "final_eval": eval_orig}
        log.write(rec)
        return

    candidates_raw = generate_corrections(plan["prompt"], port=port,
                                          k=args.k_cor, temperature=0.7)
    parsed_candidates = []
    for raw in candidates_raw:
        parsed = parse_premises_output(raw)
        parsed["evidence_verified"] = verify_evidence_quotes(parsed["evidence_quotes"], note)
        parsed["n_verified"] = sum(1 for v in parsed["evidence_verified"] if v)
        parsed["raw"] = raw
        parsed_candidates.append(parsed)

    rec["correction"] = {
        "skipped_reason": None,
        "error_type": plan["error_type"],
        "spans": plan["spans"],
        "queries": plan.get("queries", []),
        "retriever_used": plan.get("retriever_used"),
        "llm_retrieval": plan.get("llm_retrieval"),
        "best_sim": plan["best_sim"],
        "contrast_ex": plan.get("contrast_ex"),
        "candidates": parsed_candidates,
    }

    if parsed_candidates:
        ranked = sorted(
            enumerate(parsed_candidates),
            key=lambda kv: (-kv[1]["n_verified"], not kv[1]["parse_ok"], kv[0]),
        )
        best_idx, best = ranked[0]
        proposed = best.get("conclusion") or best.get("answer") or candidates_raw[best_idx]
        rec["correction"]["chosen_candidate_index"] = best_idx
        rec["correction"]["chosen_n_verified"] = best["n_verified"]
        rec["correction"]["chosen_parse_ok"] = best["parse_ok"]
    else:
        proposed = item["model_answer"]
    rec["correction"]["proposed"] = proposed

    # ---- verdict V2 ----
    v = run_verdict_v2(fold, idx, note, item["question"], item["model_answer"],
                       proposed, port=port, k=args.k_ver,
                       accept_threshold=args.verdict_thresh)
    rec["verdict"] = v

    if not v["accept_correction"]:
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "kept_original", "delta": 0, "final_eval": eval_orig}
        log.write(rec)
        return

    # ---- judge_corrected (GPT-4o T=0 oracle) ----
    time.sleep(0.5)
    j_cor = judge_call(note, item["question"], item["ground_truth"], proposed,
                       n=1, temperature=0.0)
    rec["judge_corrected"] = {
        "label": j_cor["label"],
        "raw": j_cor["raws"][0] if j_cor["raws"] else "",
    }
    eval_corrected = j_cor["label"] if j_cor["label"] is not None else eval_orig
    delta = (1 if eval_corrected == 1 and eval_orig == 0
             else (-1 if eval_corrected == 0 and eval_orig == 1 else 0))
    rec["outcome"] = {
        "action": "corrected",
        "delta": delta,
        "final_eval": eval_corrected,
    }
    log.write(rec)


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def run_model(model_alias: str, *, args, notes: dict) -> dict:
    cfg = MODELS[model_alias]
    print(f"\n{'=' * 70}")
    print(f"MULTI-MODEL PILOT — {model_alias}")
    print(f"{'=' * 70}")
    print(f"  step8 dir: {cfg['step8_dir']}")
    print(f"  expected vLLM model id substring: {cfg['expected_id_substring']}")

    # Verify the right model is loaded in vLLM
    served = served_model_id(args.port).lower()
    print(f"  vLLM is currently serving: {served}")
    if cfg["expected_id_substring"] not in served:
        print(f"  ❌ vLLM is not serving {model_alias}. Expected substring "
              f"'{cfg['expected_id_substring']}' in '{served}'.")
        if args.no_prompt:
            return {"model": model_alias, "skipped": True,
                    "reason": "wrong vLLM model loaded"}
        input(f"\n  Please restart vLLM with the {model_alias} model and press Enter to continue...")
        served = served_model_id(args.port).lower()
        if cfg["expected_id_substring"] not in served:
            print(f"  Still wrong model loaded; skipping {model_alias}.")
            return {"model": model_alias, "skipped": True,
                    "reason": "wrong vLLM model loaded after retry"}

    # Sample test items
    items = sample_test_items(cfg["step8_dir"], args.n_wrong, args.n_correct,
                              seed=args.seed)
    print(f"  Sampled {len(items)} items "
          f"({sum(1 for i in items if i['label']==0)} wrong, "
          f"{sum(1 for i in items if i['label']==1)} correct) "
          f"across 5 folds")

    # Per-model audit log
    model_dir = OUT_DIR / cfg["step8_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "audit.jsonl"
    log = AuditLog(log_path)
    print(f"  Audit log: {log_path}")
    print(f"  Already done: {len(log.all())}")

    for i, item in enumerate(items, 1):
        try:
            run_one_item(item, notes, port=args.port, model_cfg=cfg,
                         args=args, log=log)
        except Exception as e:
            print(f"  ❌ ({item['fold']},{item['idx']}): {e}", flush=True)
            continue
        if i % 5 == 0:
            done = log.all()
            from collections import Counter
            actions = Counter((r.get("outcome") or {}).get("action", "?") for r in done)
            fixes = sum(1 for r in done
                        if (r.get("outcome") or {}).get("action") == "corrected"
                        and (r.get("outcome") or {}).get("delta") == 1)
            brks = sum(1 for r in done
                       if (r.get("outcome") or {}).get("action") == "corrected"
                       and (r.get("outcome") or {}).get("delta") == -1)
            print(f"  [{i}/{len(items)}] log={len(done)} {dict(actions)} fixes={fixes} brk={brks}", flush=True)

    # Per-model summary
    done = log.all()
    from collections import Counter
    actions = Counter((r.get("outcome") or {}).get("action", "?") for r in done)
    fixes = sum(1 for r in done
                if (r.get("outcome") or {}).get("action") == "corrected"
                and (r.get("outcome") or {}).get("delta") == 1)
    brks = sum(1 for r in done
               if (r.get("outcome") or {}).get("action") == "corrected"
               and (r.get("outcome") or {}).get("delta") == -1)
    return {
        "model": model_alias,
        "step8_dir": cfg["step8_dir"],
        "served_id": served,
        "n_items": len(done),
        "actions": dict(actions),
        "fixes": fixes,
        "breaks": brks,
        "log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="qwen2.5,llama3,deepseek,qwen3",
                   help="comma-separated subset of " + ",".join(MODELS.keys()))
    p.add_argument("--n-wrong", type=int, default=10)
    p.add_argument("--n-correct", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--k-det", type=int, default=5)
    p.add_argument("--k-cor", type=int, default=3)
    p.add_argument("--k-ver", type=int, default=5)
    p.add_argument("--severity-thresh", type=int, default=3)
    p.add_argument("--verdict-thresh", type=int, default=3)
    p.add_argument("--span-thresh", type=float, default=0.45)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-prompt", action="store_true",
                   help="don't pause for vLLM swap; skip models that aren't loaded")
    args = p.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    if not selected:
        print(f"!! no valid models in --models. choices: {list(MODELS.keys())}")
        return 1

    notes = _load_notes_lookup()

    summaries = []
    for m in selected:
        try:
            s = run_model(m, args=args, notes=notes)
        except Exception as e:
            print(f"❌ failed on {m}: {e}", flush=True)
            s = {"model": m, "error": str(e)}
        summaries.append(s)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"MULTI-MODEL PILOT SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<14} {'N':>5} {'fix':>5} {'brk':>5} {'kept':>6} {'keep':>6}")
    for s in summaries:
        if s.get("skipped") or s.get("error"):
            print(f"{s['model']:<14} SKIPPED/ERROR: {s.get('reason') or s.get('error','')}")
            continue
        a = s["actions"]
        kept = a.get("kept_original", 0) + a.get("kept_original_low_evidence", 0)
        keep = a.get("keep", 0)
        print(f"{s['model']:<14} {s['n_items']:>5} {s['fixes']:>5} {s['breaks']:>5} {kept:>6} {keep:>6}")

    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
