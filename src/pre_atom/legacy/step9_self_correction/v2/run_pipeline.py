#!/usr/bin/env python3
"""
Module 6 — Pipeline runner.

Glues Modules 0-5 together:
    judge_orig → detect → (correct if detected & evidence found) → verdict
                       → judge_corrected → outcome → audit

Resume-safe via the audit log (keyed on (fold, idx)).

Two modes:
    --mode pilot   60-item sample (the same 60 used in detection bake-off)
    --mode full    all 962 items

Detection variant and verdict variant are pluggable; default to F1 + v1f
until Module 1 / Module 4a bake-offs select the winners.

Usage:
    python run_pipeline.py --port 8003 --mode pilot \
        --detect F1 --verdict v1f --k-det 5 --k-cor 3 --k-ver 5
"""
from __future__ import annotations

import os
import argparse
import time
from pathlib import Path

from audit_log import AuditLog, make_record
from correction import build_correction_prompt, generate_corrections
from detection_d2 import detect_d2
from detection_format_bakeoff import build_test_set
from judge import _load_notes_lookup, judge as judge_call
from verdict_v2 import run_verdict_v2

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUT_DIR = RUN_ROOT / "output" / "step9_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(item: dict, notes: dict[str, str], *, args, log: AuditLog) -> None:
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

    # ---- judge_orig ----
    j_orig = judge_call(note, item["question"], item["ground_truth"], item["model_answer"],
                        n=1, temperature=0.0)
    rec["judge_orig"] = {
        "label": j_orig["label"],
        "raw": j_orig["raws"][0] if j_orig["raws"] else "",
        "label_T01_legacy": int(item["label"]),
    }
    eval_orig = j_orig["label"] if j_orig["label"] is not None else int(item["label"])

    # ---- detection (D2: atomic yes/no, NO GPT-4o) ----
    det = detect_d2(note, item["question"], item["model_answer"], args.port,
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
        rec["outcome"] = {"action": "keep", "delta": 0,
                          "final_eval": eval_orig}
        log.write(rec)
        return

    # ---- correction (rule template + note-span retrieval, NO GPT-4o) ----
    # Pass the full D2 detection dict so build_correction_prompt can collect
    # all K=5 reason variants for multi-query retrieval (and use Qwen2.5
    # cite-by-number as the primary retriever)
    detection_final_for_correction = {
        "verdict": "INCORRECT",
        "error_type": det["error_type"],
        "error_statement": det["error_statement"],
        "correct_statement": "",
        "contradiction": det["contradiction"],
        "qmis": det["qmis"],
    }
    plan = build_correction_prompt(detection_final_for_correction, note, item["question"],
                                   item["model_answer"], fold,
                                   similarity_threshold=args.span_thresh,
                                   port=args.port,
                                   retriever=args.retriever)
    if plan["skipped_reason"]:
        rec["correction"] = {
            "skipped_reason": plan["skipped_reason"],
            "best_sim": plan.get("best_sim", 0.0),
            "spans": plan["spans"],
            "queries": plan.get("queries", []),
            "retriever_used": plan.get("retriever_used"),
            "llm_retrieval": plan.get("llm_retrieval"),
            "error_type": plan["error_type"],
        }
        rec["verdict"] = None
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "kept_original_low_evidence", "delta": 0,
                          "final_eval": eval_orig}
        log.write(rec)
        return

    candidates_raw = generate_corrections(plan["prompt"], port=args.port,
                                          k=args.k_cor, temperature=0.7)

    # Parse + verify each candidate (factored-CoVe premises/conclusion format)
    from correction_prompt_v2 import parse_premises_output, verify_evidence_quotes
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

    # Best-of-K candidate selection: prefer candidates with the most verified
    # quotes; tie-break by parse_ok then by index. Falls back to the first raw
    # if everything failed parsing.
    if parsed_candidates:
        ranked = sorted(
            enumerate(parsed_candidates),
            key=lambda kv: (-kv[1]["n_verified"], not kv[1]["parse_ok"], kv[0]),
        )
        best_idx, best = ranked[0]
        proposed = best.get("conclusion") or candidates_raw[best_idx]
        rec["correction"]["chosen_candidate_index"] = best_idx
        rec["correction"]["chosen_n_verified"] = best["n_verified"]
        rec["correction"]["chosen_parse_ok"] = best["parse_ok"]
    else:
        proposed = item["model_answer"]
    rec["correction"]["proposed"] = proposed

    # ---- verdict (V2 pairwise A/B, NO GPT-4o) ----
    v = run_verdict_v2(fold, idx, note, item["question"],
                       item["model_answer"], proposed,
                       port=args.port, k=args.k_ver,
                       accept_threshold=args.verdict_thresh)
    rec["verdict"] = v

    if not v["accept_correction"]:
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "kept_original", "delta": 0,
                          "final_eval": eval_orig}
        log.write(rec)
        return

    # ---- judge_corrected ----
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--mode", choices=["pilot", "full"], required=True)
    p.add_argument("--k-det", type=int, default=5)
    p.add_argument("--k-cor", type=int, default=3)
    p.add_argument("--k-ver", type=int, default=5)
    p.add_argument("--severity-thresh", type=int, default=3,
                   help="min D2 votes (out of K) to fire detection")
    p.add_argument("--verdict-thresh", type=int, default=3,
                   help="min V2 verdict votes (out of K) to accept correction")
    p.add_argument("--span-thresh", type=float, default=0.45,
                   help="for embed retriever: cosine threshold; "
                        "for LLM retriever: vote-fraction threshold (votes/K)")
    p.add_argument("--retriever", default="union",
                   choices=["union", "llm_then_embed", "llm_only", "embed_only"],
                   help="which note-span retriever to use (union = top-3 R3 + top-3 R2 deduped)")
    p.add_argument("--n-wrong", type=int, default=30)
    p.add_argument("--n-correct", type=int, default=30)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default=None,
                   help="run on a single item, format 'fold,idx' (uses full base, ignores --mode for selection)")
    args = p.parse_args()

    notes = _load_notes_lookup()
    if args.mode == "pilot":
        items = build_test_set(args.n_wrong, args.n_correct)
        log_path = OUT_DIR / "pilot_audit_log.jsonl"
    else:
        # Full: every (fold, idx) pair from the qwen2.5 step8 csvs
        import pandas as pd
        parts = []
        for fold in range(5):
            f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
            if f.exists():
                df = pd.read_csv(f); df["fold"] = fold; parts.append(df)
        base = pd.concat(parts, ignore_index=True)
        items = []
        for _, r in base.iterrows():
            items.append({
                "fold": int(r["fold"]),
                "idx": int(r["idx"]),
                "patient_id": int(r["patient_id"]),
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "model_answer": str(r["model_answer"]),
                "label": int(r["binary_correct"]),
            })
        log_path = OUT_DIR / "full_audit_log.jsonl"

    if args.only:
        f, i = args.only.split(",")
        f, i = int(f), int(i)
        import pandas as pd
        path = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{f}" / "zeroshot_evaluated_binary.csv"
        df = pd.read_csv(path)
        row = df[df["idx"] == i].iloc[0]
        items = [{
            "fold": f, "idx": i,
            "patient_id": int(row["patient_id"]),
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "model_answer": str(row["model_answer"]),
            "label": int(row["binary_correct"]),
        }]
    if args.limit:
        items = items[:args.limit]

    log = AuditLog(log_path)
    print(f"Pipeline mode={args.mode}  N={len(items)}  D2 K={args.k_det} sev={args.severity_thresh}/{args.k_det}  V2 K={args.k_ver} acc={args.verdict_thresh}/{args.k_ver}", flush=True)
    print(f"  Audit log: {log_path}", flush=True)
    print(f"  Already done: {len(log.all())}", flush=True)

    for i, item in enumerate(items, 1):
        try:
            run_one(item, notes, args=args, log=log)
        except Exception as e:
            print(f"  ❌ ({item['fold']},{item['idx']}): {e}", flush=True)
            continue
        if i % 5 == 0:
            done = log.all()
            actions: dict[str, int] = {}
            fixes = breaks = 0
            for r in done:
                o = r.get("outcome") or {}
                a = o.get("action", "?")
                actions[a] = actions.get(a, 0) + 1
                if a == "corrected":
                    if o.get("delta") == 1: fixes += 1
                    elif o.get("delta") == -1: breaks += 1
            print(f"  [{i}/{len(items)}] log={len(done)} actions={actions} fixes={fixes} brk={breaks}", flush=True)

    print(f"\nDone. Audit log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
