#!/usr/bin/env python3
"""
Regen + count-compare correction pilot.

Replicates the DeepSeek "regen + count + Qwen3 parse" method (Apr 2026
self_critique findings: +34 net → 80.5% from 76.9% on DeepSeek-R1-Distill).

Key differences from the D2+V2 evidence-based pipeline:

  1. NO detection gate. Every item is corrected.
  2. Correction = REGEN at zero shot — same prompt as the original step8
     zeroshot generation. No evidence retrieval, no error signal, no
     factored CoVe. Just re-ask the question.
  3. Verdict = COUNT_COMPARE. Show both answers, ask the target model to
     count contradictions in each, pick the one with fewer.
  4. Output parser = Qwen3-32B (Mac Studio). Reasoning models can't be
     parsed reliably with regex; Qwen3 extracts the A/B pick.

Each target model runs as both the regen-er AND the verdict-er
(self-correction). vLLM is assumed to already be serving the right model.

Sampling: same 10W + 30C across 5 folds as the D2+V2 multi-model pilot.

Output: output/step9_v2/multi_model/{model_dir}/regen_audit.jsonl
"""
from __future__ import annotations

import os
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
sys.path.insert(0, str(Path(__file__).parent))
from audit_log import AuditLog, make_record
from detection_format_bakeoff import served_model_id, vllm_chat, set_default_chat_template_kwargs
from judge import _load_notes_lookup, judge as judge_call
from multi_model_pilot import MODELS, sample_test_items, sample_random_per_fold

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Regen prompt — matches the structure of the original step8 zeroshot prompt
# (medical-expert system + note + question + open answer). Different prompts
# may be needed per model family but vLLM applies the chat template via
# tokenizer config so the same content works across models.
REGEN_SYS = "You are a medical expert."

REGEN_USER_TMPL = """Discharge note:
{note}

Question: {question}

Answer the question using only information from the discharge note. Be specific
and complete. If the question asks about multiple visits, conditions, or events,
cover all of them."""


# Count-compare verdict — verbatim from the DeepSeek pipeline
# (workspace/self_critique/scripts/run_regen_count_qwen_fullscale.py:62-76)
COUNT_COMPARE_SYS = "You are a strict medical expert."

COUNT_COMPARE_TMPL = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count how many factual claims in each answer contradict the discharge notes.
Different wording for the same fact is NOT a contradiction.

A_ERRORS: <number>
B_ERRORS: <number>"""


# Qwen3-32B parser — the reasoning model's output is hard to parse with regex,
# so we delegate the A/B extraction to a smaller, instruction-following model.
QWEN_PARSE_SYS = ("You interpret a medical expert's analysis and extract their "
                  "decision about which of two answers is more reliable.")

QWEN_PARSE_TMPL = """A medical expert compared two clinical answers (A and B) against discharge notes. Here is their analysis:

---
{analysis}
---

Based on this analysis, which answer should we keep? The expert counted contradictions in each — pick the answer with FEWER contradictions. If both have the same count, pick A.

DECISION: A or B
REASON: <one sentence>
/no_think"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def regen_zeroshot(note: str, question: str, port: int) -> str:
    """One-shot regen via the target model. Same prompt content for every
    model; vLLM applies the right tokenizer chat template.

    Note truncation: 18000 chars (~6000 tokens) keeps the prompt below
    BioMistral's 8192-token window.
    """
    user = REGEN_USER_TMPL.format(note=note[:18000], question=question)
    return vllm_chat(REGEN_SYS, user, port,
                     max_tokens=1024, temperature=0.0)


def count_compare(note: str, question: str, answer_a: str, answer_b: str,
                  port: int, max_tokens: int = 1024) -> str:
    """Run the count-compare prompt on the target model. Returns the raw
    text (with <think> blocks already stripped by vllm_chat).

    Default max_tokens=1024 (was 2048) so the prompt fits BioMistral's
    8192-token context window. The count-compare output is short
    ("A_ERRORS: N\\nB_ERRORS: N" + maybe a brief explanation), 1024 is
    enough for all models including reasoning ones (their thinking gets
    stripped by strip_think upstream).
    """
    # Truncate note for BioMistral's 8K window safety. Most notes fit but
    # the longest discharge summaries are 8K+ tokens by themselves.
    user = COUNT_COMPARE_TMPL.format(
        note=note[:18000],
        question=question,
        answer_a=answer_a[:1500], answer_b=answer_b[:1500],
    )
    return vllm_chat(COUNT_COMPARE_SYS, user, port,
                     max_tokens=max_tokens, temperature=0.0)


def qwen3_parse_decision(analysis: str) -> tuple[str, str]:
    """Use Qwen3-32B (Mac Studio) to extract A/B from the count-compare
    output. Returns (pick, reason). Falls back to 'A' (keep original) if
    extraction fails."""
    user = QWEN_PARSE_TMPL.format(analysis=analysis[:1500])
    for attempt in range(3):
        try:
            r = requests.post(QWEN32B_URL, json={
                "model": "Qwen/Qwen3-32B-MLX-bf16",
                "messages": [
                    {"role": "system", "content": QWEN_PARSE_SYS},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 128, "temperature": 0.0,
            }, timeout=120)
            text = r.json()["choices"][0]["message"]["content"].strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            m = re.search(r"DECISION:\s*([AB])", text.upper())
            if m:
                pick = m.group(1)
                rm = re.search(r"REASON:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
                reason = rm.group(1).strip()[:200] if rm else ""
                return pick, reason
        except Exception as e:
            print(f"  qwen3 parse retry {attempt+1}/3: {e}", flush=True)
            time.sleep(3)
    return "A", "parse failed → keep original"


# ---------------------------------------------------------------------------
# Per-item runner
# ---------------------------------------------------------------------------

def run_one(item: dict, notes: dict[str, str], *, port: int,
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

    # ---- judge_orig: reuse the existing binary_correct label from step8 ----
    # The step8 CSVs already contain GPT-4o binary judgments; no need to
    # re-judge the original answer (saves ~962 GPT-4o calls per model).
    eval_orig = int(item["label"])
    rec["judge_orig"] = {
        "label": eval_orig,
        "raw": "",
        "source": "step8_binary_correct",
    }

    # ---- regenerate from zero shot (target model) ----
    try:
        regen = regen_zeroshot(note, item["question"], port)
    except Exception as e:
        print(f"  regen err ({fold},{idx}): {e}", flush=True)
        rec["correction"] = {"skipped_reason": "regen_failed", "error": str(e)}
        rec["verdict"] = None
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "keep", "delta": 0, "final_eval": eval_orig}
        log.write(rec)
        return

    # ---- count-compare verdict (blind A/B placement) ----
    rng = random.Random(42 + (fold << 16) + idx)
    orig_in_a = rng.random() > 0.5
    ans_a = item["model_answer"] if orig_in_a else regen
    ans_b = regen if orig_in_a else item["model_answer"]
    try:
        cc_raw = count_compare(note, item["question"], ans_a, ans_b, port)
    except Exception as e:
        print(f"  cc err ({fold},{idx}): {e}", flush=True)
        cc_raw = ""

    pick, reason = qwen3_parse_decision(cc_raw) if cc_raw else ("A", "empty cc output")
    accept = (pick == "B") if orig_in_a else (pick == "A")  # accept = pick the regen

    rec["correction"] = {
        "skipped_reason": None,
        "method": "regen_zeroshot",
        "candidates": [{"raw": regen, "n_verified": 0, "parse_ok": True}],
        "proposed": regen,
    }
    rec["verdict"] = {
        "variant": "count_compare_qwen3parse",
        "orig_in_slot_A": orig_in_a,
        "cc_raw": cc_raw[:1000],
        "qwen3_pick": pick,
        "qwen3_reason": reason,
        "accept_correction": accept,
    }

    if not accept:
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "kept_original", "delta": 0, "final_eval": eval_orig}
        log.write(rec)
        return

    # ---- judge_corrected (oracle) ----
    time.sleep(0.5)
    j_cor = judge_call(note, item["question"], item["ground_truth"], regen,
                       n=1, temperature=0.0)
    rec["judge_corrected"] = {
        "label": j_cor["label"],
        "raw": j_cor["raws"][0] if j_cor["raws"] else "",
    }
    eval_cor = j_cor["label"] if j_cor["label"] is not None else eval_orig
    delta = (1 if eval_cor == 1 and eval_orig == 0
             else (-1 if eval_cor == 0 and eval_orig == 1 else 0))
    rec["outcome"] = {
        "action": "corrected",
        "delta": delta,
        "final_eval": eval_cor,
    }
    log.write(rec)


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def run_model(model_alias: str, *, args, notes: dict) -> dict:
    cfg = MODELS[model_alias]
    print(f"\n{'=' * 70}")
    print(f"REGEN+COUNT PILOT — {model_alias}")
    print(f"{'=' * 70}")
    served = served_model_id(args.port).lower()
    print(f"  vLLM serving: {served}")
    if cfg["expected_id_substring"] not in served:
        print(f"  ❌ wrong model loaded; skipping")
        return {"model": model_alias, "skipped": True}

    # Apply per-model chat_template_kwargs (e.g. Qwen3's enable_thinking=False)
    # for the duration of this model's run.
    ctk = cfg.get("chat_template_kwargs")
    set_default_chat_template_kwargs(ctk)
    if ctk:
        print(f"  chat_template_kwargs: {ctk}")

    if args.random_per_fold:
        items = sample_random_per_fold(cfg["step8_dir"],
                                       n_per_fold=args.random_per_fold,
                                       seed=args.seed)
        n_w = sum(1 for i in items if i["label"] == 0)
        n_c = sum(1 for i in items if i["label"] == 1)
        print(f"  Random sample: {len(items)} items ({n_w}W + {n_c}C, natural distribution)")
    else:
        items = sample_test_items(cfg["step8_dir"], args.n_wrong, args.n_correct,
                                  seed=args.seed)
        print(f"  Stratified sample: {len(items)} items "
              f"({sum(1 for i in items if i['label']==0)}W + "
              f"{sum(1 for i in items if i['label']==1)}C)")

    model_dir = OUT_DIR / cfg["step8_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / args.audit_name
    log = AuditLog(log_path)
    print(f"  Audit log: {log_path}")
    print(f"  Already done: {len(log.all())}")

    for i, item in enumerate(items, 1):
        try:
            run_one(item, notes, port=args.port, args=args, log=log)
        except Exception as e:
            print(f"  ❌ ({item['fold']},{item['idx']}): {e}", flush=True)
            continue
        if i % 5 == 0:
            done = log.all()
            actions = Counter((r.get("outcome") or {}).get("action", "?") for r in done)
            fixes = sum(1 for r in done
                        if (r.get("outcome") or {}).get("action") == "corrected"
                        and (r.get("outcome") or {}).get("delta") == 1)
            brks = sum(1 for r in done
                       if (r.get("outcome") or {}).get("action") == "corrected"
                       and (r.get("outcome") or {}).get("delta") == -1)
            print(f"  [{i}/{len(items)}] log={len(done)} {dict(actions)} fixes={fixes} brk={brks}",
                  flush=True)

    done = log.all()
    actions = Counter((r.get("outcome") or {}).get("action", "?") for r in done)
    fixes = sum(1 for r in done
                if (r.get("outcome") or {}).get("action") == "corrected"
                and (r.get("outcome") or {}).get("delta") == 1)
    brks = sum(1 for r in done
               if (r.get("outcome") or {}).get("action") == "corrected"
               and (r.get("outcome") or {}).get("delta") == -1)
    return {
        "model": model_alias,
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
    p.add_argument("--models", default="biomistral,qwen2.5,llama3,deepseek,qwen3",
                   help="comma-separated subset of " + ",".join(MODELS.keys()))
    p.add_argument("--n-wrong", type=int, default=10)
    p.add_argument("--n-correct", type=int, default=30)
    p.add_argument("--random-per-fold", type=int, default=None,
                   help="if set, sample N random items per fold (overrides n-wrong/n-correct)")
    p.add_argument("--audit-name", default="regen_audit.jsonl",
                   help="filename for the per-model audit log")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    if not selected:
        print(f"!! no valid models")
        return 1

    notes = _load_notes_lookup()
    summaries = []
    for m in selected:
        try:
            s = run_model(m, args=args, notes=notes)
        except Exception as e:
            print(f"❌ {m}: {e}", flush=True)
            s = {"model": m, "error": str(e)}
        summaries.append(s)

    print(f"\n{'=' * 70}")
    print(f"REGEN+COUNT PILOT SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<14} {'N':>5} {'fix':>5} {'brk':>5} {'kept':>6} {'keep':>6}")
    for s in summaries:
        if s.get("skipped") or s.get("error"):
            print(f"{s['model']:<14} SKIPPED/ERROR")
            continue
        a = s["actions"]
        kept = a.get("kept_original", 0)
        keep = a.get("keep", 0)
        print(f"{s['model']:<14} {s['n_items']:>5} {s['fixes']:>5} {s['breaks']:>5} {kept:>6} {keep:>6}")

    out_path = OUT_DIR / "regen_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
