"""Error Location pipeline — main runner.

Steps (selected via --step):
  probe      : Step 0 — token-budget probe (5 items, max_tokens=2048)
  gold       : Step 1 — GPT-4o gold narrative gen (5-fold, ~113 items, ~$0.90)
  smoke      : Step 3a — locator format smoke (3 items, read raw)
  iterate    : Step 3b — locator iteration on fold_0 (~26 items × N versions)
  lockdown   : Step 4 — winning prompt × 5 folds
  spotcheck  : Step 5 — GPT-4o re-judge 10 random items (~$0.10)
  report     : Step 6 — write summary JSON + log table for Finding

Per Implementation Discipline Rule 4: smoke before any batch.
Per Truncation Detection: every LLM call passes through detect_truncation.
Per Misc(WIP) + Execution Discipline § Deterministic by default: temperature=0.

Plan: https://www.notion.so/3506be46cf3c817e99f0fa38d288e0bd
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ichl.error_location.prompts import (
    GOLD_SYSTEM,
    GOLD_USER_TMPL,
    LOCATOR_VERSIONS,
    COMPARATOR_SYSTEM,
    COMPARATOR_USER_TMPL,
    COMPARATOR_V4_SYSTEM,
    COMPARATOR_V4_USER_TMPL,
    SPOT_CHECK_SYSTEM,
    SPOT_CHECK_USER_TMPL,
)
from ichl.prompt_engineering.correction.truncation_detector import detect_truncation


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "ichl" / "error_location"
STEP8 = ROOT / "output" / "step8"
NOTES_FILE = ROOT / "output" / "EHRNoteQA_processed.jsonl"


# ============================================================
# Data loading
# ============================================================
def load_full_notes() -> dict[int, dict]:
    """patient_id -> {note, question, choice_*, answer (letter), ground_truth}"""
    notes_by_pid: dict[int, dict] = {}
    for line in NOTES_FILE.open():
        if not line.strip():
            continue
        r = json.loads(line)
        pid = int(r["patient_id"])
        # Build step8-format full note
        note_parts = []
        for i in (1, 2, 3):
            n = r.get(f"note_{i}")
            if n:
                note_parts.append(f"[Note {i}]\n{n}")
        note_text = "\n\n".join(note_parts)
        # GT = letter + choice text
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        notes_by_pid[pid] = {
            "patient_id": pid,
            "note": note_text,
            "question": str(r.get("question", "")),
            "ground_truth": gt,
            "answer_letter": letter,
        }
    return notes_by_pid


def load_wrong_zs(target: str = "qwen2.5-7b-instruct", folds: list[int] | None = None) -> list[dict]:
    """Load wrong-zs items (binary_correct=0) per fold. Returns list of records.

    Schema: patient_id, fold, zs_answer, binary_correct (always 0), question, gt, note.
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    notes_by_pid = load_full_notes()
    out = []
    for fold in folds:
        csv_path = STEP8 / target / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        df = pd.read_csv(csv_path)
        df = df[df["binary_correct"] == 0]
        for _, r in df.iterrows():
            pid = int(r["patient_id"])
            if pid not in notes_by_pid:
                continue
            n = notes_by_pid[pid]
            out.append({
                "patient_id": pid,
                "fold": fold,
                "zs_answer": str(r.get("model_answer") or ""),
                "binary_correct": 0,
                "question": n["question"],
                "ground_truth": n["ground_truth"],
                "note": n["note"],
            })
    return out


# ============================================================
# Wrapper logging interface — Implementation Discipline Rule 3
# ============================================================
def _wrapper_record(text: str, finish_reason: str | None, usage: Any | None,
                    raw_response: str, max_tokens: int,
                    target: str, sub_variant: str, latency_s: float,
                    err: str | None = None) -> dict:
    """Standard wrapper output. Inherits detect_truncation."""
    usage_dict = None
    prompt_tokens = None
    completion_tokens = None
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None) or (
            usage.get("prompt_tokens") if isinstance(usage, dict) else None)
        completion_tokens = getattr(usage, "completion_tokens", None) or (
            usage.get("completion_tokens") if isinstance(usage, dict) else None)
        usage_dict = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    trunc = detect_truncation(
        raw_response=raw_response, text_clean=text,
        finish_reason=finish_reason, usage=usage_dict,
        max_tokens=max_tokens, target=target, sub_variant=sub_variant,
    )
    return {
        "text": text,
        "finish_reason": finish_reason,
        "truncation_report": trunc.as_dict() if hasattr(trunc, "as_dict") else {
            "is_truncated_certain": trunc.is_truncated_certain,
            "is_truncated_likely": trunc.is_truncated_likely,
            "signals": trunc.signals,
            "notes": trunc.notes,
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_s": round(latency_s, 2),
        "_err": err,
    }


# ============================================================
# vLLM target call (Qwen2.5-7B-Instruct)
# ============================================================
def vllm_call(client, model_name: str, system: str, user: str,
              max_tokens: int, temperature: float = 0.0,
              target: str = "qwen2.5-7b-instruct",
              enable_thinking: bool | None = None,
              max_model_len: int | None = None,
              max_retries: int = 2) -> dict:
    """Per Prompt Design per Model: Qwen3 toggles thinking via chat_template_kwargs.

    Per `Truncation Detection on Every LLM Output § Required actions`:
    auto-retry on certain-truncation at 2× max_tokens (capped at max_model_len),
    up to max_retries retries. Preserve retry history in record.
    """
    retry_history = []
    cur_max = max_tokens
    for attempt in range(max_retries + 1):
        t0 = time.monotonic()
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": temperature, "max_tokens": cur_max,
            }
            if enable_thinking is not None:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
            r = client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or ""
            # Strip <think>...</think> for downstream judge per Prompt Design per Model.
            if "</think>" in text:
                import re as _re
                text = _re.sub(r"^.*?</think>\s*", "", text, flags=_re.DOTALL).strip()
            rec = _wrapper_record(
                text=text, finish_reason=r.choices[0].finish_reason,
                usage=r.usage, raw_response=text, max_tokens=cur_max,
                target=target, sub_variant="locator",
                latency_s=time.monotonic() - t0,
            )
            rec["retry_history"] = list(retry_history)
            rec["final_max_tokens"] = cur_max
            # Auto-retry trigger per Truncation Detection § Required actions
            if rec["truncation_report"]["is_truncated_certain"] and attempt < max_retries:
                # Cap retry so prompt + max_tokens + safety ≤ max_model_len.
                # Use observed prompt_tokens to compute true headroom.
                pt = rec.get("prompt_tokens") or 0
                safety = 200
                hard_cap = (max_model_len - pt - safety) if (max_model_len and pt) else (max_model_len or cur_max * 2)
                next_max = min(int(cur_max * 2), hard_cap)
                if next_max > cur_max:
                    retry_history.append({
                        "attempt": attempt, "max_tokens": cur_max,
                        "completion_tokens": rec["completion_tokens"],
                        "trunc_signals": rec["truncation_report"]["signals"],
                    })
                    cur_max = next_max
                    continue
            return rec
        except Exception as e:
            return _wrapper_record(
                text="", finish_reason=None, usage=None, raw_response="",
                max_tokens=cur_max, target=target, sub_variant="locator",
                latency_s=time.monotonic() - t0, err=str(e)[:300],
            )
    return rec  # exhausted retries; return last attempt's record


# ============================================================
# GPT-4o call (gold + spot-check)
# ============================================================
def _load_openai_keys():
    """Return ordered list of OPENAI_API_KEY values (primary first, then backups).

    Reads env vars matching OPENAI_API_KEY[_*] in declared order, then falls back
    to .env file (also preserving declaration order). Used by _openai_client() for
    priority + 429/auth fallback.
    """
    keys = []
    seen = set()
    # 1. Env vars: primary first, then any OPENAI_API_KEY_BACKUP* / OPENAI_API_KEY_*
    primary = os.environ.get("OPENAI_API_KEY")
    if primary:
        keys.append(primary); seen.add(primary)
    for k in sorted(os.environ.keys()):
        if k.startswith("OPENAI_API_KEY") and k != "OPENAI_API_KEY":
            v = os.environ.get(k)
            if v and v not in seen:
                keys.append(v); seen.add(v)
    # 2. .env file (preserve declaration order)
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OPENAI_API_KEY") and "=" in line:
                v = line.split("=", 1)[1].strip()
                if v and v not in seen:
                    keys.append(v); seen.add(v)
    return keys


class _FallbackOpenAIClient:
    """OpenAI client wrapper that retries on 429/auth errors with backup keys.

    Drop-in for openai.OpenAI() — exposes .chat.completions.create with the same
    signature; on RateLimitError or AuthenticationError, switches to the next key.
    """
    def __init__(self, keys, timeout=300):
        from openai import OpenAI
        if not keys:
            raise RuntimeError("No OPENAI_API_KEY found in env or .env")
        self._keys = keys
        self._timeout = timeout
        self._idx = 0
        self._client = OpenAI(api_key=keys[0], timeout=timeout)

    def _switch_to_backup(self):
        if self._idx + 1 >= len(self._keys):
            return False
        self._idx += 1
        from openai import OpenAI
        self._client = OpenAI(api_key=self._keys[self._idx], timeout=self._timeout)
        print(f"[openai] primary key failed; switched to backup #{self._idx}")
        return True

    @property
    def chat(self):
        return _FallbackChat(self)

    def models_list(self):
        return self._client.models.list()


class _FallbackChat:
    def __init__(self, owner): self._o = owner
    @property
    def completions(self): return _FallbackCompletions(self._o)


class _FallbackCompletions:
    def __init__(self, owner): self._o = owner

    def create(self, **kwargs):
        from openai import RateLimitError, AuthenticationError, PermissionDeniedError
        last_err = None
        for _ in range(len(self._o._keys)):
            try:
                return self._o._client.chat.completions.create(**kwargs)
            except (RateLimitError, AuthenticationError, PermissionDeniedError) as e:
                last_err = e
                if not self._o._switch_to_backup():
                    raise
        raise last_err if last_err else RuntimeError("all openai keys exhausted")


def _openai_client():
    """Per Execution Discipline § OPENAI_API_KEY pattern.
    Returns a fallback-capable client that tries primary key first, then any
    OPENAI_API_KEY_BACKUP_* keys on 429/auth errors.
    """
    keys = _load_openai_keys()
    return _FallbackOpenAIClient(keys, timeout=300)


def gpt4o_call(client, system: str, user: str,
               max_tokens: int = 400, temperature: float = 0.0,
               sub_variant: str = "gpt4o") -> dict:
    t0 = time.monotonic()
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens,
        )
        text = (r.choices[0].message.content or "").strip()
        return _wrapper_record(
            text=text, finish_reason=r.choices[0].finish_reason,
            usage=r.usage, raw_response=text, max_tokens=max_tokens,
            target="gpt-4o", sub_variant=sub_variant,
            latency_s=time.monotonic() - t0,
        )
    except Exception as e:
        return _wrapper_record(
            text="", finish_reason=None, usage=None, raw_response="",
            max_tokens=max_tokens, target="gpt-4o", sub_variant=sub_variant,
            latency_s=time.monotonic() - t0, err=str(e)[:300],
        )


# ============================================================
# Qwen3-235B-MLX comparator call (C=1 per MEMORY)
# ============================================================
QWEN3_235B_MODEL_HINT = (
    "/Users/madblade/.lmstudio/models/lmstudio-community/"
    "Qwen3-235B-A22B-Instruct-2507-MLX-4bit"
)  # exact id from curl /v1/models on 192.168.68.107:8800 (verified 2026-04-28)


def _mlx_client():
    from ichl.clients.mlx_openai_client import MLXOpenAIClient
    # Override default 27B model to 235B
    return MLXOpenAIClient(model=QWEN3_235B_MODEL_HINT)


def mlx_comparator_call(client, system: str, user: str,
                        max_tokens: int = 200, temperature: float = 0.0) -> dict:
    t0 = time.monotonic()
    try:
        # MLXOpenAIClient.chat returns MLXResponse with content/finish_reason/etc.
        resp = client.chat(
            system=system, user=user,
            enable_thinking=False,
            temperature=temperature, max_tokens=max_tokens,
        )
        return _wrapper_record(
            text=resp.content, finish_reason=resp.finish_reason,
            usage={"prompt_tokens": resp.prompt_tokens,
                   "completion_tokens": resp.completion_tokens},
            raw_response=resp.content, max_tokens=max_tokens,
            target="qwen3-235b-mlx", sub_variant="comparator",
            latency_s=time.monotonic() - t0,
        )
    except Exception as e:
        return _wrapper_record(
            text="", finish_reason=None, usage=None, raw_response="",
            max_tokens=max_tokens, target="qwen3-235b-mlx", sub_variant="comparator",
            latency_s=time.monotonic() - t0, err=str(e)[:300],
        )


# ============================================================
# Output parsing (lightweight; the comparator reads narratives directly,
# but we extract the structured fields for record-keeping)
# ============================================================
_CLAIM_RE = re.compile(r"CLAIM:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_CONTRA_RE = re.compile(r"CONTRADICTION:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_SECTION_RE = re.compile(r"SECTION:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_NO_CONTRA_RE = re.compile(r"NO\s+CONTRADICTION\s+FOUND", re.IGNORECASE)
_MATCH_RE = re.compile(r"MATCH:\s*(YES|NO)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def parse_narrative(text: str) -> dict:
    if _NO_CONTRA_RE.search(text or ""):
        return {"claim": None, "contradiction": None, "section": None,
                "no_contradiction": True, "format_violation": False}
    claim = _CLAIM_RE.search(text or "")
    contra = _CONTRA_RE.search(text or "")
    section = _SECTION_RE.search(text or "")
    fmt_ok = bool(claim and contra)
    return {
        "claim": (claim.group(1).strip() if claim else None),
        "contradiction": (contra.group(1).strip() if contra else None),
        "section": (section.group(1).strip() if section else None),
        "no_contradiction": False,
        "format_violation": not fmt_ok,
    }


def parse_match(text: str) -> dict:
    m = _MATCH_RE.search(text or "")
    r = _REASON_RE.search(text or "")
    if not m:
        return {"match": None, "reason": (text or "")[:200], "format_violation": True}
    return {
        "match": 1 if m.group(1).upper() == "YES" else 0,
        "reason": (r.group(1).strip() if r else ""),
        "format_violation": False,
    }


# ============================================================
# Step 0 — Token-budget probe
# ============================================================
def step_probe(args):
    """Run locator on 5 fold_0 wrong items at max_tokens=2048; compute p95."""
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT / "step0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_wrong_zs(folds=[0])[:5]
    print(f"[Step 0] Probing on {len(items)} fold_0 wrong items at max_tokens=2048...")

    from ichl.common import vllm_manager
    vllm_manager.stop()
    vllm_manager.ensure_model(args.target, log_dir=out_dir / "vllm_logs")
    from openai import OpenAI
    vllm = OpenAI(base_url=args.vllm_url, api_key="not-needed", timeout=600)

    sys, tmpl = LOCATOR_VERSIONS[args.locator_version]
    out_file = out_dir / "probe_outputs.jsonl"
    comp_toks = []
    # Generous probe budget: thinking adds 200-800 tok per Prompt Design per Model.
    # Default probe budget: thinking models need much more room (Qwen3 thinking blew 4096 in prior probe).
    # Can override via --probe-max-tokens. The point of the probe is to find the tail; pick big.
    probe_max_tok = args.probe_max_tokens if args.probe_max_tokens else (12000 if args.enable_thinking else 2048)
    print(f"  probe max_tokens={probe_max_tok} (thinking={args.enable_thinking})")
    with out_file.open("w") as f:
        for it in items:
            user = tmpl.format(note=it["note"], question=it["question"], zs_answer=it["zs_answer"])
            r = vllm_call(vllm, args.vllm_model, sys, user, max_tokens=probe_max_tok, temperature=0.0,
                          target=args.target, enable_thinking=(True if args.enable_thinking else (False if args.disable_thinking else None)))
            row = {**it, **r, "step": "probe"}
            f.write(json.dumps(row) + "\n"); f.flush()
            ct = r.get("completion_tokens")
            print(f"  pid={it['patient_id']}  comp_tok={ct}  finish={r.get('finish_reason')}  trunc={r['truncation_report']['is_truncated_certain']}")
            if ct: comp_toks.append(ct)

    if not comp_toks:
        raise SystemExit("[Step 0] No completion_tokens recorded; cannot compute budget")
    # Probe-itself-truncation check per Truncation Detection § Step 0:
    # "the probe is exactly where unrealistic settings should be caught."
    n_probe_trunc = sum(1 for r in (json.loads(l) for l in (out_dir / "probe_outputs.jsonl").open())
                        if r.get("truncation_report", {}).get("is_truncated_certain"))
    if n_probe_trunc > 0:
        print(f"\n[Step 0] FAIL: {n_probe_trunc}/{len(comp_toks)} probe items hit max_tokens={probe_max_tok} (certain truncation).")
        print(f"  Probe budget too small. p95 is biased downward.")
        print(f"  Re-run Step 0 with --max-gen-tokens {probe_max_tok * 2} (or higher) before computing production budget.")
        raise SystemExit("[Step 0] Probe was truncated; cannot compute reliable production budget.")
    arr = np.array(comp_toks)
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    max_obs = int(arr.max())
    production = max(int(2 * p95), int(max_obs * 1.2))
    summary = {
        "n_probed": len(comp_toks),
        "n_probe_truncated": n_probe_trunc,
        "completion_tokens": {"min": int(arr.min()), "p50": int(np.percentile(arr, 50)),
                              "p95": int(p95), "p99": int(p99), "max": max_obs},
        "production_max_tokens": production,
        "formula": "max(2 * p95, max * 1.2)",
        "principle": "Truncation Detection on Every LLM Output § Step 0 probe pattern",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[Step 0] DONE.")
    print(f"  comp_tok: min={summary['completion_tokens']['min']}  p50={summary['completion_tokens']['p50']}  p95={int(p95)}  p99={int(p99)}  max={max_obs}")
    print(f"  → production max_tokens = max(2*p95={int(2*p95)}, max*1.2={int(max_obs*1.2)}) = {production}")
    print(f"  Saved: {out_dir / 'summary.json'}")


# ============================================================
# Step 1 — GPT-4o gold narrative gen
# ============================================================
def step_gold(args):
    """Gold gen, target-aware. Output dir: gold/<target>/fold_N/ (Qwen2.5 stays at gold/fold_N/ for back-compat)."""
    """Generate gold narratives for all wrong-zs items via GPT-4o."""
    # Qwen2.5 stays at OUT/gold for back-compat; other targets at OUT/gold/<target>
    out_dir = OUT / "gold" if args.target == "qwen2.5-7b-instruct" else OUT / "gold" / args.target
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0, 1, 2, 3, 4]
    items = load_wrong_zs(target=args.target, folds=folds)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[Step 1] Generating gold narratives for {len(items)} wrong-zs items (folds={folds})...")
    print(f"  Cost estimate: {len(items)} × $0.008 = ${len(items)*0.008:.2f}")

    client = _openai_client()
    n_done = n_err = n_trunc = n_format_viol = 0
    cost_total = 0.0

    by_fold: dict[int, list[dict]] = {f: [] for f in folds}
    for it in items:
        user = GOLD_USER_TMPL.format(
            note=it["note"], question=it["question"],
            ground_truth=it["ground_truth"], zs_answer=it["zs_answer"])
        r = gpt4o_call(client, GOLD_SYSTEM, user, max_tokens=400, temperature=0.0,
                       sub_variant="gold")
        n_done += 1
        if r["_err"]:
            n_err += 1
        elif r["truncation_report"]["is_truncated_certain"]:
            n_trunc += 1
        narrative_text = r["text"]
        parsed = parse_narrative(narrative_text)
        if parsed["format_violation"]:
            n_format_viol += 1
        if r["prompt_tokens"]:
            cost_total += r["prompt_tokens"] * 5e-6 + (r["completion_tokens"] or 0) * 1.5e-5
        rec = {
            "patient_id": it["patient_id"], "fold": it["fold"],
            "question": it["question"], "ground_truth": it["ground_truth"],
            "zs_answer": it["zs_answer"],
            "gold_narrative_text": narrative_text,
            "gold_claim": parsed["claim"],
            "gold_contradiction": parsed["contradiction"],
            "gold_section": parsed["section"],
            "gold_no_contradiction": parsed["no_contradiction"],
            "gold_format_violation": parsed["format_violation"],
            "truncation_report": r["truncation_report"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "latency_s": r["latency_s"],
            "_err": r["_err"],
        }
        by_fold[it["fold"]].append(rec)
        if n_done % 25 == 0:
            print(f"  done {n_done}/{len(items)}  err={n_err} trunc={n_trunc} fmt_viol={n_format_viol}  cost~${cost_total:.2f}")

    for f, rows in by_fold.items():
        fold_dir = out_dir / f"fold_{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "gold_narratives.jsonl").open("w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
    print(f"\n[Step 1] DONE. {n_done} items, err={n_err} trunc={n_trunc} fmt_viol={n_format_viol}")
    print(f"  Total cost: ${cost_total:.2f}")
    print(f"  Saved: {out_dir}/fold_*/gold_narratives.jsonl")


# ============================================================
# Step 3a — Locator format smoke (3 items)
# Step 3b — Iteration on fold_0
# ============================================================
def step_locate(args):
    """Run locator on items. Use --limit 3 for smoke; default = full fold_0 (or 5-fold for lockdown)."""
    out_dir = OUT / "locator" / args.locator_version
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0]
    items = load_wrong_zs(target=args.target, folds=folds)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[Locate] {len(items)} wrong-zs items (folds={folds}, version={args.locator_version})...")

    from ichl.common import vllm_manager
    vllm_manager.stop()
    vllm_manager.ensure_model(args.target, log_dir=out_dir / "vllm_logs")
    from openai import OpenAI
    vllm = OpenAI(base_url=args.vllm_url, api_key="not-needed", timeout=600)

    sys, tmpl = LOCATOR_VERSIONS[args.locator_version]
    n_done = n_err = n_trunc = n_format_viol = n_no_contra = 0

    # Resolve max_model_len from vllm_manager TARGETS (used as retry cap)
    from ichl.common import vllm_manager as _vmgr
    _spec = _vmgr.TARGETS.get(args.target)
    _max_model_len = _spec.max_model_len if _spec else 16384

    by_fold: dict[int, list[dict]] = {f: [] for f in folds}
    for it in items:
        user = tmpl.format(note=it["note"], question=it["question"], zs_answer=it["zs_answer"])
        r = vllm_call(vllm, args.vllm_model, sys, user,
                      max_tokens=args.max_gen_tokens, temperature=0.0,
                      target=args.target, enable_thinking=(True if args.enable_thinking else (False if args.disable_thinking else None)),
                      max_model_len=_max_model_len, max_retries=2)
        n_done += 1
        if r["_err"]:
            n_err += 1
        elif r["truncation_report"]["is_truncated_certain"]:
            n_trunc += 1
        parsed = parse_narrative(r["text"])
        if parsed["no_contradiction"]:
            n_no_contra += 1
        elif parsed["format_violation"]:
            n_format_viol += 1
        rec = {
            "patient_id": it["patient_id"], "fold": it["fold"],
            "prompt_version": args.locator_version,
            "model_narrative_text": r["text"],
            "model_claim": parsed["claim"],
            "model_contradiction": parsed["contradiction"],
            "model_section": parsed["section"],
            "model_no_contradiction": parsed["no_contradiction"],
            "model_format_violation": parsed["format_violation"],
            "truncation_report": r["truncation_report"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "latency_s": r["latency_s"],
            "_err": r["_err"],
        }
        by_fold[it["fold"]].append(rec)
        if n_done % 25 == 0:
            print(f"  done {n_done}/{len(items)}  err={n_err} trunc={n_trunc} fmt_viol={n_format_viol} no_contra={n_no_contra}")
        # Per-50 abort gate per Per-50 In-Flight Sanity Checkpoint
        if n_done >= 50:
            if n_err > 0:
                print(f"  ABORT: HTTP/API error count > 0 after {n_done} items"); raise SystemExit(1)
            if n_trunc / n_done > 0.05:
                print(f"  ABORT: trunc rate {100*n_trunc/n_done:.1f}% > 5%"); raise SystemExit(1)

    # Post-batch truncation gate per Truncation Detection § Triggers requiring run rejection.
    # For small batches (n < 50) the per-50 gate never fires; this catches them.
    trunc_rate = n_trunc / max(n_done, 1)
    print(f"\n[Locate] post-batch trunc rate: {100*trunc_rate:.1f}% ({n_trunc}/{n_done})")
    if trunc_rate > 0.20:
        print(f"  RUN REJECTED per Truncation Detection § Triggers (>20% in single cell). Re-probe Step 0 and rerun.")
        raise SystemExit(2)
    elif trunc_rate > 0.10:
        print(f"  RUN REJECTED per Truncation Detection § Triggers (>10% overall). Re-probe Step 0 and rerun.")
        raise SystemExit(2)
    elif trunc_rate > 0.05:
        print(f"  WARNING: trunc rate {100*trunc_rate:.1f}% > 5%; results suspect, flag in Finding.")

    for f, rows in by_fold.items():
        fold_dir = out_dir / f"fold_{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / f"qwen25_{args.locator_version}.jsonl").open("w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
    print(f"\n[Locate] DONE. {n_done} items, err={n_err} trunc={n_trunc} fmt_viol={n_format_viol} no_contra={n_no_contra}")
    print(f"  Saved: {out_dir}/fold_*/qwen25_{args.locator_version}.jsonl")


# ============================================================
# Step 3-4 judge — Qwen3-235B-MLX comparator
# ============================================================
def step_judge(args):
    """Compare gold vs model narratives via Qwen3-235B-MLX (C=1)."""
    out_dir = OUT / "judge" / args.locator_version
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0]

    client = _mlx_client()
    n_done = n_err = n_match = n_format_viol = 0
    by_fold: dict[int, list[dict]] = {f: [] for f in folds}
    for fold in folds:
        gold_path = OUT / "gold" / f"fold_{fold}" / "gold_narratives.jsonl"
        loc_path = OUT / "locator" / args.locator_version / f"fold_{fold}" / f"qwen25_{args.locator_version}.jsonl"
        if not gold_path.exists() or not loc_path.exists():
            print(f"  fold_{fold}: missing gold or locator outputs; skip")
            continue
        gold = {r["patient_id"]: r for r in [json.loads(l) for l in gold_path.open()]}
        loc = {r["patient_id"]: r for r in [json.loads(l) for l in loc_path.open()]}
        common_pids = sorted(set(gold) & set(loc))
        for pid in common_pids:
            # v4 uses multi-candidate comparator; others use single-narrative comparator.
            if args.locator_version == "v4":
                user = COMPARATOR_V4_USER_TMPL.format(
                    gold_narrative=gold[pid]["gold_narrative_text"],
                    model_narrative=loc[pid]["model_narrative_text"],
                )
                cmp_sys = COMPARATOR_V4_SYSTEM
            else:
                user = COMPARATOR_USER_TMPL.format(
                    gold_narrative=gold[pid]["gold_narrative_text"],
                    model_narrative=loc[pid]["model_narrative_text"],
                )
                cmp_sys = COMPARATOR_SYSTEM
            r = mlx_comparator_call(client, cmp_sys, user, max_tokens=200, temperature=0.0)
            parsed = parse_match(r["text"])
            n_done += 1
            if r["_err"]:
                n_err += 1
            if parsed["format_violation"]:
                n_format_viol += 1
            if parsed["match"] == 1:
                n_match += 1
            rec = {
                "patient_id": pid, "fold": fold,
                "prompt_version": args.locator_version,
                "judge_match": parsed["match"],
                "judge_reason": parsed["reason"],
                "judge_format_violation": parsed["format_violation"],
                "raw_judge_text": r["text"],
                "truncation_report": r["truncation_report"],
                "latency_s": r["latency_s"],
                "_err": r["_err"],
            }
            by_fold[fold].append(rec)
            if n_done % 25 == 0:
                print(f"  done {n_done}  err={n_err} fmt_viol={n_format_viol} match={n_match}")

    for f, rows in by_fold.items():
        fold_dir = out_dir / f"fold_{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / f"qwen3_235b_{args.locator_version}.jsonl").open("w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
    print(f"\n[Judge] DONE. {n_done} items, err={n_err} fmt_viol={n_format_viol} match={n_match}")
    print(f"  Saved: {out_dir}/fold_*/qwen3_235b_{args.locator_version}.jsonl")


# ============================================================
# Step 5 — GPT-4o spot-check (10 random items)
# ============================================================
def step_spotcheck(args):
    """Re-judge 10 random items with GPT-4o; agreement vs Qwen3-235B."""
    out_dir = OUT / "spot_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0, 1, 2, 3, 4]

    candidates = []
    for fold in folds:
        gold_path = OUT / "gold" / f"fold_{fold}" / "gold_narratives.jsonl"
        loc_path = OUT / "locator" / args.locator_version / f"fold_{fold}" / f"qwen25_{args.locator_version}.jsonl"
        judge_path = OUT / "judge" / args.locator_version / f"fold_{fold}" / f"qwen3_235b_{args.locator_version}.jsonl"
        if not all(p.exists() for p in (gold_path, loc_path, judge_path)):
            continue
        gold = {r["patient_id"]: r for r in [json.loads(l) for l in gold_path.open()]}
        loc = {r["patient_id"]: r for r in [json.loads(l) for l in loc_path.open()]}
        judge = {r["patient_id"]: r for r in [json.loads(l) for l in judge_path.open()]}
        for pid in set(gold) & set(loc) & set(judge):
            if judge[pid]["judge_match"] in (0, 1):
                candidates.append({"pid": pid, "fold": fold,
                                   "gold": gold[pid], "loc": loc[pid],
                                   "judge": judge[pid]})
    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.spotcheck_n, len(candidates)))
    print(f"[Step 5] Spot-checking {len(sample)} items via GPT-4o (cost ~${len(sample)*0.008:.2f})...")

    client = _openai_client()
    out_file = out_dir / f"gpt4o_{args.locator_version}_sample{len(sample)}.jsonl"
    n_agree = 0
    with out_file.open("w") as fp:
        for s in sample:
            user = SPOT_CHECK_USER_TMPL.format(
                gold_narrative=s["gold"]["gold_narrative_text"],
                model_narrative=s["loc"]["model_narrative_text"],
            )
            r = gpt4o_call(client, SPOT_CHECK_SYSTEM, user, max_tokens=200, temperature=0.0,
                           sub_variant="spotcheck")
            parsed = parse_match(r["text"])
            agree = int(parsed["match"] == s["judge"]["judge_match"]) if parsed["match"] is not None else 0
            n_agree += agree
            rec = {
                "patient_id": s["pid"], "fold": s["fold"],
                "prompt_version": args.locator_version,
                "gpt4o_match": parsed["match"], "gpt4o_reason": parsed["reason"],
                "qwen3_235b_match": s["judge"]["judge_match"],
                "agreement": agree,
                "truncation_report": r["truncation_report"],
                "_err": r["_err"],
            }
            fp.write(json.dumps(rec) + "\n")
    rate = n_agree / max(len(sample), 1)
    print(f"\n[Step 5] DONE. Agreement: {n_agree}/{len(sample)} = {rate:.1%}")
    print(f"  Threshold: ≥80% per plan Slot 9.")
    print(f"  Saved: {out_file}")


# ============================================================
# Step 6 — Reporting
# ============================================================
def step_report(args):
    """Pool judgments across folds; compute match_rate + Wilson CI per Slot 11i."""
    from scipy.stats import beta
    folds = [int(f) for f in args.folds.split(",")] if args.folds else [0, 1, 2, 3, 4]

    by_fold = {}
    pooled = {"n_total": 0, "n_judged": 0, "n_match": 0,
              "n_truncated": 0, "n_format_violation": 0,
              "n_no_contradiction": 0, "n_judge_failed": 0}
    for fold in folds:
        gold_path = OUT / "gold" / f"fold_{fold}" / "gold_narratives.jsonl"
        loc_path = OUT / "locator" / args.locator_version / f"fold_{fold}" / f"qwen25_{args.locator_version}.jsonl"
        judge_path = OUT / "judge" / args.locator_version / f"fold_{fold}" / f"qwen3_235b_{args.locator_version}.jsonl"
        if not all(p.exists() for p in (gold_path, loc_path, judge_path)):
            print(f"  fold_{fold}: missing files; skip")
            continue
        loc_rows = [json.loads(l) for l in loc_path.open()]
        judge_rows = {r["patient_id"]: r for r in [json.loads(l) for l in judge_path.open()]}
        cell = {"n_total": len(loc_rows), "n_judged": 0, "n_match": 0,
                "n_truncated": 0, "n_format_violation": 0,
                "n_no_contradiction": 0, "n_judge_failed": 0}
        for r in loc_rows:
            pid = r["patient_id"]
            if r.get("model_no_contradiction"): cell["n_no_contradiction"] += 1
            if r.get("model_format_violation"): cell["n_format_violation"] += 1
            if r["truncation_report"]["is_truncated_certain"]: cell["n_truncated"] += 1
            j = judge_rows.get(pid)
            if not j or j.get("judge_match") not in (0, 1):
                cell["n_judge_failed"] += 1
            else:
                cell["n_judged"] += 1
                if j["judge_match"] == 1: cell["n_match"] += 1
        # Wilson CI for match rate
        k, n = cell["n_match"], cell["n_judged"]
        if n > 0:
            p = k / n
            lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
            hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        else:
            p, lo, hi = 0.0, 0.0, 1.0
        cell["match_rate"] = round(p, 4)
        cell["ci95"] = [round(lo, 4), round(hi, 4)]
        by_fold[fold] = cell
        for k_ in pooled: pooled[k_] += cell[k_]

    # Pooled CI
    k, n = pooled["n_match"], pooled["n_judged"]
    if n > 0:
        p = k / n
        lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    else:
        p, lo, hi = 0.0, 0.0, 1.0
    pooled["match_rate"] = round(p, 4)
    pooled["ci95"] = [round(lo, 4), round(hi, 4)]

    summary = {
        "prompt_version": args.locator_version,
        "by_fold": {str(f): v for f, v in by_fold.items()},
        "pooled": pooled,
        "judge": "qwen3-235b-mlx",
        "comparator_prompt": "verbatim Slot 11c",
        "scope": {
            "target": "qwen2.5-7b-instruct",
            "wrong_filter": "binary_correct=0 from step8 GPT-4o judge",
            "temperature": 0.0,
        },
    }
    out_path = OUT / f"summary_{args.locator_version}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Step 6] Summary saved: {out_path}")
    print(f"  Pooled: {pooled['n_match']}/{pooled['n_judged']} = {pooled['match_rate']:.3f}  CI95={pooled['ci95']}")
    for f, c in by_fold.items():
        print(f"  fold_{f}: {c['n_match']}/{c['n_judged']} = {c['match_rate']:.3f}  trunc={c['n_truncated']} fmt_viol={c['n_format_violation']} no_contra={c['n_no_contradiction']} judge_fail={c['n_judge_failed']}")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["probe", "gold", "smoke", "iterate", "lockdown",
                             "judge", "spotcheck", "report"])
    ap.add_argument("--folds", default="", help="comma-separated, e.g. '0' or '0,1,2,3,4'")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--locator-version", default="v1", choices=list(LOCATOR_VERSIONS.keys()))
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--target", default="qwen2.5-7b-instruct",
                    help="vllm_manager TARGETS key (qwen2.5-7b-instruct / qwen3-8b / deepseek-r1-distill-llama-8b).")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 thinking via chat_template_kwargs (default Qwen3 vLLM = think-on).")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Qwen3 explicit no-think via chat_template_kwargs={'enable_thinking': False}.")
    ap.add_argument("--max-gen-tokens", type=int, default=600,
                    help="Locator generation budget; set from Step 0 probe result.")
    ap.add_argument("--probe-max-tokens", type=int, default=0,
                    help="Step 0 probe max_tokens (0 = default: 2048 base, 12000 thinking).")
    ap.add_argument("--spotcheck-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.step == "probe":
        step_probe(args)
    elif args.step == "gold":
        step_gold(args)
    elif args.step in ("smoke", "iterate", "lockdown"):
        # smoke = locate with --limit 3; iterate = fold_0 full; lockdown = 5-fold full
        if args.step == "smoke":
            args.limit = args.limit or 3
            args.folds = args.folds or "0"
        elif args.step == "iterate":
            args.folds = args.folds or "0"
        elif args.step == "lockdown":
            args.folds = args.folds or "0,1,2,3,4"
        step_locate(args)
    elif args.step == "judge":
        step_judge(args)
    elif args.step == "spotcheck":
        step_spotcheck(args)
    elif args.step == "report":
        step_report(args)


if __name__ == "__main__":
    main()
