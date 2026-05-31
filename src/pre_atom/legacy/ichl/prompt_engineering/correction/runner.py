"""T0 correction runner — per-item regen for one (target, sub_variant).

Usage (programmatic):
    from ichl.prompt_engineering.correction.runner import run_correction

    results = run_correction(
        target="deepseek-r1-distill-llama-8b",
        sub_variant_id="a",
        items=items,                     # list[dict] from data_loader
        max_tokens=16384,
        temperature=1.0,
        raw_log_dir=Path(".../raw_outputs/deepseek/a"),
    )

Returns one dict per item with the full response payload. Does NOT parse
verdicts — Stage III handles verdict design.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ichl.clients.base import LLMClient
from ichl.clients.factory import make_client
from ichl.prompt_engineering.correction.sub_variants import (
    SYSTEM_MSG,
    format_prompt,
    get_enable_thinking,
    get_sub_variant,
)
from ichl.prompt_engineering.correction.truncation_detector import detect_truncation


_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

# Retry-on-truncation defaults
_DEFAULT_TRUNC_RETRIES = 2          # retry up to 2 times (so up to 3 total attempts)
_DEFAULT_TRUNC_MAX_CAP = 32768      # absolute ceiling; target's max_model_len may be lower


def _detect_unclosed_think(raw_text: str) -> bool:
    """Kept for backward compat; the full TruncationDetector is used in run_correction_one_item."""
    if not raw_text:
        return False
    n_open = len(_THINK_OPEN_RE.findall(raw_text))
    n_close = len(_THINK_CLOSE_RE.findall(raw_text))
    return n_open > n_close


def _call_one(
    client: LLMClient,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    """Single call, return a diagnostics dict with everything the probe + later runs need."""
    t0 = time.monotonic()
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    resp = client.call(
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    wall_s = time.monotonic() - t0
    usage = resp.usage or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    finish_reason = resp.finish_reason
    truncated = (finish_reason == "length")
    unclosed_think = _detect_unclosed_think(resp.raw_text)
    return {
        "success": resp.success,
        "error": resp.error,
        "raw_text": resp.raw_text,
        "text": resp.text,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "unclosed_think_block": unclosed_think,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "latency_s": round(wall_s, 3),
        "client_latency_s": resp.latency,
        "client_name": resp.client,
        "temperature": temperature,
        "max_tokens_cap": max_tokens,
        "enable_thinking": enable_thinking,
    }


def run_correction_one_item(
    client: LLMClient,
    target: str,
    sub_variant_id: str,
    item: dict[str, Any],
    *,
    max_tokens: int,
    temperature: float = 1.0,
    raw_log_dir: Path | None = None,
    truncation_retries: int = _DEFAULT_TRUNC_RETRIES,
    truncation_max_cap: int = _DEFAULT_TRUNC_MAX_CAP,
) -> dict[str, Any]:
    """Run one correction call for one item with auto-retry on certain-truncation.

    On `is_truncated_certain=True` from the TruncationDetector, max_tokens is doubled
    and the call retried up to `truncation_retries` times (or until the cap hits
    `truncation_max_cap`). Full retry history is preserved in the record.
    """
    sv = get_sub_variant(sub_variant_id)
    user_prompt = format_prompt(
        sub_variant_id=sub_variant_id,
        note=item["note"],
        question=item["question"],
        a0=item.get("A0", ""),
    )
    enable_thinking = get_enable_thinking(target, sub_variant_id)

    attempts: list[dict[str, Any]] = []
    current_cap = max_tokens
    for attempt_idx in range(truncation_retries + 1):
        call_out = _call_one(
            client=client,
            system=SYSTEM_MSG,
            user=user_prompt,
            temperature=temperature,
            max_tokens=current_cap,
            enable_thinking=enable_thinking,
        )
        report = detect_truncation(
            raw_response=call_out["raw_text"],
            text_clean=call_out["text"],
            finish_reason=call_out["finish_reason"],
            usage={
                "completion_tokens": call_out["completion_tokens"],
                "prompt_tokens": call_out["prompt_tokens"],
            },
            max_tokens=current_cap,
            target=target,
            sub_variant=sub_variant_id,
        )
        call_out["truncation_report"] = report.as_dict()
        attempts.append(call_out)

        if not report.is_truncated_certain:
            break
        next_cap = current_cap * 2
        if next_cap > truncation_max_cap or attempt_idx + 1 > truncation_retries:
            break
        current_cap = next_cap

    final = dict(attempts[-1])
    final["retry_attempts"] = len(attempts)
    final["retry_triggered"] = len(attempts) > 1
    if len(attempts) > 1:
        final["retry_history"] = [
            {
                "attempt_idx": i,
                "max_tokens_cap": a["max_tokens_cap"],
                "finish_reason": a["finish_reason"],
                "completion_tokens": a["completion_tokens"],
                "truncation_report": a["truncation_report"],
            }
            for i, a in enumerate(attempts[:-1])
        ]

    record = {
        "pilot_item_id": item["pilot_item_id"],
        "patient_id": item["patient_id"],
        "fold": item["fold"],
        "target_model": target,
        "sub_variant_id": sub_variant_id,
        "sub_variant_name": sv.name,
        "note_chars": len(item["note"]),
        "A0": item.get("A0", ""),
        "A0_binary_correct": item.get("A0_binary_correct"),
        "system_prompt": SYSTEM_MSG,
        "user_prompt": user_prompt,
        **final,
    }

    if raw_log_dir is not None:
        raw_log_dir = Path(raw_log_dir)
        raw_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = raw_log_dir / f"{item['pilot_item_id']}.json"
        out_path.write_text(json.dumps(record, indent=2, default=str))
    return record


def run_correction(
    target: str,
    sub_variant_id: str,
    items: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 1.0,
    raw_log_dir: Path | None = None,
    client: LLMClient | None = None,
    progress_every: int = 10,
    truncation_retries: int = _DEFAULT_TRUNC_RETRIES,
    truncation_max_cap: int = _DEFAULT_TRUNC_MAX_CAP,
    truncation_alert_rate: float = 0.05,
) -> list[dict[str, Any]]:
    """Run one sub-variant over items; auto-retry certain-truncation; alert if rate > threshold."""
    if client is None:
        client = make_client(target)
    records: list[dict[str, Any]] = []
    n_retried = 0
    n_certain = 0
    n_likely = 0
    for i, item in enumerate(items):
        rec = run_correction_one_item(
            client=client,
            target=target,
            sub_variant_id=sub_variant_id,
            item=item,
            max_tokens=max_tokens,
            temperature=temperature,
            raw_log_dir=raw_log_dir,
            truncation_retries=truncation_retries,
            truncation_max_cap=truncation_max_cap,
        )
        records.append(rec)
        if rec.get("retry_triggered"):
            n_retried += 1
        tr = rec.get("truncation_report") or {}
        if tr.get("is_truncated_certain"):
            n_certain += 1
        if tr.get("is_truncated_likely"):
            n_likely += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  [{target}/{sub_variant_id}] {i+1}/{len(items)} "
                f"(last ct={rec['completion_tokens']} fr={rec['finish_reason']} "
                f"retries={rec.get('retry_attempts', 1) - 1} "
                f"trunc_certain={n_certain}/{i+1})"
            )
    n = len(records) or 1
    print(
        f"  [{target}/{sub_variant_id}] DONE n={n}  "
        f"retried={n_retried}  certain_after_retry={n_certain} ({100*n_certain/n:.1f}%)  "
        f"likely={n_likely} ({100*n_likely/n:.1f}%)"
    )
    if n_certain / n > truncation_alert_rate:
        print(
            f"  [{target}/{sub_variant_id}] ⚠ TRUNCATION ALERT: certain-truncation rate "
            f"{100*n_certain/n:.1f}% exceeds threshold {100*truncation_alert_rate:.1f}% "
            f"\u2014 review raw outputs + increase max_tokens cap."
        )
    return records
