"""Evaluator: run one (candidate × target) cell of the pilot matrix.

A "cell" = one candidate prompt, one target model, the full pilot dataset.
The runner orchestrates cells; this module runs a single cell end-to-end:

  1. For each pilot item, format the prompt template.
  2. Call the target model via its vLLM client.
  3. Save the raw response to `log_dir/<pilot_item_id>.json`.
  4. Run every parser in `parsers` on the response.
  5. Pick a chosen verdict by `chosen_parser_order` (default: regex → llm).
  6. Record whether the chosen verdict matches `binary_correct`.
  7. Append per-item rows to `per_item_path` (JSONL, flush after every write).
  8. Call the accuracy metric to get the scalar score.

The caller is expected to pre-build the parser list (so the LLM parser's
MLX client is reused across a full run). See `run_detection_pilot.py`.

Contract matches the Notion 'Claude: Plan: Detection — Pilot Runner Design'
§ Runner architecture and § Per-item result schema.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ichl.clients.base import LLMClient
from ichl.clients.factory import make_client
from ichl.prompt_engineering.metrics.accuracy import AccuracySummary, summarize
from ichl.prompt_engineering.metrics.base import get_metric  # re-exported for callers
from ichl.prompt_engineering.parsers.base import Parser


SYSTEM_MSG = (
    "You are a medical expert checking an AI model's answer for factual "
    "correctness against discharge notes."
)


@dataclass
class CellResult:
    """Result of evaluating one (candidate × target) cell."""
    candidate_name: str
    target_model: str
    metric_name: str
    score: float                              # primary metric value (accuracy)
    summary: AccuracySummary                  # expanded per-cell stats
    per_item: list[dict[str, Any]] = field(default_factory=list)
    n_context_too_long: int = 0
    n_call_failed: int = 0
    raw_output_dir: Path | None = None
    per_item_path: Path | None = None


@dataclass
class EvaluationResult:
    """Back-compat wrapper retained for the optimizer.

    New code should use CellResult directly.
    """
    candidate_name: str
    metric_name: str
    score: float
    per_item: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


# ─────────────────────── formatting ───────────────────────

def _format_prompt(template: str, item: dict[str, Any]) -> str:
    """Fill prompt template variables. Unknown vars raise KeyError.

    Template vars supported: {note}, {question}, {model_answer}, plus optional
    {choices} (falls back to empty string if template doesn't reference it).
    """
    # format_map is tolerant: it only errors on referenced keys.
    values = {
        "note": item.get("note", ""),
        "question": item.get("question", ""),
        "model_answer": item.get("model_answer", ""),
        "answer": item.get("model_answer", ""),      # legacy alias
        "choices": item.get("choices", ""),
    }
    return template.format_map(values)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _save_raw(dir_: Path, item_id: str, payload: dict[str, Any]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{item_id}.json").write_text(json.dumps(payload, default=str, indent=2))


# ─────────────────────── the one-cell runner ───────────────────────

def evaluate_cell(
    *,
    candidate_name: str,
    prompt_template: str,
    pilot_data: list[dict[str, Any]],
    target_client: LLMClient | str,
    parsers: list[Parser],
    max_tokens: int,
    temperature: float = 0.0,
    enable_thinking: bool | None = None,
    chosen_parser_order: tuple[str, ...] = ("regex", "llm"),
    log_dir: Path | None = None,
    per_item_path: Path | None = None,
) -> CellResult:
    """Run one cell end-to-end.

    Args:
        candidate_name:  e.g. 'A1_minimal_neutral'
        prompt_template: text with {note}/{question}/{model_answer} placeholders
        pilot_data:      list of PilotItem dicts from the loader
        target_client:   pre-built LLMClient OR a config name (will be lazily made)
        parsers:         iterable of Parser instances (runner pre-builds so the
                         MLX client is reused across items)
        max_tokens:      from Step 0 (per target model)
        temperature:     0.0 for deterministic detection
        enable_thinking: passed through to vLLM client for Qwen3-style servers;
                         None means "use client's default"
        chosen_parser_order:
                         parser names to try in order; first non-UNKNOWN wins
        log_dir:         directory for raw_outputs/; may be None for tests
        per_item_path:   JSONL path for per-item rows; may be None for tests

    Returns:
        CellResult with score, summary, per_item, and error counts.
    """
    if isinstance(target_client, str):
        target_client = make_client(target_client)
    target_name = getattr(target_client, "name", "<unknown>")

    if not parsers:
        raise ValueError("Need at least one parser")
    parsers_by_name = {p.name: p for p in parsers}

    per_item: list[dict[str, Any]] = []
    n_ctx = 0
    n_fail = 0

    for item in pilot_data:
        pid_id = item["pilot_item_id"]
        user_prompt = _format_prompt(prompt_template, item)

        t0 = time.monotonic()
        resp = target_client.call(
            system=SYSTEM_MSG,
            user=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        call_lat = time.monotonic() - t0

        # Save raw output for this item (every call, success or failure).
        if log_dir is not None:
            _save_raw(log_dir, pid_id, {
                "pilot_item_id": pid_id,
                "candidate": candidate_name,
                "target_model": target_name,
                "user_prompt": user_prompt,
                "system_prompt": SYSTEM_MSG,
                "raw_response": resp.raw_text,
                "text_clean": resp.text,
                "latency_s": resp.latency,
                "finish_reason": resp.finish_reason,
                "usage": resp.usage,
                "success": resp.success,
                "error": resp.error,
            })

        if not resp.success:
            if resp.error == "context_too_long":
                n_ctx += 1
            else:
                n_fail += 1
            row = _no_verdict_row(item, candidate_name, target_name, resp, parsers)
            per_item.append(row)
            if per_item_path:
                _append_jsonl(per_item_path, row)
            continue

        # Run every parser on the cleaned response text.
        parser_results: dict[str, Any] = {}
        for p in parsers:
            pr = p.parse(resp.text)
            parser_results[p.name] = {
                "verdict": pr.verdict, "match_text": pr.match_text,
                "match_pos": pr.match_pos, "latency_s": pr.latency_s,
                "raw_response": pr.raw_response, "notes": pr.notes,
                "extra": pr.extra,
            }

        # Pick the chosen verdict per audit-rule (2026-04-22, Iteration Process
        # Audit, Issue A): on disagreement the LLM parser is right 80 % of the
        # time (vs regex 20 %), so prefer LLM on disagreement. Policy:
        #   1. If parsers agree and are non-UNKNOWN → use that verdict (regex wins tie)
        #   2. If they disagree and LLM is non-UNKNOWN → use LLM
        #   3. If LLM is UNKNOWN but regex is non-UNKNOWN → use regex
        #   4. Otherwise → UNKNOWN
        regex_v = parser_results.get("regex", {}).get("verdict", "UNKNOWN")
        llm_v = parser_results.get("llm", {}).get("verdict", "UNKNOWN")
        if regex_v == llm_v and regex_v != "UNKNOWN":
            chosen_parser, chosen_verdict = "regex", regex_v
        elif llm_v != "UNKNOWN" and regex_v != llm_v:
            chosen_parser, chosen_verdict = "llm", llm_v
        elif regex_v != "UNKNOWN":
            chosen_parser, chosen_verdict = "regex", regex_v
        else:
            chosen_parser, chosen_verdict = "llm", "UNKNOWN"

        # Agreement: only defined if both named parsers actually ran.
        regex_v = parser_results.get("regex", {}).get("verdict", "UNKNOWN")
        llm_v = parser_results.get("llm", {}).get("verdict", "UNKNOWN")
        agree = (regex_v == llm_v)

        expected = "CORRECT" if item["binary_correct"] == 1 else "INCORRECT"
        verdict_correct = 1 if chosen_verdict == expected else 0

        row = {
            "pilot_item_id": pid_id,
            "patient_id": item["patient_id"],
            "fold": item["fold"],
            "candidate": candidate_name,
            "target_model": target_name,
            "raw_response": resp.raw_text,
            "text_clean": resp.text,
            "response_latency_ms": round(resp.latency * 1000, 1),
            "call_latency_ms": round(call_lat * 1000, 1),
            "finish_reason": resp.finish_reason,
            "usage": resp.usage,
            "regex_verdict": regex_v,
            "regex_match_pos": parser_results.get("regex", {}).get("match_pos", -1),
            "regex_match_text": parser_results.get("regex", {}).get("match_text", ""),
            "llm_verdict": llm_v,
            "llm_latency_ms": round(parser_results.get("llm", {}).get("latency_s", 0.0) * 1000, 1),
            "llm_raw": parser_results.get("llm", {}).get("raw_response", ""),
            "agree": agree,
            "chosen_parser": chosen_parser,
            "chosen_verdict": chosen_verdict,
            "binary_correct": item["binary_correct"],
            "verdict_correct": verdict_correct,
        }
        per_item.append(row)
        if per_item_path:
            _append_jsonl(per_item_path, row)

    summary = summarize(per_item)
    score = summary.accuracy

    return CellResult(
        candidate_name=candidate_name,
        target_model=target_name,
        metric_name="accuracy",
        score=score,
        summary=summary,
        per_item=per_item,
        n_context_too_long=n_ctx,
        n_call_failed=n_fail,
        raw_output_dir=log_dir,
        per_item_path=per_item_path,
    )


def _no_verdict_row(
    item: dict[str, Any], candidate_name: str, target_name: str,
    resp: Any, parsers: list[Parser],
) -> dict[str, Any]:
    """Build a UNKNOWN-verdict row for failed calls (context overflow, HTTP errors)."""
    return {
        "pilot_item_id": item["pilot_item_id"],
        "patient_id": item["patient_id"],
        "fold": item["fold"],
        "candidate": candidate_name,
        "target_model": target_name,
        "raw_response": resp.raw_text,
        "text_clean": resp.text,
        "response_latency_ms": round(max(resp.latency, 0.0) * 1000, 1),
        "call_latency_ms": 0.0,
        "finish_reason": resp.finish_reason,
        "usage": resp.usage,
        "regex_verdict": "UNKNOWN",
        "regex_match_pos": -1,
        "regex_match_text": "",
        "llm_verdict": "UNKNOWN",
        "llm_latency_ms": 0.0,
        "llm_raw": "",
        "agree": True,
        "chosen_parser": parsers[0].name,
        "chosen_verdict": "UNKNOWN",
        "binary_correct": item["binary_correct"],
        "verdict_correct": 0,
        "call_error": resp.error,
    }


# ─────────────────────── back-compat shim for optimizer.py ───────────────────────

def evaluate(
    candidate_name: str,
    prompt_template: str,                          # noqa: ARG001
    pilot_data: list[dict[str, Any]],              # noqa: ARG001
    metric: str,
    target_client_name: str,                       # noqa: ARG001
    log_dir: Path | None = None,                   # noqa: ARG001
    **metric_kwargs: Any,
) -> EvaluationResult:
    """Legacy shim: kept so the existing `optimizer.py` still imports.

    New code should call `evaluate_cell` directly. The optimizer will be
    upgraded to this interface in a follow-up pass.
    """
    metric_fn = get_metric(metric)
    score = metric_fn(
        prompt_template=prompt_template,
        pilot_data=pilot_data,
        target_client_name=target_client_name,
        log_dir=log_dir,
        **metric_kwargs,
    )
    return EvaluationResult(
        candidate_name=candidate_name, metric_name=metric, score=float(score),
    )
