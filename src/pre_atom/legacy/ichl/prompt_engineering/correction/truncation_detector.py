"""Truncation detector for correction-model outputs.

Runs multiple heuristics and returns a TruncationReport with per-signal flags
plus `is_truncated_certain` (high-confidence) and `is_truncated_likely` (adds
soft signals).

Certain signals — wired into the runner as retry triggers:
  1. `api_length`      — `finish_reason == "length"` (definitive).
  2. `near_ceiling`    — `completion_tokens >= 0.98 * max_tokens`.
  3. `unclosed_think`  — model-specific:
        - DS: no `</think>` in raw at all (DS never emits the opening tag, so
          absence of the closing tag means the reasoning block was cut off).
        - Qwen3 think-ON: `count("<think>") > count("</think>")`.
        - Others: N/A.

Likely-but-not-certain signals — logged for audit, not retried:
  4. `incomplete_ending` — text_stripped does not end with terminal punctuation
        AND is longer than 50 chars. Short answers (< 50 chars) exempt.
  5. `dangling_connective` — text ends with a connective (and / or / the / to / …).
  6. `missing_markers` — optional sub-variant-specific expected markers absent.

Usage:
    report = detect_truncation(
        raw_response=resp.raw_text, text_clean=resp.text,
        finish_reason=resp.finish_reason, usage=resp.usage,
        max_tokens=mt, target="deepseek-r1-distill-llama-8b", sub_variant="c",
    )
    if report.is_truncated_certain:
        # retry with 2x max_tokens
        ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TruncationReport:
    is_truncated_certain: bool
    is_truncated_likely: bool
    signals: dict[str, bool]
    notes: list[str] = field(default_factory=list)

    def fired_signals(self) -> list[str]:
        return [name for name, fired in self.signals.items() if fired]

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_truncated_certain": self.is_truncated_certain,
            "is_truncated_likely": self.is_truncated_likely,
            "signals": self.signals,
            "notes": self.notes,
        }


_TERMINAL_PUNCT_RE = re.compile(r"[.!?][\"')\]]*\s*$")
# List-format endings are acceptable "complete" terminations: last line starts with
# a numbered / bulleted marker, or the whole last line is a short medical-list item.
_LIST_ITEM_LAST_LINE_RE = re.compile(
    r"(?m)(^|\n)\s*(?:[\-\*\u2022]|\d+[.)])\s+[^\n]{1,120}\s*$",
)
_DANGLING_CONNECTIVE_RE = re.compile(
    r"\b(and|or|but|the|a|an|of|to|in|at|on|with|by|for|from|that|which|because|if|when|while|since|as|into|onto|upon)\s*$",
    re.IGNORECASE,
)

_SHORT_ANSWER_CHAR_THRESHOLD = 50
_NEAR_CEILING_RATIO = 0.98


def detect_truncation(
    *,
    raw_response: str,
    text_clean: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    target: str = "",
    sub_variant: str = "",
    expected_markers: list[str] | None = None,
) -> TruncationReport:
    """Apply all truncation heuristics to a model response."""
    signals: dict[str, bool] = {}
    notes: list[str] = []

    text = text_clean if text_clean is not None else raw_response
    text_stripped = (text or "").rstrip()

    # ── Certain signals ───────────────────────────────
    signals["api_length"] = finish_reason == "length"
    if signals["api_length"]:
        notes.append("finish_reason=length")

    completion_tokens = (usage or {}).get("completion_tokens") or 0
    if max_tokens and completion_tokens:
        ratio = completion_tokens / max_tokens
        signals["near_ceiling"] = ratio >= _NEAR_CEILING_RATIO
        if signals["near_ceiling"]:
            notes.append(f"completion_tokens={completion_tokens}/{max_tokens} ratio={ratio:.3f}")
    else:
        signals["near_ceiling"] = False

    target_l = target.lower()
    sv_l = sub_variant.lower()
    if "deepseek" in target_l:
        signals["unclosed_think"] = "</think>" not in raw_response
        if signals["unclosed_think"]:
            notes.append("DS: no </think> in raw — think block cut off")
    elif "qwen3" in target_l and "think" in sv_l and "nothink" not in sv_l:
        n_open = raw_response.count("<think>")
        n_close = raw_response.count("</think>")
        signals["unclosed_think"] = n_open > n_close
        if signals["unclosed_think"]:
            notes.append(f"Qwen3-think: n_open={n_open} n_close={n_close}")
    else:
        signals["unclosed_think"] = False

    # ── Likely (soft) signals ─────────────────────────
    if text_stripped and len(text_stripped) >= _SHORT_ANSWER_CHAR_THRESHOLD:
        has_terminal = bool(_TERMINAL_PUNCT_RE.search(text_stripped))
        # List-format endings count as complete (medical answers often end on a list item)
        has_list_ending = bool(_LIST_ITEM_LAST_LINE_RE.search(text_stripped))
        signals["incomplete_ending"] = not (has_terminal or has_list_ending)
        if signals["incomplete_ending"]:
            notes.append(f"no terminal punct or list ending: ...{text_stripped[-80:]!r}")
        signals["dangling_connective"] = bool(_DANGLING_CONNECTIVE_RE.search(text_stripped))
        if signals["dangling_connective"]:
            notes.append(f"dangling connective: ...{text_stripped[-60:]!r}")
    else:
        signals["incomplete_ending"] = False
        signals["dangling_connective"] = False
        if not text_stripped:
            signals["incomplete_ending"] = True
            notes.append("empty text")

    if expected_markers and text:
        missing = [m for m in expected_markers if m not in text]
        signals["missing_markers"] = len(missing) > 0
        if missing:
            notes.append(f"missing expected markers: {missing}")
    else:
        signals["missing_markers"] = False

    # ── Aggregate ─────────────────────────────────────
    certain_keys = ("api_length", "near_ceiling", "unclosed_think")
    likely_extra = ("incomplete_ending", "dangling_connective", "missing_markers")
    is_certain = any(signals[k] for k in certain_keys)
    is_likely = is_certain or any(signals[k] for k in likely_extra)

    return TruncationReport(
        is_truncated_certain=is_certain,
        is_truncated_likely=is_likely,
        signals=signals,
        notes=notes,
    )
