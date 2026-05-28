"""MLX judge wrapper: binary (0/1) verdict via Qwen3.5-27B-6bit on Mac Studio.

Usage:
    from ichl.prompt_engineering.mlx_judge.mlx_judge import MLXJudge
    judge = MLXJudge()
    out = judge.judge(note=..., question=..., ground_truth=..., model_answer=...,
                      prompt_version='V0', icl_examples=None)
    # → {"label": 0|1|None, "raw": "...", "latency_s": ..., "prompt_tokens": ..., "completion_tokens": ...}
"""
from __future__ import annotations

import re
from typing import Sequence

from ichl.clients.factory import make_client
from ichl.prompt_engineering.mlx_judge.prompts import (
    JUDGE_SYSTEM,
    build_user_message,
)


_FIRST_DIGIT_RE = re.compile(r"[01]")


def parse_binary(text: str) -> int | None:
    """Parse the MLX output for 0 or 1. Return None if uncertain.

    Rules (in order):
        1. Try to match trailing VERDICT: N (for consistency with few-shot style).
        2. Look at the first line; take first 0/1 there.
        3. Otherwise, the first 0/1 occurring in the text.
        4. Return None if neither appears.
    """
    if not text:
        return None
    t = text.strip()
    # Trailing "VERDICT: X"
    m = re.search(r"VERDICT\s*[:=]\s*([01])", t, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # First line
    first_line = t.splitlines()[0].strip() if t else ""
    m = _FIRST_DIGIT_RE.search(first_line)
    if m:
        return int(m.group(0))
    # Fallback: first 0/1 anywhere
    m = _FIRST_DIGIT_RE.search(t)
    if m:
        return int(m.group(0))
    return None


class MLXJudge:
    def __init__(self, client_name: str = "mlx-qwen35", max_tokens: int = 16):
        self.client = make_client(client_name)
        self.max_tokens = max_tokens

    def judge(
        self,
        note: str,
        question: str,
        ground_truth: str,
        model_answer: str,
        prompt_version: str = "V0",
        icl_examples: Sequence[dict] | None = None,
    ) -> dict:
        user_msg = build_user_message(
            note=note,
            question=question,
            ground_truth=ground_truth,
            model_answer=model_answer,
            icl_examples=icl_examples,
        )
        resp = self.client.call(
            system=JUDGE_SYSTEM,
            user=user_msg,
            temperature=0.0,
            max_tokens=self.max_tokens,
            enable_thinking=False,
        )
        usage = resp.usage or {}
        raw = resp.text or resp.raw_text or ""
        label = parse_binary(raw) if resp.success else None
        return {
            "label": label,
            "raw": raw,
            "latency_s": resp.latency if resp.latency is not None else -1.0,
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "success": resp.success,
            "error": resp.error,
            "prompt_version": prompt_version,
        }
