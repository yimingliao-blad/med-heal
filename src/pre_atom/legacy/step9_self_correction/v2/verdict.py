#!/usr/bin/env python3
"""
Module 4 — Verdict gate.

Decide whether a candidate correction should replace the original answer.
Per-user direction: strictness is *not* a fixed knob; it's chosen after
Module 4a measures verdict-variant quality on a labeled set. Until then,
we ship V1f (free-form contradiction count, multi-sample with conservative
tie-break: ties → keep original).

Variants implemented:
  v1f  free-form pairwise contradiction count + Qwen3-32B JSON extract of pick
  v1j  direct-JSON pairwise contradiction count

Both blind A/B placement using a deterministic seed so the same item gets
the same placement across runs.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from typing import Callable

from detection_format_bakeoff import (
    SYS_DET, build_chatml, vllm_gen, q32_extract,
    _parse_inline_json,
)

# ---------- Prompts ----------

V1F_PROMPT = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count how many factual claims in each answer CONTRADICT the discharge notes.

Answer A contradictions: <number>
Answer B contradictions: <number>
Better answer: A or B"""

V1J_PROMPT = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count how many factual claims in each answer CONTRADICT the discharge notes,
then pick the answer with fewer contradictions.

Output ONLY a JSON object:
{{"a_contradictions": <integer>, "b_contradictions": <integer>, "pick": "A" or "B"}}"""

EXTRACT_VERDICT = """/nothink
Which answer was picked?

TEXT:
{raw}

{{"pick": "A" or "B"}}"""

VERDICT_SYS = "You are a medical expert comparing answers."


# ---------- Vote helpers ----------

def _seed_for(fold: int, idx: int) -> int:
    return 42 + (fold << 16) + idx


def _ab_placement(fold: int, idx: int) -> bool:
    """Returns True if the original answer goes in slot A. Deterministic."""
    rng = random.Random(_seed_for(fold, idx))
    return rng.random() > 0.5


def _vote(picks: list[str | None]) -> tuple[str, dict, float]:
    """Return (majority_pick, distribution, unanimity).
    Majority is one of 'A','B','TIE','UNCLEAR'. Ties between A and B → 'TIE'."""
    valid = [p for p in picks if p in ("A", "B")]
    if not valid:
        return "UNCLEAR", {}, 0.0
    counts = Counter(valid)
    if len(counts) == 2 and counts["A"] == counts["B"]:
        return "TIE", dict(counts), 0.5
    top, n = counts.most_common(1)[0]
    return top, dict(counts), n / len(valid)


# ---------- Variant runners ----------

def _v1f_call(prompt: str, port: int) -> tuple[str | None, str]:
    raw = vllm_gen(prompt, port, max_tokens=512, temperature=0.7)
    obj = q32_extract(raw, EXTRACT_VERDICT) or {}
    pick = str(obj.get("pick", "")).upper().strip() or None
    if pick not in ("A", "B"):
        pick = None
    return pick, raw


def _v1j_call(prompt: str, port: int) -> tuple[str | None, str]:
    raw = vllm_gen(prompt, port, max_tokens=300, temperature=0.7)
    obj = _parse_inline_json(raw) or {}
    pick = str(obj.get("pick", "")).upper().strip() or None
    if pick not in ("A", "B"):
        pick = None
    return pick, raw


VARIANTS: dict[str, tuple[str, Callable[[str, int], tuple[str | None, str]]]] = {
    "v1f": (V1F_PROMPT, _v1f_call),
    "v1j": (V1J_PROMPT, _v1j_call),
}


def run_verdict(variant: str, fold: int, idx: int, note: str, question: str,
                original_answer: str, corrected_answer: str, *,
                port: int, k: int = 5) -> dict:
    """Multi-sample verdict; ties go to ORIGINAL.

    Returns:
        {
          variant, orig_in_slot_A, samples: [{pick, raw}],
          vote_distribution, majority_pick, unanimity, accept_correction
        }
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown verdict variant: {variant}")
    template, call_fn = VARIANTS[variant]

    orig_is_a = _ab_placement(fold, idx)
    ans_a = original_answer if orig_is_a else corrected_answer
    ans_b = corrected_answer if orig_is_a else original_answer
    user = template.format(note=note, question=question,
                           answer_a=ans_a[:600], answer_b=ans_b[:600])
    prompt = build_chatml(VERDICT_SYS, user)

    samples = []
    for _ in range(k):
        pick, raw = call_fn(prompt, port)
        samples.append({"pick": pick, "raw": raw})

    majority_pick, dist, unanimity = _vote([s["pick"] for s in samples])
    if majority_pick == "TIE" or majority_pick == "UNCLEAR":
        accept = False  # conservative tie-break: keep original
    else:
        # accept if the majority picked the corrected answer's slot
        accept = (majority_pick == "B") if orig_is_a else (majority_pick == "A")

    return {
        "variant": variant,
        "orig_in_slot_A": orig_is_a,
        "samples": samples,
        "vote_distribution": dist,
        "majority_pick": majority_pick,
        "unanimity": unanimity,
        "accept_correction": accept,
    }
