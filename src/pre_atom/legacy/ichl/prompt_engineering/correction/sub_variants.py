"""T0 Anchor sub-variant prompt templates.

Five single-prompt regen sub-variants at temperature=1, k=1. System prompt
uniform across all sub-variants:
    "You are a medical expert answering clinical questions from discharge notes."

Template placeholders (substituted by runner.py):
    {note}      — concatenated [Note 1] ... [Note 2] ... block
    {question}  — the clinical question
    {A0}        — the zeroshot answer (only present in 'c' and 'e')

Thinking-mode table per target (for future multi-target use):
    DS        — always thinks (no toggle)
    Qwen3-8B  — per-sub-variant toggle (TBD per Stage III design)
    Others    — N/A (no think mode)

For the DS smoke test, all sub-variants run with the model's default
thinking behaviour (always-on).
"""
from __future__ import annotations

from dataclasses import dataclass


SYSTEM_MSG = "You are a medical expert answering clinical questions from discharge notes."


@dataclass(frozen=True)
class SubVariant:
    id: str                   # 'a'..'e'
    name: str                 # short human-readable name
    template: str             # user-message template with {note}{question}{A0?}
    a0_hidden: bool           # if True, {A0} is NOT shown (A₀ hidden)
    uses_a0: bool             # if True, the template contains {A0}


# --- templates -----------------------------------------------------------
_BARE_REGEN = (
    "Discharge notes:\n{note}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_COT_REGEN = (
    "Discharge notes:\n{note}\n\n"
    "Question: {question}\n\n"
    "Think step by step, then give your final answer."
)

_PACKED_COT = (
    "Discharge notes:\n{note}\n\n"
    "Question: {question}\n\n"
    "Previous answer: {A0}\n\n"
    "The previous answer may have errors. "
    "Step 1: identify any error. "
    "Step 2: find supporting evidence in the notes. "
    "Step 3: give the corrected final answer."
)

_RE_READ = (
    "Discharge notes:\n{note}\n\n"
    "Question: {question}\n\n"
    "Re-read the notes carefully. Find the relevant information, then answer."
)

_CHALLENGE_AND_REGEN = (
    "Discharge notes:\n{note}\n\n"
    "Question: {question}\n\n"
    "Your previous answer was: {A0}\n\n"
    "This answer may be wrong. Identify the problem with it, then give the correct answer."
)


SUB_VARIANTS: dict[str, SubVariant] = {
    "a": SubVariant(
        id="a",
        name="bare_regen",
        template=_BARE_REGEN,
        a0_hidden=True,
        uses_a0=False,
    ),
    "b": SubVariant(
        id="b",
        name="cot_regen",
        template=_COT_REGEN,
        a0_hidden=True,
        uses_a0=False,
    ),
    "c": SubVariant(
        id="c",
        name="packed_cot",
        template=_PACKED_COT,
        a0_hidden=False,
        uses_a0=True,
    ),
    "d": SubVariant(
        id="d",
        name="re_read",
        template=_RE_READ,
        a0_hidden=True,
        uses_a0=False,
    ),
    "e": SubVariant(
        id="e",
        name="challenge_and_regen",
        template=_CHALLENGE_AND_REGEN,
        a0_hidden=False,
        uses_a0=True,
    ),
}


# Thinking-mode map — per-target per-sub-variant.
# DS: always-on; caller doesn't need to toggle (model thinks regardless).
# Qwen3-8B: the detection experiment used enable_thinking=True by default;
#   a per-sub-variant toggle table may be set later in Stage III. For T0 we
#   leave it as the model default (think-on) for now.
# Others: N/A.
THINK_MODE: dict[str, dict[str, bool | None]] = {
    "deepseek-r1-distill-llama-8b": dict.fromkeys(SUB_VARIANTS),   # always thinks
    "qwen3-8b": dict.fromkeys(SUB_VARIANTS, True),                       # think-on default
    "qwen2.5-7b-instruct": dict.fromkeys(SUB_VARIANTS),            # no think mode
    "llama-3.1-8b-instruct": dict.fromkeys(SUB_VARIANTS),          # no think mode
    "biomistral-7b": dict.fromkeys(SUB_VARIANTS),                  # no think mode
}


def get_sub_variant(sub_variant_id: str) -> SubVariant:
    if sub_variant_id not in SUB_VARIANTS:
        raise KeyError(
            f"Unknown sub_variant_id '{sub_variant_id}'. "
            f"Valid: {sorted(SUB_VARIANTS.keys())}"
        )
    return SUB_VARIANTS[sub_variant_id]


def format_prompt(sub_variant_id: str, note: str, question: str, a0: str) -> str:
    """Substitute {note} / {question} / {A0} in the given sub-variant's template."""
    sv = get_sub_variant(sub_variant_id)
    # All templates use {note} and {question}; {A0} only if uses_a0.
    kwargs = {"note": note, "question": question}
    if sv.uses_a0:
        kwargs["A0"] = a0
    return sv.template.format(**kwargs)


def get_enable_thinking(target: str, sub_variant_id: str) -> bool | None:
    """Return per-call enable_thinking override (None means use client default)."""
    return THINK_MODE.get(target, {}).get(sub_variant_id)
