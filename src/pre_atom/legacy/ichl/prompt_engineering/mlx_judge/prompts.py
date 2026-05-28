"""Prompt templates for the MLX binary judge.

Mirrors the GPT-4o Stage-1 binary judge semantics from
src/ichl/prompt_engineering/correction/pilot_round0.py::judge_binary.

Variants:
    V0: zero-shot, same structure as GPT-4o Stage-1.
    V4: 4-shot (2 right + 2 wrong) prepended.
    V8: 8-shot (4 right + 4 wrong) prepended.

For ICL examples, the note is truncated (default 800 chars) to keep input
manageable; the current-query note stays full-length.
"""
from __future__ import annotations

from typing import Sequence

JUDGE_SYSTEM = (
    "You are a medical expert evaluating an AI model's answer to a clinical question."
)

# The canonical GPT-4o Stage-1 user template, used for both the ICL examples and
# the query itself.
QUERY_TEMPLATE = (
    "DISCHARGE SUMMARY:\n{note}\n\n"
    "QUESTION:\n{question}\n\n"
    "CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
    "MODEL'S ANSWER:\n{model_answer}\n\n"
    "Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
    "Respond with ONLY a single digit:\n"
    "1 = Correct\n"
    "0 = Incorrect"
)

ICL_INTRO = (
    "Below are example judgments demonstrating how to evaluate answers. Each example ends with "
    "the correct verdict (0 or 1). After the examples, you will be given a new case to judge."
)


def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "\n[...truncated]"


def format_example(ex: dict, note_trunc: int = 800) -> str:
    """Format a single ICL example (example case + its gold verdict)."""
    note = _truncate(ex.get("note", "") or "", note_trunc)
    body = QUERY_TEMPLATE.format(
        note=note,
        question=ex["question"],
        ground_truth=ex["ground_truth"],
        model_answer=ex["model_answer"],
    )
    verdict = int(ex["binary_correct"])
    return f"--- EXAMPLE ---\n{body}\nVERDICT: {verdict}\n--- END EXAMPLE ---"


def build_user_message(
    note: str,
    question: str,
    ground_truth: str,
    model_answer: str,
    icl_examples: Sequence[dict] | None = None,
    note_trunc_examples: int = 800,
) -> str:
    """Assemble the user message. ICL examples prepended if provided."""
    query = QUERY_TEMPLATE.format(
        note=note,
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )
    if not icl_examples:
        return query
    blocks = [ICL_INTRO]
    for ex in icl_examples:
        blocks.append(format_example(ex, note_trunc=note_trunc_examples))
    blocks.append("--- NEW CASE TO JUDGE ---")
    blocks.append(query)
    return "\n\n".join(blocks)
