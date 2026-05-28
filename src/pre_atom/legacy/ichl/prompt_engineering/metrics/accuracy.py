"""Accuracy metric against `binary_correct`.

For each pilot item, compare the detector's chosen verdict to the ground
truth. The metric itself returns a single float (accuracy), but it also
populates `per_item` + per-cell class-precision/recall via a side-channel
dict on the evaluator, so the Finding page can print the full table.

Verdict → expected-label mapping:
    binary_correct == 1  → expected = 'CORRECT'
    binary_correct == 0  → expected = 'INCORRECT'

'UNKNOWN' counts as incorrect (neither class predicted).

Because pilot data is stratified 50/50 (20 correct + 20 incorrect out of 40),
accuracy is directly comparable across candidates and targets — unlike the
natural 89% correct distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ichl.prompt_engineering.metrics.base import register_metric


@dataclass
class AccuracySummary:
    """Per-cell accuracy report (logged by the runner for the Finding page)."""
    accuracy: float
    n: int
    n_correct_class: int                 # items where binary_correct == 1
    n_incorrect_class: int               # items where binary_correct == 0
    # Per-class precision/recall (1 = positive class per class).
    precision_correct: float
    recall_correct: float
    precision_incorrect: float
    recall_incorrect: float
    n_unknown: int                       # items where chosen_verdict == 'UNKNOWN'
    n_parser_disagree: int               # items where regex_verdict != llm_verdict
    per_item: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "n": self.n,
            "n_correct_class": self.n_correct_class,
            "n_incorrect_class": self.n_incorrect_class,
            "precision_correct": self.precision_correct,
            "recall_correct": self.recall_correct,
            "precision_incorrect": self.precision_incorrect,
            "recall_incorrect": self.recall_incorrect,
            "n_unknown": self.n_unknown,
            "n_parser_disagree": self.n_parser_disagree,
        }


def _precision_recall(
    per_item: list[dict[str, Any]],
    positive_verdict: str,
    positive_label: int,
) -> tuple[float, float]:
    """Return (precision, recall) for a given verdict treated as positive."""
    tp = sum(1 for r in per_item
             if r["chosen_verdict"] == positive_verdict and r["binary_correct"] == positive_label)
    fp = sum(1 for r in per_item
             if r["chosen_verdict"] == positive_verdict and r["binary_correct"] != positive_label)
    fn = sum(1 for r in per_item
             if r["chosen_verdict"] != positive_verdict and r["binary_correct"] == positive_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def summarize(per_item: list[dict[str, Any]]) -> AccuracySummary:
    """Build an AccuracySummary from the runner's per-item rows.

    Each row must have keys: 'chosen_verdict' ('CORRECT'|'INCORRECT'|'UNKNOWN'),
    'binary_correct' (0 or 1), optional 'agree' (bool).
    """
    n = len(per_item)
    if n == 0:
        return AccuracySummary(
            accuracy=0.0, n=0, n_correct_class=0, n_incorrect_class=0,
            precision_correct=0.0, recall_correct=0.0,
            precision_incorrect=0.0, recall_incorrect=0.0,
            n_unknown=0, n_parser_disagree=0,
        )

    n_correct_class = sum(1 for r in per_item if r["binary_correct"] == 1)
    n_incorrect_class = sum(1 for r in per_item if r["binary_correct"] == 0)
    n_unknown = sum(1 for r in per_item if r["chosen_verdict"] == "UNKNOWN")
    n_parser_disagree = sum(1 for r in per_item if not r.get("agree", True))

    n_hit = sum(
        1 for r in per_item
        if (r["chosen_verdict"] == "CORRECT" and r["binary_correct"] == 1)
        or (r["chosen_verdict"] == "INCORRECT" and r["binary_correct"] == 0)
    )
    accuracy = n_hit / n

    p_correct, r_correct = _precision_recall(per_item, "CORRECT", 1)
    p_incorrect, r_incorrect = _precision_recall(per_item, "INCORRECT", 0)

    return AccuracySummary(
        accuracy=accuracy, n=n,
        n_correct_class=n_correct_class,
        n_incorrect_class=n_incorrect_class,
        precision_correct=p_correct, recall_correct=r_correct,
        precision_incorrect=p_incorrect, recall_incorrect=r_incorrect,
        n_unknown=n_unknown,
        n_parser_disagree=n_parser_disagree,
        per_item=per_item,
    )


@register_metric("accuracy")
def accuracy_metric(
    prompt_template: str,                  # noqa: ARG001
    pilot_data: list[dict[str, Any]],      # noqa: ARG001
    target_client_name: str,               # noqa: ARG001
    *,
    per_item: list[dict[str, Any]] | None = None,
    **kwargs: Any,                         # noqa: ARG001
) -> float:
    """Thin metric entry-point.

    The runner builds `per_item` (with parser verdicts), then calls this
    with `per_item=...` and gets back a float accuracy. The full
    AccuracySummary is available via `summarize(per_item)` when the runner
    wants the expanded per-cell stats for the Finding page.

    Signature matches the `MetricFn` registry contract; the prompt_template
    / pilot_data / target_client_name args are unused here because the
    runner does the heavy lifting and passes in `per_item` directly.
    """
    if per_item is None:
        return 0.0
    summary = summarize(per_item)
    return summary.accuracy
