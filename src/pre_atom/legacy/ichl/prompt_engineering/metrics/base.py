"""Metric ABC + registry.

A Metric is a callable that takes:
  - a candidate prompt template
  - a pilot dataset (list of items with ground truth)
  - a client config (for running the prompt against a target model)
and returns a scalar score (higher = better).

Concrete metric implementations live in sibling files and register themselves:
    from ichl.prompt_engineering.metrics.base import register_metric

    @register_metric("selectivity")
    def selectivity(prompt, pilot_data, target_client, **kwargs):
        # run prompt against each pilot item; compute TPR / FPR
        return tpr / fpr
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


MetricFn = Callable[..., float]

_METRIC_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str):
    def _decorator(fn: MetricFn) -> MetricFn:
        _METRIC_REGISTRY[name] = fn
        return fn
    return _decorator


def get_metric(name: str) -> MetricFn:
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Metric '{name}' not registered. Known: {list(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[name]


class Metric(ABC):
    """Class-based metric (alternative to @register_metric function).

    Implement `__call__` and register the instance's `.name` in the registry.
    """
    name: str = "<base>"

    @abstractmethod
    def __call__(
        self,
        prompt_template: str,
        pilot_data: list[dict[str, Any]],
        target_client_name: str,
        **kwargs: Any,
    ) -> float:
        ...
