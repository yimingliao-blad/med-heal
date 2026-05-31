"""Metric registry. Each step registers its own metric (e.g., 'selectivity').

See Notion "Claude: Module: Prompt Engineering Iteration" — Conventions.
"""
from ichl.prompt_engineering.metrics.base import Metric, register_metric, get_metric

__all__ = ["Metric", "register_metric", "get_metric"]
