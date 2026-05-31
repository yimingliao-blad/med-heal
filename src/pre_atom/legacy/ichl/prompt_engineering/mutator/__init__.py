"""Mutator: rule-based (bone) + LLM polish (flesh).

See Notion "Claude: Module: Prompt Engineering Iteration" — Variation strategy.
"""
from ichl.prompt_engineering.mutator.rules import rule_based_mutate
from ichl.prompt_engineering.mutator.llm import llm_polish

__all__ = ["rule_based_mutate", "llm_polish"]
