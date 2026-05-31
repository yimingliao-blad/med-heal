"""Seed-prompt and variation-primitive loaders.

YAML format for seed prompts (one file per step):
    name: str              # short label
    purpose: str
    template: str          # the actual prompt (may use {variables})
    tags: [str]            # optional labels for grouping
    provenance: str        # e.g., "adapted from P1_minimal (pilot 2026-04)"

YAML format for variation primitives:
    word_level:
      - name: "synonym"
        action: "replace 'contradict' with 'conflict'"
    structural:
      - name: "add_write_own_answer_first"
        action: "prepend: 'First, write your own answer. Then compare.'"
    constraint:
      - name: "loosen_json"
        action: "remove requirement to output strict JSON"

STUB — implementations fill in as we start optimizing each step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SeedPrompt:
    name: str
    template: str
    purpose: str = ""
    tags: list[str] = field(default_factory=list)
    provenance: str = ""


@dataclass
class VariationPrimitive:
    name: str
    kind: str                # 'word_level' | 'structural' | 'constraint'
    action: str              # human-readable description, consumed by LLM polisher
    params: dict[str, Any] = field(default_factory=dict)


def load_seeds(path: Path | str) -> list[SeedPrompt]:
    """Load a seeds YAML (list of seed dicts). Returns list of SeedPrompt."""
    data = yaml.safe_load(Path(path).read_text()) or []
    if isinstance(data, dict):
        data = [data]
    seeds = []
    for i, d in enumerate(data):
        seeds.append(SeedPrompt(
            name=d.get("name", f"seed_{i:02d}"),
            template=d["template"],
            purpose=d.get("purpose", ""),
            tags=list(d.get("tags", []) or []),
            provenance=d.get("provenance", ""),
        ))
    return seeds


def load_variations(path: Path | str) -> list[VariationPrimitive]:
    """Load a variations YAML (dict keyed by kind → list of primitives)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    primitives: list[VariationPrimitive] = []
    for kind, items in data.items():
        for d in items or []:
            primitives.append(VariationPrimitive(
                name=d.get("name", "<unnamed>"),
                kind=kind,
                action=d.get("action", ""),
                params=d.get("params", {}) or {},
            ))
    return primitives
