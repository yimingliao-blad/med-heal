"""Select balanced ICL examples from train_pool.jsonl.

Strategy:
    - (k_right, k_wrong): total examples = k_right + k_wrong.
    - Balance across targets as much as possible: aim for equal counts per target
      within each label.
    - Deterministic given seed.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def load_pool(pool_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(pool_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _bucket(rows: Sequence[dict]) -> dict[tuple[str, int], list[dict]]:
    """Bucket by (target, label)."""
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["target"], int(r["binary_correct"]))
        buckets[key].append(r)
    return buckets


def select_balanced(
    pool: Sequence[dict],
    k_right: int,
    k_wrong: int,
    seed: int = 42,
) -> list[dict]:
    """Return k_right + k_wrong examples, balanced across targets."""
    rng = random.Random(seed)
    buckets = _bucket(pool)
    targets = sorted({r["target"] for r in pool})

    selected: list[dict] = []
    for label, k in [(1, k_right), (0, k_wrong)]:
        # Distribute k across targets as evenly as possible.
        base = k // len(targets)
        rem = k % len(targets)
        # Shuffle targets for fairness of the remainder.
        tshuf = list(targets)
        rng.shuffle(tshuf)
        per_target = {t: base + (1 if i < rem else 0) for i, t in enumerate(tshuf)}
        for t in targets:
            need = per_target[t]
            candidates = list(buckets.get((t, label), []))
            rng.shuffle(candidates)
            picked = candidates[:need]
            if len(picked) < need:
                # Fallback: pull from any target with this label
                extra_pool = [r for r in pool if int(r["binary_correct"]) == label and r not in picked]
                rng.shuffle(extra_pool)
                picked.extend(extra_pool[: need - len(picked)])
            selected.extend(picked)

    # Final shuffle so right/wrong aren't clumped
    rng.shuffle(selected)
    return selected
