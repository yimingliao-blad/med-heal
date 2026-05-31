#!/usr/bin/env python3
"""
Module 5 — Full audit / persistence module.

Single source of truth for V2 pipeline runs. Every item the pipeline touches
produces ONE JSON line in the audit log containing every input, every raw
output, every parsed value, every vote distribution, and the final outcome.

This is what was missing from v1 (run_fullscale.py only persisted the final
flags). The audit script `audit_view.py` reads this log instead of re-running
the pipeline.

The schema is intentionally permissive — any stage may be missing if the
pipeline branched away from it (e.g. detection said CORRECT → no correction
record). Missing stages must be present as `null`, not absent, so a future
reader can distinguish "this stage was not invoked" from "this stage failed".
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


# A single global lock for append-safe writes from multiple coroutines/threads.
_lock = threading.Lock()


_REQUIRED_TOP_KEYS = [
    "fold", "idx", "item",
    "judge_orig",
    "detection",
    "correction",
    "verdict",
    "judge_corrected",
    "outcome",
]


def _empty_record(fold: int, idx: int) -> dict[str, Any]:
    return {
        "fold": fold,
        "idx": idx,
        "item": None,
        "judge_orig": None,
        "detection": None,
        "correction": None,
        "verdict": None,
        "judge_corrected": None,
        "outcome": None,
    }


class AuditLog:
    """JSON-Lines audit log. One record per (fold, idx). Append-only."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[int, int], dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (int(obj["fold"]), int(obj["idx"]))
                self._cache[key] = obj

    def has(self, fold: int, idx: int) -> bool:
        return (int(fold), int(idx)) in self._cache

    def get(self, fold: int, idx: int) -> dict[str, Any] | None:
        return self._cache.get((int(fold), int(idx)))

    def all(self) -> list[dict[str, Any]]:
        return list(self._cache.values())

    def write(self, record: dict[str, Any]) -> None:
        for k in _REQUIRED_TOP_KEYS:
            if k not in record:
                raise ValueError(f"audit record missing required key: {k}")
        key = (int(record["fold"]), int(record["idx"]))
        with _lock:
            self._cache[key] = record
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w") as f:
                for r in self._cache.values():
                    f.write(json.dumps(r, default=str) + "\n")
            os.replace(tmp, self.path)


def make_record(fold: int, idx: int, item: dict | None = None) -> dict[str, Any]:
    """Convenience: empty record with optional item field already filled."""
    rec = _empty_record(fold, idx)
    if item is not None:
        rec["item"] = item
    return rec


# ---------- Self-test ----------

if __name__ == "__main__":
    p = Path("/tmp/_audit_log_test.jsonl")
    if p.exists():
        p.unlink()
    log = AuditLog(p)
    rec = make_record(0, 51, {"question": "demo"})
    rec["judge_orig"] = {"label": 0, "raw": "0"}
    rec["detection"] = {"variant": "J3", "majority": {"verdict": "INCORRECT"}}
    rec["correction"] = {"action": "regenerated"}
    rec["verdict"] = {"accept": True}
    rec["judge_corrected"] = {"label": 1, "raw": "1"}
    rec["outcome"] = {"action": "corrected", "delta": 1}
    log.write(rec)

    # Reload and verify
    log2 = AuditLog(p)
    assert log2.has(0, 51)
    print("audit_log self-test OK")
    print(json.dumps(log2.get(0, 51), indent=2))
