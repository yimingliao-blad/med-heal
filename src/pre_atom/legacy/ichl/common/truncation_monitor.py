"""Truncation monitor — live alerter for LLM output truncation.

Wraps `vllm_call` / `gpt4o_call` (or any LLM call returning a dict with
`truncation_report.is_truncated_certain` + `completion_tokens`).

Per call: log to trunc_log.jsonl + classify severity + alert on STDERR for
CRITICAL events. Aborts run if ≥5 CRITICAL events occur in last 50 calls.

Implements the `/monitor-truncation` skill (see .claude/skills/monitor-truncation/SKILL.md).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRUNC_BASE = ROOT / "output" / "_truncation"


class TruncationMonitor:
    """Per-process singleton monitor. Multiple stages share the same run_id."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, run_id: str | None = None,
                 critical_threshold: int = 5, window_size: int = 50):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self.run_dir = TRUNC_BASE / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.recent_alerts = deque(maxlen=window_size)
        self.counts = {"INFO": 0, "WARN": 0, "CRITICAL": 0}
        self._files = {}  # stage_model → file handle

    @classmethod
    def get(cls, run_id: str | None = None) -> "TruncationMonitor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(run_id=run_id)
            return cls._instance

    def _file_for(self, stage: str, model: str):
        key = f"{stage}__{model}"
        if key not in self._files:
            p = self.run_dir / f"{key}__trunc.jsonl"
            self._files[key] = p.open("a")
        return self._files[key]

    def record(self, *, stage: str, model: str, patient_id, max_tokens: int,
               result: dict, parser_succeeded: bool | None = None):
        """Record one LLM call. result must have `truncation_report` and `text` keys.

        Returns the alert_level. Raises SystemExit if abort threshold exceeded.
        """
        tr = result.get("truncation_report") or {}
        is_trunc = bool(tr.get("is_truncated_certain"))
        finish = tr.get("finish_reason", "?")
        completion = result.get("completion_tokens", 0) or 0
        text = result.get("text", "") or ""
        out_chars = len(text)
        starts = (text[:80] + "...") if len(text) > 80 else text
        ends = ("..." + text[-60:]) if len(text) > 60 else text

        # Classify
        if not is_trunc:
            level = "INFO"
        elif parser_succeeded is False:
            level = "CRITICAL"
        else:
            level = "WARN"

        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "stage": stage, "model": model, "patient_id": patient_id,
            "max_tokens_requested": max_tokens,
            "completion_tokens": completion,
            "finish_reason": finish,
            "is_truncated_certain": is_trunc,
            "output_chars": out_chars,
            "output_starts_with": starts,
            "output_ends_with": ends,
            "parser_succeeded": parser_succeeded,
            "alert_level": level,
        }

        with self._lock:
            self._file_for(stage, model).write(json.dumps(rec) + "\n")
            self._file_for(stage, model).flush()
            self.counts[level] += 1
            self.recent_alerts.append(level)

        # STDERR alert for CRITICAL
        if level == "CRITICAL":
            print(f"[trunc] CRITICAL {model} {stage} pid={patient_id} "
                  f"trunc={is_trunc} chars={out_chars} parser_ok=False",
                  file=sys.stderr, flush=True)

        # Periodic mention
        n_total = sum(self.counts.values())
        if n_total % 25 == 0 and n_total > 0:
            recent_critical = sum(1 for x in self.recent_alerts if x == "CRITICAL")
            if recent_critical >= 2:
                print(f"[trunc] alert: {recent_critical} CRITICAL events in last "
                      f"{len(self.recent_alerts)} items — review max_gen budget",
                      file=sys.stderr, flush=True)

        # Abort gate
        recent_critical = sum(1 for x in self.recent_alerts if x == "CRITICAL")
        if recent_critical >= self.critical_threshold:
            self._write_summary()
            raise SystemExit(
                f"[trunc] ABORT: {recent_critical} CRITICAL events in last "
                f"{len(self.recent_alerts)} items (threshold={self.critical_threshold})"
            )

        return level

    def _write_summary(self):
        with self._lock:
            summary = {
                "run_id": self.run_id,
                "counts": dict(self.counts),
                "total": sum(self.counts.values()),
                "recent_window_alerts": list(self.recent_alerts),
            }
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def close(self):
        self._write_summary()
        for f in self._files.values():
            f.close()


# ============================================================
# Convenience wrappers
# ============================================================

def wrap_call(stage: str, model: str, patient_id, max_tokens: int):
    """Decorator factory: wraps a function returning the standard LLM result dict.

    Usage:
        @wrap_call(stage="detection", model="qwen3-8b", patient_id=pid, max_tokens=8192)
        def my_call():
            return vllm_helper.call(...)
    """
    def deco(fn):
        def inner(*args, **kwargs):
            r = fn(*args, **kwargs)
            mon = TruncationMonitor.get()
            mon.record(stage=stage, model=model, patient_id=patient_id,
                       max_tokens=max_tokens, result=r,
                       parser_succeeded=None)  # caller can update later
            return r
        return inner
    return deco


def record_call(*, stage: str, model: str, patient_id, max_tokens: int,
                result: dict, parser_succeeded: bool | None = None):
    """Functional API: call this directly after each LLM call to log + classify."""
    return TruncationMonitor.get().record(
        stage=stage, model=model, patient_id=patient_id,
        max_tokens=max_tokens, result=result,
        parser_succeeded=parser_succeeded,
    )


def close_monitor():
    """Call at end of run to flush summary."""
    if TruncationMonitor._instance is not None:
        TruncationMonitor._instance.close()


if __name__ == "__main__":
    # Self-test cases (per skill SKILL.md)
    print("--- self-test 1: clean output (INFO) ---")
    mon = TruncationMonitor("test_run_clean")
    r = {"text": "VERDICT: CORRECT", "completion_tokens": 5,
         "truncation_report": {"is_truncated_certain": False, "finish_reason": "stop"}}
    level = mon.record(stage="detection", model="qwen2.5-7b-instruct",
                       patient_id=1, max_tokens=8192, result=r,
                       parser_succeeded=True)
    print(f"  level={level} (expected INFO)")
    assert level == "INFO"

    print("--- self-test 2: truncated but parseable (WARN) ---")
    r = {"text": "VERDICT: INCORRECT\nCLAIM: ...\n[truncated mid-sentence",
         "completion_tokens": 8192,
         "truncation_report": {"is_truncated_certain": True, "finish_reason": "length"}}
    level = mon.record(stage="detection", model="llama-3.1-8b-instruct",
                       patient_id=2, max_tokens=8192, result=r,
                       parser_succeeded=True)
    print(f"  level={level} (expected WARN)")
    assert level == "WARN"

    print("--- self-test 3: truncated AND parser failed (CRITICAL) ---")
    r = {"text": "Let me think about this...",
         "completion_tokens": 8192,
         "truncation_report": {"is_truncated_certain": True, "finish_reason": "length"}}
    level = mon.record(stage="detection", model="deepseek-r1",
                       patient_id=3, max_tokens=8192, result=r,
                       parser_succeeded=False)
    print(f"  level={level} (expected CRITICAL)")
    assert level == "CRITICAL"

    print("--- self-test 4: 5 CRITICAL → ABORT ---")
    abort_mon = TruncationMonitor("test_run_abort", critical_threshold=5, window_size=50)
    try:
        for i in range(6):
            abort_mon.record(stage="x", model="y", patient_id=i, max_tokens=8192,
                             result={"text": "", "completion_tokens": 8192,
                                     "truncation_report": {"is_truncated_certain": True,
                                                            "finish_reason": "length"}},
                             parser_succeeded=False)
        print("  FAIL: should have aborted")
    except SystemExit as e:
        print(f"  level=ABORT (expected)  msg={e}")

    print("\n--- all self-tests passed ---")
    mon.close()
