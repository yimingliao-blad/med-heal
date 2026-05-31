"""Shared LLM-call ledger. Records every model call's input and output to a
per-run JSONL file so calls can be audited after the fact.

Usage in a script:

    from llm_audit import set_ledger, log_call
    set_ledger(out_dir / "llm_calls.jsonl")          # once, in main()
    ...
    text = vllm_chat(system, user, ...)
    log_call("detection.confirm", model, system, user, text,
             temperature=t, max_tokens=mt, fold=row["fold"], idx=row["idx"])

Design notes:
- Append-only JSONL, one record per call, flushed immediately (correctness over speed).
- Thread-safe: the phase scripts use ThreadPoolExecutor, so writes are locked.
- If no ledger is set, log_call is a no-op (scripts stay runnable without wiring).
- Full prompts and outputs are stored verbatim (no truncation) — the whole point
  is to recover exactly what the model saw and produced.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_ledger_path: Path | None = None
_run_meta: dict[str, Any] = {}


def set_ledger(path: str | os.PathLike, **run_meta: Any) -> None:
    """Point the ledger at a file and record run-level metadata on every record."""
    global _ledger_path, _run_meta
    _ledger_path = Path(path)
    _ledger_path.parent.mkdir(parents=True, exist_ok=True)
    _run_meta = dict(run_meta)
    # Write a header record so the file always exists and carries run context.
    with _lock:
        with _ledger_path.open("a") as f:
            f.write(json.dumps({"_ledger_start": time.time(), **_run_meta}, ensure_ascii=False, default=str) + "\n")


def log_call(call_type: str, model: str, system: str, user: str, output: str, **meta: Any) -> None:
    """Append one call record. No-op if no ledger is set."""
    if _ledger_path is None:
        return
    rec = {
        "ts": time.time(),
        "call_type": call_type,
        "model": model,
        "system": system,
        "user": user,
        "output": output,
        "system_chars": len(system or ""),
        "user_chars": len(user or ""),
        "output_chars": len(output or ""),
    }
    if meta:
        rec["meta"] = meta
    with _lock:
        with _ledger_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
