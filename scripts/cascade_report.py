#!/usr/bin/env python3
"""One-line cascade progress + audit report (progress, note-presence, parser-sentinel).
Printed by the Monitor loop each cycle. Keeps it to a single stdout line per call."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = "/home/ra/Projects/llm-ehr-hallucination/.venv/bin/python"
DIR = PROJECT_ROOT / "runs" / "expK_cascade" / "qwen25_nw-1_nc200_seed42"
LEDGER = DIR / "llm_calls.jsonl"
RECORDS = DIR / "records.jsonl"
TOTAL = 309


def main():
    done = sum(1 for _ in RECORDS.open()) if RECORDS.exists() else 0
    errs = 0
    if RECORDS.exists():
        for line in RECORDS.open():
            try:
                if json.loads(line).get("error"):
                    errs += 1
            except Exception:
                pass
    # note-presence audit (quick)
    note = "note?"
    try:
        out = subprocess.run([PY, str(PROJECT_ROOT / "scripts" / "audit_ledger.py"), str(LEDGER)],
                             capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"OVERALL:\s*(.+)", out)
        note = "notesOK" if (m and m.group(1).strip().startswith("OK")) else "NOTES_WARN"
    except Exception:
        note = "note_audit_err"
    # ledger call count (throughput signal)
    ncalls = sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0
    ts = time.strftime("%H:%M")
    # NOTE: the parser-sentinel (MLX Qwen3.5) is SLOW and runs as a SEPARATE patient job,
    # not in this fast per-cycle report — MLX cannot answer a fresh sample every cycle.
    print(f"[{ts}] CASCADE {done}/{TOTAL} ({done*100//TOTAL}%) | calls {ncalls} | errors {errs} | {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
