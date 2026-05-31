#!/usr/bin/env python3
"""Dry-run orchestration entrypoint.

This intentionally defaults to printing commands. Use individual scripts with
`--execute` when the selected variant and server state are confirmed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    steps = [
        [sys.executable, str(ROOT / "scripts" / "build_decision_matrix.py")],
        [sys.executable, str(ROOT / "scripts" / "preflight.py")],
        [sys.executable, str(ROOT / "scripts" / "run_baselines.py")],
        [sys.executable, str(ROOT / "scripts" / "validate_judge.py")],
        [sys.executable, str(ROOT / "scripts" / "run_self_correction.py"), "--limit", "5"],
    ]
    for cmd in steps:
        print("\n$", " ".join(cmd))
        subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

