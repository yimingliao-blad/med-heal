#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pre_atom.config import env_for_legacy, load_config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    p.add_argument("--source", choices=["step2", "step8"], default="step2")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    script = ROOT / "src" / "pre_atom" / "legacy" / "step9_self_correction" / "v2" / "judge.py"
    cmd = [sys.executable, str(script), "--validate-against-gold", "--source", args.source]
    print(" ".join(cmd))
    if args.execute:
        subprocess.run(cmd, check=True, env=env_for_legacy(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

