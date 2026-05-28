#!/usr/bin/env python3
"""Config-driven wrapper for Step 8 baseline/RA-ICL/CoT runs.

By default this prints the exact commands. Pass `--execute` to run them.
"""
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
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--folds", nargs="+", default=None)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    models = args.models or cfg["models"]
    conditions = args.conditions or cfg["conditions"]
    folds = args.folds or [str(f) for f in cfg["folds"]]
    script = Path(cfg.get("stage8", {}).get("script", ROOT / "src" / "pre_atom" / "legacy" / "step8_multimodel_icl" / "generate_step8.py"))
    if not script.is_absolute():
        script = (ROOT / script).resolve()
    env = env_for_legacy(cfg)

    for model in models:
        cmd = [
            sys.executable,
            str(script),
            "--model",
            model,
            "--conditions",
            *conditions,
            "--folds",
            *map(str, folds),
            "--port",
            str(cfg["servers"]["vllm_port"]),
        ]
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

