#!/usr/bin/env python3
"""Wrapper for the selected and pilot self-correction variants."""
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
    p.add_argument("--variant", choices=["step9_v2", "regen", "regen_v3"], default="step9_v2")
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    p.add_argument("--model-aliases", nargs="+", default=["qwen2.5"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    env = env_for_legacy(cfg)
    v2 = ROOT / "src" / "pre_atom" / "legacy" / "step9_self_correction" / "v2"
    port = str(cfg["servers"]["vllm_port"])

    if args.variant == "step9_v2":
        script = v2 / "run_pipeline.py"
        cmd = [sys.executable, str(script), "--port", port, "--mode", args.mode]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
    elif args.variant == "regen":
        script = v2 / "regen_pilot.py"
        cmd = [sys.executable, str(script), "--models", *args.model_aliases, "--port", port]
    else:
        script = v2 / "regen_v3_pilot.py"
        cmd = [sys.executable, str(script), "--models", *args.model_aliases, "--port", port]

    print(" ".join(cmd))
    if args.execute:
        subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

