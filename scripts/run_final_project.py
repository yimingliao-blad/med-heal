#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], execute: bool) -> None:
    print("\n$", " ".join(cmd), flush=True)
    if execute:
        subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Final refactor project command hub.")
    ap.add_argument("--execute", action="store_true", help="run commands instead of printing them")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default="qwen2.5")
    ap.add_argument("--smoke", action="store_true", help="use 2 wrong + 2 correct for multirun commands")
    args = ap.parse_args()

    n_wrong = "2" if args.smoke else "-1"
    n_correct = "2" if args.smoke else "-1"
    commands = [
        [sys.executable, str(ROOT / "scripts" / "prepare_clean_project.py")],
        [
            sys.executable,
            str(ROOT / "scripts" / "run_baselines.py"),
            "--models",
            "biomistral-7b",
            "qwen2.5-7b-instruct",
            "qwen3-8b",
            "deepseek-r1-distill-llama-8b",
            "llama-3.1-8b-instruct",
            "--conditions",
            "zeroshot",
            "gtr_note_pos_k1",
            "gtr_note_neg_k1",
            "gtr_note_posneg_k1",
            "cot_evidence",
            "cot_conclusion",
            "multiturn",
            "gtr_note_any_unlabeled_k1",
            "--folds",
            "0",
            "1",
            "2",
            "3",
            "4",
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "run_natural_mini_parser_pipeline.py"),
            "--port",
            str(args.port),
            "--concurrency",
            str(args.concurrency),
            "--n-wrong",
            n_wrong,
            "--n-correct",
            n_correct,
            "--det-prompt",
            "cot_route",
            "--det-temperature",
            "0.0",
            "--judge",
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "run_regen_verdict_mini.py"),
            "--model",
            args.model,
            "--port",
            str(args.port),
            "--concurrency",
            str(args.concurrency),
            "--n-wrong",
            n_wrong,
            "--n-correct",
            n_correct,
            "--judge",
        ],
    ]
    for cmd in commands:
        run(cmd, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
