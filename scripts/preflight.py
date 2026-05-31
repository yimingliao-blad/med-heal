#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pre_atom.config import load_config, resolve_path  # noqa: E402


def check_file(label: str, path: Path) -> bool:
    ok = path.exists()
    print(f"{'OK ' if ok else 'MISS'} {label}: {path}")
    return ok


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    ok = True
    ok &= check_file("processed_ehrnoteqa", resolve_path(cfg["paths"]["processed_ehrnoteqa"], base=ROOT))
    ok &= check_file("human_eval_csv", resolve_path(cfg["paths"]["human_eval_csv"], base=ROOT))
    folds_dir = resolve_path(cfg["paths"]["folds_dir"], base=ROOT)
    for fold in cfg["folds"]:
        ok &= check_file(f"fold_{fold}/test.jsonl", folds_dir / f"fold_{fold}" / "test.jsonl")
    step8_dir = resolve_path(cfg["paths"]["step8_output_dir"], base=ROOT)
    for model in cfg["models"]:
        for fold in cfg["folds"]:
            p = step8_dir / model / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
            ok &= check_file(f"{model} fold_{fold} zeroshot labels", p)
    print(f"{'OK ' if os.environ.get('OPENAI_API_KEY') else 'MISS'} OPENAI_API_KEY")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

