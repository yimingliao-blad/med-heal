#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pre_atom.config import load_config
from pre_atom.data_assets import prepare_data_assets, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "final_project.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest = prepare_data_assets(cfg)
    out = write_manifest(manifest, ROOT / "reports" / "final_project_data_manifest.json")
    print(json.dumps({"manifest": str(out), "summary": manifest["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
