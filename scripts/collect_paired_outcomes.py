#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pre_atom.config import load_config, resolve_path  # noqa: E402


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def row_from_record(rec: dict, source_log: Path, model: str | None = None) -> dict | None:
    item = rec.get("item") or {}
    outcome = rec.get("outcome") or {}
    judge_orig = rec.get("judge_orig") or {}
    if not outcome:
        return None
    z = judge_orig.get("label")
    if z is None:
        z = item.get("label")
    f = outcome.get("final_eval")
    if f is None:
        return None
    return {
        "fold": rec.get("fold"),
        "idx": rec.get("idx"),
        "patient_id": item.get("patient_id"),
        "model": model or source_log.parent.name,
        "source_log": str(source_log),
        "action": outcome.get("action"),
        "delta": outcome.get("delta"),
        "zeroshot_correct": int(z),
        "final_correct": int(f),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    ap.add_argument("--logs", nargs="*", default=None, help="Explicit JSONL audit logs")
    ap.add_argument("--glob", default="multi_model/*/audit.jsonl", help="Glob under step9_v2 output when --logs is omitted")
    ap.add_argument("--output", default=str(ROOT / "output" / "paired_outcomes.csv"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    step9_dir = resolve_path(cfg["paths"]["step9_v2_output_dir"], base=ROOT)
    if args.logs:
        logs = [Path(x) for x in args.logs]
    else:
        logs = sorted(step9_dir.glob(args.glob))
    rows = []
    for log in logs:
        if not log.exists():
            print(f"MISS {log}")
            continue
        model = log.parent.name if log.parent.name != "step9_v2" else None
        for rec in iter_jsonl(log):
            row = row_from_record(rec, log, model=model)
            if row:
                rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} rows={len(rows)} logs={len(logs)}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
