#!/usr/bin/env python3
"""
Module 5 — audit_view CLI.

Pretty-print one or more audit records from the V2 audit log file.
This replaces audit_one_case.py: instead of re-running the pipeline, we
just read what was already persisted.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from audit_log import AuditLog

DEFAULT_LOG = Path(__file__).resolve().parents[3] / "output" / "step9_v2" / "pilot_audit_log.jsonl"


def _hr(label: str) -> str:
    return "\n" + ("=" * 100) + f"\n{label}\n" + ("=" * 100)


def _show(label: str, value, max_chars: int = 1500) -> None:
    if value is None:
        print(f"\n[{label}] (none)")
        return
    s = json.dumps(value, indent=2, default=str) if not isinstance(value, str) else value
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n... [TRUNCATED total={len(s)}]"
    print(f"\n[{label}]\n{textwrap.indent(s, '  ')}")


def show_record(rec: dict) -> None:
    print(_hr(f"AUDIT RECORD  fold={rec['fold']}  idx={rec['idx']}"))
    item = rec.get("item") or {}
    _show("question", item.get("question"))
    _show("ground_truth", item.get("ground_truth"))
    _show("original_answer", item.get("original_answer"))
    _show("note (truncated)", item.get("note"), max_chars=2000)
    print(_hr("JUDGE — original answer"))
    _show("judge_orig", rec.get("judge_orig"))
    print(_hr("DETECTION"))
    _show("detection", rec.get("detection"), max_chars=4000)
    print(_hr("CORRECTION"))
    _show("correction", rec.get("correction"), max_chars=3500)
    print(_hr("VERDICT"))
    _show("verdict", rec.get("verdict"), max_chars=2500)
    print(_hr("JUDGE — corrected answer"))
    _show("judge_corrected", rec.get("judge_corrected"))
    print(_hr("OUTCOME"))
    _show("outcome", rec.get("outcome"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=DEFAULT_LOG,
                   help="path to audit log JSONL (default: pilot_audit_log.jsonl)")
    p.add_argument("--fold", type=int, required=False)
    p.add_argument("--idx", type=int, required=False)
    p.add_argument("--list", action="store_true", help="list all (fold, idx) in the log")
    p.add_argument("--summary", action="store_true",
                   help="print aggregate counts (fixes/breaks/etc.)")
    args = p.parse_args()

    if not args.log.exists():
        print(f"!! audit log not found: {args.log}")
        return 1
    log = AuditLog(args.log)

    if args.list:
        for r in sorted(log.all(), key=lambda r: (r["fold"], r["idx"])):
            o = r.get("outcome") or {}
            print(f"fold={r['fold']:>2} idx={r['idx']:>4}  "
                  f"action={o.get('action','?'):<14} delta={o.get('delta','?')}")
        return 0

    if args.summary:
        recs = log.all()
        n = len(recs)
        actions: dict[str, int] = {}
        fixes = 0
        breaks = 0
        for r in recs:
            o = r.get("outcome") or {}
            a = o.get("action", "?")
            actions[a] = actions.get(a, 0) + 1
            if a == "corrected" and o.get("delta") == 1:
                fixes += 1
            elif a == "corrected" and o.get("delta") == -1:
                breaks += 1
        print(f"Total records: {n}")
        for k, v in sorted(actions.items()):
            print(f"  action={k}: {v}")
        print(f"  fixes (corrected, delta=+1):  {fixes}")
        print(f"  breaks (corrected, delta=-1): {breaks}")
        print(f"  net: {fixes - breaks:+d}")
        return 0

    if args.fold is None or args.idx is None:
        p.print_help()
        return 1
    rec = log.get(args.fold, args.idx)
    if not rec:
        print(f"!! no record for fold={args.fold} idx={args.idx}")
        return 1
    show_record(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
