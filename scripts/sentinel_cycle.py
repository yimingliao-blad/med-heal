#!/usr/bin/env python3
"""One progress + audit cycle for the cascade (local/fast, no MLX).

Prints two lines:
  1. cascade progress + error count,
  2. running parse audit: how often GPT-4o-mini (the FINAL parser) overrode the regex
     first pass, per the logged parse.flag.*/parse.verdict.* ledger calls.

The GPT-4o-mini judge IS the authoritative parse; this audit just surfaces how much it
is correcting regex (high override = regex was unreliable there). Everything is logged;
the full review artifact is scripts/parse_divergence.py at the end.

Usage: python scripts/sentinel_cycle.py
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import expK_cascade_collect as K

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIR = PROJECT_ROOT / "runs" / "expK_cascade" / "qwen25_nw-1_nc200_seed42"
LEDGER = DIR / "llm_calls.jsonl"
RECORDS = DIR / "records.jsonl"
TOTAL = 309

FLAG_PRE = "<<<ANALYSIS\n"
FLAG_POST = "\nANALYSIS>>>"
VERD_PRE = "<<<RESPONSE\n"
VERD_POST = "\nRESPONSE>>>"


def _ex(u, pre, post):
    i, j = u.find(pre), u.find(post)
    return u[i + len(pre):j] if (i >= 0 and j >= 0) else ""


def progress_line() -> str:
    done = sum(1 for _ in RECORDS.open()) if RECORDS.exists() else 0
    errs = 0
    if RECORDS.exists():
        for ln in RECORDS.open():
            try:
                if json.loads(ln).get("error"):
                    errs += 1
            except Exception:
                pass
    ncalls = sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0
    ts = time.strftime("%H:%M")
    return f"[{ts}] CASCADE {done}/{TOTAL} ({done*100//TOTAL}%) | calls {ncalls} | errors {errs}"


def audit_line() -> str:
    if not LEDGER.exists():
        return "PARSE-AUDIT: (no ledger yet)"
    n = agree = override = 0
    per = defaultdict(lambda: [0, 0])  # stage -> [override, n]
    for ln in LEDGER.open():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        ct = r.get("call_type", "")
        if ct.startswith("parse.flag."):
            raw = _ex(r.get("user", ""), FLAG_PRE, FLAG_POST)
            regex = K.parse_flag(raw)
            up = (r.get("output", "") or "").upper()
            g = True if re.search(r"\bYES\b", up) else (False if (re.search(r"\bNO\b", up) or "UNCLEAR" in up) else None)
            st = "f." + ct[len("parse.flag."):]
        elif ct.startswith("parse.verdict."):
            raw = _ex(r.get("user", ""), VERD_PRE, VERD_POST)
            regex = K.parse_verdict_letter(raw)
            up = (r.get("output", "") or "").upper()
            if re.search(r"\bU\b", up) and not re.search(r"\b[AB]\b", up):
                g = "U"
            else:
                m = re.search(r"\b([AB])\b", up)
                g = m.group(1) if m else None
            st = "v." + ct[len("parse.verdict."):]
        else:
            continue
        if g is None:
            continue
        n += 1
        per[st][1] += 1
        if g == regex:
            agree += 1
        else:
            override += 1
            per[st][0] += 1
    if n == 0:
        return "PARSE-AUDIT: (no parse calls yet)"
    parts = " ".join(f"{s} {o}/{t}" for s, (o, t) in sorted(per.items()))
    return f"PARSE-AUDIT [gpt-override {override}/{n}={override*100//n}%] {parts}"


def main():
    print(progress_line(), flush=True)
    print(audit_line(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
