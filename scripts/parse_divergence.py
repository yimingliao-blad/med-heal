#!/usr/bin/env python3
"""End-of-run parse audit: regex first-pass vs GPT-4o-mini final decision.

The cascade parses every gate/diagnoser/verdict output two ways: a regex first pass,
then GPT-4o-mini reads the raw and gives the FINAL authoritative decision (robust to
Qwen2.5 not following format AND to regex edge cases). Every judge call is logged to the
ledger (parse.flag.<stage> / parse.verdict.<stage>: user=raw analysis, output=YES/NO|A/B).

This report recomputes the regex candidate from each logged raw and reports, per stage,
how often GPT-4o-mini AGREED with regex and where it OVERRODE it (with examples) — so the
divergence is fully reviewable after the run.

Usage: python scripts/parse_divergence.py runs/expK_cascade/<dir>/llm_calls.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import expK_cascade_collect as K  # parse_flag, parse_verdict_letter, prompt templates

FLAG_PRE = "<<<ANALYSIS\n"
FLAG_POST = "\nANALYSIS>>>"
VERD_PRE = "<<<RESPONSE\n"
VERD_POST = "\nRESPONSE>>>"


def extract(user: str, pre: str, post: str) -> str:
    i = user.find(pre)
    j = user.find(post)
    if i < 0 or j < 0:
        return ""
    return user[i + len(pre):j]


def gpt_flag(out: str):
    up = (out or "").upper()
    if "UNCLEAR" in up:
        return False  # pipeline maps UNCLEAR -> not flagged
    if re.search(r"\bYES\b", up):
        return True
    if re.search(r"\bNO\b", up):
        return False
    return None


def gpt_letter(out: str):
    up = (out or "").upper()
    if re.search(r"\bU\b", up) and not re.search(r"\b[AB]\b", up):
        return "U"
    m = re.search(r"\b([AB])\b", up)
    return m.group(1) if m else None


def main(path: str) -> int:
    stats = defaultdict(lambda: {"n": 0, "agree": 0, "override": 0, "unclear": 0, "ex": []})
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ct = r.get("call_type", "")
        if ct.startswith("parse.flag."):
            stage = ct[len("parse.flag."):]
            raw = extract(r.get("user", ""), FLAG_PRE, FLAG_POST)
            regex = K.parse_flag(raw)
            g = gpt_flag(r.get("output", ""))
            kind = "flag"
        elif ct.startswith("parse.verdict."):
            stage = ct[len("parse.verdict."):]
            raw = extract(r.get("user", ""), VERD_PRE, VERD_POST)
            regex = K.parse_verdict_letter(raw)
            g = gpt_letter(r.get("output", ""))
            kind = "verdict"
        else:
            continue
        key = f"{kind}:{stage}"
        s = stats[key]
        s["n"] += 1
        if g is None:
            s["unclear"] += 1
        elif g == regex:
            s["agree"] += 1
        else:
            s["override"] += 1
            if len(s["ex"]) < 4:
                s["ex"].append((regex, g, (raw or "")[:140].replace("\n", " ")))
    print(f"=== parse divergence (regex first-pass vs GPT-4o-mini final): {path} ===\n")
    print(f"{'stage':28} {'n':>6} {'agree':>7} {'override':>9} {'unclear':>8} {'override%':>9}")
    tot = defaultdict(int)
    for key in sorted(stats):
        s = stats[key]
        ov = s["override"] / max(1, s["n"])
        print(f"{key:28} {s['n']:>6} {s['agree']:>7} {s['override']:>9} {s['unclear']:>8} {ov*100:>8.1f}%")
        tot["n"] += s["n"]; tot["agree"] += s["agree"]; tot["override"] += s["override"]; tot["unclear"] += s["unclear"]
    print("-" * 72)
    print(f"{'TOTAL':28} {tot['n']:>6} {tot['agree']:>7} {tot['override']:>9} {tot['unclear']:>8} {tot['override']/max(1,tot['n'])*100:>8.1f}%")
    print("\n=== override examples (GPT-4o-mini disagreed with regex — these are the cases regex got wrong or the model formatted oddly) ===")
    for key in sorted(stats):
        for regex, g, snip in stats[key]["ex"]:
            print(f"  [{key}] regex={regex} gpt={g} :: {snip}")
    print("\nNOTE: override% is how often the GPT-4o-mini judge changed the regex result. High override on a")
    print("stage means regex was unreliable there (or Qwen2.5 formatted loosely) — GPT-4o-mini is the final answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
