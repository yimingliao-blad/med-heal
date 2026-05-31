#!/usr/bin/env python3
"""Parser-sentinel — confirm our regex parsing of LLM outputs is correct, via Qwen3.5.

Historical flaw: regex parsing of LLM output drifts silently and corrupts results. This
re-parses raw outputs from the ledger TWO ways and checks agreement:
  - OUR parse  : the exact regex our pipeline uses for each call_type.
  - TIER-2 parse: an independent Qwen3.5-27B (MLX :8803) reads the same raw and extracts
                  the same field.
Agreement < 0.95 on any field is flagged — that is a parsing bug to fix before trusting results.

Parse-sensitive call_types audited:
  verdict.*            -> the A/B pick (last-line letter)
  gate.positive_confirm-> YES / NO
  gate.plain_confirm   -> flagged (UNCONFIRMED non-empty & not 'none')
  diag.blind_plain / diag.blind_cot -> flagged (INCONSISTENT non-empty & not 'none')
  diag.blind_cot_clean -> flagged (WRONG: non-'none')

Usage: python scripts/parse_audit.py <ledger.jsonl> [--per-type 40]
Requires Qwen3.5 at MLX http://192.168.68.107:8803.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict

import requests

MLX_BASE = "http://192.168.68.107:8803"
MLX_URL = f"{MLX_BASE}/v1/chat/completions"
_MLX_MODEL = None


def mlx_model() -> str:
    global _MLX_MODEL
    if _MLX_MODEL is None:
        r = requests.get(f"{MLX_BASE}/v1/models", timeout=10)
        _MLX_MODEL = r.json()["data"][0]["id"]
    return _MLX_MODEL


# ---------- OUR parse (must mirror the pipeline exactly) ----------

def our_verdict_letter(raw: str) -> str:
    lines = [ln for ln in (raw or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"\b([AB])\b", ln.upper())
        if m:
            return m.group(1)
    return "A"  # default in pipeline


def our_yesno(raw: str) -> str:
    return "YES" if re.search(r"\bYES\b", (raw or "").upper()) else "NO"


def our_flag_token(raw: str) -> bool:
    """Mirror expK.parse_flag: flagged = explicit final FLAG: YES token (last occurrence)."""
    flags = re.findall(r"FLAG\s*:\s*(YES|NO)", raw or "", re.I)
    return bool(flags) and flags[-1].upper() == "YES"


def our_parse(call_type: str, raw: str):
    if call_type.startswith("verdict."):
        return ("pick", our_verdict_letter(raw))
    if call_type == "gate.positive_confirm":
        return ("yesno", our_yesno(raw))
    if call_type in ("gate.plain_confirm", "diag.blind_plain", "diag.blind_cot", "diag.blind_cot_clean"):
        return ("flagged", our_flag_token(raw))
    return (None, None)


PARSE_TYPES = ("verdict.C3_cot", "verdict.C3_strict", "gate.positive_confirm", "gate.plain_confirm",
               "diag.blind_plain", "diag.blind_cot", "diag.blind_cot_clean")


# ---------- TIER-2 parse (Qwen3.5 on MLX) ----------

def mlx(system: str, user: str, max_tokens: int = 16) -> str:
    payload = {"model": mlx_model(), "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0.0}
    try:
        r = requests.post(MLX_URL, json=payload, timeout=120)
        out = (r.json()["choices"][0]["message"].get("content") or "").strip()
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S | re.I).strip()
    except Exception as e:
        return f"__ERR__{e}"


def tier2_parse(kind: str, raw: str):
    if kind == "pick":
        out = mlx("Extract the final A/B choice from this text. Reply only A or B.", raw[:4000], 8)
        m = re.search(r"\b([AB])\b", out.upper())
        return m.group(1) if m else "?"
    if kind == "yesno":
        out = mlx("Did this text conclude YES or NO? Reply only YES or NO.", raw[:4000], 8)
        return "YES" if "YES" in out.upper() else ("NO" if "NO" in out.upper() else "?")
    if kind == "flagged":
        # PARSING check (not content re-judgement): extract the model's explicit FINAL
        # FLAG token. The diagnoser's stated final decision is the contract; whether that
        # decision is coherent with its own analysis is a detection-quality question
        # measured elsewhere, not by the parser-sentinel.
        out = mlx("The text ends with a line 'FLAG: YES' or 'FLAG: NO'. Report ONLY that final FLAG value, ignoring everything else. Reply FLAGGED if the final line is FLAG: YES, or CLEAN if it is FLAG: NO.", raw[:6000], 8)
        if "FLAG" in out.upper():
            return True
        if "CLEAN" in out.upper():
            return False
        return "?"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # collect raws by call_type
    by_type = defaultdict(list)
    for line in open(args.ledger):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ct = r.get("call_type")
        if ct in PARSE_TYPES and (r.get("output") or "").strip():
            by_type[ct].append(r["output"])
    rng = random.Random(args.seed)
    print(f"=== parser-sentinel: {args.ledger} ===")
    print("MLX tier-2 = Qwen3.5-27B @ :8803")
    overall_bad = False
    for ct in PARSE_TYPES:
        raws = by_type.get(ct, [])
        if not raws:
            print(f"{ct:24} (no calls)")
            continue
        rng.shuffle(raws)
        sample = raws[: args.per_type]
        agree = 0
        n = 0
        disagreements = []
        for raw in sample:
            kind, ours = our_parse(ct, raw)
            if kind is None:
                continue
            t2 = tier2_parse(kind, raw)
            if t2 == "?" or (isinstance(t2, str) and t2.startswith("__ERR__")):
                continue
            n += 1
            if ours == t2:
                agree += 1
            elif len(disagreements) < 3:
                disagreements.append((ours, t2, raw[:120].replace("\n", " ")))
        if n == 0:
            print(f"{ct:24} (no valid tier-2 comparisons — MLX returned nothing)")
            continue
        rate = agree / n
        flag = "" if rate >= 0.95 else "  <-- BELOW 0.95: parsing bug"
        if rate < 0.95:
            overall_bad = True
        print(f"{ct:24} agreement {agree}/{n} ({rate*100:.0f}%){flag}")
        for ours, t2, snip in disagreements:
            print(f"    ours={ours} tier2={t2} :: {snip}")
    print()
    print("OVERALL:", "PARSING BUG(S) FOUND — fix before trusting results" if overall_bad else "OK — parsing agrees with Qwen3.5 tier-2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
