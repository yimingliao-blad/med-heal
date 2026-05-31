#!/usr/bin/env python3
"""Live ledger auditor — run any time while a collection is in progress.

Reads an llm_calls.jsonl ledger and reports:
  - call counts by call_type (stage)
  - failures (call_type ending .fail, or empty output)
  - NOTE-PRESENCE check: of calls whose prompt should contain a discharge note
    (extract / gate / verdict / correction stages), how many actually have a
    substantial note in the user prompt. Catches the empty-note bug LIVE.
  - input/output char stats per stage

Usage:
  python scripts/audit_ledger.py runs/expK_cascade/<dir>/llm_calls.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


# Stages whose user-prompt MUST contain the discharge note. Excludes by design:
#   compare.*  (uses extracted facts, not the note)
#   corr.*     (uses the error+evidence, not the full note — B2 finding)
#   r1.*, parse.*, gpt.localize (no full note by design)
NOTE_STAGES = ("extract.answer", "extract.question", "gate.", "diag.", "verdict.", "judge", "summary")


def main(path: str) -> int:
    n = 0
    by_type = Counter()
    fails = Counter()
    empty_out = Counter()
    note_present = defaultdict(lambda: [0, 0])  # stage -> [with_note, total]
    in_chars = defaultdict(int)
    out_chars = defaultdict(int)
    header = None
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "_ledger_start" in r:
            header = r
            continue
        ct = r.get("call_type", "?")
        n += 1
        by_type[ct] += 1
        if ct.endswith(".fail"):
            fails[ct] += 1
        if not (r.get("output") or "").strip():
            empty_out[ct] += 1
        in_chars[ct] += r.get("user_chars", 0)
        out_chars[ct] += r.get("output_chars", 0)
        # note presence: does the user prompt look like it contains a real note?
        if any(ct.startswith(s) or ct == s for s in NOTE_STAGES):
            u = r.get("user", "") or ""
            # heuristic: a real discharge note slice is long and contains clinical markers
            has_note = len(u) > 2000 and ("[Note" in u or "Discharge" in u or "Admission" in u or "Patient" in u)
            note_present[ct][0] += 1 if has_note else 0
            note_present[ct][1] += 1
    print(f"=== ledger: {path} ===")
    if header:
        print(f"script={header.get('script')} served={header.get('served')}")
    print(f"total calls: {n}")
    print()
    print(f"{'call_type':28} {'count':>7} {'fails':>6} {'empty_out':>10} {'avg_in':>8} {'avg_out':>8}")
    for ct, c in by_type.most_common():
        print(f"{ct:28} {c:>7} {fails.get(ct,0):>6} {empty_out.get(ct,0):>10} {in_chars[ct]//max(1,c):>8} {out_chars[ct]//max(1,c):>8}")
    print()
    print("=== NOTE-PRESENCE CHECK (catches empty-note bug) ===")
    bad = False
    for ct in sorted(note_present):
        wn, tot = note_present[ct]
        rate = wn / max(1, tot)
        flag = "" if rate > 0.9 else "  <-- WARNING: notes missing!"
        if rate <= 0.9:
            bad = True
        print(f"  {ct:28} note-present {wn}/{tot} ({rate*100:.0f}%){flag}")
    print()
    print("OVERALL:", "WARNING — some note-bearing prompts have no note" if bad else "OK — all note-bearing prompts contain a note")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
