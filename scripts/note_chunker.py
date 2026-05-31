#!/usr/bin/env python3
"""Machinery chunker prototype: admission split -> section split -> structured key->value.

Deterministic, no LLM. Turns a multi-admission discharge note into provenance-tagged structure:
  admissions (sorted by chartdate, numbered 1st/2nd/3rd)
    -> sections (HEADER: content)
       -> structured key->value where the pattern is clean (header fields, lab codes)
Narrative prose is left as-is for the LLM layer. Run as __main__ to demo on a couple of cases.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ADM = re.compile(r"(?=Patient ID\s*:)")
HDR = re.compile(r"^([A-Z][A-Za-z /]{2,45}):\s*(.*)")
LAB = re.compile(r"\b([A-Za-z][A-Za-z0-9%/]{1,8})-(\d+\.?\d*\*?#?)")

# Whitelist of TRUE discharge-note section headers. Splitting only on these keeps list-form
# sections (e.g. Discharge Medications) intact — a generic colon-regex shatters them at
# mid-entry sub-labels like "Tablet Refills:" / "RX *" / "Disp #" and loses most of a list.
SECTION_HEADERS = [
    "Chief Complaint", "Major Surgical or Invasive Procedure", "History of Present Illness",
    "Past Medical History", "Surgical History", "SURGICAL HISTORY", "OB/GYN HISTORY", "PAST MEDICAL HISTORY",
    "Social History", "Family History", "Physical Exam", "Physical Examination on Discharge",
    "Pertinent Results", "Brief Hospital Course", "Medications on Admission", "Discharge Medications",
    "Discharge Disposition", "Discharge Diagnosis", "Discharge Condition", "Discharge Instructions",
    "Followup Instructions", "Allergies", "Service", "Chief Complaint",
]
WHITE = re.compile(r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:\s*(.*)", re.I)


def split_sections_clean(text: str) -> list[tuple[str, str]]:
    """Section split on WHITELISTED headers only — keeps lists intact (see SECTION_HEADERS)."""
    secs, cur = [], None
    for ln in text.splitlines():
        m = WHITE.match(ln.strip())
        if m:
            if cur:
                secs.append((cur[0], cur[1].strip()))
            cur = [m.group(1), m.group(2)]
        elif cur is not None:
            cur[1] += " " + ln.strip()
    if cur:
        secs.append((cur[0], cur[1].strip()))
    return [(h, c) for h, c in secs if c]


def split_admissions(note: str) -> list[dict]:
    blocks = [b for b in ADM.split(note) if b.strip()]
    adms = []
    for b in blocks:
        cd = re.search(r"Chartdate\s*:\s*([\d-]+)", b)
        aid = re.search(r"Admission ID\s*:\s*(\d+)", b)
        adms.append({"adm_id": aid.group(1) if aid else "?", "chartdate": cd.group(1) if cd else "9999-99-99", "text": b})
    adms.sort(key=lambda a: a["chartdate"])
    for i, a in enumerate(adms, 1):
        a["n"] = i
    return adms


def split_sections(text: str) -> list[tuple[str, str]]:
    secs, cur = [], None
    for ln in text.splitlines():
        m = HDR.match(ln.strip())
        if m:
            if cur:
                secs.append((cur[0], cur[1].strip()))
            cur = [m.group(1), m.group(2)]
        elif cur is not None:
            cur[1] += " " + ln.strip()
    if cur:
        secs.append((cur[0], cur[1].strip()))
    return [(h, c) for h, c in secs if c]


def parse_labs(text: str) -> dict:
    return {m.group(1): m.group(2) for m in LAB.finditer(text)}


def structured(note: str) -> list[dict]:
    out = []
    for a in split_admissions(note):
        secs = split_sections(a["text"])
        out.append({"n": a["n"], "chartdate": a["chartdate"], "adm_id": a["adm_id"],
                    "fields": {h: c for h, c in secs}, "labs": parse_labs(a["text"])})
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phase2b_extract_compare_detection as P2
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    for kw, show_fields in [("lung nodule", ["Chief Complaint", "Major Surgical or Invasive Procedure", "Discharge Diagnosis"]),
                            ("previous surgeries", ["SURGICAL HISTORY", "Major Surgical or Invasive Procedure"])]:
        row = next(r for r in rows.values() if kw in r["question"])
        print("=" * 95)
        print(f"CASE: {kw}\nQ: {row['question'][:120]}\nGOLD: {row['ground_truth'][:120]}")
        for a in structured(row["note"]):
            print(f"\n  -- Admission #{a['n']} (chartdate {a['chartdate']}, id {a['adm_id']}) --")
            for f in show_fields:
                if a["fields"].get(f):
                    print(f"     {f}: {a['fields'][f][:120]}")
            nlab = {k: v for k, v in a["labs"].items() if k.lower() in ("prot", "24prot", "creat", "wbc", "hgb")}
            if nlab:
                print(f"     [labs] {nlab}")
