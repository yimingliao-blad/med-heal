#!/usr/bin/env python3
"""
Detection D2 — atomic yes/no detection.

Per user direction (after the F1 + GPT-4o-validity-gate experiment was rejected
for using GPT-4o inside the runtime gate):

  - one simple yes/no question per LLM call
  - two atomic checks: (1) does the answer contradict the notes?
                       (2) does the answer address the question?
  - K samples per question, plain-text first-line yes/no parse
  - Qwen3-32B is the FALLBACK extractor when the first-line is unclear or
    when the model uses hedging language ("possibly", "it seems", etc.)
  - the gate is multi-sample agreement (3/5 majority by default), with the
    full vote distribution persisted so we can interpret weak vs strong
    signals after the fact
  - GPT-4o is NEVER called inside this module

Return shape (per item):
  {
    "contradiction": {
        "samples": [{"raw": "...", "first_line": "yes|no|unclear",
                     "parse_path": "regex|q32_fallback|unparseable",
                     "reason": "<one-liner from the model>"}],
        "votes": {"yes": int, "no": int, "unclear": int},
        "n_valid": int,
        "majority": "yes|no|unclear",
        "majority_count": int,
        "confidence": float (majority_count / n_valid),
    },
    "qmis": {... same shape ... where "yes" means addresses the question ...},
    "fired": bool,                          # severity gate decision
    "fired_reason": str | None,             # which signal triggered
    "fire_threshold": int,                  # majority threshold used
    "error_type": "CONTRADICTION" | "QUESTION_MISALIGNMENT" | None,
    "error_statement": str,                 # one-liner from the firing sample
    "correct_statement": str,               # left empty here; correction step
                                            #   pulls evidence from note spans
  }
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

# Reuse the vLLM transport already in detection_format_bakeoff
from detection_format_bakeoff import build_chatml, vllm_gen, vllm_chat, q32_extract

SYS = "You are a strict medical expert checking clinical answers."

CONTRA_PROMPT = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

Does the answer make any factual claim that DIRECTLY contradicts the discharge notes?
By "contradict" I mean the answer states something that the notes say differently
(wrong medication, wrong date, wrong diagnosis, wrong outcome, wrong dose, etc.).

Reply on the FIRST line with exactly one word: yes  or  no
On the SECOND line, give one short sentence saying which claim contradicts the
notes (or "none" if you answered no)."""

QMIS_PROMPT = """Question: {question}

Answer: {answer}

Does the answer DIRECTLY address what the question asks? Consider whether the
answer is about the right visit, the right time period, the right aspect, and
actually responds to the asked question rather than a related but different one.

Reply on the FIRST line with exactly one word: yes  or  no
On the SECOND line, give one short sentence saying what the answer addresses vs
what the question asks (or "addresses fully" if you answered yes)."""

# ---------- Robust parsing chain ----------

# Strict first-line yes/no
_RE_FIRST_LINE = re.compile(r"^[\s\-*>#]*([A-Za-z]+)\b")
# Hedging / soft phrases that should NOT be treated as a clean yes
_HEDGE_WORDS = {"possibly", "perhaps", "maybe", "likely", "unclear",
                "might", "could", "seems", "appears", "partially",
                "somewhat", "kind", "sort"}

# Cheap Qwen3-32B fallback extractor — single yes/no question
EXTRACT_YESNO = """/nothink
Read the text below. The author was asked a yes/no question.

TEXT:
{raw}

Did the author answer YES or NO to the original question? If they hedged or
gave a non-committal answer, reply UNCLEAR.

Reply with exactly one word: YES, NO, or UNCLEAR.
"""


def _parse_first_line(raw: str) -> str | None:
    if not raw:
        return None
    first = raw.strip().splitlines()[0] if raw.strip() else ""
    m = _RE_FIRST_LINE.match(first)
    if not m:
        return None
    word = m.group(1).lower()
    if word in ("yes", "y"):
        return "yes"
    if word in ("no", "n"):
        return "no"
    return None


def _has_hedging(raw: str) -> bool:
    if not raw:
        return False
    head = raw.strip().splitlines()[0].lower() if raw.strip() else ""
    return any(h in head for h in _HEDGE_WORDS)


def _q32_fallback_yesno(raw: str) -> str:
    """Returns 'yes' / 'no' / 'unclear'. Uses Qwen3-32B."""
    obj_text = q32_extract_text(raw)
    text = (obj_text or "").strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "unclear"


def q32_extract_text(raw: str) -> str:
    """Plain-text Qwen3-32B call (no JSON parsing). Returns the raw answer."""
    import requests
    from detection_format_bakeoff import QWEN32B_URL
    try:
        r = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Read carefully. Answer in one word only."},
                {"role": "user", "content": EXTRACT_YESNO.format(raw=raw[:1500])},
            ],
            "max_tokens": 20, "temperature": 0.0,
        }, timeout=60)
        text = r.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        print(f"  q32 fallback err: {e}", flush=True)
        return ""


def parse_yesno(raw: str) -> tuple[str, str]:
    """Returns (verdict, parse_path).
    verdict ∈ {'yes','no','unclear'}; parse_path ∈ {'regex','q32_fallback','unparseable'}."""
    first = _parse_first_line(raw)
    if first in ("yes", "no"):
        if first == "yes" and _has_hedging(raw):
            # Looks like yes but hedged → fallback
            v = _q32_fallback_yesno(raw)
            return v, "q32_fallback"
        return first, "regex"
    # Fallback
    v = _q32_fallback_yesno(raw)
    if v in ("yes", "no", "unclear"):
        return v, "q32_fallback"
    return "unclear", "unparseable"


def _second_line_reason(raw: str) -> str:
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return ""
    # The first line is the yes/no token; the second line is the reason
    return lines[1][:300]


# ---------- Single-question multi-sample call ----------

def _call_yesno(prompt_user: str, port: int, k: int) -> dict:
    """Sample one question K times and aggregate."""
    samples = []
    for _ in range(k):
        raw = vllm_chat(SYS, prompt_user, port,
                        max_tokens=200, temperature=0.7)
        verdict, path = parse_yesno(raw)
        samples.append({
            "raw": raw,
            "first_line": verdict,
            "parse_path": path,
            "reason": _second_line_reason(raw),
        })
    votes = Counter(s["first_line"] for s in samples)
    n_valid = sum(votes[v] for v in ("yes", "no"))
    if n_valid == 0:
        majority = "unclear"
        majority_count = votes.get("unclear", 0)
    else:
        # Majority among yes/no only
        if votes.get("yes", 0) >= votes.get("no", 0):
            majority = "yes"; majority_count = votes.get("yes", 0)
        else:
            majority = "no"; majority_count = votes.get("no", 0)
    return {
        "samples": samples,
        "votes": dict(votes),
        "n_valid": n_valid,
        "majority": majority,
        "majority_count": majority_count,
        "confidence": (majority_count / n_valid) if n_valid else 0.0,
    }


# ---------- Top-level detection ----------

def detect_d2(note: str, question: str, answer: str, port: int, *,
              k: int = 5, severity_threshold: int = 3) -> dict:
    """Run D2 detection (contradiction + qmis) with K samples each.

    severity_threshold is the minimum number of agreeing samples (out of K) for
    a signal to count as fired. Default 3/5 = simple majority (per user).
    """
    contra = _call_yesno(
        CONTRA_PROMPT.format(note=note, question=question, answer=answer[:800]),
        port, k,
    )
    # qmis: yes = addresses the question; for an error we want NO (does not address)
    qmis = _call_yesno(
        QMIS_PROMPT.format(question=question, answer=answer[:800]),
        port, k,
    )

    fired = False
    fired_reason = None
    error_type = None
    error_statement = ""
    # Contradiction signal: majority "yes" (a contradiction exists)
    if contra["majority"] == "yes" and contra["majority_count"] >= severity_threshold:
        fired = True
        fired_reason = f"contradiction yes={contra['majority_count']}/{contra['n_valid']}"
        error_type = "CONTRADICTION"
        error_statement = next((s["reason"] for s in contra["samples"]
                                if s["first_line"] == "yes" and s["reason"]), "")

    # qmis signal: majority "no" (does not address the question)
    if qmis["majority"] == "no" and qmis["majority_count"] >= severity_threshold:
        # Only override if not already contradiction (priority order)
        if not fired:
            fired = True
            fired_reason = f"qmis no={qmis['majority_count']}/{qmis['n_valid']}"
            error_type = "QUESTION_MISALIGNMENT"
            error_statement = next((s["reason"] for s in qmis["samples"]
                                    if s["first_line"] == "no" and s["reason"]), "")

    return {
        "contradiction": contra,
        "qmis": qmis,
        "fired": fired,
        "fired_reason": fired_reason,
        "fire_threshold": severity_threshold,
        "error_type": error_type,
        "error_statement": error_statement,
        # correct_statement is filled in by the correction step from note spans;
        # D2 doesn't try to capture "what the notes say" — that's where F1 hallucinated.
        "correct_statement": "",
    }


# ---------- Self-test ----------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from judge import _load_notes_lookup

    notes = _load_notes_lookup()
    note = notes["12152580"]  # idx=51 patient
    question = "What was the main postop complication from the infraclavicular first rib resection procedure?"
    answer = ("Based on the discharge summary provided, there does not appear to be any "
              "mention of a postoperative complication from the infraclavicular first rib "
              "resection procedure. The patient was described as tolerating the procedure "
              "well and was discharged without any significant issues noted.")
    print("Running D2 self-test on idx=51 (urinary retention case)...")
    d = detect_d2(note, question, answer, port=8003, k=5, severity_threshold=3)
    print(f"\ncontradiction: votes={d['contradiction']['votes']} majority={d['contradiction']['majority']}/{d['contradiction']['n_valid']}")
    for s in d['contradiction']['samples']:
        print(f"  [{s['parse_path']}] {s['first_line']}: {s['reason'][:120]}")
    print(f"\nqmis: votes={d['qmis']['votes']} majority={d['qmis']['majority']}/{d['qmis']['n_valid']}")
    for s in d['qmis']['samples']:
        print(f"  [{s['parse_path']}] {s['first_line']}: {s['reason'][:120]}")
    print(f"\nFIRED: {d['fired']} reason={d['fired_reason']}")
    print(f"error_type: {d['error_type']}")
    print(f"error_statement: {d['error_statement']}")
