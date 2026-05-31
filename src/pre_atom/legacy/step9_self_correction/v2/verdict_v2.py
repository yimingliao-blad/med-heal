#!/usr/bin/env python3
"""
Verdict V2 — pairwise A/B comparison, single simple question.

Per user direction (DeepSeek finding: models reliably pick the better of two
answers when asked simply). Replaces the v1f contradiction-count prompt with
the simplest possible pairwise question.

  - one prompt asking "which answer is more consistent with the notes"
  - K samples, blind A/B placement seeded by item idx
  - first-line A/B regex parse, Qwen3-32B fallback for unclear/hedging
  - majority-3-of-5 default; ties or unclear → keep original (conservative)
  - GPT-4o is NEVER called inside this module
"""
from __future__ import annotations

import random
import re
from collections import Counter

from detection_format_bakeoff import build_chatml, vllm_gen, vllm_chat
from detection_d2 import q32_extract_text  # reuse plain-text fallback transport

VERDICT_SYS = "You are a medical expert comparing two answers to the same clinical question."

VERDICT_PROMPT = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Read the discharge summary and decide which answer better answers the question.
Evaluate both answers on three criteria:

1. CONSISTENCY — does the answer match the discharge notes (no contradictions)?
2. COMPLETENESS — does it cover ALL parts of the question? If the question asks
   about multiple visits, multiple conditions, multiple body parts, or events
   "before and after", the answer must cover ALL of them. An answer that omits
   half the question is worse than one that covers everything.
3. DIRECTNESS — does it answer the question without filler?

Pick the answer that is more consistent AND more complete. If both are equally
consistent and complete, pick the more direct one. **If one answer is missing
information the other has, the more complete answer wins**, even if the more
complete one is longer.

Reply on the FIRST line with exactly one letter: A  or  B
On the SECOND line, give one short sentence saying why."""

EXTRACT_AB = """/nothink
Read the text below. The author was asked to pick A or B.

TEXT:
{raw}

Did the author pick A or B? If they hedged or didn't pick clearly, reply UNCLEAR.

Reply with exactly one word: A, B, or UNCLEAR.
"""


_RE_FIRST_LETTER = re.compile(r"^[\s\-*>#]*([AaBb])\b")


def _parse_pick(raw: str) -> tuple[str, str]:
    """Returns (pick, parse_path). pick ∈ {'A','B','unclear'}."""
    if not raw:
        return "unclear", "unparseable"
    first = raw.strip().splitlines()[0] if raw.strip() else ""
    m = _RE_FIRST_LETTER.match(first)
    if m:
        return m.group(1).upper(), "regex"
    # Fallback: Qwen3-32B
    try:
        import requests
        from detection_format_bakeoff import QWEN32B_URL
        r = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Read carefully. Answer in one word only."},
                {"role": "user", "content": EXTRACT_AB.format(raw=raw[:1500])},
            ],
            "max_tokens": 20, "temperature": 0.0,
        }, timeout=60)
        text = r.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().upper()
        if text.startswith("A"):
            return "A", "q32_fallback"
        if text.startswith("B"):
            return "B", "q32_fallback"
        return "unclear", "q32_fallback"
    except Exception as e:
        print(f"  v2 verdict q32 err: {e}", flush=True)
        return "unclear", "unparseable"


def _seed_for(fold: int, idx: int) -> int:
    return 42 + (fold << 16) + idx


def run_verdict_v2(fold: int, idx: int, note: str, question: str,
                   original_answer: str, corrected_answer: str, *,
                   port: int, k: int = 5,
                   accept_threshold: int = 3) -> dict:
    """Pairwise A/B verdict; ties / unclear → keep original.

    accept_threshold = minimum number of votes (out of K) for the corrected
    answer's slot for the correction to be accepted. Default 3/5 majority.
    """
    rng = random.Random(_seed_for(fold, idx))
    orig_in_a = rng.random() > 0.5
    ans_a = original_answer if orig_in_a else corrected_answer
    ans_b = corrected_answer if orig_in_a else original_answer

    user = VERDICT_PROMPT.format(note=note, question=question,
                                 answer_a=ans_a[:1500], answer_b=ans_b[:1500])

    samples = []
    for _ in range(k):
        raw = vllm_chat(VERDICT_SYS, user, port, max_tokens=200, temperature=0.7)
        pick, path = _parse_pick(raw)
        # Second line for interpretation
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        reason = lines[1][:200] if len(lines) >= 2 else ""
        samples.append({"raw": raw, "pick": pick, "parse_path": path, "reason": reason})

    counts = Counter(s["pick"] for s in samples)
    n_valid = counts.get("A", 0) + counts.get("B", 0)
    if n_valid == 0:
        majority_pick = "UNCLEAR"
        majority_count = 0
    elif counts.get("A", 0) > counts.get("B", 0):
        majority_pick = "A"; majority_count = counts["A"]
    elif counts.get("B", 0) > counts.get("A", 0):
        majority_pick = "B"; majority_count = counts["B"]
    else:
        majority_pick = "TIE"; majority_count = counts.get("A", 0)

    # Map back to "is the corrected answer the one chosen?"
    corrected_slot = "B" if orig_in_a else "A"
    corrected_count = counts.get(corrected_slot, 0)
    accept = (corrected_count >= accept_threshold
              and majority_pick != "TIE"
              and majority_pick != "UNCLEAR"
              and (majority_pick == corrected_slot))

    # Length sanity gate: if the corrected answer is less than half the
    # original length and the original is non-trivially long, REJECT the
    # correction regardless of vote count. This catches the multi-part-
    # question coverage failure mode observed on idx=184: Qwen2.5 verdict
    # picks the more concise answer 5/5 even when explicitly told to value
    # completeness, so the gate fires on length alone, not on votes.
    length_sanity_failed = False
    short_threshold = 0.5
    if (accept
        and len(original_answer) > 200
        and len(corrected_answer) < short_threshold * len(original_answer)):
        accept = False
        length_sanity_failed = True

    return {
        "variant": "v2_pairwise",
        "k": k,
        "accept_threshold": accept_threshold,
        "orig_in_slot_A": orig_in_a,
        "samples": samples,
        "votes": dict(counts),
        "n_valid": n_valid,
        "majority_pick": majority_pick,
        "majority_count": majority_count,
        "corrected_slot": corrected_slot,
        "corrected_vote_count": corrected_count,
        "accept_correction": accept,
        "length_sanity_failed": length_sanity_failed,
        "len_original": len(original_answer),
        "len_corrected": len(corrected_answer),
    }
