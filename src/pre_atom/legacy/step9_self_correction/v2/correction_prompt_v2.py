#!/usr/bin/env python3
"""
Correction prompt V2 (Step D).

Implements the redesign that came out of the literature review:

  - Factored CoVe: the previous wrong answer is NOT shown to the regeneration
    step. The regeneration starts from a clean context (Dhuliawala et al.,
    ACL Findings 2024). This is the single most important change vs v1.

  - Quote-first / verbatim grounding: the model is forced to copy verbatim
    sentences from the note BEFORE writing the conclusion. The token-level
    copying breaks the prior-answer trajectory (Sivarajkumar 2025).

  - Premises → Derivation → Conclusion structured output: the model commits
    to evidence-grounded premises before sampling the conclusion token (Wu
    et al. 2024 "Key Condition Verification" + practitioner literature on
    structured decoding).

  - Optional contrast example from the BM contrast pool (built in Step C).
    Used as a worked transformation example, not a semantic match.

  - Post-hoc verbatim-quote verifier: every "evidence:" line in the model's
    output is checked against the actual note (case-insensitive,
    whitespace-tolerant). Hallucinated quotes are flagged so the verdict
    step can take low-confidence ones into account.

Usage:
    from correction_prompt_v2 import build_v2_prompt, parse_premises_output, verify_evidence_quotes

    prompt = build_v2_prompt(note, question, retrieved_spans, contrast_example=None)
    raws = generate_corrections(prompt, port=8003, k=3, temperature=0.7)
    for raw in raws:
        parsed = parse_premises_output(raw)
        verified = verify_evidence_quotes(parsed["evidence_quotes"], note)
"""
from __future__ import annotations

import re


# ---------- The prompt template ----------

V2_TEMPLATE = """You are a medical expert reading a patient's discharge note. Answer the question using facts from the note.

Discharge note:
{note}

Question: {question}

A retrieval system has highlighted the following sentences as likely relevant.
They are HINTS, not constraints. Read the WHOLE note and use any sentence that
bears on the question — multiple admissions, results, medication changes,
procedures, diagnoses. Do not feel limited to the highlighted sentences.

Highlighted sentences (hints):
{spans_block}
{contrast_block}
Write your answer in this format:

EVIDENCE:
- "<verbatim quote 1 from the note>"
- "<verbatim quote 2 from the note>"
- "<verbatim quote 3 from the note>"
(list as many quotes as you need — 2 to 6 — covering ALL the facts your answer relies on)

ANSWER:
<your final answer to the question, 1-3 sentences, written naturally>

Rules:
- Each EVIDENCE line must be a verbatim quote from the note in double quotes.
- The ANSWER must be supported by the quotes you listed.
- If a question asks about multiple visits or events, cover all of them in the quotes.
- If the note does not contain the information, say so explicitly in the ANSWER."""


def render_spans_block(spans: list[dict]) -> str:
    if not spans:
        return "(no spans pre-selected; read the full note carefully)"
    lines = []
    for i, s in enumerate(spans, 1):
        sentence = s.get("sentence", "")
        # The score may be a vote_fraction (0..1) or a cosine sim (0..1) or
        # a sum-of-positives (>1). Show whichever the retriever gave us.
        score = s.get("similarity", s.get("score", 0.0))
        source = s.get("source", "")
        tag = f" [{source}]" if source else ""
        lines.append(f"  {i}.{tag} \"{sentence}\"  (relevance {score:.2f})")
    return "\n".join(lines)


def render_contrast_block(contrast_example: dict | None) -> str:
    """Render a worked example from the BM contrast pool. Optional."""
    if not contrast_example:
        return ""
    return (
        "\nWorked example for a similar clinical question (use only as a stylistic"
        " guide; the facts below are about a DIFFERENT patient):\n"
        f"  Question: {contrast_example.get('question', '')}\n"
        f"  Wrong answer (do NOT imitate): {contrast_example.get('wrong_answer', '')[:300]}\n"
        f"  What was wrong: {contrast_example.get('what_was_wrong', '')}\n"
        f"  Correct answer: {contrast_example.get('ground_truth', '')[:300]}\n"
    )


def build_v2_prompt(note: str, question: str, spans: list[dict],
                    contrast_example: dict | None = None) -> str:
    return V2_TEMPLATE.format(
        note=note,
        question=question,
        spans_block=render_spans_block(spans),
        contrast_block=render_contrast_block(contrast_example),
    )


# ---------- Output parser ----------

# Bullet evidence quotes (EVIDENCE: section). Match: -, *, or numbered, then a
# quoted string (straight or curly quotes).
_EVIDENCE_RE = re.compile(
    r'^\s*[-*\u2022]\s*["\u201c](?P<quote>.+?)["\u201d]\s*$',
    re.MULTILINE,
)
# Plain quoted lines (no bullet) inside the EVIDENCE section
_EVIDENCE_NOBULLET_RE = re.compile(
    r'^\s*["\u201c](?P<quote>.+?)["\u201d]\s*$',
    re.MULTILINE,
)
# ANSWER section: everything after the first "ANSWER:" tag (also accept
# legacy "CONCLUSION:" tag for backward compatibility with older outputs)
_ANSWER_RE = re.compile(
    r'(?:ANSWER|CONCLUSION)\s*:?\s*(?P<answer>.+?)$',
    re.IGNORECASE | re.DOTALL,
)
_EVIDENCE_HEADER_RE = re.compile(r'EVIDENCE\s*:?', re.IGNORECASE)


def parse_premises_output(raw: str) -> dict:
    """Parse the EVIDENCE / ANSWER format.

    Returns:
        {
          'premises': [{'claim': '', 'quote': str}, ...]   # legacy field name
          'evidence_quotes': [str, ...],
          'conclusion': str,                                # legacy field name (now ANSWER)
          'answer': str,                                    # alias
          'parse_ok': bool,
        }
    """
    if not raw:
        return {"premises": [], "evidence_quotes": [], "conclusion": "",
                "answer": "", "parse_ok": False}

    # Split into evidence section and answer section
    answer_match = _ANSWER_RE.search(raw)
    if answer_match:
        evidence_zone = raw[:answer_match.start()]
        answer = answer_match.group("answer").strip()
    else:
        evidence_zone = raw
        answer = ""

    # Strip the EVIDENCE: header line if present
    header = _EVIDENCE_HEADER_RE.search(evidence_zone)
    if header:
        evidence_zone = evidence_zone[header.end():]

    # Extract quoted strings from evidence_zone
    quotes: list[str] = []
    for m in _EVIDENCE_RE.finditer(evidence_zone):
        quotes.append(m.group("quote").strip())
    if not quotes:
        for m in _EVIDENCE_NOBULLET_RE.finditer(evidence_zone):
            quotes.append(m.group("quote").strip())

    premises = [{"claim": "", "quote": q} for q in quotes]
    parse_ok = bool(quotes and answer)
    return {
        "premises": premises,
        "evidence_quotes": quotes,
        "conclusion": answer,  # legacy alias
        "answer": answer,
        "parse_ok": parse_ok,
    }


# ---------- Verbatim quote verifier ----------

def verify_evidence_quotes(quotes: list[str], note: str) -> list[bool]:
    """Check whether each quoted string is actually present in the note
    (case-insensitive, whitespace-tolerant).
    """
    if not note:
        return [False] * len(quotes)
    norm_note = re.sub(r"\s+", " ", note).lower()
    out = []
    for q in quotes:
        norm_q = re.sub(r"\s+", " ", q).lower().strip()
        # Strip trailing punctuation that often differs
        norm_q = norm_q.rstrip(".,:;")
        if not norm_q:
            out.append(False)
            continue
        # Try direct substring; if that fails, try without trailing
        # ellipsis-style truncation
        if norm_q in norm_note:
            out.append(True)
        else:
            # Allow partial: at least 60 chars in common starting from the
            # quote's start
            if len(norm_q) >= 30:
                head = norm_q[:int(len(norm_q) * 0.6)]
                out.append(head in norm_note)
            else:
                out.append(False)
    return out


# ---------- Self-test ----------

if __name__ == "__main__":
    # Build a synthetic example to verify parsing + verification
    note = """[Note 1]
Patient was admitted with venous thoracic outlet syndrome. She underwent
left first rib resection. She tolerated the procedure well. After transfer
to the floor, her Foley was discharged at midnight on POD0. She was unable
to void and was bladder scanned showing 700cc of retained urine. She was
straight cathed in the morning of POD1. The JP drain was removed and she
was then able to void after ambulating. Discharged in stable condition."""
    spans = [
        {"sentence": "She was unable to void and was bladder scanned showing 700cc of retained urine.",
         "similarity": 1.00, "source": "R3_llm_cite"},
        {"sentence": "She was straight cathed in the morning of POD1.",
         "similarity": 1.00, "source": "R3_llm_cite"},
        {"sentence": "The JP drain was removed and she was then able to void after ambulating.",
         "similarity": 1.00, "source": "R3_llm_cite"},
    ]
    question = "What was the main postop complication from the infraclavicular first rib resection procedure?"
    p = build_v2_prompt(note, question, spans)
    print(p)
    print()
    print("=" * 60)
    fake_output = """PREMISES:
1. The patient could not void after surgery — evidence: "She was unable to void and was bladder scanned showing 700cc of retained urine."
2. She required catheterization to relieve the retention — evidence: "She was straight cathed in the morning of POD1."
3. Voiding eventually returned after JP drain removal — evidence: "The JP drain was removed and she was then able to void after ambulating."

CONCLUSION:
The main postoperative complication was urinary retention, which required straight catheterization and resolved after the JP drain was removed."""
    parsed = parse_premises_output(fake_output)
    print(f"premises: {len(parsed['premises'])}")
    for pm in parsed["premises"]:
        print(f"  - {pm['claim']}")
        print(f"    quote: {pm['quote']}")
    print(f"conclusion: {parsed['conclusion']}")
    print(f"parse_ok: {parsed['parse_ok']}")
    verified = verify_evidence_quotes(parsed["evidence_quotes"], note)
    print(f"verified: {verified}  ({sum(verified)}/{len(verified)})")
