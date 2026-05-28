"""Verbatim corrector prompts for Stage IV Track-Loc (locator-guided correction).

Per Plan Slot 11, Implementation Discipline Rule 5: canonical text at call site.
Plan: https://app.notion.com/p/3516be46cf3c818495a7f3ed974c78d1
"""
from __future__ import annotations


# ============================================================
# Corrector v1 — simple revise instruction with structured contradiction info
# Phase A input: GPT-4o gold narrative (CLAIM/CONTRADICTION/SECTION)
# Phase B input: locator's TOP-1 candidate (same format)
# ============================================================
CORRECTOR_V1_SYSTEM = (
    "You are a medical expert revising a clinical answer. The previous answer is "
    "known to be incorrect; you have been told the specific contradiction. Produce "
    "a revised answer that resolves the contradiction by stating what the discharge "
    "notes actually say."
)

CORRECTOR_V1_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

PREVIOUS ANSWER (known to be incorrect):
{zs_answer}

IDENTIFIED CONTRADICTION:
{contradiction_narrative}

Produce a revised answer that resolves this contradiction by stating what the discharge notes actually say. Output the revised answer directly, without preamble or explanation. Be specific and complete.

Revised answer:"""


# ============================================================
# T0.a baseline — bare-regen, no contradiction info (control)
# ============================================================
T0A_SYSTEM = "You are a medical expert answering questions about discharge summaries."

T0A_USER_TMPL = """DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

Answer:"""


# ============================================================
# Corrector v2 — structured swap (uses parsed CLAIM / CONTRADICTION / SECTION)
# Tests whether explicit "replace X with Y" framing improves over v1's free-form revise.
# ============================================================
CORRECTOR_V2_SYSTEM = (
    "You are a medical expert revising a clinical answer. The previous answer is "
    "known to be incorrect; you have been told the specific incorrect claim and "
    "what the discharge notes actually say. Produce a revised answer that replaces "
    "the incorrect claim with the correct fact while preserving anything else in "
    "the previous answer that is unrelated to the contradiction."
)

CORRECTOR_V2_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

PREVIOUS ANSWER (known to be incorrect):
{zs_answer}

INCORRECT CLAIM in the previous answer:
{contradiction_claim}

WHAT THE NOTES ACTUALLY STATE:
{contradiction_truth}

LOCATION IN NOTES:
{contradiction_section}

Produce a revised answer that:
1. Replaces the incorrect claim with what the notes actually state.
2. Preserves any portions of the previous answer that are correct and unrelated to the contradiction.
3. Reads as a complete, coherent answer to the question.

Output the revised answer directly, without preamble or explanation. Be specific and complete.

Revised answer:"""


# ============================================================
# Corrector v3 — structured fields, no "preserve" instruction.
# Tests whether v2's regression was caused by the "preserve correct portions"
# instruction (which may have kept wrong content from zs_answer).
# ============================================================
CORRECTOR_V3_SYSTEM = (
    "You are a medical expert producing a clinical answer. The previous answer "
    "is wrong; you have been told the specific incorrect claim and what the "
    "discharge notes actually say. Produce a correct answer based on the notes."
)

CORRECTOR_V3_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

PREVIOUS (INCORRECT) ANSWER:
{zs_answer}

INCORRECT CLAIM in the previous answer:
{contradiction_claim}

WHAT THE NOTES ACTUALLY STATE:
{contradiction_truth}

LOCATION IN NOTES:
{contradiction_section}

Produce the correct answer to the question using the discharge notes. Output the answer directly, without preamble or explanation. Be specific and complete.

Answer:"""


# ============================================================
# Corrector v4 — answer-focused framing (not error-focused).
# Hypothesis: v1's "previous answer is wrong" framing anchors the corrector
# on the wrong answer. v4 reframes as "answer the question, here's relevant
# clinical context" — corrector treats contradiction info as supplementary
# evidence, not as "what to fix".
# ============================================================
CORRECTOR_V4_SYSTEM = (
    "You are a medical expert answering a clinical question about a patient. "
    "You have the discharge notes, the question, a previous candidate answer, "
    "and an analysis of where that candidate diverges from the notes. Use all "
    "of this to produce the correct answer."
)

CORRECTOR_V4_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

A PREVIOUS CANDIDATE ANSWER:
{zs_answer}

ANALYSIS OF WHERE THE CANDIDATE DIVERGES FROM THE NOTES:
{contradiction_narrative}

Based on the discharge notes and the analysis, write the correct answer to the question. Output only the answer, without preamble. Be specific and complete.

Answer:"""


# ============================================================
# Corrector v5 — explicit two-step CoT corrector.
# Hypothesis: forcing reasoning before final answer might help; though
# Magistral M5 (CoT for binary judging) regressed, generation tasks may
# benefit from explicit decomposition. Worth testing.
# ============================================================
CORRECTOR_V5_SYSTEM = (
    "You are a medical expert revising a clinical answer. Reason step-by-step "
    "before producing the final answer."
)

CORRECTOR_V5_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

PREVIOUS ANSWER (incorrect):
{zs_answer}

IDENTIFIED CONTRADICTION:
{contradiction_narrative}

Step 1: Identify the specific factual error in the previous answer.
Step 2: State what the discharge notes actually say about that fact.
Step 3: Write the corrected answer.

Output format:
STEP 1: <error>
STEP 2: <correct fact from notes>
STEP 3 (FINAL ANSWER): <revised answer>"""


# ============================================================
# Corrector v6 — surgical edit pattern.
# Hypothesis: explicit "output the answer with X replaced by Y" preserves
# correct content of zs while fixing the contradiction. v2 had a similar
# preserve hint and regressed; v6 makes the surgical pattern explicit
# rather than a soft hint.
# ============================================================
CORRECTOR_V6_SYSTEM = (
    "You are a medical expert performing a surgical edit on a clinical answer. "
    "You will replace the incorrect statement with the correct fact while "
    "leaving the rest of the answer unchanged."
)

CORRECTOR_V6_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

PREVIOUS ANSWER:
{zs_answer}

IDENTIFIED CONTRADICTION:
{contradiction_narrative}

Produce a revised answer by:
1. Locating the incorrect statement in the previous answer.
2. Replacing it with what the discharge notes actually say.
3. Preserving every other word of the previous answer verbatim.

Do not rewrite. Do not paraphrase the correct portions. Only change the contradiction.

Revised answer:"""


# ============================================================
# Corrector v7 — context-only framing (no flagging that prior answer was wrong).
# Hypothesis: telling the corrector the answer is "wrong" induces over-correction.
# v7 supplies the contradiction as raw clinical context, no judgement language.
# ============================================================
CORRECTOR_V7_SYSTEM = (
    "You are a medical expert answering a clinical question using the discharge "
    "notes and any additional clinical context provided."
)

CORRECTOR_V7_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

ADDITIONAL CLINICAL CONTEXT (relevant facts from the notes):
{contradiction_narrative}

PRIOR DRAFT ANSWER (for reference):
{zs_answer}

Write the answer to the question. Be specific and complete. Output only the answer.

Answer:"""


# Registered versions (extended during iteration)
CORRECTOR_VERSIONS = {
    "v1": (CORRECTOR_V1_SYSTEM, CORRECTOR_V1_USER_TMPL),
    "v2": (CORRECTOR_V2_SYSTEM, CORRECTOR_V2_USER_TMPL),
    "v3": (CORRECTOR_V3_SYSTEM, CORRECTOR_V3_USER_TMPL),
    "v4": (CORRECTOR_V4_SYSTEM, CORRECTOR_V4_USER_TMPL),
    "v5": (CORRECTOR_V5_SYSTEM, CORRECTOR_V5_USER_TMPL),
    "v6": (CORRECTOR_V6_SYSTEM, CORRECTOR_V6_USER_TMPL),
    "v7": (CORRECTOR_V7_SYSTEM, CORRECTOR_V7_USER_TMPL),
    "t0a": (T0A_SYSTEM, T0A_USER_TMPL),
}
