"""Verbatim prompts for the Error Location pipeline.

Per [Workflow] Complete Plan Acceptance Criteria Slot 11 + Implementation Discipline Rule 5:
the canonical text appears at call site (here) AND verbatim in the plan page.

Plan page: 3506be46-cf3c-817e-99f0-fa38d288e0bd
"""
from __future__ import annotations

# ============================================================
# 11a. GPT-4o gold narrative — Step 1, one-time labeling.
# Has access to ground truth; produces the contradiction reference.
# ============================================================
GOLD_SYSTEM = (
    "You are a medical expert analyzing a clinical Q&A. You have access to "
    "the discharge notes, the question, the correct answer, and a model's "
    "wrong answer. Your task is to identify the single most direct contradiction "
    "between the model's answer and the discharge notes."
)

GOLD_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S WRONG ANSWER:
{zs_answer}

Identify the single most direct contradiction between the model's answer and the discharge notes. Output exactly:

CLAIM: <one sentence stating what the model answer asserts that is wrong>
CONTRADICTION: <one sentence quoting or paraphrasing what the note actually says>
SECTION: <which discharge note (Note 1 / 2 / 3) and which clinical section>

Be specific and concrete. Cite exact phrasing from the note where possible."""


# ============================================================
# 11b. Qwen2.5-7B locator v1 — Step 3, blind (no GT).
# ============================================================
LOCATOR_V1_SYSTEM = (
    "You are a medical expert reviewing a clinical Q&A for errors. "
    "The provided answer may contain a contradiction with the discharge notes. "
    "Identify it."
)

LOCATOR_V1_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

ANSWER UNDER REVIEW:
{zs_answer}

The answer above may contradict the discharge notes. Identify the single most direct contradiction. Output exactly:

CLAIM: <one sentence stating what the answer asserts that is contradicted by the notes>
CONTRADICTION: <one sentence quoting or paraphrasing what the notes actually say>
SECTION: <which discharge note (Note 1 / 2 / 3) and which clinical section>

If you cannot find any contradiction, output exactly: NO CONTRADICTION FOUND."""


# ============================================================
# 11c. Qwen3-235B-MLX comparator — Step 3-4, judges narrative match.
# ============================================================
COMPARATOR_SYSTEM = (
    "You are a medical expert comparing two narratives that describe what is "
    "wrong with a clinical answer."
)

COMPARATOR_USER_TMPL = """GOLD NARRATIVE (produced with knowledge of the correct answer):
{gold_narrative}

MODEL NARRATIVE (produced blind, without the correct answer):
{model_narrative}

Do these two narratives point to the same contradiction? Consider:
- Do they describe the same incorrect claim?
- Do they cite the same supporting fact from the discharge notes?

Reply with ONLY:
MATCH: YES / NO
REASON: <one sentence>"""


# ============================================================
# 11d. GPT-4o spot-check judge — Step 5, validates the comparator.
# Identical semantics to 11c.
# ============================================================
SPOT_CHECK_SYSTEM = COMPARATOR_SYSTEM
SPOT_CHECK_USER_TMPL = COMPARATOR_USER_TMPL


# ============================================================
# 11b-v2. Locator v2 — question-anchored.
# v1 result on fold_0: 3/27 = 11.1% (failure threshold 25%).
# Failure mode: locator picks a different contradiction than gold because
# it doesn't know the GT-relevant aspect. v2 anchors the search on the
# question (the answer is known wrong; find the contradiction most relevant
# to the question being asked).
# ============================================================
LOCATOR_V2_SYSTEM = (
    "You are a medical expert reviewing a clinical Q&A. The answer is known to be "
    "incorrect. Your task is to identify the specific contradiction with the discharge "
    "notes that explains why the answer is wrong on this particular question."
)

LOCATOR_V2_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

ANSWER UNDER REVIEW (known to be incorrect):
{zs_answer}

The answer above is wrong. Find the single most direct contradiction with the discharge notes that explains the error — i.e., the part of the answer that, if corrected, would make the answer to the question correct.

Output exactly:

CLAIM: <one sentence stating the specific incorrect statement in the answer that prevents it from correctly answering the question>
CONTRADICTION: <one sentence quoting or paraphrasing what the notes actually say about that aspect>
SECTION: <which discharge note (Note 1 / 2 / 3) and which clinical section>

If after careful review you cannot find any contradiction relevant to the question, output exactly: NO CONTRADICTION FOUND."""


# ============================================================
# 11b-v3. Locator v3 — same prompt as v2, format-first reinforced.
# Tested with thinking DISABLED on Qwen3-8B per 2026-04-29 user direction:
# v2 + Qwen3-8B-think hit 11% inherent truncation on items where thinking
# exceeds 32K native context. v3 tests whether Qwen3-8B base (no CoT) can
# produce decent narratives without the truncation risk.
# ============================================================
LOCATOR_V3_SYSTEM = LOCATOR_V2_SYSTEM

LOCATOR_V3_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

ANSWER UNDER REVIEW (known to be incorrect):
{zs_answer}

The answer above is wrong. Find the single most direct contradiction with the discharge notes that explains the error.

Output the structured answer immediately, in this exact format:

CLAIM: <one sentence stating the specific incorrect statement in the answer that prevents it from correctly answering the question>
CONTRADICTION: <one sentence quoting or paraphrasing what the notes actually say about that aspect>
SECTION: <which discharge note (Note 1 / 2 / 3) and which clinical section>

Begin your response with CLAIM: directly. If after careful review you cannot find any contradiction relevant to the question, output exactly: NO CONTRADICTION FOUND."""


# Locator versions (registered by version key for iteration)
# ============================================================
# 11b-v4. Locator v4 — multi-hypothesis (top-3 candidates).
# v3 fold_0 plateau at 24.8% pooled → relax the strict-match constraint:
# emit 3 candidate contradictions; comparator checks if gold matches ANY.
# Tests whether locator FINDS the right contradiction but RANKS it wrong.
# ============================================================
LOCATOR_V4_SYSTEM = (
    "You are a medical expert reviewing a clinical Q&A. The answer is known to be "
    "incorrect. Your task is to identify the top 3 candidate contradictions with the "
    "discharge notes, ranked by relevance to the question."
)

LOCATOR_V4_USER_TMPL = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

ANSWER UNDER REVIEW (known to be incorrect):
{zs_answer}

The answer above is wrong. List the top 3 candidate contradictions with the discharge notes, ranked by relevance to the question (most relevant first).

Output exactly:

CANDIDATE 1:
CLAIM: <what the answer asserts that is wrong>
CONTRADICTION: <what the notes actually say>
SECTION: <which note + clinical section>

CANDIDATE 2:
CLAIM: <...>
CONTRADICTION: <...>
SECTION: <...>

CANDIDATE 3:
CLAIM: <...>
CONTRADICTION: <...>
SECTION: <...>

Begin your response with CANDIDATE 1: directly. If you cannot find 3 distinct contradictions, fill remaining slots with NO MORE CONTRADICTIONS FOUND."""


LOCATOR_VERSIONS = {
    "v1": (LOCATOR_V1_SYSTEM, LOCATOR_V1_USER_TMPL),
    "v2": (LOCATOR_V2_SYSTEM, LOCATOR_V2_USER_TMPL),
    "v3": (LOCATOR_V3_SYSTEM, LOCATOR_V3_USER_TMPL),
    "v4": (LOCATOR_V4_SYSTEM, LOCATOR_V4_USER_TMPL),
    # v5: same prompt as v4, used with --enable-thinking for deeper reasoning.
    # Hypothesis: thinking helps catch conceptual contradictions (e.g., Hydromorphone =
    # Dilaudid same drug) and wrong-direction errors that v4 no-think misses.
    "v5": (LOCATOR_V4_SYSTEM, LOCATOR_V4_USER_TMPL),
}


# ============================================================
# 11c-v4. Comparator for v4 — checks if gold matches ANY candidate.
# ============================================================
COMPARATOR_V4_SYSTEM = COMPARATOR_SYSTEM
COMPARATOR_V4_USER_TMPL = """GOLD NARRATIVE (produced with knowledge of the correct answer):
{gold_narrative}

MODEL CANDIDATES (3 candidate contradictions, ranked by model's confidence):
{model_narrative}

Does the GOLD narrative point to the same contradiction as ANY of the model's 3 candidates? Consider:
- Does any candidate describe the same incorrect claim as gold?
- Does any candidate cite the same supporting fact from the discharge notes as gold?

Reply with ONLY:
MATCH: YES / NO
WHICH_CANDIDATE: <1, 2, 3, or NONE>
REASON: <one sentence>"""
