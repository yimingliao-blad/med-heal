"""Verdict prompts — 5 V-variants for Round 0 iteration.

V1-V3 inherited verbatim from src/step9_self_correction/error_taxonomy/test_correction_verdict.py
(the step9 final-pipeline winners: V1>V2>V3 on Qwen2.5 with 0 FP breaks per master plan).
V4-V5 fresh per master plan Stage III spec.

All output the same parseable token: PICK: A | B | UNCERTAIN.
Bare format per user 2026-04-30 (Q6 = no reasoning, matches Magistral M4 pattern).

Position randomization (A/B labels) is applied per-item via seed=42 in the runner;
candidates are NEVER labeled "original"/"corrected" in the prompt (bias guard).
"""
from __future__ import annotations


VERDICT_SYSTEM = (
    "You are a medical expert evaluating two candidate answers to a clinical question. "
    "Pick the candidate that is correct based on the discharge notes."
)


# ============================================================
# V1 — Which-is-better (master plan's defending champion).
# Step9's V1 was actually contra-count; this V1 is a clean which-is-better baseline.
# ============================================================
V1_WHICH_BETTER = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

CANDIDATE A:
{answer_a}

CANDIDATE B:
{answer_b}

Which candidate is the correct answer to the question, based on the discharge notes?

Output exactly one of:
PICK: A
PICK: B
PICK: UNCERTAIN"""


# ============================================================
# V2 — Count contradictions (step9 V1, verbatim except output format)
# ============================================================
V2_CONTRA_COUNT = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

CANDIDATE A:
{answer_a}

CANDIDATE B:
{answer_b}

Count how many factual claims in each candidate CONTRADICT the discharge notes. Different wording for the same fact is NOT a contradiction.

Output exactly one of:
PICK: A
PICK: B
PICK: UNCERTAIN"""


# ============================================================
# V3 — Principle-based three criteria (step9 V2, output format normalized)
# ============================================================
V3_PRINCIPLE = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

CANDIDATE A:
{answer_a}

CANDIDATE B:
{answer_b}

Compare on three criteria:
1. Which better addresses what the question specifically asks?
2. Which has fewer factual conflicts with the discharge notes?
3. Which better covers the critical information needed to answer the question?

Output exactly one of:
PICK: A
PICK: B
PICK: UNCERTAIN"""


# ============================================================
# V4 — Evidence-trace (master plan V3): cite a note line for each candidate
# ============================================================
V4_EVIDENCE_TRACE = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

CANDIDATE A:
{answer_a}

CANDIDATE B:
{answer_b}

For each candidate, identify whether the discharge notes contain direct evidence supporting it. Then pick the candidate whose claims are best supported by the notes.

Output exactly one of:
PICK: A
PICK: B
PICK: UNCERTAIN"""


# ============================================================
# V5 — Blind-pick (master plan V5): minimal framing
# ============================================================
V5_BLIND_PICK = """DISCHARGE NOTES:
{note}

QUESTION:
{question}

CANDIDATE A:
{answer_a}

CANDIDATE B:
{answer_b}

Based on the discharge notes, output exactly one of:
PICK: A
PICK: B
PICK: UNCERTAIN"""


# (system, template, max_gen) — per-V budget. V3 needs more because its prompt
# enumerates 3 criteria, prompting reasoning before the PICK line. Others
# emit bare PICK so 50 tokens is enough. DS-R1 (always-think) requires
# separate model-level override, set in pipeline.VERDICT_MODELS.
VERDICT_VERSIONS = {
    "v1": (VERDICT_SYSTEM, V1_WHICH_BETTER, 256),
    "v2": (VERDICT_SYSTEM, V2_CONTRA_COUNT, 256),
    "v3": (VERDICT_SYSTEM, V3_PRINCIPLE, 256),
    "v4": (VERDICT_SYSTEM, V4_EVIDENCE_TRACE, 256),
    "v5": (VERDICT_SYSTEM, V5_BLIND_PICK, 256),
}
