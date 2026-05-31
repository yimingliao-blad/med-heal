# What Does Correction Actually Need — Research Plan

Date: 2026-05-30
Motivation: Phase 1/2b showed a paradox — fix-rate (25%) is higher than the AGREE rate (13%), i.e. correction often fixes even from a PARTIAL diagnosis. So "how much does the diagnosis need to be right for correction to work?" is unanswered, and it determines everything detection must produce. This plan decomposes the question into oracle-driven experiments.

## The central question

A wrong answer can have several flaws. We do not know what the correction step actually needs:

- **Any-one-correct-flaw**: pointing at any single real flaw is enough for correction to fix the answer.
- **Critical-flaw**: only the single most important flaw enables the fix; lesser flaws don't.
- **All-flaws**: correction needs the complete set of flaws.

This is the user's question: *"is one correct flaw enough for correction to make the work, or must it be the critical one?"* The answer dictates the detection target:
- if any-one → detection just needs high recall on *some* real flaw (natural's strength).
- if critical → detection must *rank/prioritize* flaws (much harder).
- if all → detection must be *complete*.

## Experiment B1 — oracle-ranked error list (the foundational test)

For each wrong case, use GPT-4o to decompose the answer's errors into a **ranked list** (most critical → least), each with its note evidence. Then run correction three ways and compare fix-rate:

| Arm | Correction is given… |
|---|---|
| critical-only | only the #1 (most critical) error |
| lesser-only | only a lower-ranked (non-critical) error |
| all-errors | the full ranked list |

Decision read:
- critical-only ≈ all-errors, and >> lesser-only → correction needs **the critical flaw**; detection must prioritize.
- critical-only ≈ lesser-only ≈ all → **any one real flaw** suffices; detection just needs recall.
- all >> critical-only → correction needs **completeness**.

This is oracle-driven (no detection dependency), so it is clean and fast. It is the first experiment because it defines the detection target.

## Experiment B2 — correction input form

Independently, what evidence should correction receive? Compare fix-rate across:

| Arm | Correction input (besides the prior answer) |
|---|---|
| full-note | the whole discharge note |
| error-only | the detection error description (by type) |
| q-summary | a summary of only the question-relevant note content |
| error + q-summary | both |

Read: which input form gives correction the most fixes with the fewest breaks. Tests whether a focused question-summary beats the raw note (the long-note attention issue from the root-cause analysis), and whether the error description adds on top.

## Experiment A — detection feedback quality (which detector feeds correction best)

Once B1 tells us the target, compare ways of *generating* the feedback. Base: **k=3 greedy natural** error-finding (the recall engine). Then three downstream feedback builders:

| Feedback builder | What it does |
|---|---|
| cot | reflects/reconsiders the k=3 findings step by step, verifying which are correct |
| planner-verify | the existing planner: checklist + per-item confirm/contradict/silent (careful verifier) |
| planner-context | NEW — given natural already flagged it, this planner does NOT re-find what's wrong; it gathers the **question-relevant context** the correction needs (the slot, the relevant note facts), providing correction *context* rather than a verdict |

The key distinction the user drew: cot and planner-verify both *reconsider whether the finding is correct*; planner-context instead *builds the correction's working context from the question*, taking natural's flag decision as given. Compare which produces feedback that yields the best downstream fix-rate.

## How the pieces connect

- B1 defines WHAT correction needs (any/critical/all flaw).
- B2 defines the best INPUT FORM for correction (note/error/summary).
- A defines how to GENERATE that feedback live (cot vs planner-verify vs planner-context), once we know the target from B1.

"Correction is a correct signal, or it must direct the point it can fix" — B1 measures exactly that: does a generic correct signal suffice, or must the feedback pinpoint the critical fixable point.

## Proposed order

1. **B1 (oracle-ranked errors)** — foundational, oracle-driven, decisive. Defines the detection target. Start here.
2. **B2 (correction input form)** — oracle-driven, independent of detection. Can run alongside B1.
3. **A (detection feedback builders)** — built to hit the target B1 reveals; compares cot / planner-verify / planner-context on downstream fix-rate.

All on Qwen2.5 wrong cases, c=8, full LLM ledger, via the pipeline driver so stages auto-advance with gates.

## Open clarifications before building

1. B1 ranking: GPT-4o ranks errors by *clinical criticality for answering the question*, or by *how wrong the answer is*? (I'll use "criticality for answering the exact question" unless you prefer otherwise.)
2. "k=3 greedy natural" — greedy = T=0 (deterministic, 3 identical) doesn't make sense; I read it as k=3 at low temp taking the UNION of distinct errors found (recall engine). Confirm.
3. planner-context — confirm its job is: take natural's flag as final, output the question-required slot + the relevant note facts for correction, NOT a second verdict.
