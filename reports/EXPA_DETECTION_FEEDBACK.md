# Experiment A — Detection Feedback Form Does Not Drive Correction

Date: 2026-05-30
Base detector (shared across arms): k=3 natural compare at T=0.7, UNION flag. Sample: 96 wrong + 63 correct, real notes, c=8.
Output: `runs/expA_detection_feedback/qwen25_nw-1_nc50_seed42/`.

## Result

Three feedback builders, identical correction prompt, only the feedback block differs:

| builder | what it feeds correction | fix | break | net | fix-rate |
|---|---|---:|---:|---:|---:|
| cot | reflected/verified diagnosis | 23 | 8 | 15 | 24% |
| planner_verify | checklist confirm/contradict/silent diagnosis | 24 | 7 | 17 | 25% |
| planner_context | source evidence + target slot, NO verdict | 24 | 10 | 14 | 25% |

Shared detector: union recall 0.93 (best recall engine to date, vs natural 0.83), over-flag 0.79.

## Finding: the feedback form is not the lever

All three builders land at the same ~25% fix-rate (23/24/24 fixes). A verified diagnosis, a careful checklist, and pure context-without-verdict are interchangeable for correction success. Differences are in breaks (7–10), which is noise at this N.

Since all three arms share the same note + spans + flag and differ only in the feedback block, the conclusion is direct: **correction's fix-rate is set by the shared note + spans + the flag, not by the form of the detection feedback.**

## Reconciles the whole arc into three tiers

| Condition | fix-rate |
|---|---:|
| no feedback (Phase 1 baseline) | ~6% |
| any live feedback form + note + spans (Exp A) | ~25% |
| oracle-precise diagnosis (Phase 1 contradiction_quote) | ~60% |

- 6→25%: having *something* plus the note.
- 25→60%: diagnosis **correctness** (AGREE), NOT feedback form. All three live builders are equally imperfect (~13–17% AGREE), so all plateau at 25%.

## Decisions this sets

1. **Stop optimizing detection feedback form.** cot / planner_verify / planner_context are interchangeable for fix-rate; pick planner_verify (marginal best net) or the cheapest.
2. **k=3 union = the flag engine.** Recall 0.93. Lock it for the flag decision; pair with a precision/verdict gate for the over-flag (0.79).
3. **The real lever is diagnosis correctness (→ B1) and input form (→ B2).** A removes "feedback form" from the search space and points at correctness and shared-input.

## Reframes B1 and B2

- **B1 (next):** the 25→60% gap is about diagnosis *correctness* on the right flaw. B1 hands correction the oracle-correct critical flaw vs a lesser flaw vs all — tests whether hitting the critical flaw correctly is the unlock.
- **B2:** since note + spans do the work, B2's full-note vs summary vs error+summary tests which shared input drives the base 25%.
