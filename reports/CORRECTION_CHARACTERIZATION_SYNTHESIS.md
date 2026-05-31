# Correction Characterization — Synthesis (A, B1, B1b, B2)

Date: 2026-05-30
Model: Qwen2.5-7B. All on the 109 wrong cases (+50 correct where breaks matter), real notes (24k), GPT-4o judge, c=8, full LLM ledgers.

## The four experiments

| Exp | Question | Result | Conclusion |
|---|---|---|---|
| A | Does the detection feedback FORM matter? | cot 24% / planner_verify 25% / planner_context 25% fix-rate | Form is irrelevant. |
| B1 | Which flaw does correction need (oracle-ranked)? | critical 66.7% / lesser 40.9% / all 71.0% | Needs THE critical flaw; completeness adds only +4pp. |
| B1b | Can a live prompt PRIORITIZE to the critical flaw? | baseline-all 31.6% ≈ best prio 30.5% (tight correction) | Prioritization is not the lever — live candidates lack the correct critical flaw. |
| B2 | Does correction need the note / summary / error-only? | full_note 64.1% / summary 60.9% / error_only 63.0% | The note is redundant given a correct error+evidence. |

## The single conclusion

The whole self-correction problem reduces to **producing one correct, evidence-backed statement of the critical error.** Given that statement:

- correction fixes ~63–67% (B1 critical, B2 error_only),
- it needs nothing else — not the note, not retrieved spans, not a summary, not a particular feedback form, not a prioritization step.

Everything downstream of the diagnosis is solved. The entire remaining difficulty is upstream: **detection's ability to state the correct critical error.**

## Why the live pipeline plateaus at ~25–31%

Qwen2.5 produces a correct critical-error statement only ~13–17% of the time (the AGREE rate measured in Phase 2/2b). When it states the critical error correctly → correction fixes it; when it states a lesser or wrong flaw → mostly fails (lesser-only is 41%, wrong is ~0). So live fix-rate ≈ the rate at which detection happens to state the correct critical error. The 31%→63% gap is **100% detection diagnosis correctness** — confirmed by elimination:

- not feedback form (A),
- not flaw completeness (B1: all ≈ critical),
- not flaw selection / prioritization (B1b),
- not correction input form (B2).

## What this rules in for the next step

The only lever left is raising the rate at which detection states the **correct critical error with evidence**. Concretely:

1. **Stronger detector test (highest value):** can GPT-4o-mini (or GPT-4o) — WITHOUT ground truth, given only note+question+answer — produce the correct critical error more often than Qwen2.5? This separates "7B capability limit" from "method limit." If a stronger model lifts diagnosis correctness toward the oracle, the small model is the bottleneck and we report the per-model ceiling. If even a strong model without the answer can't, the task itself is hard and we cap expectations.
2. **Evidence-forced diagnosis:** require detection to quote the exact contradicting note sentence and state the corrected value — the form that the oracle error takes. Tested on Qwen2.5 first.
3. **Accept and report the ceiling:** Qwen2.5 self-correction tops out near 25–31% live fix-rate (≈ +1.2pp at full scale with the 70% verdict gate), and the gap to 63–67% is a diagnosis-quality ceiling, not a pipeline-design gap.

## What this rules OUT (stop working on)

- Correction prompt engineering — solved; correction works at 63% from a correct error alone.
- Correction input (note / summary / spans) — redundant given the error.
- Detection feedback form (cot / planner / context) — interchangeable.
- Live prioritization of flaws — not the lever.
- Cross-patient analogy — dropped earlier (3 confirmations).
- Persona tuning — weak second-order lever (Phase 3).

## Practical pipeline implication

The deployable pipeline is minimal: detection produces a critical-error statement → correction applies it (no note needed) → verdict gate (70% break-catch) filters. Net at full scale is set almost entirely by detection's diagnosis-correctness rate. To move the headline number, that rate is the only thing to improve.
