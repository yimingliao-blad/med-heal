# Detection Failure — Root-Cause Analysis and Proposal

Date: 2026-05-29
Data: Phase 1 + Phase 2 re-run with REAL notes (the earlier runs had an empty-note bug; this analysis uses the corrected data). Model: Qwen2.5-7B. Sample: 93 originally-wrong cases (joined across Phase 1 oracle-fix labels, Phase 2 detection, offline taxonomy, real note lengths).

This replaces the speculative "detection fails, it's a capability wall" reading. The failure is not one thing — it is three separable causes, and they point to different fixes.

## The three root causes, with evidence

### Cause 1 — Long notes collapse detection recall (attention / context-length)

Detection flag-rate falls monotonically as the note gets longer, while the cases stay just as fixable:

| Note length | oracle-fixable | detection flags it | diagnosis AGREE |
|---|---:|---:|---:|
| < 6k chars | 78% | **89%** | 0/8 |
| 6-10k | 74% | **55%** | 1/21 |
| 10-15k | 67% | **47%** | 0/7 |
| > 15k | 65% | **29%** | 0/9 |

On short notes detection flags 89% of wrong answers; on long notes only 29%. The wrong answers in long notes are still fixable (oracle-fix only drops 78%→65%) — detection just stops *finding* them. This is the user's "is the note overwhelming the model's attention?" hypothesis, and the data says yes, clearly. The recall collapse is a context-length / attention-dilution effect, and it is the most tractable of the three causes.

### Cause 2 — Diagnosis quality is a flat capability ceiling (not length-driven)

AGREE rate (live diagnosis matches the oracle's "what is wrong and why") is near-zero in EVERY note-length bucket and EVERY error type — including short notes:

- < 6k notes: AGREE 0/8.
- MISREADING: AGREE 1 / PARTIAL 15 / WRONG 12.
- OMISSION: AGREE 0 / PARTIAL 2 / WRONG 3.
- Overall: ~5% AGREE, mostly PARTIAL.

Because this does not improve on short notes, it is NOT an attention problem. Qwen2.5-7B can roughly sense something is off (it flags, and produces a PARTIAL description) but cannot pin the exact error with oracle precision. This is a model-capability ceiling — the oracle diagnoses were written by GPT-4o. This cause is what caps the deployable fix-rate at 5-10% vs the 60% oracle ceiling.

### Cause 3 — A hard core no diagnosis can fix (reasoning / knowledge)

28 of 93 wrong cases (30%) are NOT fixed even when handed the exact oracle diagnosis:

| Error type | hard-core count |
|---|---:|
| MISREADING | 15 |
| OMISSION | 6 |
| QUESTION_MISALIGNMENT | 5 |
| FABRICATION | 2 |

These are cases where telling the model precisely what is wrong still does not produce a correct answer — so the bottleneck is the model's clinical reasoning or knowledge, not detection or correction. 11 of the 28 are in >15k notes (long-note reasoning is harder). This 30% is a ceiling on the whole method for Qwen2.5, independent of any prompt or detection work.

## What each cause rules in as a fix

| Cause | Mechanism | Tractable fix |
|---|---|---|
| 1. Recall collapse on long notes | attention dilution | **Summarize / focus the note before detection** so the relevant facts are not buried |
| 2. Diagnosis quality ceiling | 7B capability | Force note-span quoting; multi-sample agreement; or a stronger detector |
| 3. Hard core (30%) | reasoning/knowledge | Out of scope — mark as the method ceiling; do not chase |

## Proposal

### Proposal A — Summary-first / focused-note detection (targets Cause 1, highest leverage)

This is the user's "can we provide a summary to help the error be detected?" idea, and the data now strongly motivates it. On long notes detection misses 71% of wrong answers. If a question-focused summary surfaces the relevant facts, detection should recover recall.

Three arms on the 93 wrong + ~50 correct, all at the SAME detection prompt:

| Arm | Note context for detection |
|---|---|
| A0 full-note (current) | first 18k chars of the raw note |
| A1 question-focused spans | retrieve top-k note spans for the question + answer, detect on those |
| A2 structured summary + spans | LLM summary of the note focused on the question, plus the source spans |

Metrics: recall (esp. on >10k notes), over-flag on correct, diagnosis AGREE, downstream fix-rate. Decision: pick the context that recovers long-note recall without raising over-flag. This directly tests whether the attention problem is fixable by focusing the input.

Cost: ~143 cases x 3 arms x (detect + parse + correct + judge) at c=8 ~ 30-40 min + ~$4 oracle.

### Proposal B — Force the diagnosis to quote the note (targets Cause 2)

The PARTIAL diagnoses are vague paraphrases. Require detection to output the exact contradicting note sentence(s) verbatim, not a paraphrase. Hypothesis: a quote-grounded diagnosis is more precise and raises AGREE + fix-rate.

Two arms: current free-form diagnosis vs quote-forced diagnosis ("cite the exact note sentence that proves the answer wrong"). Same 93 cases. Measure AGREE and fix-rate. If quote-forcing lifts AGREE materially, it is a cheap capability boost; if not, Cause 2 needs a stronger detector (Proposal D).

### Proposal C — Infer the hint from ZS + question + note directly (targets Cause 2, the user's "can it infer from ZS or question/notes" idea)

Instead of the two-step plan→confirm detection, test whether a single direct pass ("here is the note, the question, and the prior answer — state the one specific thing wrong, quoting the note") produces a better diagnosis than the current structured pipeline. The structured multi-field parse may be losing focus (the user's hypothesis #5: "forced to exact command, loses focus"). Compare: structured pipeline diagnosis vs direct one-shot diagnosis, on AGREE + fix-rate.

### Proposal D — Stronger / multi-sample detector (targets Cause 2, only if B+C insufficient)

If quote-forcing and direct-inference do not lift AGREE, the ceiling is the 7B model. Test: (a) K=3 detection with agreement (already built), (b) GPT-4o-mini as the detector as an upper bound. This separates "method limit" from "7B limit" — if GPT-4o-mini detection lifts fix-rate toward 60%, the gap is model capability and the paper reports the small-model ceiling honestly.

### What NOT to do

- Persona tuning — Phase 3 showed it is a weak second-order lever.
- Cross-patient analogy — dropped (multiple confirmations).
- More correction-prompt engineering — correction works at 60% given a good diagnosis; it is not the bottleneck.

## Recommended order

1. **Proposal A (summary-first detection)** — biggest, most tractable lever; directly tests the attention hypothesis the data most strongly supports.
2. **Proposal C (direct vs structured diagnosis)** — cheap, tests whether the pipeline itself loses focus.
3. **Proposal B (quote-forced diagnosis)** — cheap precision boost.
4. **Proposal D (stronger detector)** — only if B+C cannot lift AGREE; settles method-vs-model.

All at c=8. The 30% hard core is logged as the method ceiling and excluded from "fixable" denominators going forward.

## Open questions

1. For Proposal A's summary arm, summarize with Qwen2.5 (the model under test, keeps it self-contained) or GPT-4o-mini (cleaner but adds an external dependency)? The note-grounding rule says quoted spans must be trusted over summary wording — the A2 arm keeps source spans alongside the summary for that reason.
2. Should the hard-core 30% be carved out into its own analysis (what kinds of reasoning/knowledge they need), or just reported as a ceiling?
3. Run Proposal A first alone, or A+C together overnight at c=8?
