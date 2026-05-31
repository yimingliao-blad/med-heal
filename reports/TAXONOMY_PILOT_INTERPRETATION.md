# Interpreting the Taxonomy Alignment Pilot — What We Learned

Date: 2026-05-29

The 3-arm pilot's headline says "drop the cross-patient analogy." That's the right call by the pre-registered rule. But the more important reading is underneath. This memo answers the user's four questions: *what was the purpose, how does it help next steps, what did we learn, what more do we need to design.*

## Purpose of the pilot

The narrow question was: **should the cross-patient analogy retrieval taxonomy match the correction taxonomy?** The premise: if the correction prompt says "operation = REPLACE_VALUE," and the analogy is also a REPLACE_VALUE case, the two prompt inputs reinforce each other. The arms tested three variants of "match-by-T2" against the runner's existing lexical baseline.

The implicit assumption baked into the test design: **the analogy is contributing measurable signal to begin with**. If it isn't, no retrieval-side change matters.

## What the numbers actually say

| Arm | Fix | Break | Net | What changed |
|---|---:|---:|---:|---|
| C-1 lexical untyped | 0 | 0 | 0 | baseline |
| C-2 T2-typed | 1 | 0 | +1 | 36/50 saw a T2-matched analogy; 14/50 saw no analogy |
| C-3 T2-typed + fallback | 1 | 0 | +1 | 36/50 saw a T2-matched analogy; 14/50 fell back to lexical |

Per-operation breakdown for C-2:

| Detection op | N | C-2 fixes |
|---|---:|---:|
| KEEP_ORIGINAL | 14 | 0 (correction not attempted) |
| ADD_MISSING_SLOT | 30 | 1 |
| REPLACE_VALUE | 4 | 0 |
| REMOVE_UNSUPPORTED_CLAIM | 2 | 0 |
| REFOCUS_TIME_OR_VISIT | 0 | n/a |

Two observations from this:

1. **Across all three arms, total fixes were 0, 1, 1.** The analogy retrieval method moved the needle by exactly one case. We did not learn whether taxonomy alignment is a good idea — we learned that the analogy in this pipeline isn't doing enough work for the question to be answerable on N=50.

2. **30 of 36 routed cases are ADD_MISSING_SLOT.** The taxonomy match concentrates almost entirely on one operation, so the test was effectively asking "does T2-matching help on ADD_MISSING_SLOT cases?" The other operations had so few cases that we couldn't have detected an effect even if it existed.

## The buried finding — pipeline shape is the real story

Yesterday's 218-case Qwen2.5 correction-only pilot reached **net +27** under `taxonomy_evidence`. Today's 50-case pilot with the same Qwen2.5, same correction-operation framing, same span retriever, but with the natural-pipeline detection ADDED in front, reached **net +1 at best**.

That gap is about 6 fixes per 50 cases vs essentially zero today. Three suspects, in order of suspicion:

1. **Detection gating throws away half the upside.** 14 of 50 wrong cases (28%) got `correction_operation = KEEP_ORIGINAL`. Those cases never enter correction. Yesterday's pipeline corrected every case. With detection in front, the recoverable population is 36/50 = 72% of the wrong-case pool, before correction even runs.
2. **The live hint is weaker than the offline taxonomy.** Yesterday's `taxonomy_evidence` used the offline `phase1_wrong_gpt4o.json` PRIMARY_ERROR (MISREADING / FABRICATION / etc — derived by GPT-4o on each wrong case). Today's hint is `correction_operation` from the natural pipeline parser. Coarser vocabulary, derived live, sometimes wrong.
3. **The correction prompt got more constrained.** Today's prompt threads operation + spans + analogy + decisive_evidence + do_not_change. Yesterday's `taxonomy_evidence` had hint + spans only. More inputs to reconcile, less room for the model to write the answer.

Disentangling these three is exactly the 7A bake-off's job. We now have direct evidence that the question matters — the gap between cell 4 (offline oracle + spans, +27) and cell 6 (live detection + spans + analogy, today ~+1) is roughly 26 fixes per 218 cases. That's an enormous design ceiling sitting between deployable and ideal.

## What we did and didn't learn about the taxonomy

What we DID confirm:

- The T2 tagging step is mechanically sound — 1664 entries tagged HIGH confidence with zero UNCLEAR, distribution sensible.
- The runtime cost of typed retrieval is negligible — embedding lookup adds <1 ms vs lexical.
- The retrieval mechanism doesn't break on the "no matching subset" case (C-2's 14/50 "none" path).

What we DID NOT learn:

- Whether taxonomy alignment IS a good idea — only one case differed between arms. The effect (if any) is below the noise floor at N=50.
- Whether the analogy block in the prompt is doing ANYTHING. We didn't run a no-analogy control. Yesterday's `evidence_only` arm (no analogy, no hint) hit +20 on 218 cases. Today's three arms (analogy on, detection gating) hit ~0. The right control is "today's pipeline minus the analogy block."
- Whether the classifier's tags are sensible. 0 UNCLEAR is suspicious — it might mean every entry fits cleanly, or it might mean the classifier was too willing to pick a label. Hand-spot 20 entries to check.
- Whether T2 (operation) is the right axis to match on at all. We never tested matching by question topic, by answer-slot shape, or by error-type in T1.

## How this helps the next step (7A)

The taxonomy pilot was supposed to feed into 7A. With this result it does, in three ways:

1. **7A is now the highest-priority gate.** The detection-gated pipeline (cell 6 family) underperforms the simple correction-only (cell 4 family) by an order of magnitude. We need to know whether that gap is recoverable.

2. **7A's cell 5 / cell 6 just got partial data.** Today's pilot is essentially a "cell 6" datapoint for Qwen2.5 at N=50 with detection + spans + analogy + correction. Net is +1. For 7A's clean comparison we need a "cell 6 without analogy" arm at the same N — that's a one-line change to today's pilot.

3. **The +14 missing-from-correction issue puts the strict-vs-force gate decision front and center.** In the 7A scope detail, options A (strict gate) and B (force correction even on KEEP_ORIGINAL) were laid out for the user to pick. Today's data shows the strict gate caps the fix ceiling at 36/50 cases. The force option could recover the other 14, but we don't know if correction on a CORRECT-labeled wrong case helps or hurts.

## What more to design — concrete proposals

Three small additional tests would let us understand the taxonomy properly:

### Test α — no-analogy baseline at today's pipeline shape (1 arm, 50 cases)

Same code path as today's C-1, but pass an empty analogy block. If net stays at ~0, the analogy is doing nothing in the current pipeline; if net drops below 0, the analogy was helping slightly even when we couldn't see it in C-1 vs C-2. Cost: 50 cases × ~5 LLM calls × Qwen2.5 ≈ 5 min + $0.40 oracle.

### Test β — hand-audit 20 random pool tags (cheap, no compute)

Sample 4 entries per operation × 5 operations = 20 entries (well, 4 active operations). Print question + wrong_answer + what_was_wrong + ground_truth + tagged_operation. Mark each "correct tag" / "plausible alternative" / "wrong." If error rate >10%, retag with a stronger classifier or refine the prompt. Cost: 10 minutes of reading.

### Test γ — force-correction on KEEP_ORIGINAL cases (the cell 6B variant of 7A)

Today's 14 KEEP_ORIGINAL cases: rerun those 14 with correction forced regardless of detection's verdict, using `evidence_only` arm (no hint, spans only). If fix-rate on those 14 is ≥3, the strict gate is leaving easy wins on the table; if it's 0, detection's KEEP_ORIGINAL labels are honest. Cost: 14 cases × 1 LLM call each ≈ 1 min + $0.12 oracle.

These three together cost about 15 minutes and $0.55. They turn today's pilot from "drop the analogy" into "understand which of analogy, detection gate, hint vocabulary, and prompt constraints is doing what."

## Recommended order of operations from here

1. **Skip the three small tests** if user wants to move fast. Go straight to 7A (the full 6-cell ablation) at N=50, including the "no analogy" cell as a control.
2. **Run all three small tests first** if user wants the taxonomy pilot to stand on its own and the 7A bake-off to be cleaner-conditioned. Total +15 min before 7A starts.
3. **Run only test α** as the minimum: it directly answers "is the analogy block in the current pipeline doing anything," which is the most important single follow-up question.

My recommendation: **option 3 (test α only), then 7A**. Test α is the one with the highest signal/cost ratio. β can wait — the tag distribution is sensible enough that hand-spotting can happen anytime. γ is subsumed by 7A's strict-vs-force comparison.

## Bottom line for the paper

If we ship without further taxonomy work: report this pilot as a clean negative result. "Cross-patient analogy retrieval, with or without T2-operation matching, did not improve correction on N=50 Qwen2.5 wrong cases (Δ ≤ +1 fix). The analogy was dropped from the final method."

If we want to claim the taxonomy idea was rigorously tested: add test α + report alongside.

If we want to claim the live-hint pipeline is competitive with offline-oracle pipelines: run 7A and let those numbers carry the methods section.

## Decisions waiting

1. Run test α (no-analogy control at today's pipeline shape, 50 cases, ~5 min) before 7A? Y/N.
2. Hand-audit 20 pool tags (β) now? Y/N. (Cheap, takes 10 min, no compute.)
3. Go directly to 7A scope decision (MIN / STD / MAX from `reports/SUBSTUDY_7A_SCOPE_DETAIL.md`)? Pick a package now or keep deciding?
