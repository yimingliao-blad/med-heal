# Few-shot vs Correction Taxonomy Alignment — Design and Justification

Date: 2026-05-29
Status: pre-decision design discussion. No code yet. This file answers the user's question: *"the biggest thing is the taxonomy for few-shot should be the same or overlap with correction? We need to justify."*

## What is in scope

"Few-shot" here means **cross-patient analogy retrieval** — the Channel-B `{example_block}` path where the correction prompt receives one (or k) reference cases from `bm_contrast_pool/fold_X_pool.json`. Channel A (direct generation few-shot) is out per user direction. The analogy retrieval is currently dead code in `operation_guided`; whether we re-enable it depends on the recall pilot (4B-1) AND on this taxonomy decision.

"Correction" here means **the natural-pipeline correction prompt's hint** — the structured fields the correction LLM reads to know what to fix.

## The three taxonomies actually in play

| Taxonomy | Vocabulary | Where it appears | Source |
|---|---|---|---|
| **T1: Detection error_type** | CONTRADICTION / OMISSION / QUESTION_MISALIGNMENT / NONE / UNCLEAR (5 cats) | Natural-pipeline detection output, live per case | `meta_plan_confirm_natural` + `gpt4o-mini-helper-v2` parser |
| **T2: Correction operation** | REPLACE_VALUE / ADD_MISSING_SLOT / REMOVE_UNSUPPORTED_CLAIM / REFOCUS_TIME_OR_VISIT / KEEP_ORIGINAL (5 cats) | Natural-pipeline correction prompt's `correction_operation` field | Same parser, derived from T1 + memo |
| **T3: Offline PRIMARY_ERROR** | MISREADING / FABRICATION / OMISSION / QUESTION_MISALIGNMENT / HEDGING / CORRECT_OR_UNKNOWN (6 cats) | `phase1_wrong_gpt4o.json` Qwen2.5 audit | Offline GPT-4o pass |

Today's pipeline already has T1 and T2 wired together with this **one-to-one-or-many mapping** (from `parse_detection_with_backend`'s helper-v2 instructions in the runner):

| T1 (detection error_type) | T2 (correction operation) |
|---|---|
| CONTRADICTION | REPLACE_VALUE *or* REMOVE_UNSUPPORTED_CLAIM |
| OMISSION | ADD_MISSING_SLOT |
| QUESTION_MISALIGNMENT | REFOCUS_TIME_OR_VISIT |
| NONE | KEEP_ORIGINAL |
| UNCLEAR | KEEP_ORIGINAL |

The bm_contrast_pool entries are NOT currently tagged in any of T1/T2/T3 — only free-text `what_was_wrong` + `wrong_answer` + `ground_truth`.

## The design question

When the correction step is shown a retrieved analogy, the analogy is more useful when its situation matches the test case's situation. The question is **which axis of situation matters most**:

- **Same error type** (T1 / T3) — same WHAT went wrong.
- **Same correction operation** (T2) — same HOW it should be fixed.
- **Same question topic** (separate axis, not error-taxonomy) — same clinical content.
- **Same answer-slot shape** (e.g., date / medication / list) — same output form.

The literature mostly retrieves by clinical-content similarity (last two axes). The error-taxonomy axes are unusual in RA-ICL but mechanically interesting because the correction prompt is ALREADY structured around T1+T2.

## First-principles argument — same / overlap / independent

### A. Same taxonomy (analogy tagged with same vocab as correction's hint)

**Argument FOR:**

- Consistency within a single prompt. The hint says "error_type = CONTRADICTION, operation = REPLACE_VALUE." The retrieved analogy is also labeled CONTRADICTION / REPLACE_VALUE. The model sees one consistent story: "this kind of case needs this kind of fix; here's an example of exactly that."
- Retrieval becomes a hard filter — "only show me analogies of the same operation type." That collapses a 320-entry pool to a small, focused subset where the example actually demonstrates the operation we're asking for.
- Simpler method section in the paper: one taxonomy threaded through detection → correction → analogy retrieval.

**Argument AGAINST:**

- Detection's T1 is COARSE (5 cats). Analogy pool can support finer distinctions. Forcing T1 onto the pool throws away signal.
- If detection makes a T1 mistake (calls CONTRADICTION on an OMISSION case), the analogy is mis-typed too, reinforcing the wrong correction direction.
- The yesterday `error_type_router` arm in the existing 14-arm set already attempts this — it's tagged "weak routing hint" exactly because the type signal alone is brittle.

### B. Overlap (one taxonomy is a subset / coarsening of the other)

This is what the runner ALREADY DOES between T1 and T2. T1 (5 cats) → T2 (5 cats) is a many-to-one mapping (CONTRADICTION can split into REPLACE_VALUE or REMOVE_UNSUPPORTED_CLAIM). T3 (6 cats) is a finer version of T1 — MISREADING is a sub-category of CONTRADICTION; HEDGING has no T1 equivalent.

**Argument FOR overlap:**

- Detection stays simple (T1). Correction can use the finer downstream signal when available (T2 / T3).
- Analogy pool can be tagged at the finest available vocabulary (T3 or T2). Retrieval can drop down to coarser categories when no finer-typed analogy exists.
- Allows the pipeline to gracefully degrade: T3-match → T2-match → T1-match → no-type-match (lexical/embedding only).

**Argument AGAINST overlap:**

- Three taxonomies to maintain. Extra annotation work to tag the pool in T3. The pipeline has more failure modes.
- Mapping is not always clean (HEDGING in T3 has no obvious T1/T2 equivalent).
- Hardest to defend in a paper — reviewers will ask "why three taxonomies and not one."

### C. Independent (analogy uses its OWN vocabulary, no enforced match)

The analogy gets tagged by its OWN best descriptor — perhaps "answer-slot shape" (date / medication / list / paragraph), or correction style (terse-rewrite / extension / reordering), or question topic (medication change / oncology / etc). Retrieval picks the best analogy by that vocabulary's matching, independent of T1/T2.

**Argument FOR:**

- The analogy's purpose is to SHOW HOW TO ANSWER, not to identify the error. Tagging by answer-shape may be more useful than tagging by error-type.
- Decouples diagnosis (T1) from treatment (analogy). Clinically realistic — a single symptom can have multiple causes, each with different treatments.
- Side-steps the risk of detection-T1 errors propagating into analogy mis-tagging.

**Argument AGAINST:**

- Loses the consistency of "this error type → this analogy example" cleanness. The model has to integrate two different signals.
- Hard to design and validate without a fresh annotation pass.
- No prior med-heal evidence backs an independent taxonomy.

## Why this question matters specifically for med-heal

Yesterday's `taxonomy_evidence` arm (the +27 result) embedded T3 (offline PRIMARY_ERROR) into the correction prompt's TEXT INSTRUCTION — not into the analogy retrieval. The analogy retrieval was NOT used (operation_guided ignores `{example_block}`, taxonomy_evidence is a single-stage prompt without analogy). So yesterday's result tells us NOTHING about taxonomy-aligned analogy retrieval.

That gap is exactly the question the user is asking: *if we add analogy retrieval back, what taxonomy ties it to the correction step?*

The answer determines:

- Whether we need to tag bm_contrast_pool (and in which taxonomy).
- Whether the retrieval mechanism is "lexical / embedding similar to test case" (untyped) or "filter to same error_type, then lexical / embedding within" (typed).
- Whether the analogy block in the correction prompt should be PRESENTED as "an example of the same error type" (typed framing) or "an example of a similar past case" (untyped framing).

## Recommended default — justified

I recommend **OPTION B-1: overlap, with T2 (correction operation) as the primary tag**.

### Why T2 specifically (not T1 or T3):

- T2 names the EDIT the correction has to perform. The whole purpose of the analogy is to show "here is an edit of this kind." Tagging by edit type makes the analogy directly actionable.
- T2 is already produced by the natural-pipeline parser. No new detection stage needed.
- T2's vocabulary (REPLACE_VALUE / ADD_MISSING_SLOT / REMOVE_UNSUPPORTED_CLAIM / REFOCUS_TIME_OR_VISIT / KEEP_ORIGINAL) maps cleanly onto the analogy entries' `what_was_wrong` text (a one-time GPT-4o-mini classification pass over ~320 entries per fold, ~$0.07 per fold, ~$0.35 total).
- It preserves the ability to fall back to T1 or to plain embedding when T2 is unavailable (graceful degradation).

### Why "overlap" not "same":

- The detection step uses T1+T2 internally; the analogy pool gets tagged ONLY in T2. T1 stays a transient field used by the parser. The paper's method section says: "errors are categorized into 5 correction operations (T2). Each cross-patient analogy is annotated with the operation it demonstrates. At inference time, analogy retrieval is filtered to same-operation candidates, then ranked by embedding similarity."
- Single explicit taxonomy in the paper — T2. T1 is an internal parser intermediate, not a method-section taxonomy.

### Why NOT independent:

- Independent vocabulary requires fresh annotation design and lacks any baseline. Too uncertain for an ACM-BCB submission.
- The correction prompt already structures the LLM's task around T2. Misalignment between hint and analogy creates noise that we'd have to study separately.

## The empirical test that confirms

Three-arm head-to-head on Qwen2.5 wrong cases (50 cases initially, 218 if promising):

| Arm | Analogy retrieval | Analogy presentation in prompt |
|---|---|---|
| C-1 untyped | top-1 by lexical token overlap from full pool (current runner behavior) | "Here is one past case that may be informative" |
| C-2 T2-typed | top-1 from operation-matched subset of pool, ranked by embedding similarity | "Here is a past case where the correct fix was the same operation (X)" |
| C-3 T2-typed + T1-fallback | top-1 from T2-matched; if empty, fall back to T1-matched; if empty, full pool | same as C-2 with provenance tag |

Decision rules:

- If C-2 / C-3 do not improve net fix-rate over C-1 by ≥+3 on 50 cases, the taxonomy-alignment overhead isn't worth it; revert to untyped retrieval AND drop cross-patient analogy from the final method.
- If C-2 wins clearly, lock T2 as the analogy taxonomy and the paper has one taxonomy to describe.
- If C-3 beats C-2, the fallback logic earns its place in the method section.

Cost: tag 320×5 = 1600 pool entries with T2 via GPT-4o-mini (~$0.35 oracle, one-time, 5 min). Then 3 arms × 50 cases × ~3 LLM calls = 450 calls, Qwen2.5 ~10 min + ~$1.20 judge.

## Honest trade-off if the test fails

If C-2 / C-3 lose to C-1, that is itself a reportable finding: "Cross-patient analogy retrieval by error-type matching did not improve over lexical retrieval on N=50 Qwen2.5 cases; the cross-patient analogy was dropped from the final method." Cleanly defensible.

## Where this leaves the 7A scope

The 7A bake-off (the 6-cell hint × spans grid) does NOT include cross-patient analogy. T1 / T2 / T3 in 7A all refer to the correction-prompt HINT, not to retrieval. So 7A and the taxonomy-alignment substudy are independent:

- 7A answers: does the hint help, and does live detection's hint match the oracle's hint?
- Taxonomy-alignment substudy answers: if we add cross-patient analogy back, what taxonomy ties it to the correction prompt?

Either can run first. Running 7A first gives us deployable-method numbers; running the taxonomy test first decides whether cross-patient analogy is in or out before we spend time on it.

## Decisions waiting

1. Accept "overlap, T2 as primary tag" as the working default? (Or pick A same / C independent.)
2. Approve the GPT-4o-mini tagging pass (~$0.35 oracle, one-time) so the bm_contrast_pool gets T2 labels?
3. Approve the 3-arm head-to-head (C-1 / C-2 / C-3) on 50 Qwen2.5 cases to test the default?
4. Run the taxonomy substudy BEFORE 7A, AFTER 7A, or interleaved?
