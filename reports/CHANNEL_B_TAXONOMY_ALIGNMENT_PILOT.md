# Channel B Taxonomy Alignment Pilot — Result

Date: 2026-05-29
Sample: 50 Qwen2.5-7B-Instruct zero-shot wrong cases, seed 42, mixed across folds.
Pipeline per case: natural-pipeline detection (`meta_plan_confirm_natural` + `gpt4o-mini-helper-v2`) → same-patient top-5 spans via `gtr_q_answer` → cross-patient analogy top-1 (per arm) → correction (operation + spans + analogy prompt) → GPT-4o judge.

Outputs: `runs/taxonomy_alignment/qwen25_nw50_seed42/{judged_outputs.jsonl, summary.json}`.

## Result

| Arm | Retrieval | Fix | Break | Net | Retrieval path (50 cases) |
|---|---|---:|---:|---:|---|
| C-1 | lexical token overlap, full pool | 0 | 0 | **0** | lexical_full × 50 |
| C-2 | T2-typed subset, GTR cosine within | 1 | 0 | **+1** | t2_match × 36, none × 14 |
| C-3 | T2-typed + T1-fallback + lexical-fallback | 1 | 0 | **+1** | t2_match × 36, lexical_fallback × 14 |

Transition metric uses the fresh GPT-4o judge of the original answer as baseline (5/50 cases the fresh judge thought were correct, indicating stored-label drift).

Single disagreement case: fold 2 idx 135, `correction_operation=ADD_MISSING_SLOT`. C-1 wrong → C-2 correct (real flip from T2-match). C-3 inherited the same.

## Decision rule outcome

Pre-registered rule from `reports/TAXONOMY_ALIGNMENT_DESIGN.md`:

> If C-2 / C-3 do not improve net fix-rate over C-1 by ≥+3 on 50 cases, the taxonomy-alignment overhead isn't worth it; revert to untyped retrieval AND drop cross-patient analogy from the final method.

Observed delta: **+1**. Below the ≥+3 threshold.

**Decision: drop cross-patient analogy from the final method.** Channel B's `{example_block}` path stays dead code in `operation_guided` and any successor correction prompt.

## Context for the negative finding

Three reasons the delta is small, none of which change the decision:

1. **Detection routed only 36 of 50 cases for correction.** 14 cases (28%) had detection `correction_operation=KEEP_ORIGINAL` — among the 50 stored-label-wrong cases, detection's recall on the wrong-class is ~72% here. Cases that detection labels CORRECT do not consume the analogy, so the analogy effect is bounded by detection recall.
2. **Distribution skew to ADD_MISSING_SLOT.** Of the 36 routed cases, 30 are ADD_MISSING_SLOT (83%), 4 REPLACE_VALUE, 2 REMOVE_UNSUPPORTED_CLAIM, 0 REFOCUS_TIME_OR_VISIT. The taxonomy-match retrieval mostly sees the same single bucket. Within-bucket variance is what the test measures, and it produced 1 flip.
3. **Sample size.** N=50 with a 1-case delta has wide confidence intervals. A larger N could move the delta either way, but the design doc registered a +3 threshold precisely to require a real effect rather than chase noise.

## What still works without the analogy

The same correction prompt minus the analogy block IS the `operation_guided` arm that yesterday's 218-case Qwen2.5 pilot used. That pipeline (with detection + spans + correction-operation + multi-dimension verdict gate) reached net +5 in the 40-case meta_plan_confirm_natural Qwen2.5 screen. Yesterday's simpler **single-stage correction-only with `taxonomy_evidence`** (no detection, no verdict) reached net **+27** on 218 cases. Both of those are stronger than anything cross-patient analogy adds.

## Pool-tagging by-product

The T2 classification pass produced clean labels for all 5 folds:

| Fold | N | REPLACE_VALUE | ADD_MISSING_SLOT | REMOVE_UNSUPPORTED_CLAIM | REFOCUS_TIME_OR_VISIT | HIGH conf |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 320 | 116 | 72 | 102 | 30 | 320 |
| 1 | 336 | 116 | 80 | 106 | 34 | 336 |
| 2 | 337 | 120 | 74 | 105 | 38 | 337 |
| 3 | 327 | 113 | 75 | 103 | 36 | 327 |
| 4 | 344 | 120 | 76 | 111 | 37 | 344 |
| **Total** | **1664** | **585 (35%)** | **377 (23%)** | **527 (32%)** | **175 (11%)** | **1664 (100%)** |

Zero UNCLEAR. Every entry resolved into one operation with HIGH confidence. Sidecar files at `workspace/self_critique/data/bm_contrast_pool/fold_X_t2_tags.json` (sibling to the pool JSONs, originals untouched).

The tags remain useful as cross-patient analogy may be revisited later under a different retrieval design (e.g., as part of a regen baseline, or for per-error-type analysis of the natural-pipeline failures). For the current pre-fullscale plan they're an asset on disk but not in the live path.

## Effect on the pre-fullscale plan

Gate 7 substudies (`reports/PRE_FULLSCALE_CONFIRMATION_PLAN.md`):

| Substudy | Status after this pilot |
|---|---|
| 7A — correction-only with oracle / live hint × spans (six-cell ablation) | **Unblocked.** This is the next priority. |
| 7B — rebuild bm_contrast_pool from audited high-confidence pairs | **Cancelled.** With cross-patient analogy out, rebuild is moot. |
| 7C — gte-large-en-v1.5 multi-component scorer vs Step 9 R2 for note spans | **Still open.** Independent of analogy decision. |

## Outputs and provenance

- Code: `scripts/tag_bm_contrast_pool_t2.py`, `scripts/channelB_taxonomy_alignment_pilot.py`.
- Data: `runs/taxonomy_alignment/qwen25_nw50_seed42/`.
- Pool tags: `workspace/self_critique/data/bm_contrast_pool/fold_X_t2_tags.json` (in the source repo).
- Oracle cost: ~$0.33 tagging + ~$1.50 pilot judging ≈ **$1.83 total**.
- Wall time: ~17 min tagging + ~12 min pilot ≈ **~30 min total**.
