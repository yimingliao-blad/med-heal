# Pipeline Feature Matrix and Analysis

Date: 2026-05-29

This note summarizes which pipeline features should be considered for the current Med-Heal correction pipeline, based on the Qwen2.5 20/20 screens and GPT-4o audits run so far.

## Current Best Reference

Best small-screen pipeline before the added context ablations; after the added tests, `first18k` is the leading 40-case result and `dynamic_spans` is the main comparator:

```bash
--det-prompt meta_plan_confirm_natural \
--det-parse-backend gpt4o-mini-helper-v2 \
--correction-prompt operation_guided \
--verdict-prompt multi_dimension \
--note-context dynamic_spans \
--det-temperature 0.0 \
--correction-temperature 0.0 \
--verdict-temperature 0.0 \
--verdict-k 1
```

20 wrong / 20 correct screen:

| Pipeline | Detected | Accepted | Fixes | Breaks | Net |
|---|---:|---:|---:|---:|---:|
| Natural + helper-v2 original | 17 | 17 | 4 | 2 | +2 |
| Natural + centrality tuning | 11 | 10 | 3 | 1 | +2 |
| Natural + background-slot tuning | 16 | 16 | 5 | 1 | +4 |
| Formatted structured | 7 | 6 | 2 | 1 | +1 |


## Additional Qwen2.5 Feature Ablations

After the initial feature matrix, three more 20 wrong / 20 correct Qwen2.5 ablations were run on the same seed with GPT-4o judging and GPT-4o intermediate audits where useful. These use the tuned natural detection prompt, GPT-4o-mini helper-v2 parser, operation-guided correction, and multi-dimension verdict.

| Variant | Detected | Accepted | Stored-label Fixes | Stored-label Breaks | Stored-label Net | Judge-transition Net | Intermediate Audit | Decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `first18k`, `vk1` | 12 | 12 | 5 | 0 | +5 | +5 | 12/12 routed cases valid, stage failure NONE | Promote to larger comparison |
| `dynamic_spans`, `vk1` | 16 | 16 | 5 | 1 | +4 | +3 | 15/16 routed cases valid in prior audit | Keep as candidate, not automatic default |
| `dynamic_spans`, `vk3` | 16 | 16 | 4 | 0 | +4 | +3 | 14/16 routed cases valid | Safer than `vk1`, but loses fixes |
| `dynamic_summary`, `vk1` | 16 | 16 | 4 | 2 | +2 | +1 | 13/16 routed cases valid, 3 detection failures | Do not use by default |

The transition metric compares `judge_original` to `judge_final` within the same run. This is important because the stored `orig_label` and a fresh GPT-4o judgment can disagree on individual cases. Future summaries now export both stored-label and transition metrics.

Updated context decision:

- `first18k` is the best result on this 40-case seed, with no judged final breaks.
- `dynamic_spans` remains plausible because it avoids truncation risk on long notes, but it did not beat `first18k` here.
- `dynamic_summary` should not be in the main pipeline; it increased invalid detections and judged breaks.
- The 100-case validation should compare `first18k` vs `dynamic_spans` directly before deciding the full-962 default.

## Feature Matrix

| Feature | Current Decision | Evidence | Rationale |
|---|---|---|---|
| Natural detection memo | Keep testing | Best tuned run reached net +4 | Natural prose appears better at finding real correction opportunities than rigid field output. |
| GPT-4o-mini as interpreter/parser | Keep | Improved helper-v2 produced usable payloads and enabled net +4 run | Mini should act as a stage interpreter, not a medical judge. It should preserve the tested model's diagnosis into structured fields. |
| Qwen2.5 self-parse | Do not use as default yet | Routed only 2/40; both valid, but missed many correction signals | High precision, low recall. Could be revisited with helper-v2-style parser prompt. |
| Rigid formatted detection | Keep as baseline only | Structured rerun net +1 | Lower recall and weaker evidence/correction quality than tuned natural path. Useful for ablation. |
| Operation-guided correction | Keep | Works best when detection payload contains operation/evidence fields | Needs reliable parser fields: operation, evidence sufficiency, decisive evidence, do-not-change. |
| Multi-dimension verdict gate | Keep, but tighten | Best verdict direction; natural tuned net +4 | Checks error removed, no new contradiction, question complete. Still accepted one harmful correction, so needs list/history guard. |
| `dynamic_spans` context | Keep as candidate | Strong prior, but `first18k` beat it on the added 40-case ablation | Gives focused evidence and avoids long-note truncation, but may omit useful global context. Compare against `first18k` at 100 cases. |
| `dynamic_summary` context | Do not include by default | 20/20 ablation: stored-label net +2, transition net +1, 3 invalid detections | Summary may focus facts, but it can compress qualifiers or overstate evidence. Use only as a later ablation. |
| Full note first18k | Promote to candidate default | Added ablation reached stored-label net +5 and transition net +5 with 0 breaks | Simple and strong on this seed, but may truncate long notes. Compare against `dynamic_spans` before full scale. |
| `verdict-k=3` | Not default for now | Reduced breaks but also reduced accepted fixes | Useful when the gate is noisy, but current tuned natural pipeline uses k=1 and gets better net. |
| GPT-4o answer judge | Keep for evaluation only | Used for original/candidate/final labels | Measures answer correctness transitions. Not part of tested-model decision path. |
| GPT-4o intermediate audit | Keep for development analysis | Identified detection/correction/gate failure modes | Useful for prompt engineering, not final pipeline inference. |

## Summary vs Dynamic Spans

Summary is intuitively useful because it focuses the LLM on salient facts. The risk is that summarization becomes an extra reasoning stage that can drop qualifiers, merge visits, or overstate weak evidence.

Current decision:

- Compare `first18k` and `dynamic_spans` in the next 100-case validation before choosing the full-962 default.
- Do not include `dynamic_summary` in the main method.
- If summary is revisited later, use it only as an auxiliary view together with quoted source spans, and instruct every downstream step to trust quoted spans over summary wording.

Suggested future summary ablation:

| Variant | Summary Role | Expected Risk |
|---|---|---|
| `dynamic_spans` only | Source evidence only | Lowest risk |
| `summary + source spans` | Focus aid plus evidence | Moderate risk |
| `summary only` | Compressed evidence | High risk, not recommended |

## What Should Be in the Pipeline Now

Recommended candidate pipeline for the next larger screen:

1. Qwen2.5 natural audit plan and memo.
2. GPT-4o-mini helper-v2 parser into structured correction payload.
3. Operation-guided correction.
4. Multi-dimension verdict gate.
5. Compare `first18k` and `dynamic_spans` as note context; do not use summary by default.
6. Temperatures at 0.0.

Important guard to add before larger runs:

- For broad list/history questions, do not remove a listed item unless the note directly contradicts it or the question explicitly asks only for current-note/current-admission items.

## What Should Not Be in the Main Pipeline Yet

- `dynamic_summary` as the only evidence context.
- Old GPT-4o-mini parser prompt.
- Qwen2.5 self-parser without helper-style prompt engineering.
- Pure rigid formatted detection as the primary method.
- Verdict `k=3` as default before testing on larger samples.

## Professional Knowledge / Domain Knowledge Questions

Yes, some questions require professional clinical-domain understanding beyond surface string matching. Examples include:

| Question Type | Domain Knowledge Needed | Risk |
|---|---|---|
| Medication changes | Understanding drug names, substitutions, dosing, routes, and timing | Missing whether a medication was continued vs replaced. |
| Procedure history | Distinguishing past surgical history from current admission procedures | Removing background history incorrectly. |
| Etiology / contributing factors | Knowing what counts as a cause, risk factor, complication, or association | Adding/removing plausible but unsupported factors. |
| Temporal clinical course | Understanding admissions, discharge dates, first vs second visit, pre-op vs post-op | Wrong visit/time focus. |
| Oncology treatment | Interpreting node excision/dissection, metastasis, anticoagulation, follow-up | Confusing interventions with supportive care. |
| Obstetric cases | Understanding twins, gestational age, contractions, fetal fibronectin, tocolysis | Missing central temporal/clinical differences. |

Pipeline implication:

- The tested model can use its clinical knowledge to interpret the note, but the final correction must still be grounded in note evidence.
- Parser/helper stages should not add medical knowledge; they should only translate the tested model's memo into structured fields.
- The gate should reject corrections that rely on plausible clinical knowledge without note support.

## Immediate Next Recommendation

Before scaling, add one more prompt guard:

```text
For broad history/list questions, do not narrow or remove listed background items unless the discharge note directly contradicts the item, or the question explicitly asks for only current-admission/current-note items. Absence from a focused span is not contradiction.
```

The added ablations already show the break can disappear under `first18k` or `dynamic_spans/vk3`. Next, run the 100-case stratified comparison of `first18k` vs `dynamic_spans`; use the result to choose the full-962 context mode.

## Regen Baseline Note

A focused regen inventory and smoke-test report is now in `reports/regen_pipeline_inventory_and_smoke.md`. Existing 962-case cached `regen_fullscale` results show strong model dependence: DeepSeek +45 net, Llama +10, Qwen3 +10, Qwen2.5 -6, BioMistral -19. Treat regen as a per-model comparator/fallback family, not as a universal default. The added smoke test confirms Step 9 `regen+count`, Step 9 `regen_v3`, and the five legacy T0 regen subvariants are wired; Qwen3 legacy T0 regen now uses `enable_thinking=False`.
