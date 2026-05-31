# Natural Pipeline Findings and Scale Plan

Date: 2026-05-29

This report documents the current evidence for the natural-tone correction pipeline, the known weaknesses, and the plan for testing other models or the full 962-case set. The current evidence is from a 40-case Qwen2.5 screen, so it should be treated as a prompt-engineering signal, not a final estimate of production performance.

## Current Best Pipeline

Best stored-label and transition result on the current 40-case seed used `first18k` context. `dynamic_spans` remains the main comparator because it avoids long-note truncation risk.

```bash
--det-prompt meta_plan_confirm_natural \
--det-parse-backend gpt4o-mini-helper-v2 \
--correction-prompt operation_guided \
--verdict-prompt multi_dimension \
--note-context first18k   # compare with dynamic_spans at 100 cases \
--det-temperature 0.0 \
--correction-temperature 0.0 \
--verdict-temperature 0.0 \
--verdict-k 1
```

The intended role split is:

| Stage | Model Role | Should Reason Clinically? | Should Decide Final Correctness? |
|---|---|---:|---:|
| Natural detection | Tested model, e.g. Qwen2.5 | Yes | No |
| Parser / interpreter | GPT-4o-mini helper-v2 | No, preserve the memo | No |
| Correction | Tested model | Yes, grounded in note evidence | No |
| Multi-dimension verdict | Tested model | Yes, compare original/candidate/evidence | Yes, for pipeline acceptance |
| GPT-4o answer judge | Evaluation only | Yes | Offline only |
| GPT-4o intermediate audit | Development only | Yes | Offline diagnosis only |

## Small-Screen Results

20 wrong / 20 correct Qwen2.5 screen:

| Variant | Detected | Accepted | Fixes | Breaks | Net | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Natural + helper-v2 original | 17 | 17 | 4 | 2 | +2 | Natural memo had useful recall, but over-edited some correct answers. |
| Natural + centrality tuning | 11 | 10 | 3 | 1 | +2 | Better precision, lower route rate. |
| Natural + background-slot tuning | 16 | 16 | 5 | 1 | +4 | Best current result. It preserved most gains while reducing avoidable edits. |
| Formatted structured | 7 | 6 | 2 | 1 | +1 | Useful baseline, but weaker recall and evidence handling. |
| Qwen2.5 self-parse | 2 | 2 | 1 | 0 | +1 | High precision, too much recall loss. Needs more parser engineering before reuse. |
| Old GPT-4o-mini parse | 0 | 0 | 0 | 0 | 0 | Parser contract failed; not representative of GPT-4o-mini when prompted correctly. |

Best natural `dynamic_spans` run before the added context ablation:

| Metric | Value |
|---|---:|
| Detected | 16 / 40 |
| Accepted | 16 / 40 |
| Final fixes | 5 |
| Final breaks | 1 |
| Final net | +4 |
| Correction-candidate fixes | 4 |
| Correction-candidate breaks | 1 |
| Correction-candidate net | +3 |
| Intermediate detection valid | 15 / 16 |
| Intermediate correction target supported | 15 / 16 |
| Intermediate candidate matches detection | 15 / 16 |
| Intermediate stage failure NONE | 15 / 16 |


## Added Feature-Matrix Tests

Additional Qwen2.5 ablations were run after the first report to reduce uncertainty before scaling. All used the tuned natural prompt, GPT-4o-mini helper-v2 parser, operation-guided correction, multi-dimension verdict, zero temperatures, and the same 20/20 seed.

| Variant | Detected | Accepted | Stored-label Net | Judge-transition Net | Intermediate Audit | Decision |
|---|---:|---:|---:|---:|---|---|
| `first18k`, `vk1` | 12 | 12 | +5 | +5 | 12/12 routed cases valid | Test against spans at 100 cases |
| `dynamic_spans`, `vk1` | 16 | 16 | +4 | +3 | 15/16 routed cases valid | Keep as candidate |
| `dynamic_spans`, `vk3` | 16 | 16 | +4 | +3 | 14/16 routed cases valid | Safer but not higher net |
| `dynamic_summary`, `vk1` | 16 | 16 | +2 | +1 | 13/16 routed cases valid, 3 detection failures | Do not default |

The most important change is that `first18k` performed best on this seed. This does not prove it is the full-scale default because long notes can still be truncated, but it means the next validation should compare `first18k` and `dynamic_spans` directly instead of assuming retrieval spans are always better.

The summary result is negative. It routed too many invalid detections and had worse transition net, so summary should not be part of the main pipeline unless a later prompt specifically fixes summary-induced evidence drift.

## Main Findings

Natural reasoning is helpful, but only when constrained. The natural memo lets the tested model explain uncertainty, evidence, and correction intent more flexibly than a rigid form. The best run came after adding explicit centrality and background-slot rules, which suggests the benefit is not "free-form output" by itself; it is natural reasoning plus a disciplined interpreter and gate.

GPT-4o-mini is acceptable as an interpreter when prompted correctly. The old mini parser lost nearly all correction signals. The helper-v2 parser recovered useful structure by preserving the tested model's memo, extracting the correction operation, and separating must-fix claims from optional details. In this pipeline, GPT-4o-mini should not be treated as the judge; it is a translation layer between a natural memo and structured downstream fields.

Qwen2.5 self-parse is not ready as the default parser. Direct log review showed that Qwen self-parse missed many correction signals from its own natural memo. It may be possible to improve it with helper-v2-style prompting, but the current result is too low-recall for the main pipeline.

The multi-dimension verdict gate is useful but not sufficient alone. It checks whether the original detection error was removed, whether the correction introduces a new contradiction, and whether the question is completely answered. It still accepted one harmful change, so the detection/correction prompts need stronger handling for broad list and history questions.

`first18k` and `dynamic_spans` should be compared before full scale. `first18k` was best on this seed, while `dynamic_spans` may still be safer on long notes because it avoids truncation. Summaries can focus the model on facts, but the added ablation showed evidence drift and invalid detections; they should not be part of the main path.

## Known Weaknesses

The remaining break in the best `dynamic_spans/vk1` run was a broad past-surgical-history case. The added `first18k` and `dynamic_spans/vk3` runs avoided final judged breaks on this seed, but the case remains useful as a prompt-risk example:

| Field | Detail |
|---|---|
| Case | fold0 idx68 |
| Question | "What is the past surgical history of the patient, and why is she undergoing total thyroidectomy?" |
| Pipeline diagnosis | Total thyroidectomy is correct; TAH/BSO is unsupported or incorrect. |
| Correction behavior | Removed TAH/BSO from the answer. |
| GPT-4o answer judge | Original correct, correction incorrect. |
| Intermediate audit | Mostly considered the detection/correction valid. |

This exposes a hard category: broad history/list questions. A listed background item should not be removed merely because the question also asks for the current procedure rationale. Absence from a focused span is not enough to prove a history item false.

Other risk categories:

| Weakness | Why It Matters | Proposed Treatment |
|---|---|---|
| Broad list/history questions | Correct answers may include extra background items that are not central but still valid. | Add a stricter keep-list rule; do not remove listed history unless directly contradicted. |
| Optional completeness vs required answer slot | The model may add or remove details because they are clinically plausible but not required. | Parser should label optional/background details separately from correction targets. |
| Evidence sufficiency | Some structured runs routed cases despite weak evidence. | Keep `EVIDENCE_SUFFICIENT_FOR_CORRECTION` and decisive evidence fields. |
| Judge disagreement | GPT-4o final judge and intermediate audit can disagree about whether a correction is appropriate. | Track both answer-level outcome and stage-level diagnosis; inspect disagreements. |
| Distribution shift | 40 cases cannot estimate rare failure modes. | Move to 100 stratified, then full 962. |

## Professional Knowledge Questions

Some EHRNoteQA-style questions require clinical-domain understanding beyond lexical matching:

| Question Type | Domain Knowledge Needed | Pipeline Risk |
|---|---|---|
| Medication changes | Drug names, substitutions, route, dose, timing, discontinued vs continued | Missing whether a medication was actually changed. |
| Procedure history | Past surgical history vs current-admission procedure | Removing valid history or mixing procedures across visits. |
| Etiology / contributing factors | Cause vs risk factor vs association vs complication | Adding plausible but unsupported medical factors. |
| Temporal course | Admission vs discharge, first vs second visit, pre-op vs post-op | Correct fact attached to wrong time point. |
| Oncology | Metastasis, node excision/dissection, anticoagulation, follow-up | Confusing treatment, diagnostic workup, and supportive care. |
| Obstetrics | Gestational age, twins, contractions, fetal fibronectin, tocolysis | Misreading central clinical status. |

The tested model can use clinical knowledge to interpret notes, but the final correction still needs note-grounded evidence. Parser/helper stages should not introduce clinical knowledge; they should only interpret the tested model's answer into structured fields.

## What to Keep in the Pipeline

Use for the next larger run:

| Component | Decision |
|---|---|
| Natural-tone detection memo | Keep |
| GPT-4o-mini helper-v2 parser | Keep |
| Operation-guided correction | Keep |
| Multi-dimension verdict | Keep |
| `first18k` and dynamic spans | Compare at 100 cases |
| Temperatures at 0.0 | Keep |
| GPT-4o final/candidate/original judge | Keep for evaluation |
| GPT-4o intermediate audit | Keep for sampled development diagnostics |

Do not use as main defaults yet:

| Component | Reason |
|---|---|
| Summary-only context | Too much evidence compression risk; added 20/20 ablation had 3 invalid detections and transition net +1. |
| Old GPT-4o-mini parser | Failed to preserve correction signals. |
| Current Qwen2.5 self-parse | Low recall. |
| Rigid formatted detection as primary | Lower net gain than tuned natural prompt. |
| `verdict-k=3` as default | Conservative, but reduced accepted fixes in the small screen. |

## Recommended Prompt Guard Before Scaling

Add or keep this guard in the natural detection/parser/correction flow:

```text
For broad history/list questions, preserve listed background items unless the discharge note directly contradicts them, or the question explicitly asks for only current-admission/current-note items. Absence from a focused span is not contradiction. Do not remove a background item simply because it is not necessary to answer the main rationale question.
```

This is the main known weakness from the best run.

## Validation Plan

Stage A: 100-case stratified validation.

| Setting | Value |
|---|---|
| Sample | 50 originally wrong, 50 originally correct if available |
| Pipeline | Run both `first18k` and `dynamic_spans` with the natural helper-v2 pipeline and broad-list guard |
| Evaluation | GPT-4o judge original, correction candidate, and final |
| Intermediate audit | GPT-4o audit for all routed cases, or at least all breaks plus a sample of fixes |
| Promotion criterion | Positive net gain, low break rate on originally correct answers, and no repeated catastrophic failure type |

Stage B: full 962-case run.

Run only after Stage A is stable. The full run should include all wrong and correct cases, not just the 20/20 screen. Before this run, make the output run id unique by timestamp or explicit suffix so repeated prompt edits do not overwrite previous results.

Metrics to export:

| Metric | Purpose |
|---|---|
| Baseline correct / incorrect | Measures starting point. |
| Detected count | Measures route rate. |
| Accepted count | Measures gate aggressiveness. |
| Candidate fixes / breaks / net | Separates correction quality from verdict gate. |
| Judge-transition fixes / breaks / net | Measures fresh `judge_original` to `judge_final` transitions and reduces stored-label drift risk. |
| Final fixes / breaks / net | Main outcome. |
| Accepted precision | How often accepted edits are beneficial. |
| Break rate on originally correct answers | Safety metric. |
| Route rate by original correctness | Detects over-routing. |
| Intermediate validity | Shows if detection/correction stage is coherent. |
| Failure taxonomy | Guides next prompt/model changes. |

Stage C: targeted ablations.

| Ablation | Purpose |
|---|---|
| `dynamic_spans` vs `summary + spans` | Test whether summary adds focus without evidence loss. |
| Natural helper-v2 vs formatted structured | Confirm natural benefit on larger sample. |
| GPT-4o-mini parser vs engineered Qwen self-parser | Test whether parser can migrate off external helper. |
| `verdict-k=1` vs `verdict-k=3` | Measure safety/recall tradeoff on more cases. |
| Broad-list guard on/off | Confirm it fixes history/list breaks without suppressing real fixes. |

## Cross-Model Migration

Once Qwen2.5 is validated on at least the 100-case set, run the same pipeline design on other local models. Keep GPT-4o-mini as the interpreter initially so the model comparison tests the detection/correction reasoning, not each model's parser formatting.

Suggested comparison:

| Model | Keep Fixed | Vary |
|---|---|---|
| Qwen2.5-7B-Instruct | Parser, correction gate, context, judge | Baseline tested model |
| Larger Qwen or Qwen3 variant | Parser, correction gate, context, judge | Model reasoning capacity |
| Llama-family instruct model | Parser, correction gate, context, judge | Generalization across model family |
| Medical-domain model if available | Parser, correction gate, context, judge | Domain specialization effect |

If another model improves the natural memo quality but has weak structured output, that still supports the current method: let the model reason naturally, then use a stable parser to bridge stages.

## Current Recommendation

Do not claim a final result from the 40-case screen. The best added run reached stored-label and transition net +5 with no judged breaks, but the sample is too small and the context choice is not settled. The broad-history/list failure mode remains important even though some later ablations avoided the final break on this seed.

Next action:

1. Add or verify the broad-list/history guard in the natural prompt path.
2. Run a 100-case stratified validation comparing `first18k` against `dynamic_spans` with GPT-4o final and intermediate audits.
3. If one context mode has stable net gain and safety, run the full 962 cases with that context mode.
4. After full Qwen2.5 validation, migrate the same pipeline to other models with the parser and judge held fixed.

## Regen Baseline Note

A focused regen inventory and smoke-test report is now in `reports/regen_pipeline_inventory_and_smoke.md`. Existing 962-case cached `regen_fullscale` results show strong model dependence: DeepSeek +45 net, Llama +10, Qwen3 +10, Qwen2.5 -6, BioMistral -19. Treat regen as a per-model comparator/fallback family, not as a universal default. The added smoke test confirms Step 9 `regen+count`, Step 9 `regen_v3`, and the five legacy T0 regen subvariants are wired; Qwen3 legacy T0 regen now uses `enable_thinking=False`.
