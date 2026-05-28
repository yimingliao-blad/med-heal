# Overall Process Conditions and Decision Map

This is the working map for deciding the final reproducible pipeline one stage at a time. It is intentionally not a final selection yet. The goal is to put code sources, outputs, and judgment choices shoulder by shoulder before promoting one source per stage into the final runnable path.

## Decision Order

1. MIMIC-IV + EHRNoteQA dataset start point (`output/EHRNoteQA_processed.jsonl` selected)
2. Zero-shot generation for each model (`generate_step8.py` shared script selected; BioMistral first)
3. LLM judge on BioMistral
4. Human judge on BioMistral
5. Ground-truth/gold-label selection
6. ICL and prompting tests
7. Later correction/regeneration/self-correction tests
8. Statistical testing and final report

## Stage Matrix

| Stage | Purpose | Candidate Code Sources | Candidate Data / Outputs | Current Default for Review | Decision Needed |
|---|---|---|---|---|---|
| Dataset generation | Use the existing EHRNoteQA records already joined to MIMIC-IV discharge-note context as the fixed start point. Raw preprocessing code remains provenance. | Start artifact: `output/EHRNoteQA_processed.jsonl`; provenance only: `src/step1_preprocessing/preprocess.py`, `src/step1_preprocessing/create_split.py`. | `output/EHRNoteQA_processed.jsonl`; downstream relative folders stay aligned with existing `output/folds`, `output/step8`, `output/step9_v2`, and refactor-local `output`/`reports`. | **Selected:** start from `output/EHRNoteQA_processed.jsonl`; do not regenerate dataset unless explicitly needed later. | Resolved for now. Next decision is zero-shot source/template. |
| Zero-shot generation | Generate baseline free-form answers for all five models through one shared script, with BioMistral as the reference/start model. | **Selected:** `src/step8_multimodel_icl/generate_step8.py`; model-specific differences live in `MODEL_CONFIGS`. Legacy Step 2 generators stay as provenance/human-validation context. | Step 8: `output/step8/<model>/fold_*/zeroshot_generated.csv`; judged labels: `zeroshot_evaluated_binary.csv`; legacy Step 2: `output/ours_*_EHRNoteQA_processed.csv`. | **Selected:** shared Step 8 generator for `biomistral-7b`, `qwen2.5-7b-instruct`, `qwen3-8b`, `deepseek-r1-distill-llama-8b`, and `llama-3.1-8b-instruct`; BioMistral runs first. | Resolved for now. Next decision is LLM judge source/label policy. |
| LLM judge on BioMistral | Automatically judge BioMistral correctness and compare with human evaluation. | `src/step3_evaluation/stage1_gpt4_accuracy.py`; `src/step3_evaluation/stage1_gpt4_eval_combined.py`; `src/step8_multimodel_icl/evaluate_step8_binary.py`; `src/ichl/judges/gpt4o_stage1_binary_judge.py`; `src/step9_self_correction/v2/judge.py`. | `output/step3_evaluation/stage1_gpt4/biomistral-7b_stage1_accuracy.csv`; `output/step8/biomistral-7b/fold_*/zeroshot_evaluated_binary.csv`; `output/step9_v2/judge_agreement_*.json`. | Use Step 9 v2 / ICHL canonical GPT-4o binary judge at temperature 0 for final validation; retain legacy Step 8 labels for comparison. | Decide if final judge is GPT-4o T=0 only, legacy T=0.1 labels, or both with a label-drift audit. |
| Human judge on BioMistral | Human expert correctness labels for BioMistral calibration and GPT-judge selection. | **Selected raw export:** `datasets/external/all_users_openended_BioMistral-7B_1775740232208.csv`; derived sets: `output/step9_v2/sample100_gold_seed42.csv`, `output/step9_v2/sample200_with_gold.csv`; provenance: `datasets/human_judge_Stage2/*`. | Raw export has Sara=328, Jose=310, Caitlin=100, Kushali=50; Sara/Jose shared=145, agreed=112; final 100-case file has resolved `gold` labels. | **Selected:** use the 788-row raw export as authoritative; use `Answer Quality == 5` as correct; use Sara/Jose A∩B N=112 for GPT judge prompt alignment; keep sample100 as broader final human set. | Provisional only; next compare GPT judge prompts against the 112-case baseline. |
| Ground-truth/gold selection | Select trusted labels for evaluating judge and downstream decisions. | Gold subset logic in `src/step9_self_correction/v2/judge.py`; `src/ichl/prompt_engineering/mlx_judge/build_gold_112.py`; human-agreement reports. | A∩B human agreement subset; majority-of-human labels in extended sample; GPT-4o labels for full Step 8 outputs. | Use A∩B human gold subset to validate the judge; use GPT-4o binary labels for full-scale paired experiments after validation. | Decide whether final gold is strict human agreement only, majority human labels, GPT-4o after validation, or a hybrid hierarchy. |
| ICL / prompting tests | Compare zero-shot, RA-ICL, positive/negative examples, CoT, multiturn, and related prompt variants. | `src/step8_multimodel_icl/generate_step8.py`; `src/pilot_12_ra_icl/*`; older Step 4/6/8 prompt optimization scripts; ICHL prompt-engineering variants. | `output/step8/<model>/fold_*/*_{generated,evaluated_binary}.csv`; `output/pilot_12_ra_icl/*`; many BioMistral pilot outputs under `output/pilot*`, `output/step4*`, `output/step6*`. | Use Step 8 full 5-fold conditions as the current main ICL comparison; keep older BioMistral pilots as evidence/provenance. | Decide which prompt family/condition names form the final ICL table, and which pilots are only historical evidence. |
| Regeneration / self-correction | Test correction after baseline answer: detection, correction, verdict, regeneration. | Current: `src/step9_self_correction/v2/*`; older: `src/step9_self_correction/error_taxonomy/*`; ICHL correction/verdict prompt-engineering. | `output/step9_v2/*`; `output/step9_v2/multi_model/*`; older taxonomy JSONs. | Use Step 9 v2 as current canonical implementation; compare with regen/count and older taxonomy results. | Decide which correction path, if any, is included in final test after ICL results are fixed. |
| Significance tests | Quantify whether final method improves over baseline. | New refactor: `src/pre_atom/stats.py`; older docs: `src/step9_self_correction/error_taxonomy/STATISTICAL_PLAN.md`; analysis scripts in Step 8/Pilot 12. | `refactor/pre_atom_pipeline/output/paired_outcomes.csv`; `reports/stats_*`; existing summary JSONs. | Use paired item-level zero-shot vs final labels, McNemar exact/binomial, bootstrap CI, and per-fold summaries. | Decide final paired outcome table and whether to test each model separately or pooled/cross-model. |

## Immediate Review Queue

### 1. Dataset Generation

Candidate final source:

- `src/step1_preprocessing/preprocess.py`
- `src/step1_preprocessing/create_split.py`
- Existing artifacts: `output/EHRNoteQA_processed.jsonl`, `output/folds/fold_*/test.jsonl`

Decision recorded:

- Use `output/EHRNoteQA_processed.jsonl` as the beginning spot for the refactor pipeline.
- Keep the original Step 1 preprocessing scripts as provenance only.
- Maintain the downstream relative folder structure for folds, Step 8 outputs, Step 9 outputs, and refactor-local summaries.

### 2. Zero-Shot Baseline

Decision recorded:

- Use `src/step8_multimodel_icl/generate_step8.py` as the shared zero-shot generator for all five models.
- Run/review BioMistral first, then `qwen2.5-7b-instruct`, `qwen3-8b`, `deepseek-r1-distill-llama-8b`, and `llama-3.1-8b-instruct`.
- Keep the older Step 2 generation scripts visible as provenance and human-judge validation context.
- Treat correctness labels as a later judge/ground-truth decision, not part of generation.

### 3. Judge and Gold Labels

Candidate final judge:

- `src/step9_self_correction/v2/judge.py` / `src/ichl/judges/gpt4o_stage1_binary_judge.py`

Provisional evidence recorded, not final:

- Candidate raw export: `datasets/external/all_users_openended_BioMistral-7B_1775740232208.csv`.
- Final 100-case human/gold artifact: `output/step9_v2/sample100_gold_seed42.csv`.
- Extended human/gold artifact: `output/step9_v2/sample200_with_gold.csv`.
- Binary mapping: `Answer Quality == 5` means correct; otherwise incorrect.
- GPT judge prompt selection should use the clean Sara/Jose A∩B agreement baseline (N=112).

### 4. ICL Tests

Candidate main source:

- `src/step8_multimodel_icl/generate_step8.py`

Main candidate conditions:

- `zeroshot`
- `gtr_note_pos_k1`
- `gtr_note_neg_k1`
- `gtr_note_posneg_k1`
- `cot_evidence`
- `cot_conclusion`
- `multiturn`
- `gtr_note_any_unlabeled_k1`

Pilot/historical sources to compare before final selection:

- `src/pilot_12_ra_icl/*`
- BioMistral Step 4/6/8 pilot outputs
- ICHL prompt-engineering prompt families

## How to Use This Map

For each stage, we should make one explicit decision before moving to the next:

1. `selected_code_source`
2. `selected_input_artifacts`
3. `selected_output_schema`
4. `selected_metric_or_label_definition`
5. `legacy_sources_to_keep_for_comparison`

Those choices should then be recorded in `configs/process_decisions.json` and mirrored in the runnable scripts.

## Current Rerun Plan

See `reports/RERUN_AND_VALIDATION_PLAN.md`. Working assumptions: Step 8 full-scale will be rerun as the main repeated experiment family, full-scale regeneration/self-correction will also be included, and method details such as retrieval embeddings, prompts, pool construction, and true multirun design remain unfinalized pending validation tests.
