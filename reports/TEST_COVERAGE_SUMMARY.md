# Test Coverage Summary

This is a working inventory of tests already present on disk. It is not a final selection of what goes into the paper/report.

## Full-Scale Step 8: 5 Models x 5 Folds

All five target models have complete 5-fold generated and binary-evaluated outputs for the core Step 8 condition set:

- `zeroshot`
- `gtr_note_pos_k1`
- `gtr_note_neg_k1`
- `gtr_note_posneg_k1`
- `cot_evidence`
- `cot_conclusion`
- `multiturn`
- `gtr_note_any_unlabeled_k1`

Models:

- `biomistral-7b`
- `qwen2.5-7b-instruct`
- `qwen3-8b`
- `deepseek-r1-distill-llama-8b`
- `llama-3.1-8b-instruct`

Primary location:

```text
output/step8/<model>/fold_<0-4>/<condition>_{generated,evaluated_binary}.csv
```

## Full-Scale Extra Step 8 Tests

Additional conditions exist beyond the core set:

- `contrastive_random` and `contrastive_targeted`: complete 5-fold generated and binary-evaluated for all five models.
- Negative k-sweep `gtr_note_neg_k2` to `gtr_note_neg_k5`: complete 5-fold generated and binary-evaluated for Qwen2.5, Qwen3, DeepSeek, and Llama3; not present for BioMistral.
- Random controls:
  - DeepSeek: `random_pos_k1` and `random_neg_k1` complete across 5 folds.
  - Qwen2.5: generated across 5 folds, but binary labels are incomplete for random controls.
  - Qwen3: only `random_pos_k1` fold 0 generated, not fully labeled.
- `oracle_concise`: BioMistral fold 0 generated only, not binary-evaluated.

## Pilot 12: RA-ICL / Retrieval ICL

Location:

```text
output/pilot_12_ra_icl/
```

Status from `experiment_log.json`: completed through phase 5.

Main tested families:

- Retrieval indices: BM25, GTR, KATE across all 5 folds.
- Positive retrieved example: `gtr_note_pos_k1`.
- Negative retrieved example: `gtr_note_neg_k1`.
- Positive + negative: `gtr_note_posneg_k1`.
- Full-context positive: `gtr_note_fullctx_pos_k1`.
- Type + note retrieval: `gtr_type_note_pos_k1`.
- Mixed-pool retrieval: `gtr_note_any_unlabeled_k1`, `gtr_note_any_labeled_k1`.
- K-shot sweep: `gtr_note_pos_k1` through `gtr_note_pos_k5`.
- Composite prompt variants: guideline, annotated, negative-guideline/full, positive-negative annotated/guideline.

Recorded pilot finding:

- `gtr_note_pos_k1` was best in Pilot 12: 80.14% +/- 2.43.
- k=1 was best; k>1 degraded.
- Mixed-pool retrieval returned to about zero-shot level.
- Composite guideline/annotation variants degraded relative to bare note-positive retrieval.

## Other Pilot Families Present

Pilot output directories with generated/evaluated-style files:

| Directory | Broad purpose from filenames / prior structure |
|---|---|
| `output/pilot` | early BioMistral/Meditron/Qwen prompt tests |
| `output/pilot_4_new_prompt` | new prompt variants |
| `output/pilot_5_context_enhancement` | context-enhancement tests |
| `output/pilot_6_relevance` | relevance prompt tests |
| `output/pilot_7_fullscale` | early fullscale pilot |
| `output/pilot_8_qwen3` | Qwen3 pilot |
| `output/pilot_9_qwen3_icl` | Qwen3 ICL pilot |
| `output/pilot_10_annotated_icl` | annotated ICL pilot |
| `output/pilot_11_qwen3_annotated_icl` | Qwen3 annotated ICL pilot |
| `output/pilot_13_biomistral_prompt` | BioMistral prompt check |
| `output/pilot_14_multimodel` | multimodel pilot |
| `output/pilot_v2` | earlier train/test transfer and context-window study |
| `output/pilot_v2_study` | larger prompt-study family |
| `output/pilot_v3`, `output/pilot_v3_comprehensive` | comprehensive BioMistral prompt/eval pilots |
| `output/pilot_v4_proper` | later proper phase-2 BioMistral pilot family |

These should be treated as provenance/pilot evidence until each family is reviewed against the final test plan.

## Step 9 V2: Regeneration / Self-Correction

Primary location:

```text
output/step9_v2/
output/step9_v2/multi_model/
```

Tests present:

- Regen+count pilots: 100-item and 40-item pilot logs per model.
- V2 pairwise pilot: 40-item audit per model.
- V3 CoVe pilot: 40-item audits for four models, plus additional Llama/Qwen3 phase-2 files.
- RA-ICL pilot in Step 9 V2: 100 items per model.
- Detection-format bakeoff: F1/J1/J2/J3 plus GPT-4o validity gate on 30 items.
- Retriever bakeoff: 12 wrong+detected items across R1-R4 retrievers.
- Verdict quality comparison: v1f versus v1j on accepted corrections.
- Full-scale regen+count: 962/962 items for all five models.
- Cross-detection with Qwen3: 100/100 items for Qwen2.5, Llama3, DeepSeek, and BioMistral.

Recorded full-scale self-correction finding from `FINDINGS_FULLSCALE.md`:

| Model | Baseline | Final | Delta | Notes |
|---|---:|---:|---:|---|
| DeepSeek-R1-8B | 76.92% | 81.60% | +4.68pp | significant, survives Holm correction |
| Llama-3.1-8B | 89.09% | 90.12% | +1.04pp | positive, not significant |
| Qwen3-8B | 92.41% | 93.45% | +1.04pp | positive, not significant |
| Qwen2.5-7B | 88.67% | 88.05% | -0.62pp | negative |
| BioMistral-7B | 53.85% | 51.87% | -1.98pp | negative |

## Human/GPT Judge Samples Preserved

The two provisional 100-case samples requested for later decision are saved at:

```text
refactor/pre_atom_pipeline/reports/human_gt_sample_comparison_seed42.json
```

They compare:

- 100 sampled from the 112 Sara/Jose-agreed pool.
- 100 sampled from the 146 majority-union pool.

These are provisional only and should not be treated as final ground truth selection.

## Current Practical Interpretation

For final-decision review, the strongest already-complete full-scale result family is Step 8 core conditions across all 5 models and 5 folds. Pilot 12 is the strongest provenance for why `gtr_note_pos_k1` / RA-ICL variants were promoted. Step 9 V2 contains the regeneration/self-correction full-scale tests and significance results, but those should be considered after the judge and ground-truth policy is fixed.
