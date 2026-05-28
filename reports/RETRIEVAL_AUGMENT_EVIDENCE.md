# Retrieval-Augment Evidence Review

Status: evidence review, not final selection. Checked Stage 8, Pilot 12 RA-ICL, Step 9 retrieval bakeoff, and newer ICHL retrieval-study outputs.

## What Stage 8 Uses

Source: `src/step8_multimodel_icl/generate_step8.py`

Stage 8 retrieval is example retrieval for in-context learning, not note-span retrieval for correction.

### Embeddings / Retrievers

Stage 8 uses fold-specific pools and indices:

- Qwen/Llama/DeepSeek/Qwen3 index source: `output/pilot_12_ra_icl/indices/fold_*`
- BioMistral index source: `output/fullscale_4_biomistral/indices/fold_*`
- Embedding model for active Step 8 retrieval: `sentence-transformers/gtr-t5-base`
- Retrieval target for the selected RA-ICL families: discharge-note similarity, not question similarity.

Older Pilot 12 also tried:

- BM25 over questions.
- GTR over questions.
- KATE / MiniLM over questions.
- Type-filtered GTR over questions.
- GTR over note text.
- Type+note retrieval.
- Full-context note retrieval variants.

### Retrieved Content

Stage 8 retrieves training examples from fold-safe pools:

- `correct_pool.json` for positive examples.
- `incorrect_pool.json` for negative examples.
- mixed pool by concatenating correct and incorrect pools for `gtr_note_any_unlabeled_k1`.

The generated prompt shows only compact example fields, not the whole retrieved note:

- Positive example: retrieved `[Question]` + `[Answer]`.
- Negative example: retrieved `[Question]` + `[Incorrect Answer]` + `[Correct Answer]`.
- Pos+neg: one negative example plus one positive example.
- Multiturn: retrieved positive example becomes a prior user/assistant turn.
- Mixed unlabeled: retrieved `[Question]` + `[Answer]` without telling the model whether it was correct.

### Prompt Families Tried In Stage 8

Core Stage 8 conditions:

- `zeroshot`
- `gtr_note_pos_k1`
- `gtr_note_neg_k1`
- `gtr_note_posneg_k1`
- `cot_evidence`
- `cot_conclusion`
- `multiturn`
- `gtr_note_any_unlabeled_k1`

Extra retrieval sweeps:

- `gtr_note_neg_k2` through `gtr_note_neg_k5`
- `random_pos_k1`
- `random_neg_k1`

## Pilot 12 Findings

### Phase 1, Fold 0, N=50

Source: `output/pilot_12_ra_icl/results/pilot12_summary.json`

Best methods:

| Method | Accuracy |
|---|---:|
| `gtr_note_pos_k1` | 76% |
| `gtr_note_fullctx_pos_k1` | 76% |
| `random_pos_k1` | 74% |
| `gtr_type_pos_k1` | 74% |
| `bm25_pos_k1` | 66% |

Interpretation: note-similar positive k=1 was the best simple retrieval candidate, but the margin over random positive k=1 was small in this pilot.

### Pilot 12 Fullscale, Qwen2.5, 5 Folds

Source: `output/pilot_12_ra_icl/fullscale/results/fullscale_pilot12_summary.json`

| Method | Mean | Delta vs zeroshot |
|---|---:|---:|
| `zeroshot` | 77.13 | 0.00 |
| `gtr_note_pos_k1` | 80.14 | +3.01 |
| `gtr_note_fullctx_pos_k1` | 79.93 | +2.80 |
| `guideline_pos_ann` | 79.21 | +2.08 |
| `gtr_note_posneg_k1` | 78.79 | +1.66 |
| `gtr_note_neg_k1` | 76.71 | -0.42 |

None of the fold-level t-tests were significant in this file, but `gtr_note_pos_k1` was the best mean and became the main Stage 8 retrieval candidate.

### Phase 3 Prompt Additions

Source: `output/pilot_12_ra_icl/pilot_phase3/fold_0/phase3_pilot_summary.json`

Tried guideline, annotated, negative-full, and posneg annotated/guideline variants. None beat the simple Phase 1 `gtr_note_pos_k1` / `gtr_note_fullctx_pos_k1` fold-0 baselines of 76%.

## Stage 8 Multi-Model Results

Source: `output/step8/results/exp1_summary.json` and `output/step8/results/step8_results_summary.csv`

The retrieval effect is model-dependent.

Compact deltas from `exp1_summary.json`:

| Model | Best retrieval/prompt family | Delta vs zeroshot |
|---|---|---:|
| BioMistral | none; zeroshot best | retrieval hurt |
| Qwen2.5 | `multiturn`, then `gtr_note_pos_k1` | +3.32 / +2.59 |
| Llama3 | `gtr_note_pos_k1` | +1.04 |
| Qwen3 | `multiturn`, `gtr_note_neg_k1`, `gtr_note_pos_k1` | +3.96 / +3.64 / +3.54 |

The newer `step8_results_summary.csv` has higher judged accuracies for Qwen2.5/Qwen3/Llama and shows more conservative effects:

- Qwen2.5: zeroshot 88.67, `gtr_note_pos_k1` 88.57, no gain.
- Qwen3: zeroshot 92.41, CoT/multiturn best; `gtr_note_pos_k1` 93.14, +0.73.
- Llama3: zeroshot 89.09, `gtr_note_neg_k5` 89.51, small gain; `gtr_note_pos_k1` roughly tied.
- BioMistral: zeroshot 53.85, retrieval variants mostly lower.

Interpretation: the old Qwen2.5 Pilot 12 evidence favored `gtr_note_pos_k1`; later judged Step 8 summaries make retrieval less uniformly beneficial and show model-specific behavior.

## Negative k-Sweep

Source: `output/step8/results/exp2_neg_ksweep_summary.json`

- Qwen2.5: `gtr_note_neg_k2` was the best negative-example k, but only +0.31 over zeroshot in the older summary; larger k generally degraded.
- Llama3: all negative k values were below zeroshot in the older summary.
- Qwen3: `gtr_note_neg_k1` was best; larger k declined monotonically or near-monotonically.

Working lesson: if using negative examples, keep k small. k=1 is the safest default; k=2 is only a Qwen2.5 candidate.

## Newer ICHL Retrieval Study

Sources:

- `src/ichl/retrieval_study/embedding_bakeoff.py`
- `src/ichl/retrieval_study/multi_component_scorer.py`
- `src/ichl/retrieval_study/raicl_pilot.py`
- `output/ichl/retrieval_study/bakeoff_table.md`
- `output/ichl/retrieval_study/multi_component_table.md`
- `output/ichl/retrieval_study/phase0_redo_gte_table.md`

### Embedding Bakeoff

The newer retrieval study moved beyond GTR and compared longer-context embeddings on full notes.

Embedding candidates tested:

- `nomic-embed-text-v1.5`
- `bge-m3`
- `gte-large-en-v1.5`

From `bakeoff_table.md`, `gte-large-en-v1.5` was strongest overall:

| Embedding | Clinical context rho | Critical detail rho | NDCG@3 clinical |
|---|---:|---:|---:|
| nomic | 0.348 | 0.281 | 0.875 |
| bge-m3 | 0.471 | 0.373 | 0.865 |
| gte-large-en-v1.5 | 0.566 | 0.455 | 0.905 |

Working retrieval embedding from this newer study: `Alibaba-NLP/gte-large-en-v1.5`.

### What To Retrieve / Score

The newer study found note-only retrieval is not enough for all desired dimensions.

`multi_component_table.md`:

| Method | Composite |
|---|---:|
| nomic note-only | 0.452 |
| nomic question-only | 0.445 |
| nomic GT-only | 0.414 |
| equal-weighted note+question+GT | 0.627 |
| fitted Ridge note+question+GT | 0.637 |

`phase0_redo_gte_table.md`:

- `cos_note only`: composite 0.508
- `cos_q only`: composite 0.389
- `cos_gt_gt only`: composite 0.479
- `2-comp (q + note)`: composite 0.581
- `3-comp + cos(GT,GT)`: composite 0.643

Working retrieval scorer from newer study: multi-component scoring, especially question + note + ground-truth-style alignment, beats note-only retrieval for selecting similar examples.

### RA-ICL Prompt Modes Tried

Source: `src/ichl/retrieval_study/raicl_pilot.py`

Modes:

- Mode A: error-similarity, pool filtered to BioMistral wrong items; prompt shows reference case with incorrect answer and correct answer.
- Mode B: correct-similarity, pool filtered to BioMistral correct items; prompt shows reference case with correct answer.
- Mode C: GT-similarity, full pool; prompt shows reference case with correct answer.
- Mode D: self-revision with exemplar; prompt shows reference case and the model's own zero-shot answer, then asks for final answer.

Generation params in this newer RA-ICL pilot were still matching historical Step 8: temperature 0.1, max_tokens 1024. This conflicts with the new policy to use Stage 8 T=0, so final reruns need updated temperature.

### Newer RA-ICL Outcome

The newer RA-ICL pilots are mixed and often negative relative to zero-shot, especially under GPT-4o judging for Qwen2.5:

- Qwen2.5 lockdown Mode C was consistently below zero-shot across folds under GPT-4o summaries.
- Mode D occasionally improved a fold but did not show stable broad gain.
- Qwen3 Mode C/D had some fold-level wins under Magistral, but not stable enough to call final.
- DeepSeek Mode C/D was usually below zero-shot under Magistral summaries.

Working lesson: newer embedding/scoring improves retrieval-quality metrics, but the RA-ICL prompt/application has not yet shown a reliable downstream accuracy gain.

## Step 9 Retrieval Bakeoff For Correction

Source: `output/step9_v2/retriever_bakeoff.md`

This is note-span retrieval for correction, not Stage 8 example retrieval.

| Retriever | Sufficient spans |
|---|---:|
| R1 single-query embedding on error statement, top-3 | 2/12 = 17% |
| R2 multi-query embedding + agreement scoring, top-5 | 5/12 = 42% |
| R3 Qwen cite-by-number K=5, question-only | 4/12 = 33% |
| R4 union(R3,R2) top-5 | 4/12 = 33% |

Working correction-retrieval candidate: R2 multi-query embedding + agreement scoring, top-5. This should not be confused with Stage 8 RA-ICL example retrieval.

## Current Best Working Takeaways

1. For classic Stage 8 RA-ICL, the best historical simple retrieval condition is `gtr_note_pos_k1`: GTR-T5 over discharge-note text, retrieve one correct example, prompt with the example question and answer.
2. Full multi-model Stage 8 results are not uniformly pro-retrieval. Retrieval helps Qwen3 and sometimes Llama/Qwen2.5, hurts BioMistral, and can be weaker than CoT or multiturn depending on model and judge version.
3. Negative examples should stay at k=1 if used; larger k often degrades.
4. Newer ICHL retrieval-quality evidence favors `gte-large-en-v1.5` and multi-component scoring over GTR note-only retrieval, but the RA-ICL downstream prompts have not yet produced stable gains.
5. For correction/regen workflows, the best retrieved content is not an example QA pair; it is evidence spans. The best current correction retrieval is R2 multi-query span retrieval, top-5.

## Candidate Next Validation

Before finalizing retrieval augmentation, run a small deterministic validation using the new Stage 8 temperature policy:

- Baseline: zero-shot T=0.
- Historical simple RA-ICL: `gtr_note_pos_k1`, k=1, prompt as Stage 8.
- Newer retrieval-quality candidate: `gte-large-en-v1.5` multi-component scorer, Mode C or Mode D prompt.
- Optional negative example: `gtr_note_neg_k1` only, not larger k.
- Models: Qwen3 and Qwen2.5 first.
- Judge: fixed old Stage 1 GPT-4o prompt/data construction, T=0.1, sequential.
