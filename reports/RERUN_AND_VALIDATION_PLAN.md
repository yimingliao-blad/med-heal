# Rerun and Validation Plan

Status: working plan, not final. Details will be revised after the next validation tests.

## Scope We Expect To Rerun

### 1. Step 8 Full-Scale Multi-Model Experiment

Step 8 is the main experiment family to repeat.

Planned repeat surface:

- All 5 models:
  - `biomistral-7b`
  - `qwen2.5-7b-instruct`
  - `qwen3-8b`
  - `deepseek-r1-distill-llama-8b`
  - `llama-3.1-8b-instruct`
- All 5 folds.
- Shared script for all models.
- Processed EHRNoteQA JSONL as the starting point.
- Existing relative folder structure preserved.

Core condition family currently expected to repeat:

- `zeroshot`
- `gtr_note_pos_k1`
- `gtr_note_neg_k1`
- `gtr_note_posneg_k1`
- `cot_evidence`
- `cot_conclusion`
- `multiturn`
- `gtr_note_any_unlabeled_k1`

Important note: Step 8 is repeatable full-scale, but not yet defined as a multirun experiment. We still need to decide what true multirun design means for this project.

### Stage 8 Temperature Policy

Final rerun policy: use `temperature=0.0` for Stage 8 conditions unless the run is explicitly a multirun or regen/self-correction experiment. Existing historical Step 8 outputs used `temperature=0.1` and are retained as provenance.

### 2. Full-Scale Regeneration / Self-Correction

Full-scale regeneration is also in final-test scope.

Current candidate source family:

```text
output/step9_v2/multi_model/
src/step9_self_correction/v2/
```

Existing full-scale regen/count outputs cover all 5 models and 962 items per model. These results will be included in the final test plan, but implementation details and rerun protocol still need confirmation.

## Details Still To Validate Before Finalizing

The following are not final yet. They need targeted tests before we choose the final version.

### Embedding / Retrieval Method

Open questions:

- Which embedding model is final for retrieval?
- Whether GTR note retrieval remains the default.
- Whether BM25, KATE, type+note, mixed pools, or other retrieval choices stay as pilots/provenance only.
- Whether retrieval uses note text, question text, type labels, full context, or combinations.

Existing evidence:

- Pilot 12 favored `gtr_note_pos_k1`.
- k=1 performed best in Pilot 12.
- Mixed pools and larger k generally degraded.

Still needed:

- Small confirmatory tests before finalizing the retrieval method.
- Clear build recipe for retrieval indices and pools.

### Prompt Design

Open questions:

- Final exact prompts for zero-shot, RA-ICL, CoT, multiturn, and regen.
- Whether BioMistral calibration prompt remains isolated from the other model prompts.
- Whether Step 8 prompt wording is preserved exactly or updated.
- Whether CoT variants are included as final methods or as comparison-only methods.

Existing evidence:

- Step 8 has a complete shared script and existing full-scale outputs.
- Stage 1 binary GPT judge is the current working judge prompt.

Still needed:

- Side-by-side prompt comparison table before final selection.
- Confirm exact output schemas for generated and judged files.

### Retrieval Content Construction

Open questions:

- Which records go into positive pools and negative pools.
- Which judge/human labels are trusted when building pools.
- Whether pools are built from BioMistral calibration outputs, Step 8 outputs, or another source.
- Whether retrieval content includes question, answer, note snippets, full notes, or metadata.

Still needed:

- Document pool construction inputs and exclusions.
- Validate no fold leakage.
- Validate index/fold alignment.

### Ground Truth / Judge Alignment

Open questions:

- Final human-ground-truth selection is not decided.
- Two provisional 100-case samples are preserved, but not final.
- The 112 Sara/Jose agreement set is currently only a clean baseline for GPT judge prompt comparison.
- Majority-union human labels remain an alternative for review.

Current working judge prompt:

- GPT-4o Stage 1 binary prompt.
- Step 8 reason-producing judge prompt should not be used for new labels unless explicitly selected for comparison.

## Multirun Decision Still Pending

We have not yet decided the real multirun design.

Current component-level working plan:

- Detection working default: F1 free-form detection plus GPT-4o validity gate.
- Correction working baseline: regen plus count-compare, with span-guided, few-shot, and best-of-K variants still to validate.
- Verdict working default: v1f pairwise contradiction count with ties keeping the original answer.
- Detailed component plan: `reports/MULTIRUN_COMPONENT_PLAN.md`.

Questions to answer next:

- Which stages need multiple stochastic runs?
- Are multiruns for generation, judging, regeneration, retrieval sampling, or all of these?
- How many runs per condition are feasible?
- What random seeds are fixed?
- Which model settings are allowed to vary?
- Are multiruns done for all models or only selected representative models?
- Which subset is used for pilot multirun tests before full rerun?

Candidate designs to test:

1. Single full rerun only for Step 8 and regen; no multirun except pilot validation.
2. Small multirun pilot on one or two models and selected conditions, then one full rerun.
3. Full multirun for a reduced method set only.
4. Full multirun for all selected final methods if compute budget allows.

No option is selected yet.

## Near-Term Next Work

1. Build a side-by-side matrix of candidate Step 8 methods and prompt variants.
2. Define small validation tests for retrieval method and pool construction.
3. Decide what `multirun` means operationally.
4. After those tests, update the final rerun scripts and configs.
5. Only then rerun Step 8 and full-scale regen under the final protocol.
