# Zero-Shot Smoke Confirmation

Status: checked from existing files only; no rerun performed.

## Wiring Check

BioMistral Step 8 zero-shot files are present for all five folds:

- `output/step8/biomistral-7b/fold_*/zeroshot_generated.csv`
- `output/step8/biomistral-7b/fold_*/zeroshot_evaluated_binary.csv`

The Step 8 generator is wired through:

- `src/step8_multimodel_icl/generate_step8.py`
- BioMistral system prompt: `You are a helpful, respectful and honest assistant.`
- Prompt format: Llama2 `[INST]` style with discharge summary, question, and `Answer:`

## Temperature

The zero-shot generation temperature in `generate_step8.py` is **not T=0**.

- Local vLLM zero-shot generation temperature: `0.1`
- Old Stage 1 GPT judge temperature now selected as fixed config: `0.1`
- GPT judge calls should remain sequential.
- Local vLLM runs may use `c=8`.

The previous T=0 setting belonged to the newer Step 9 deterministic judge wrapper, not the old Stage 1 judge configuration we selected.

## Matching Check

The existing Step 8 BioMistral zero-shot evaluated files do not match the old Stage 1 BioMistral record label-by-label because they are mostly different generated answers.

Observed from existing files:

- Step 8 BioMistral rows: 962
- Old Stage 1 BioMistral rows: 962
- Patient overlap: 962
- Label agreement: 702/962 = 72.97%
- Exact answer text match: 35/962 = 3.64%

This means Step 8 BioMistral zero-shot is wired and executable, but it should be treated as the Step 8 generated artifact, not as a direct reproduction of the old Step 2/Stage 1 BioMistral record.

## Decision

For final judge configuration, use old Stage 1 prompt/data construction and `temperature=0.1`.

For Step 8 zero-shot smoke confirmation, the existing files are complete and wired. The generation test temperature is `0.1`, not `0.0`.


## Updated Temperature Decision

Final decision for future Stage 8 reruns:

- Use `temperature=0.0` for Stage 8 generation/evaluation conditions.
- Exception: multirun and regen/self-correction experiments keep their separate tested settings.
- Existing historical Step 8 files were generated with `temperature=0.1`; they remain provenance and smoke-confirmation artifacts, not the final rerun temperature policy.
- Old Stage 1 GPT judge provenance still used `temperature=0.1` to reproduce the old GPT record; this is separate from the Stage 8 generation temperature policy.
