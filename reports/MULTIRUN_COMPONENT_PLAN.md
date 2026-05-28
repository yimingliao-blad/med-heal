# Multirun Component Plan

Status: working component decision, not final. This captures the current best choices from Step 9 V2 and defines the next validation tests before selecting the final multirun protocol.

## Source Evidence

- `output/step9_v2/detection_bakeoff_summary.md`
- `output/step9_v2/STEP9_V2_RESULTS.md`
- `output/step9_v2/STATE_AND_FINDINGS.md`
- `output/step9_v2/retriever_bakeoff.md`
- `src/step9_self_correction/v2/`

## Working Defaults

### Detection

Working default: **F1 free-form detection plus GPT-4o validity gate**.

Reason:

- F1 alone had the best recall signal but noisy rationales.
- Direct JSON variants were too conservative.
- F1 plus validity gate reduced false positives while preserving usable signal.
- Existing bakeoff result: TP 33%, FP 7%, validity 100% by construction on the small N=30 bakeoff.

Operational rule:

- If F1 detects an error but the GPT-4o validity gate says the captured correction rationale is not note-supported or does not address the actual error, treat the item as no detection and keep the original answer.

Alternative to keep for comparison:

- J2 direct JSON with three sub-prompts as a high-precision, low-recall detection baseline.

### Correction

Current best empirical result: **regen plus count-compare verdict**.

Evidence:

| Model | Baseline | Final | Delta | Net |
|---|---:|---:|---:|---:|
| DeepSeek-R1-8B | 76.92% | 81.60% | +4.68pp | +45 |
| Llama-3.1-8B | 89.09% | 90.12% | +1.04pp | +10 |
| Qwen3-8B | 92.41% | 93.45% | +1.04pp | +10 |
| Qwen2.5-7B | 88.67% | 88.05% | -0.62pp | -6 |
| BioMistral-7B | 53.85% | 51.87% | -1.98pp | -19 |

Interpretation:

- Regen is the strongest completed full-scale correction family so far.
- It is model-dependent: clearly positive for DeepSeek, modest for Llama/Qwen3, negative for Qwen2.5/BioMistral.
- It should be repeated in the final protocol, but not treated as the only correction design until targeted tests are done.

Correction variants to test next:

1. **Zero-shot regen baseline**
   - Same model regenerates the answer from scratch using the same zero-shot prompt family.
   - This remains the anchor condition because it has complete full-scale evidence.

2. **Span-guided correction**
   - Use note-span retrieval before correction.
   - Current best retrieval evidence favors R2: multi-query embedding plus agreement scoring, top-5 spans, 5/12 sufficient in the retriever bakeoff.
   - Keep the existing refusal rule when evidence quality is weak.

3. **Few-shot correction**
   - Test whether a high-quality example helps correction, not just ICL generation.
   - Do not use the BioMistral contrast pool as-is; its audited pair coherence was only 42% and RA-ICL with the BM contrast pool was net-negative for 4/5 models.
   - Build few-shot examples only from audited high-confidence fix pairs, with no fold leakage.
   - Test 1-shot first. Larger k is risky because earlier ICL evidence showed larger k often degraded performance.

4. **Best-of-K correction**
   - Generate multiple correction candidates and let the verdict choose among original and candidates.
   - Existing Step 9 notes say the pilot effectively generated multiple candidates but used the first; candidate-level pairwise verdict may improve yield.
   - Recommended first test: K=3 correction candidates.

5. **V3 CoVe comparison**
   - Keep as a comparison, not the leading candidate.
   - It was more complex and slower, with similar net performance to regen and format-following problems for Llama.

### Verdict

Working default: **v1f free-form pairwise contradiction count, Qwen3-32B extraction, ties keep original**.

Reason:

- v1f accepted more useful fixes than v1j in the 60-item pilot.
- v1j was too conservative and accepted no real fixes in that pilot.
- Ties should keep the original answer to avoid unnecessary breaks.

Known weakness:

- Count-compare has a high false-positive rate on correct items, around 55% across models in self/cross detection analyses.
- This makes verdict precision the main bottleneck.

Alternative to test:

- GPT-4o pairwise verdict as an expensive upper-bound test, to separate prompt/method limits from small-model verdict limits.

## Proposed Multirun Design

### Pilot Before Full Rerun

Use a controlled pilot before scaling.

Recommended sample:

- 100 to 150 items per selected model.
- Balanced wrong/correct where possible for detection and correction diagnostics.
- Include at least two models:
  - DeepSeek-R1-8B, because regen currently helps most.
  - Qwen2.5-7B, because regen was slightly negative and is a good fragility check.
- Optional third model:
  - Qwen3-8B or Llama-3.1-8B, because their baselines are high and gains are small.

### Detection Multirun Test

Compare:

- F1 plus validity gate, K=1.
- F1 plus validity gate, K=3 with majority or any-valid detection.
- F1 plus validity gate, K=5 with majority or any-valid detection.
- J2 direct JSON as high-precision comparison.

Metrics:

- TP rate on known-wrong items.
- FP rate on known-correct items.
- Valid rationale rate.
- Downstream fix/break effect after correction and verdict.
- Cost and runtime.

Expected decision:

- Pick the smallest K that improves TP without materially increasing FP or invalid rationales.

### Correction Few-Shot Test

Compare on the same detected candidate set:

- Regen only.
- Span-guided correction with R2 retrieval.
- 1-shot correction using audited high-confidence fix examples.
- Span-guided plus 1-shot correction.
- Best-of-3 correction candidates with verdict selection.

Metrics:

- Fix count.
- Break count.
- Neutral accepted count.
- Net change.
- Fix:break ratio.
- Correction acceptance rate.
- GPT judge agreement on accepted corrections.

Expected decision:

- A few-shot method must beat regen on net and must not increase breaks on correct cases.
- If few-shot improves DeepSeek but harms high-baseline models, keep it as model-specific rather than global.

### Verdict Multirun Test

Compare:

- v1f single verdict.
- v1f repeated K=3 with majority vote.
- GPT-4o pairwise verdict on a small upper-bound sample.

Metrics:

- Accepted fixes.
- Accepted breaks.
- Accepted neutral corrections.
- Precision among accepted corrections.
- Tie rate.
- Runtime and cost.

Expected decision:

- Keep v1f if multivote does not improve precision.
- Use GPT-4o verdict only as an upper-bound diagnostic unless the project explicitly accepts the cost and external-judge dependency.

## Recommended Near-Term Decision

For the next round, use this as the working multirun setup:

| Stage | Working Choice | Why |
|---|---|---|
| Detection | F1 plus GPT-4o validity gate | Best current balance of recall, FP control, and rationale validity |
| Correction | Regen baseline plus new span/few-shot/best-of-K tests | Regen is best completed result, but correction is not finalized |
| Verdict | v1f pairwise contradiction count, ties keep original | Better than v1j in pilot; preserves safety on ties |

The correction module is the main unresolved piece. A good few-shot correction may help, but only if the examples are audited and fold-safe. The existing BM contrast pool should not be reused directly for final few-shot correction.

## Required Reproducibility Checks

- Fixed sample seed for pilot item selection.
- Explicit model, temperature, max-token, and served-model IDs in output metadata.
- Stable patient/fold keys in every intermediate output.
- No cross-fold leakage in few-shot examples or retrieval pools.
- Save all prompts used for detection, correction, verdict, and judge.
- Save all raw model outputs before parsing.
- Save parsed outputs with parser version.
- Run McNemar tests on paired baseline vs corrected outcomes after final selection.
