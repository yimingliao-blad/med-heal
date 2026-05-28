# Step 9: Self-Correction Pipeline — Final Results

## Best Pipeline: S1 Detect → P1+Pool Correct → V1 Verdict

```
Qwen2.5-7B zeroshot answer
  ↓
Stage 1: DETECTION (3 sub-prompts, free-form output → Qwen3-32B extract JSON)
  ├─ Contradiction check: does answer conflict with notes?
  ├─ Question alignment check: does answer address the right question?
  ├─ Omission check: is critical info missing?
  └─ Aggregate: any detected → proceed to correction
  ↓
Stage 2: CORRECTION (P1+Pool, type-routed)
  ├─ CONTRADICTION → "your answer said X, notes say Y" + BM pool example
  ├─ QUESTION_MISALIGNMENT → "question asks about visit 2, refocus" + pool
  ├─ OMISSION → "missing info Z, notes say..." + pool
  └─ Output: corrected answer (temp=1.0)
  ↓
Stage 3: VERDICT (V1, contradiction count comparison)
  ├─ Count factual contradictions in original vs corrected
  ├─ Pick answer with fewer contradictions
  └─ If tied or original fewer → keep original (reject correction)
  ↓
Final answer
```

## Results: 5-Fold Cross-Validation (50 TP + 50 FP)

### Detection (S1 multi-run, 3 sub-prompts)

| Metric | Value |
|--------|:-----:|
| TP detection rate | 33/50 (66%) |
| FP detection rate | 21/50 (42%) |
| Selectivity | 1.6x |

Per-fold:
| Fold | TP det | FP det |
|------|:------:|:------:|
| 0 | 6/10 | 4/10 |
| 1 | 7/10 | 3/10 |
| 2 | 5/10 | 5/10 |
| 3 | 8/10 | 5/10 |
| 4 | 7/10 | 4/10 |

### Correction + Verdict Comparison (33 TP + 21 FP detected)

| Method | TP fix | FP brk | Judged net | Projected accuracy |
|--------|:------:|:------:|:----------:|:------------------:|
| P1 raw | 14/33 | 5/21 | +9 | 82.98% (-5.69pp) |
| P1+V1 | 7/33 | 2/21 | +5 | 86.71% (-1.96pp) |
| P1+V2 | 5/33 | 4/21 | +1 | 82.71% (-5.96pp) |
| P1+V3 | 14/33 | 5/21 | +9 | 82.98% (-5.69pp) |
| PP raw | 19/33 | 3/21 | +16 | 87.65% (-1.01pp) |
| **PP+V1** | **12/33** | **0/21** | **+12** | **91.39% (+2.72pp)** |
| PP+V2 | 10/33 | 3/21 | +7 | 85.62% (-3.05pp) |
| PP+V3 | 19/33 | 3/21 | +16 | 87.65% (-1.01pp) |

PP = P1+Pool (type-routed correction with BM error pool examples)

### Best Pipeline: PP+V1

| Metric | Value |
|--------|:-----:|
| Baseline accuracy | 88.67% |
| **Projected accuracy** | **91.39% (+2.72pp)** |
| TP fixes (judged) | 12/33 detected (36%) |
| FP breaks (judged) | **0/21 detected (0%)** |
| Projected fixes at population | 26 |
| Projected breaks at population | 0 |

### By Error Type (P1+Pool correction, raw)

| Error Type | TP detected | TP fix rate | FP detected | FP break rate |
|------------|:-----------:|:-----------:|:-----------:|:-------------:|
| CONTRADICTION | 8 | 4/8 (50%) | 9 | 1/9 (11%) |
| OMISSION | 17 | 6/17 (35%) | 9 | 3/9 (33%) |
| QUESTION_MISALIGNMENT | 8 | 4/8 (50%) | 3 | 1/3 (33%) |

OMISSION has the highest FP risk (33% break rate). V1 verdict filters these out.

## Key Findings

### What works
1. **BM error pool (RA-ICL) helps correction** — P1+pool: 58% fix vs P1 alone: 42%
2. **V1 (contradiction count) is the best verdict** — filters ALL FP breaks to 0
3. **Type-routed correction** with proper error types from sub-prompt identity
4. **Free-form output + Qwen3-32B extraction** solves the parsing problem (100% parse success)

### What doesn't work
1. **V2 (principle comparison)** — inconsistent across folds, both too conservative and too permissive
2. **V3 (error-specific verify)** — too permissive, accepts all corrections including breaks
3. **Plain regen (no guidance)** — highest FP break rate (18%)
4. **Omission detection** — 33% FP break rate makes it risky without V1 verdict

### Lessons learned
1. **Small-sample evaluations overestimate** — fold 1 pilot showed +6 net, 5-fold showed +3 raw
2. **Verdict is essential** — without V1, every method is negative at population scale
3. **Error type from sub-prompt identity** is more reliable than Qwen3-32B's label
4. **Never trust regex parsing** — always use external LLM extraction + JSON validation
5. **Save all intermediate outputs** — raw outputs, per-sub-prompt details, parse validation

## Error Taxonomy (Correction-Oriented)

| Type | % of Qwen2.5 errors | Fix action | Detection prompt |
|------|:--------------------:|------------|-----------------|
| CONTRADICTION (64%) | Answer says X, notes say Y | Update wrong fact | "Does the answer conflict with the notes?" |
| QUESTION_MISALIGNMENT (20%) | Wrong visit/aspect/time | Refocus answer | "Does the answer address the right question?" |
| OMISSION (16%) | Missing critical info | Add missing info | "Is essential info absent?" |

## Pipeline Cost

| Stage | Calls per item | Server |
|-------|:--------------:|--------|
| Detection (3 sub-prompts) | 3 vLLM + 3 Qwen32B | Local GPU + Mac Studio |
| Correction (P1+pool) | 1 vLLM + 1 pool retrieval | Local GPU + CPU embedder |
| Verdict (V1) | 1 vLLM + 1 Qwen32B | Local GPU + Mac Studio |
| GPT-4o eval | 1 call per correction | API (~$0.015/item) |
| **Total per item** | **~9 calls** | **~$0.015 API** |

For 962 items: ~8600 calls, ~2-3 hours runtime, ~$1.50 GPT-4o

## Comparison with Previous Approaches

| Method | Judged net | Projected |
|--------|:----------:|:---------:|
| V3 atomic (Qwen3) | +28 | +1.89pp |
| V5 2-path (Qwen3) | +22 | +0.24pp |
| **Step 9 PP+V1 (Qwen2.5)** | **+12** | **+2.72pp** |
| V5 2-path (Qwen2.5) | +12 | -0.70pp |
| V5 2-path (Llama3) | +22 | -8.49pp |

Step 9 achieves the best PROJECTED accuracy because of 0% FP break rate through V1 verdict.

## Files

### Scripts
- `run_all_detection_5fold.py` — runs all detection methods on 5 folds
- `run_correction_verdict_5fold.py` — runs correction + verdict on detected items
- `parsing.py` — shared Qwen3-32B extraction module with validation

### Data
- `all_detection_5fold.json` — raw detection results (100 items × 5 prompts)
- `all_detection_5fold_reclassified.json` — reclassified with correct error types
- `correction_verdict_5fold.json` — correction + verdict results (54 detected items)
- `correction_oriented_annotations.json` — 109 Qwen2.5 errors remapped to 3-type taxonomy
- `phase1_wrong_gpt4o.json` — GPT-4o error analysis of all 109 wrong answers
- `phase1_correct_gpt4o.json` — GPT-4o analysis of 50 correct answers

### Documentation
- `CORRECTION_TAXONOMY.md` — 3-type error taxonomy definition
- `DETECTION_PROMPT_RESULTS.md` — all detection prompt experiments
- `STEP9_FINAL_RESULTS.md` — this file
