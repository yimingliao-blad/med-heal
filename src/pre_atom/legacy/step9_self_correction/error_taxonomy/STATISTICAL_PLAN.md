# Statistical Validation Plan

## Goal
Demonstrate that the self-correction pipeline (S1 detect → P1+pool correct → V1 verdict) produces a statistically significant improvement over zeroshot on the full 962-item EHRNoteQA dataset.

## Full-Scale Run Plan

Run the complete pipeline on all 962 items across 5 folds:
- Detection: 3 sub-prompts per item (contra, qmis, omis)
- Correction: P1+pool (type-routed with BM error pool examples) on detected items only
- Verdict: V1 (contradiction count comparison) to accept or reject correction
- GPT-4o eval: on corrected items that pass V1 verdict

Estimated: ~962 × 9 calls for detection + ~600 detected × 6 calls for correction/verdict + ~200 GPT-4o evals
Runtime: ~6-8 hours. GPT-4o cost: ~$3.

## Statistical Tests

### Test 1: Is the pipeline significantly better than zeroshot?

**McNemar's Test** (primary)
- Data: 962 paired binary outcomes (zeroshot correct/incorrect vs pipeline correct/incorrect)
- 2×2 contingency table:
  ```
                    Pipeline correct    Pipeline incorrect
  ZS correct              a (agree)         b (BREAK)
  ZS incorrect            c (FIX)           d (agree)
  ```
- Statistic: χ² = (b - c)² / (b + c)
- H₀: P(break) = P(fix), i.e., the pipeline doesn't change accuracy
- H₁: P(fix) > P(break), i.e., the pipeline improves accuracy
- Report: p-value, odds ratio (c/b), 95% CI for odds ratio
- Significance threshold: p < 0.05

**Bootstrap Confidence Interval** (secondary)
- Resample the 962 items with replacement, 10,000 times
- For each resample: compute accuracy delta (pipeline - zeroshot)
- Report: mean delta, 95% CI, proportion of resamples where delta > 0
- If 95% CI excludes 0 → significant

**Effect Size**
- Cohen's h: standardized difference between two proportions
- Accuracy delta in percentage points
- Number needed to treat (NNT): 1 / (fix_rate - break_rate)

### Test 2: Is the per-fold variation acceptable?

**Per-Fold Net Gain**
- Report: mean ± std of per-fold net gain (fix - break)
- If mean > 0 and std < mean → consistent positive effect

**Leave-One-Fold-Out Analysis**
- For each fold k, compute projected accuracy excluding fold k
- If all 5 leave-one-out estimates are positive → no single fold drives the result
- Report min, max, mean of leave-one-out estimates

**Cochran's Q Test**
- Tests if the correction effect (fix rate) is homogeneous across folds
- H₀: fix rates are equal across folds
- If p > 0.05 → no significant heterogeneity → folds are consistent

**Wilcoxon Signed-Rank Test**
- Paired comparison of per-fold zeroshot accuracy vs pipeline accuracy
- Note: only n=5 pairs → very low power, unlikely to reach significance
- Report as supplementary, not primary evidence

**I² Heterogeneity Statistic**
- From meta-analysis framework: quantifies fraction of variation that is real vs random
- I² < 25%: low heterogeneity (good)
- I² 25-50%: moderate
- I² > 50%: high (concerning)

### Test 3: Are the 5 folds comparable?

**Chi-Square Test on Error Rates**
- Compare the zeroshot error rate across 5 folds
- H₀: all folds have the same error rate
- If p > 0.05 → folds are comparable

**Per-Fold Population Statistics**
- Report for each fold:
  - Number of items
  - Zeroshot accuracy
  - Error count and rate
  - Error type distribution (CONTRADICTION / OMISSION / QUESTION_MISALIGNMENT)
  - Number of multi-note questions
  - Question type distribution (level1 / level2)

**Fisher's Exact Test**
- For small per-fold counts where chi-square may be unreliable

## Reporting Template

```
Zeroshot accuracy: XX.XX% (N=962)
Pipeline accuracy: YY.YY% (N=962)
Improvement: +Z.ZZpp

McNemar's test: χ² = ?, p = ?
  Items fixed: F (wrong → correct)
  Items broken: B (correct → wrong)
  Odds ratio: F/B (95% CI: [?, ?])

Bootstrap 95% CI for accuracy delta: [?, ?]pp

Per-fold results:
  Fold 0: ZS=XX.X% → Pipeline=YY.Y% (net=+N)
  Fold 1: ...
  ...
  Mean net: ±X.X (std=Y.Y)

Cochran's Q: p = ? (fold homogeneity)
I²: X% (heterogeneity)

Chi-square on fold error rates: p = ? (fold comparability)
```

## What Constitutes Success

1. **McNemar's p < 0.05**: the pipeline is significantly better than zeroshot
2. **Bootstrap 95% CI excludes 0**: robust confirmation
3. **All leave-one-out estimates positive**: no single fold dependence
4. **I² < 50%**: acceptable heterogeneity
5. **Per-fold error rates not significantly different**: folds are comparable

## Potential Issues

- **Small effect size**: +2.72pp may require high power to detect. With 962 items and ~26 fixes / ~0 breaks, McNemar's should have sufficient power (expected χ² ≈ 26).
- **Non-determinism**: vLLM at temp=0 still has GPU non-determinism. Running twice may give slightly different results. Mitigation: report the variance from the pilot.
- **GPT-4o judge reliability**: 92% agreement with human gold standard (κ=0.75). Report this as a caveat.
- **Correction stochasticity**: temp=1.0 correction means different runs give different corrections. Mitigation: run correction 3 times and take majority vote (increases cost 3×).

## Pre-Registration

Before running full-scale:
- Pipeline is fixed: S1 detect → P1+pool correct → V1 verdict
- Prompts are fixed (documented in STEP9_FINAL_RESULTS.md)
- Statistical tests are pre-specified (this document)
- Success criteria are pre-specified (above)
- No further tuning after full-scale run
