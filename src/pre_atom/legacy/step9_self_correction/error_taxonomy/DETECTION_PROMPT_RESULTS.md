# Detection Prompt Test Results — Round 1

## Setup
- Model: Qwen2.5-7B-Instruct (self-critique, no ground truth)
- Test set: 50 wrong + 50 correct zeroshot answers (balanced)
- 6 prompt variants tested

## Results

| Prompt | Strategy | Wrong det | Correct det | Selectivity |
|--------|----------|:---------:|:-----------:|:-----------:|
| A1 alignment | Rule: question focus | 4/50 (8%) | 0/50 (0%) | 8.0x |
| A2 evidence | Rule: faithful to notes | 15/50 (30%) | 4/50 (8%) | 3.8x |
| A3 details | Rule: key details covered | 3/50 (6%) | 0/50 (0%) | 6.0x |
| B CoT | CoT 3-step check | 5/50 (10%) | 4/50 (8%) | 1.2x |
| C few-shot | Few-shot 4 examples | 16/50 (32%) | 14/50 (28%) | 1.1x |
| **BC CoT+few-shot** | **CoT + few-shot** | **12/50 (24%)** | **1/50 (2%)** | **12.0x** |
| A combined | Any of A1/A2/A3 | 16/50 (32%) | 4/50 (8%) | 4.0x |

## Key Findings

1. **BC (CoT + few-shot) is the best** — 12.0x selectivity, 24% recall
   - CoT structure prevents few-shot from over-triggering
   - Few-shot examples teach the model WHAT to look for
   - CoT forces step-by-step verification BEFORE verdict

2. **Few-shot alone is dangerous** — 28% FP rate, no selectivity
   - Examples prime the model to find errors everywhere
   - Without structured reasoning, model pattern-matches too aggressively

3. **CoT alone is too conservative** — 10% recall
   - Without examples, model defaults to "looks fine"
   - Self-confirmation bias dominates

4. **A2 (evidence check) is the best individual rule** — 30% recall, 8% FP
   - Direct claim-by-claim verification works
   - But 8% FP would still cause breaks at scale

5. **A1 and A3 are very safe but too conservative** — <8% recall, 0% FP

## Next Steps
- Iterate on BC prompt: add more examples, adjust CoT structure
- Test BC + A2 combination (parallel run, flag if either detects)
- Cross-model test: run BC on DeepSeek, Llama3, Qwen3

## Round 2 Results (MISREADING-focused iterations)

| Prompt | Wrong det | Correct det | Select | MISREAD det |
|--------|:---------:|:-----------:|:------:|:-----------:|
| BC (R1 best) | 12/50 (24%) | 1/50 (2%) | 12.0x | 5/30 (17%) |
| BC2 quote-check | 12/50 (24%) | 5/50 (10%) | 2.4x | 7/30 (23%) |
| BC3 misread examples | 5/50 (10%) | 6/50 (12%) | 0.8x | 4/30 (13%) |
| BC4 notes-first | 8/50 (16%) | 2/50 (4%) | 4.0x | 3/30 (10%) |

### Round 2 Findings

1. **BC remains the best prompt** — 12.0x selectivity is the ceiling for this approach
2. **MISREADING is the hardest error to self-detect** — model re-reads notes the same wrong way
3. **Targeted examples backfire** — BC3's misreading examples caused false positives (0.8x selectivity)
4. **Quote-checking helps recall but hurts precision** — BC2 found 2 more misreadings but 4 more FP
5. **Notes-first doesn't help** — BC4's extraction step didn't break the self-confirmation cycle

### Error-Specific Detection Rates (BC, best prompt)

| Error Type | % of wrong | Detection rate | Implication |
|------------|:----------:|:--------------:|-------------|
| MISREADING | 60% | 17% | Hardest — self-confirmation |
| QUESTION_MISALIGNMENT | 20% | 30% | Moderate — alignment check helps |
| OMISSION | 18% | 33% | Moderate — completeness check helps |
| FABRICATION | 2% | 100% | Easiest — not in notes = detectable |

### Conclusion
The BC (CoT + few-shot) prompt achieves 24% wrong detection with 2% FP rate (12.0x selectivity). This is significantly better than V5's detection approach. The main limitation is MISREADING (17% detection) — the dominant error type that resists self-detection.

For a correction pipeline, BC would intervene on ~24% of wrong answers while barely touching correct ones. If correction success rate is similar to V5 (~50-60%), this projects to ~12-14% net fix rate with minimal breaks.

## Cross-Model BC Detection Pilot (25 wrong + 25 correct each)

| Model | Wrong det | Correct det | Selectivity |
|-------|:---------:|:-----------:|:-----------:|
| **Qwen2.5** | 6/25 (24%) | 0/25 (0%) | **24.0x** |
| Qwen3 | 8/25 (32%) | 3/25 (12%) | 2.7x |
| Llama3 | 3/25 (12%) | 5/25 (20%) | 0.6x |
| DeepSeek | 1/25 (4%) | 0/25 (0%) | 4.0x |

### Cross-Model Findings

1. **The BC prompt is Qwen2.5-specific** — 24.0x selectivity on Qwen2.5 but doesn't generalize
2. **Llama3 has inverted detection** — flags correct answers more than wrong ones (0.6x)
3. **DeepSeek is nearly blind** — thinking model's confidence blocks self-detection (4%)
4. **Qwen3 is a middle ground** — 32% recall but 12% FP, not good enough for production
5. **Self-critique ability correlates with model architecture**, not just capability:
   - Non-thinking models (Qwen2.5, Llama3) respond differently to structured prompts
   - Thinking models (DeepSeek, Qwen3) may need prompts designed for their reasoning style

### Implications

- **No universal self-critique prompt exists** for 7-8B models
- Each model needs its own tuned detection prompt
- OR: detection should use a different approach entirely (external model, ensemble, etc.)
- The 24% detection × 12x selectivity on Qwen2.5 is a valid result for a single model

## Qwen3 No-Think Test

| Mode | Wrong det | Correct det | Selectivity |
|------|:---------:|:-----------:|:-----------:|
| Qwen3 THINK | 8/25 (32%) | 3/25 (12%) | 2.7x |
| Qwen3 NO-THINK | 5/25 (20%) | 2/25 (8%) | 2.5x |
| Qwen2.5 (ref) | 6/25 (24%) | 0/25 (0%) | 24.0x |

No-think didn't help — slightly worse on both recall and selectivity.

## CRITICAL CAVEAT: Parsing Confidence

**All detection rates reported above may be UNDERCOUNTED due to parsing failures.**

Evidence: In Round 1 BC test on Qwen2.5, 20/100 items had ERROR_TYPE output text but were counted as "not detected" because `VERDICT: INCORRECT` wasn't in the expected exact format. The model may have:
- Used markdown bold: `**VERDICT: INCORRECT**`
- Omitted VERDICT but included ERROR_TYPE
- Used different phrasing: "The answer is incorrect"
- Had the verdict buried in reasoning text

This is especially problematic for:
- **DeepSeek** (1/25 detected) — thinking model output may not follow the template
- **Llama3** (3/25 wrong, 5/25 correct) — may have different formatting

**Action needed before trusting these numbers:**
1. Re-run all tests saving raw outputs
2. Audit parse success rate per model
3. Use GPT-4o or Qwen3-32B to parse ambiguous outputs
4. Report corrected detection rates with parse-adjusted confidence intervals

## Round 4-5 Results (Free-form + Qwen32B extraction pipeline)

### Pipeline validated:
- Qwen2.5 free-form output (max_tokens=2048) → Qwen3-32B extracts JSON
- 100% parse success across all tests
- Qwen32B vs GPT-4o: 90% agreement on verdict extraction
- Verbose output is necessary — concise prompts kill detection entirely

### Round 4 (10+10 fold 0):
| Prompt | Wrong | Correct | Selectivity |
|--------|:-----:|:-------:|:-----------:|
| D1 CoT+fewshot | 5/10 | 2/10 | 2.5x |
| D2 claims verify | 3/10 | 0/10 | 30.0x |
| D3 strict teacher | 9/10 | 5/10 | 1.8x |
| D4 devil's evidence | 8/10 | 6/10 | 1.3x |
| D5 question focus | 0/10 | 2/10 | 0.0x |

### Round 5 — D2 variations (10+10 fold 0):
| Prompt | Wrong | Correct | Selectivity |
|--------|:-----:|:-------:|:-----------:|
| D2c requirements | 2/10 | 0/10 | 20.0x |
| D2d more claims | 4/10 | 2/10 | 2.0x |

### Fold 1 cross-validation (10+10):
| Prompt | Wrong | Correct | Selectivity |
|--------|:-----:|:-------:|:-----------:|
| D1 CoT+fewshot | 0/10 | 2/10 | 0.0x |
| D2c requirements | 1/10 | 0/10 | 10.0x |
| **D2d more claims** | **5/10** | **0/10** | **50.0x** |

### D2d FULL-SCALE (109 wrong + 50 correct):
| Metric | Value |
|--------|:-----:|
| Wrong detected | 49/109 (45%) |
| **Correct FP** | **16/50 (32%)** |
| **Selectivity** | **1.4x** |
| Parse fail | 0/159 |
| Error types | OMISSION 92%, MISREADING 8% |

### CRITICAL FINDING: Small-sample selectivity does NOT predict full-scale

| D2d selectivity | 10+10 fold 0 | 10+10 fold 1 | 109+50 fullscale |
|-----------------|:------------:|:------------:|:----------------:|
| **Measured** | 2.0x | 50.0x | **1.4x** |

The 10-item correct samples happened to contain "clean" items. At 50 items, many correct answers have minor omissions that trigger OMISSION detection (32% FP). The claim-by-claim + completeness check is too sensitive.

### FP Analysis
All 16 FP items were flagged as OMISSION — the model finds "missing details" in correct answers. Same root cause as V5: cannot distinguish "critical missing info" from "would be nice to include."

### Lesson Learned
- Always test detection on ≥50 correct items before claiming selectivity
- 10-item pilots are useful for rejecting bad prompts (0% detection) but NOT for estimating FP rates
- OMISSION-based detection has an inherent FP problem: correct answers can also omit peripheral details
