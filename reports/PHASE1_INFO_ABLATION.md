# Phase 1 — What Information Helps the Model Correct Its Error

Date: 2026-05-29
Sample: 109 Qwen2.5-7B-Instruct zero-shot wrong cases (all wrong cases, seed 42). Single-stage correction; oracle information from the offline GPT-4o error taxonomy (`phase1_wrong_gpt4o.json`). Note context fixed at first18k. GPT-4o judge.
Output: `runs/phase1_info_ablation/qwen25_nw-1_seed42/`.

## Result

Each arm adds ONE information source to the baseline (question + wrong answer + same-patient spans).

| Arm | Adds | Fix | Break | Net |
|---|---|---:|---:|---:|
| baseline | nothing | 3 | 1 | +2 |
| question_type | DATE/NUMBER/LAB/... label | 2 | 1 | +1 |
| question_focus | what is asked | 3 | 1 | +2 |
| error_type | PRIMARY_ERROR category | 5 | 1 | +4 |
| error_location | where the wrong claim sits | 4 | 0 | +4 |
| **contradiction_quote** | **what is wrong and why (full description)** | **62** | **3** | **+59** |
| analogy_wrong | similar past wrong case | 0 | 1 | -1 |
| analogy_correct | similar past correct answer | 2 | 0 | +2 |
| all | union of all the above | 56 | 3 | +53 |

## The one finding that matters

**The only information source that moves the needle is the natural-language description of what is wrong and why.** It fixes 62 of 103 wrong cases (60%). Everything else sits at or near the baseline of 3 fixes.

Category-level hints are nearly worthless on their own:

| Hint granularity | Example | Fix-rate |
|---|---|---:|
| Category label | error_type = OMISSION | 5/103 (5%) |
| Location pointer | "the answer claims X about visit 2" | 4/103 (4%) |
| **Full diagnosis** | "the answer says no procedure occurred, but the note shows a thyroidectomy on day 3" | **62/103 (60%)** |

The jump from a category to a sentence-level diagnosis is the difference between +4 and +59. **Telling the model the BUCKET of the error does almost nothing. Telling the model the SPECIFIC error, grounded in the note, does almost everything.**

## More information is not better

`all` (every source combined) scored net +53 — **lower** than `contradiction_quote` alone (+59). Stacking error_type + question_type + analogy + location on top of the precise diagnosis *diluted* it: 6 fewer fixes. The single strong signal beats the kitchen sink. This matters for prompt design: do not pad the correction prompt with weak categorical fields when a precise diagnosis is present.

## The oracle caveat — this is a ceiling, not a deployable number

`contradiction_quote` is the offline GPT-4o ERROR_DESCRIPTION: a careful, correct, note-grounded statement of the exact error, produced with the ground truth in hand. Handing that to the model is close to handing it the answer. So **+59 / 60% is an upper bound on RA-based correction for Qwen2.5** — it answers "if detection were perfect, how much could correction fix?" The answer: about 60% of wrong cases.

This reframes the whole pipeline. The bottleneck is NOT the correction step. Given a correct diagnosis, Qwen2.5 corrects 60% of its own errors. The bottleneck is **producing that diagnosis at inference time** — which is Phase 2 (detection).

## Why earlier detection-gated runs underperformed

This explains the gap that looked mysterious before. The natural-pipeline detection emits `error_type` (a category) plus thin `wrong_claim` / `correct_or_missing_info` fields. Phase 1 shows category-level information is worth ~+4, not +59. The earlier detection-gated pilots (taxonomy alignment +1, the 200-case small nets) were operating near the **category-information ceiling, not the diagnosis-information ceiling.** The pipeline wasn't broken — it was feeding the correction step the weak kind of information.

The lever is clear: detection must produce a precise, correct, note-grounded "what is wrong and why," not a category label.

## The willingness tradeoff is already visible

Break-rate on originally-correct cases (fresh GPT-4o judge found 6 of 109 were actually correct despite the stored wrong label):

| Arm | Fix (of 103 wrong) | Break (of 6 correct) |
|---|---:|---:|
| baseline | 3 (3%) | 1 (17%) |
| error_type | 5 (5%) | 1 (17%) |
| contradiction_quote | 62 (60%) | 3 (50%) |

The strong hint makes the model edit aggressively: it breaks 3 of 6 originally-correct cases (50%). On a wrong-dominated sample the net is still hugely positive, but this is the over-editing signal. **A precise diagnosis raises both fix-rate and break-rate** — the willingness dial is real. The Phase 3 persona sweep (balanced wrong+correct) is designed to study exactly this: can a persona keep the 60% fix-rate while cutting the break-rate.

## Analogy is dead, confirmed again

`analogy_wrong` net -1, `analogy_correct` net +2 (vs baseline +2 — i.e. no lift). The analogy-quality side check: of 109 retrievals, 35 USEFUL / 74 WEAK / 0 IRRELEVANT. Even the "useful" analogies produced no correction lift. This is the third independent signal that cross-patient analogy does not help correction. It stays out of the method.

## What Phase 1 settles

1. **The information that drives correction is a precise note-grounded error description, not a category.** Detection's target output is redefined accordingly.
2. **Correction is not the bottleneck** — Qwen2.5 fixes 60% of its errors given a correct diagnosis.
3. **Don't stack weak information** — it dilutes the strong signal.
4. **Cross-patient analogy is out** — third confirmation.
5. **A strong hint over-edits correct answers** — willingness must be managed (Phase 3).

## What Phase 1 does NOT settle

- Whether live detection can produce a diagnosis close to the oracle's quality. That is Phase 2, and it is now the single most important question.
- How much of the 60% ceiling a deployable detector recovers.
- Whether persona/stance can hold the fix-rate while reducing the 50% break-rate on correct cases. That is Phase 3 (script ready).
- Whether the 60% ceiling holds on other models (each needs its own taxonomy or a live detector).

## Recommended next steps

1. **Phase 2 detection-quality study (highest priority).** Run the natural-pipeline detection on the same 109 cases. Measure how close its `wrong_claim` + `correct_or_missing_info` + `decisive_evidence` come to the oracle ERROR_DESCRIPTION, then feed the LIVE description into the same correction step and compare fix-rate to the +59 oracle ceiling. The gap is the detection-quality tax.
2. **Phase 3 persona sweep (ready).** Run on balanced wrong+correct with `--info-set contradiction_quote` to see which persona keeps the high fix-rate while cutting over-editing.

## Decisions waiting

1. Build + run Phase 2 detection-quality study next? (It directly answers "can we get the +59 at inference time.")
2. Or run Phase 3 persona sweep first (script ready) since it's a one-command launch?
3. Phase 3 info-set: `spans_only` (clean persona isolation) or `contradiction_quote` (persona on top of the winning information)? I recommend running BOTH — spans_only shows persona's raw effect, contradiction_quote shows persona at the strong-hint operating point where over-editing is the live risk.
