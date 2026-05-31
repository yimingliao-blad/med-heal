# Full Self-Correction Pipeline — Plan, Design Matrix, and Gaps

Date: 2026-05-30
Model: Qwen2.5-7B (model-internal). All pilot numbers below are N=40–80 with ~10 breaks — **noisy**; treat as direction, not truth. Principle for this plan (user): carry **2–3 candidates per stage** into a larger validation; do not lock a single winner from a pilot.

## The pipeline as four stages

```
ZS answer ─▶ [STAGE 0: GATE?] ─▶ [STAGE 1: DIAGNOSER] ─▶ [STAGE 2: CORRECTION] ─▶ [STAGE 3: VERDICT] ─▶ final
```

The open architecture question (user): is **GATE + DIAGNOSER** better, or **DIAGNOSER-ALONE** (the diagnoser's own flag is the gate)? We have only ever tested diagnoser-alone for the blind variants. This is the first gap.

## Architecture variants (Stage-0 placement) — mostly UNTESTED

| arch | description | tested? |
|---|---|---|
| A. diagnoser-alone | the diagnoser flags + diagnoses in one; its flag is the gate | ✓ (expJ plain: rec0.70/of0.73/net−7; expJ2 CoT: rec0.88/of1.0/net−24) |
| B. recall-gate → diagnoser | union/majority flags (high recall), diagnoser localizes only flagged | ✗ **MISSING** |
| C. precision-gate → diagnoser | conservative gate (plain confirm of0.18) flags, diagnoser localizes | ✗ **MISSING** |
| D. gate ∧ diagnoser-flag | accept only if BOTH a gate and the diagnoser agree it is wrong | ✗ **MISSING** |

Key insight to test: the blind+CoT diagnoser over-flags (of1.0) as a *gate*, but has the best *localization* (0.51). Arch B/C use it only as a diagnoser on pre-flagged cases — its over-flag then doesn't matter. **This is the most important missing experiment.**

## STAGE 0 — GATE candidates

| candidate | recall | over-flag | tested? | carry? |
|---|---:|---:|---|---|
| union (k=3 any) | 0.95 | 0.79 | ✓ | candidate (recall) |
| majority (k=3 ≥2) | 0.875 | 0.67 | ✓ | candidate (balance) |
| all-agree (k=3 ==3) | 0.72 | 0.48 | ✓ | maybe |
| plain confirm | 0.28 | 0.18 | ✓ | candidate (precision) |
| confirm+CoT | 0.90 | 0.93 | ✓ | drop (over-flags) |
| **gate = none (diagnoser-alone)** | — | — | ✓ | candidate |
| **MISSING: union∧confirm (recall flag, precision filter)** | ? | ? | ✗ | build |
| **MISSING: two-pass gate (flag then confirm-the-flag)** | ? | ? | ✗ | build |

## STAGE 1 — DIAGNOSER candidates (localization = the bottleneck metric)

| candidate | localization | recall | over-flag | tested? | carry? |
|---|---:|---:|---:|---|---|
| direct extract-compare | 0.15 | 0.95 | 0.79 | ✓ | drop |
| doubt-flipside + CoT | 0.30 | 0.93 | 0.80 | ✓ | maybe |
| two-round blind, plain R2 | 0.39 | 0.70 | 0.73 | ✓ | **candidate** |
| two-round blind, CoT R2 | **0.51** | 0.88 | 1.00 | ✓ | **candidate (best loc)** |
| **MISSING: blind CoT R2 with clean WRONG/CORRECT/EVIDENCE output** | ? (≥0.51) | ? | ? | ✗ | **build (top priority)** |
| **MISSING: blind R2 ×K samples (union/majority of inconsistencies)** | ? | ? | ? | ✗ | build |
| **MISSING: round-1 paraphrase variants (verbatim vs reworded vs claim-split)** | ? | ? | ? | ✗ | build |
| **MISSING: blind R2 fed retrieved spans instead of full note** | ? | ? | ? | ✗ | maybe |

Localization trend 0.15→0.30→0.39→0.51 is real signal; blind+CoT leads. But the handoff to correction is lossy (fix-rate 0.175 ≪ loc 0.51×corr 0.74≈0.38), so the **clean-output variant is the single highest-value missing diagnoser test.**

## STAGE 2 — CORRECTION candidates (the strong axis, F1 0.835)

| candidate | flip-rate (oracle instr) | tested? | carry? |
|---|---:|---|---|
| error-led tight (WRONG/CORRECT/EVIDENCE) | 0.74 | ✓ | **candidate** |
| giveaway (answer handed over) | 0.66 | ✓ | drop (worse) |
| note-grounded vs note-free | 0.64 vs 0.63 | ✓ | note-free fine |
| **MISSING: few-shot correction (audited fix exemplar)** | ? | ✗ | build |
| **MISSING: best-of-K correction candidates** | ? | ✗ | maybe |

Correction is largely solved; carry the tight error-led form; few-shot is the one untested lever (user-flagged earlier).

## STAGE 3 — VERDICT candidates (break-catch; caps ~0.70)

| candidate | fix-keep | break-catch | tested? | carry? |
|---|---:|---:|---|---|
| C3_cot | 0.55 | 0.60–0.70 | ✓ | **candidate (best)** |
| C3_strict | 0.45 | 0.60 | ✓ | candidate |
| false_correction_sensitive (old) | 0.45 | 0.60 | ✓ | maybe |
| C3_natural | 0.48 | 0.40 | ✓ | drop |
| count_compare (old) | 0.03 | 1.00 | ✓ | drop (too strict) |
| C1/C2 (absolute) | rubber-stamp | ~0 | ✓ | drop |
| **MISSING: adaptive verdict (confidence-routed: agree→C3, split→strict)** | ? | ? | ✗ | build |
| **MISSING: C3_cot ×K vote** | ? | ? | ✗ | maybe |

## The missing-combinations matrix (what we could be missing)

1. **Architecture B/C** — recall-gate or precision-gate → blind+CoT diagnoser (its over-flag stops mattering). UNTESTED and likely the winner.
2. **Clean-output blind+CoT diagnoser** — format the inconsistency as WRONG/CORRECT/EVIDENCE to recover the lossy handoff (fix-rate 0.175 → ~0.38).
3. **Multi-sample blind R2** — K=3 blind checks, union of inconsistencies (recall) or majority (precision).
4. **Few-shot correction.**
5. **Adaptive verdict** (confidence-routed).
6. **End-to-end full-scale net** for the assembled pipeline — every pilot measured stages in isolation; we have never run the full chain at the true base rate with a verdict.
7. **Larger-N validation** — all pilots N≤80; the eventual winner must be confirmed at 109 wrong + ≥150 correct (and ideally 5-fold for CIs).
8. **Cross-model** — everything is Qwen2.5; the base-rate math says BioMistral/DeepSeek may net positive with the same pipeline.

## Evaluation protocol (staged tournament, carry 2–3 per stage)

Pilots are noisy, so:

**Round 1 — diagnoser bake-off (the bottleneck).** On a LARGER set (all 109 wrong + 100 correct), compare the 2–3 diagnoser candidates: {blind plain R2, blind CoT R2, blind CoT R2 clean-output}, EACH under the 3 architectures {alone, recall-gate, precision-gate}. Metric: localization, recall, over-flag, and downstream fix/break with a fixed tight correction. → carry the top 2 diagnoser×arch pairs.

**Round 2 — correction.** On the carried diagnoser(s), compare {tight error-led, few-shot}. → carry top 1–2.

**Round 3 — verdict.** On the carried pipeline(s), compare {C3_cot, C3_strict, adaptive}. → carry top 1.

**Round 4 — full-scale + cross-model.** Run the 1–2 surviving full pipelines at the true base rate; project net; then run across the 5 models. Report lift over zeroshot with per-fold mean ± std.

Guardrails: every stage reports lift over zeroshot; never crown a winner whose pilot margin is within noise of the runner-up — carry both. Decisions logged; full LLM ledger per run.

## The single most important next experiment

**Architecture B/C with the blind+CoT diagnoser, clean output** — i.e., recall-gate (union or majority) OR precision-gate (plain confirm) → blind+CoT diagnosis formatted as WRONG/CORRECT/EVIDENCE → tight correction → C3_cot verdict, measured end-to-end at larger N with the full-scale projection. This tests the user's open question (gate vs diagnoser-alone) AND the clean-output handoff in one run, and it is the first configuration with a real shot at positive net.
