# Phase 2 + Phase 3 — Detection Quality and Persona Effects

Date: 2026-05-29
Model: Qwen2.5-7B-Instruct. Judge: GPT-4o Stage-1 binary.
Outputs: `runs/phase2_detection/qwen25_nw-1_nc50_seed42/`, `runs/phase3_persona/qwen25_nw50_nc50_seed42_{spans_only,contradiction_quote}/`.

## Phase 2 — can LIVE detection produce the diagnosis worth +59?

Phase 1 ceiling: a precise oracle diagnosis fixes 60% of Qwen2.5 errors. Phase 2 runs live detection (5 personas) and feeds its diagnosis into the same correction step. Sample: 117 wrong + 42 correct (by fresh judge).

| Detection persona | recall (wrong) | over-flag (correct) | diagnosis AGREE | fix-rate (wrong) | net |
|---|---:|---:|---:|---:|---:|
| neutral | 0.77 | 0.67 | 5/80 | 2.6% | -8 |
| clinical_detective | 0.90 | 0.76 | 4/93 | 4.3% | -8 |
| meticulous_auditor | 0.83 | 0.71 | 6/85 | 5.1% | 0 |
| balanced_skeptic | 0.74 | **0.55** | 7/79 | 3.4% | -4 |
| strict_judge | 0.72 | 0.62 | 2/74 | 5.1% | +1 |

Diagnosis quality across all flagged-wrong cases is overwhelmingly PARTIAL (53-76 per persona) with only 2-7 AGREE and 7-19 WRONG.

### Three hard findings

1. **Live detection does NOT recover the ceiling.** Fix-rate is 2.6-5.1% against the 60% oracle ceiling. Live detection recovers under one-tenth of what a correct diagnosis could.

2. **The bottleneck is diagnosis quality, not recall.** Detection finds the wrong cases (recall 0.72-0.90). But its statement of *what is wrong and why* matches the oracle only ~5% of the time (AGREE 2-7 of ~80). Phase 1 proved PARTIAL-quality information barely helps (category-level info was +4). Live detection produces PARTIAL ~70% of the time. So the correction step is being fed exactly the weak kind of information Phase 1 showed is worthless.

3. **Over-flagging is severe and breaks correct answers.** Detection flags 55-76% of *correct* answers as wrong. When correction then "fixes" an already-correct answer using a vague diagnosis, it breaks it. That is why most nets are negative.

This is the answer to the central question: the pipeline bottleneck is not correction (which works at 60% given a good diagnosis) — it is detection's inability to produce a precise, correct diagnosis and its tendency to over-flag.

### Persona effect on detection

Persona moves *willingness to flag*, not *diagnosis quality*:
- `balanced_skeptic` has the lowest over-flag (0.55) — GPT-4o predicted this.
- `clinical_detective` has the highest recall (0.90) but also the highest over-flag (0.76).
- AGREE rate stays 2-7 across ALL personas. **No persona improves diagnosis quality.** Quality is a model-capability limit, not a persona-tunable knob.

## Phase 3 — does correction persona manage the willingness dial?

Two info-sets: spans_only (no hint) and contradiction_quote (the oracle diagnosis). 5 correction personas. ~50 wrong + ~50 correct.

### spans_only (no diagnosis)

| Persona | fix | break | net | edit-rate(correct) |
|---|---:|---:|---:|---:|
| neutral | 5 | 0 | +5 | 0.54 |
| surgical_editor | 1 | 4 | -3 | 0.05 |
| precision_fixer | 1 | 1 | 0 | 0.36 |
| senior_attending | 1 | 3 | -2 | 0.54 |
| overzealous_improver | 2 | 1 | +1 | 0.90 |

### contradiction_quote (oracle diagnosis)

| Persona | fix | break | net | edit-rate(correct) |
|---|---:|---:|---:|---:|
| neutral | 31 | 3 | **+28** | 0.53 |
| surgical_editor | 23 | 4 | +19 | **0.16** |
| precision_fixer | 26 | 4 | +22 | 0.42 |
| senior_attending | 28 | 6 | +22 | 0.58 |
| overzealous_improver | 25 | 4 | +21 | 0.92 |

### Findings

1. **Information dwarfs persona.** Best spans_only net is +5; best contradiction_quote net is +28. The information is ~6x the effect of any persona choice. Persona is a second-order lever.

2. **Persona IS a real willingness dial — edit-rate on correct cases spans 0.16 (surgical) to 0.92 (overzealous), cleanly monotonic.** The dial works exactly as designed.

3. **But moving the dial did not beat neutral on net.** With a good diagnosis, plain `neutral` wins (+28). The conservative `surgical_editor` cut editing of correct answers to 0.16 (vs neutral 0.53) but its break count (4) was no better than neutral (3) — fewer edits did not translate to fewer breaks at this N, and it lost 8 fixes (31→23).

4. **The break count barely moved across personas (3-6).** Persona changes how much the model fiddles, but fiddling a correct answer does not reliably break it, and the safe personas did not reliably reduce breaks. Persona is not a strong safety lever here.

## Combined strategic read

| Step | Works? | Evidence |
|---|---|---|
| Correction | YES, given a good diagnosis | neutral + contradiction_quote = +28 net, 31/62 wrong fixed (50%) |
| Detection recall | YES | finds 72-90% of wrong cases |
| Detection diagnosis quality | NO | matches oracle ~5%; mostly PARTIAL |
| Detection precision | NO | over-flags 55-76% of correct cases |
| Persona (either step) | WEAK lever | real willingness dial, but does not beat neutral on net and does not fix diagnosis quality |

The single bottleneck is now isolated with direct evidence: **detection must produce a precise, correct, note-grounded diagnosis AND stop over-flagging correct answers.** Persona does not solve either. Correction is already good enough.

## What this rules out and rules in

Ruled out as the next lever:
- Persona tuning — second-order; neutral already wins with good info.
- More correction-prompt engineering — correction works at 60% given a good diagnosis.
- Cross-patient analogy — already dropped (3 prior confirmations).

Ruled in as the next lever (detection diagnosis quality + precision):
1. **Reduce over-flagging.** A verdict/gate stage, or a stricter detection that abstains unless it can name a specific note-contradicted claim. The current detection says INCORRECT on 55-76% of correct answers.
2. **Raise diagnosis precision.** Options: (a) a stronger detection model (the offline oracle was GPT-4o; the live model is Qwen2.5-7B — the gap may be capability); (b) multi-sample detection with agreement (the K=3 multirun we built); (c) force detection to quote the exact contradicting note span, not paraphrase.
3. **Decouple flag from diagnose.** Detection could first decide flag/no-flag conservatively (precision), then separately produce the diagnosis only for flagged cases.

## Open questions for the user

1. The diagnosis-quality gap (5% AGREE) may be a Qwen2.5-7B capability limit. Do we test whether a stronger detector (GPT-4o-mini, or Qwen2.5 with K=3 agreement) raises AGREE materially — i.e. is the ceiling reachable with a better detector, or is it the method?
2. Over-flagging is the more tractable half. Do we add a precision gate (abstain unless a specific note-contradicted claim is named) and re-measure net?
3. Should the next study hold the WINNING correction setup fixed (neutral + diagnosis) and sweep DETECTION method (K=1 vs K=3-agreement vs quote-forced vs stronger model) to chase the AGREE rate?
