# Two-Axis Synthesis — Correction F1 and Detection F1

Date: 2026-05-30
Model: Qwen2.5-7B. All on the 962-case EHRNoteQA pool (109 wrong / 853 correct), real notes, GPT-4o judge.

Self-correction decomposes into two orthogonal axes, each with its own F1. Analyzing them separately (user direction) localizes the bottleneck precisely.

## Axis 1 — Correction F1 (reliability of executing a correct instruction)

Experiment F: give correction instructions of escalating explicitness; measure flip-rate on wrong (recall) and break-rate on correct (1 − precision).

| instruction | flip_rate | break_rate | correction F1 |
|---|---:|---:|---:|
| oracle_error (note-grounded diagnosis) | 0.74 | 0.04 | **0.835** |
| giveaway (ground truth handed to the model) | 0.66 | 0.04 | 0.784 |

Findings:
- **Correction caps at ~74% flip even with a correct instruction.** The giveaway hands the model the answer and says "make it match" — yet it flips only 66%. So ~26–34% of wrong cases are INTRINSICALLY unflippable: the model can't produce the right answer even when told it (awkward rewrite, partial adoption, MC-format mismatch). The unflipped third is not an instruction-quality problem.
- **A note-grounded diagnosis beats a bare answer** (0.74 vs 0.66): the model corrects better when told what is wrong + why + evidence than when handed the answer.
- **Break rate is low (~0.04) given a correct instruction** — confirms B1c: correct error statements don't break correct answers.

Correction is the STRONGER axis: F1 0.835, reliable once told the right thing, with a real but high ceiling (~74%).

## Axis 2 — Detection F1 (produce a correct, actionable instruction without false-flagging)

From the union (k=3 natural) detection and the AGREE measurements:

| metric | value | reading |
|---|---:|---|
| flag recall (catch wrong) | 0.95 | excellent |
| flag precision (avoid false flag) | 0.64 | over-flags correct |
| flag F1 | 0.76 | recall-heavy |
| **error-correctness (AGREE rate)** | **~0.13–0.17** | **the weak link** |

Detection finds wrong answers easily but states the CORRECT error only ~15% of the time. Note: over-flag (precision) is NOT the binding constraint — B1c showed a detector can over-flag freely if its errors are correct (break rate stays ~0.02–0.04). The binding constraint is error-correctness.

Detection is the WEAK axis, and specifically its error-correctness, not its recall or precision.

## The two axes localize the whole problem

| | Correction | Detection |
|---|---|---|
| F1 | 0.835 | 0.76 (flag) / ~0.15 (error-correctness) |
| Reliable? | Yes, caps ~74% | Recall yes; error-correctness no |
| Bottleneck? | No (it's strong) | **YES — error-correctness ~15%** |

The single bottleneck is **detection error-correctness**: the rate at which the model states a correct, evidence-backed error. It drives both fix-rate (correct error → correction flips 74%) and break-rate (correct error → ~0.04 break).

## Full-scale ceiling (corrected with measured correction reliability)

| pipeline | fix-rate | break-rate | full-scale net |
|---|---:|---:|---:|
| Ceiling (perfect detection + real correction) | 0.74 | 0.04 | 109×0.74 − 853×0.04 = **+47 (+4.9pp)** |
| Live (current detection error-correctness ~15%) | 0.30 | 0.16 | **−103** |

The achievable ceiling is **+4.9pp** (capped by correction's 74% reliability). The entire −103 → +47 gap is detection error-correctness. Every point of error-correctness moves net toward +4.9pp, by raising fixes and cutting breaks at the same time.

## What this means for the next move

- Stop working on: correction (strong), over-flag/precision (not binding, per B1c), feedback form, prioritization, verdict form, input form — all characterized as non-levers or solved.
- The only lever is **detection error-correctness**. Absolute self-judgment rubber-stamps (expE, expD C1/C2); only relative comparison discriminates (C3). So the candidate path to higher error-correctness is a RELATIVE framing — e.g. regen-and-compare (generate a fresh answer, contrast with the original to surface the real discrepancy) rather than asking the model in the absolute "what is wrong here."
- Cross-model: the ceiling is base-rate dependent. Lower-baseline models (BioMistral 54%, DeepSeek 77%) have more wrong cases and a higher achievable net; Qwen2.5/Qwen3 are the hardest.
