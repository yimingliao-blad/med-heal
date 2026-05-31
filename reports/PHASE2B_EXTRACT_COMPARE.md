# Phase 2b — Extract-Compare Detection (it works in principle)

Date: 2026-05-29
Model under test: Qwen2.5-7B. Parser: helper-v2 (borrowed from yesterday's tuned natural-memo parser). Sample: 96 wrong + 63 correct (by fresh judge), real notes, c=8.
Output: `runs/phase2b_extract_compare/qwen25_nw-1_nc50_seed42_helper-v2/`.

## The design (user-proposed)

Decompose detection into three natural-tone prompts, anchored on the zero-shot answer, instead of one single pass:

1. **answer-side extraction** — pull the note sentences the ZS answer makes claims about.
2. **question-side extraction** — pull the note sentences that actually answer the question.
3. **compare/judge** — hold the ZS answer against both extractions; say if it is wrong and what is contradicted or missing.

The compare memo is parsed by helper-v2 (gpt-4o-mini) into the structured diagnosis the correction step consumes. Goal: reach oracle-quality "what is wrong and why" from live ZS + note.

## Result vs the two reference points

| Metric | old plan→confirm (Phase 2) | **extract-compare (Phase 2b)** | oracle ceiling (Phase 1) |
|---|---:|---:|---:|
| recall on wrong | 0.37–0.63 | **0.83** | — |
| over-flag on correct | 0.15–0.49 | 0.68 | — |
| diagnosis AGREE rate | ~5% | **12.5%** (10/80 flagged) | 100% |
| fix | 5–9 | **24** | ~60 |
| break | 3–7 | 4 | — |
| net | 0 to +3 | **+20** | — |
| **fix-rate on wrong** | **5–10%** | **25%** | 60% |

The decomposition more than doubled the AGREE rate and 2.5–5×'d the fix-rate. It now recovers ~40% of the oracle ceiling, up from ~10–15%. The "does multi-prompt detection work in principle?" question is answered: **yes, clearly.**

## It solved the long-note recall collapse (Cause 1)

The root-cause analysis found the old detection's recall fell from 89% (short notes) to 29% (long notes) — attention dilution. Extract-compare holds recall flat across every length bucket:

| Note length | old plan→confirm recall | extract-compare recall |
|---|---:|---:|
| < 6k | 89% | 82% |
| 6–10k | 55% | 79% |
| 10–15k | 47% | 93% |
| > 15k | **29%** | **84%** |

The two extraction steps pull the relevant facts out of the long note before the comparison runs, so the judge is not drowning in 15k+ chars. Cause 1 (context-length / attention) is fixed by construction, exactly as the design intended.

## What is still open

1. **AGREE is 12.5%, not 60%.** The decomposition lifts diagnosis quality but does not yet reach the oracle. There is headroom — this is where the planner prompt / CoT layer comes in (the user's next step). The current compare memo is a single natural pass; a planner that first lists what to check, or a CoT that reasons step by step over the two extractions, may push AGREE higher.

2. **Over-flag is 0.68 (up from 0.15–0.49).** Extract-compare is more eager to flag — it flags 68% of correct answers. BUT break is only 4/63 (6%): flagging a correct answer mostly leads to keep-original or a harmless re-confirmation, not a broken answer. So over-flag costs compute, not accuracy, here. Still worth reducing: a precision gate or a stricter compare step.

3. **The 30% hard core remains.** Cases unfixable even by the oracle (mostly MISREADING) are still unfixable; this is a model-capability ceiling, not addressable by detection design.

## Why this is the key result

It reframes the bottleneck again, in the good direction. Earlier: "detection diagnosis quality is a flat 7B capability ceiling (~5% AGREE)." Now: that 5% was partly a *pipeline-design* limit, not purely a capability limit. Decomposing the task — extract, then compare — let the same 7B model produce usable diagnoses 2.5× more often and recover 25% of its own errors live (vs 5–10%). The model is more capable than the single-pass detection let it show.

## Next steps (user's plan: now that multi-prompt works, try planner / CoT)

1. **Planner layer** — before the compare step, a short prompt that lists what to verify (required answer slot, key entities, what the note must confirm), then the compare step works that checklist. Tests whether structuring the comparison lifts AGREE further.
2. **CoT compare** — replace the single natural compare memo with a step-by-step reasoning pass over the two extractions. Tests whether explicit reasoning sharpens the diagnosis.
3. **Parser sensitivity** — re-run with the qwen35 (mlx:8803) parser to confirm the gain is from the detection decomposition, not the helper-v2 parser.
4. **Precision gate** — add a conservative flag/no-flag check to cut the 0.68 over-flag without losing recall.

Recommended: try the planner layer first (cheapest, most likely to lift AGREE), then CoT, comparing both to this 12.5% AGREE / 25% fix-rate baseline.

## Provenance

- Code: `scripts/phase2b_extract_compare_detection.py` (3-prompt detection, helper-v2 / gpt4o-mini / qwen35 parser options).
- Full LLM input/output ledger: `runs/phase2b_extract_compare/qwen25_nw-1_nc50_seed42_helper-v2/llm_calls.jsonl`.
- Reference: Phase 1 (`PHASE1_INFO_ABLATION.md`), Phase 2 (`PHASE2_3_DETECTION_PERSONA.md`), root cause (`DETECTION_ROOTCAUSE_PROPOSAL.md`).
