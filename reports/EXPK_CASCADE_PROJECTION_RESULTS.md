# Cascade Projection Results — Qwen2.5-7B Self-Correction at True Base Rate

Date: 2026-05-30. Source: `runs/expK_cascade/qwen25_nw-1_nc200_seed42/` (309 cases = 109 ZS-wrong + 200 of 853 ZS-correct). Projection: `scripts/expK_project.py`. Correct-stratum weight ×4.265 to reflect the real 962 mix (109 wrong / 853 correct).

## Provenance / audit (all green)
- **Note-presence: 100%** on every note-bearing stage (309/309) — no empty-note bug.
- **Parse audit: 0.5% override** (8/1752) between regex first-pass and the GPT-4o-mini semantic judge. All 8 overrides are the judge correctly reading semantics the token-regex missed (e.g. a substantive `WRONG:/CORRECT:` whose final FLAG line said NO; trailing explanation after `Final: A`). Parsing is the authoritative semantic judge; regex is a logged cross-check only. See [[project-flag-token-parsing]].
- 0 runtime errors across 309 cases / 7549 logged LLM calls.

## Zeroshot baseline
**88.56%** accuracy at base rate (852 / 962 correct). This is the do-nothing bar; every number below is lift over it.

## Best pipeline per verdict (net #correct over 962; oracle = ceiling, not deployable)

| verdict | best pipeline | net@962 | acc lift | fixes | breaks |
|---|---|---:|---:|---:|---:|
| accept_all (no verdict) | union · blind_plain · raicl | **+9.0** | +0.94pp | 10 | 1 |
| C3_cot | union · blind_plain · raicl | +6.0 | +0.62pp | 6 | 0 |
| C3_strict | none · blind_plain · source_led | +5.0 | +0.52pp | 5 | 0 |
| **oracle (ceiling)** | none · blind_cot · raicl | **+15.5** | +1.61pp | 9 | 0 |

## Findings

1. **Self-correction nets positive at the true base rate — but modestly.** Best deployable pipeline ≈ **+6 to +9 correct over 962 (+0.6 to +0.9pp)**; perfect-verdict ceiling ≈ +15.5 (+1.6pp). The gain is small because the base is already 89% and a break costs ×4.27 a fix. Consistent with the precision-bound story.

2. **Diagnoser-alone ≈ gated. The gate adds nothing.** For matched diagnoser/arm/verdict, `gate=none` and `gate=union` give the same net (the diagnoser's own flag is the operative gate). Recall gates (union/majority/all) don't improve precision on top of the diagnoser; `plain_confirm` as a gate kills everything (0 fixes). This answers the open architecture question: **don't add a separate recall gate — the blind diagnoser's flag is sufficient.** (`positive_confirm` is the one gate with independent value: very precise, very low recall.)

3. **`blind_plain` is the best diagnoser.** It dominates the top deployable rows; `blind_cot` breaks more (CoT over-flags, as seen before); `blind_cot_clean` is middle. Simpler detection wins.

4. **`raicl` (retrieved fix example) ≥ `source_led`.** raicl edges the tight error-led correction in the top rows.

5. **The verdict currently OVER-rejects for a precise pipeline (counterintuitive).** For union·blind_plain·raicl: accept_all +9.0 (10 fix / 1 break) **beats** C3_cot +6.0 (6 fix / 0 break). C3_cot killed ~4 good fixes to avoid ~0 extra breaks. When the diagnoser+correction is already precise (1 break in 37 applied), the verdict gate costs more in lost fixes than it saves. The earlier "verdict is load-bearing" held for a noisy recall-gate detector; here the blind diagnoser is the precision filter, so the verdict is redundant-to-harmful. **About half the achievable ceiling (+15.5) is lost to verdict over-rejection (+6 to +9).**

## Caveats
- **Single run, small fix counts (5–10).** These are directional, not significant. The eventual winner needs per-fold mean ± std (5-fold) and the planned multi-run. Carry 2–3 candidates per stage, do not lock.
- Carry candidates: diagnoser {blind_plain, blind_cot_clean}; arm {raicl, source_led}; verdict {accept_all, C3_cot}; gate {none, positive_confirm}.

## Recommended next step
A focused 5-fold confirmation of the 2–3 surviving pipelines (esp. **blind_plain · raicl · {accept_all vs C3_cot}**, diagnoser-alone) to get CIs, then the verdict-over-rejection question: can a less conservative verdict recover the gap to the +15.5 oracle ceiling?
