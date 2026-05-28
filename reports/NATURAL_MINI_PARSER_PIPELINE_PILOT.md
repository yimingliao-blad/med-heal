# Natural Self-Detection + GPT-4o-mini Parser Pipeline Pilot

Status: executable pilot for the requested natural self-detection -> `gpt-4o-mini` parser/validator -> RA-ICL correction -> natural verdict -> `gpt-4o-mini` parser gate pipeline. This is a side-by-side candidate, not the final multirun method.

## Pipeline

```text
zero-shot answer
  -> natural Qwen2.5 self-audit
  -> gpt-4o-mini parses/validates audit into correction payload
  -> GTR same-patient evidence retrieval
  -> BM contrast-pool RA-ICL example retrieval
  -> Qwen2.5 correction
  -> natural conservative Qwen2.5 verdict
  -> gpt-4o-mini parses verdict
  -> final GPT-4o judge
```

## Settings

- Model: `Qwen/Qwen2.5-7B-Instruct` on vLLM port `8003`.
- Local vLLM concurrency: `8`.
- Detection temperature: `0.7`.
- Correction temperature: `0.0`.
- Verdict temperature: `0.7`, `k=3`; unclear/tie keeps original.
- Parser/validator: `gpt-4o-mini`, temperature `0.0`, JSON mode.
- Final judge: fixed old Stage 1 GPT-4o prompt, temperature `0.1`, sequential.

## Results

| Run | N | Detection style | Safe detections | Accepted | Fix | Break | Net | Projected net on 109 wrong / 853 correct |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| smoke 2 wrong + 2 correct | 4 | conservative natural | 0 | 0 | 0 | 0 | 0 | 0.0 |
| pilot 20 wrong + 20 correct | 40 | conservative natural | 0 | 0 | 1* | 0 | 1* | +5.45* |
| pilot 20 wrong + 20 correct | 40 | probe natural | 17 | 5 | 2 | 1 | 1 | -31.75 |
| pilot 20 wrong + 20 correct | 40 | stepwise CoT-style natural, T=0.0 | 32 | 8 | 2 | 1 | 1 | -31.75 |
| pilot 20 wrong + 20 correct | 40 | stepwise CoT-style natural, T=0.7 | 33 | 5 | 1 | 1 | 0 | -37.20 |

`*` The conservative pilot did not change any answers. The single apparent fix is final GPT judge drift on an unchanged original answer, so it should not be counted as a real pipeline gain.

## Comparison With Structured p5 Baseline

The existing structured p5 full-pipeline pilot on the same 20 wrong + 20 correct sample reported:

| Method | Detected | Accepted | Fix | Break | Net |
|---|---:|---:|---:|---:|---:|
| p5 structured detection + regex/mini fallback + pairwise verdict | 11 | 8 | 4 | 1 | +3 |
| natural conservative + mini parser | 0 | 0 | 0 real | 0 | 0 real |
| natural probe + mini parser | 17 | 5 | 2 | 1 | +1 |
| stepwise CoT-style + mini parser, T=0.0 | 32 | 8 | 2 | 1 | +1 |
| stepwise CoT-style + mini parser, T=0.7 | 33 | 5 | 1 | 1 | 0 |

## Interpretation

The `gpt-4o-mini` parser did what we wanted mechanically: it converted free-form Qwen2.5 audits and verdicts into structured decisions without format failures. The remaining bottleneck is not parsing. It is Qwen2.5 self-detection and the verdict gate.

The conservative natural prompt reduced hallucinated corrections but missed almost all actual errors. The probe natural prompt restored recall, but it still over-flagged OMISSION and one false correction passed the verdict gate. The stepwise CoT-style prompt increased detection recall further, but mostly by over-calling errors: 32-33 of 40 cases were marked safe to correct, including many originally correct cases. The verdict gate rejected most attempted corrections, but not enough to protect the imbalanced full set. Because Qwen2.5 zero-shot is highly imbalanced toward correct answers, even a 5% break rate on correct cases overwhelms a 5-10% fix rate on wrong cases. These pilots project negative on the real 109 wrong / 853 correct distribution.

## Decision For Now

Keep this runner as a ready-to-run candidate and parser architecture test, but do not promote it to the final multirun configuration yet. The next improvement should target false-positive suppression before correction. The CoT-style prompt alone is not enough; it makes Qwen2.5 more willing to identify errors, but it does not make the identified errors reliable. A better next test is to add an evidence-support validation gate before correction: retrieve evidence from the parsed detection payload, then ask `gpt-4o-mini` to verify whether the retrieved note spans actually support the proposed wrong/missing claim and correction target. Only cases passing that gate should reach correction.

## Outputs

- Runner: `refactor/pre_atom_pipeline/scripts/run_natural_mini_parser_pipeline.py`
- Conservative pilot summary: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_conservative_mini_parser/summary.json`
- Conservative judged rows: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_conservative_mini_parser/judged_outputs.jsonl`
- Probe pilot summary: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_probe_mini_parser/summary.json`
- Probe judged rows: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_probe_mini_parser/judged_outputs.jsonl`
- CoT T=0.0 pilot summary: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_cot_detT0p0_mini_parser/summary.json`
- CoT T=0.0 judged rows: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_cot_detT0p0_mini_parser/judged_outputs.jsonl`
- CoT T=0.7 pilot summary: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_cot_detT0p7_mini_parser/summary.json`
- CoT T=0.7 judged rows: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_cot_detT0p7_mini_parser/judged_outputs.jsonl`


## Correction: Decision Ownership

The user clarified that all decision making must stay with the tested model. `gpt-4o-mini` is allowed only for text-related jobs such as extracting structured fields from free-form output. I updated the runner accordingly:

- Qwen2.5 now explicitly states detection verdict and correction readiness/routing.
- `gpt-4o-mini` extracts `verdict`, `correction_ready`, fields, and verdict pick from the model text.
- `gpt-4o-mini` no longer validates whether a case is safe to correct.
- Pipeline routing uses the tested model's stated `Ready for correction` / `Route to correction` decision only.

### Model-Owned Decision Results

| Run | N | Detection style | Safe/routed by test model | Accepted | Fix | Break | Net | Projected net on 109 wrong / 853 correct |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| pilot 20 wrong + 20 correct | 40 | CoT readiness, T=0.0 | 1 | 0 | 0 | 0 | 0 | 0.00 |
| pilot 20 wrong + 20 correct | 40 | CoT route, T=0.0 | 33 | 7 | 2 | 1 | +1 | -31.75 |

The CoT readiness prompt caused Qwen2.5 to often say the answer was incorrect but "Not ready for correction," so it effectively blocked correction. The CoT route prompt made Qwen2.5 own the routing decision more explicitly and recovered correction attempts, but it still over-routed: 33/40 cases went to correction, yielding 2 fixes and 1 break after verdict.

This confirms the earlier qualitative issue without giving GPT decision authority: CoT-style detection makes Qwen2.5 more willing to find errors, but it still over-flags, especially OMISSION. The current verdict gate is not strong enough to make this positive on the imbalanced full Qwen2.5 set.

### Corrected Outputs

- CoT readiness, model decides: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_cot_detT0p0_testmodel_decides_mini_extracts/summary.json`
- CoT route, model decides: `refactor/pre_atom_pipeline/output/natural_mini_parser_pipeline/qwen25_nw20_nc20_seed42_natural_cot_route_detT0p0_testmodel_decides_mini_extracts/summary.json`
