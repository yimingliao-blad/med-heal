# GPT Judge 100-Case Confirmation

Status: current working judge finalized for rerun planning.

## Test Set

- Source: Caitlin Schwanke's 100 BioMistral human-judge cases.
- Runnable overlap with old GPT-4o record and BioMistral answer file: 100 / 100.
- Current sequential run output: `refactor/pre_atom_pipeline/output/quick_tests/judge_old_gpt_match_sequential/current_gpt_T0_vs_old_gpt_on_caitlin100_sequential.jsonl`.

## Current Judge Setting

- Model: `gpt-4o`.
- Temperature: `0.0`.
- Max tokens: `10`.
- API concurrency: `1` for GPT calls.
- System prompt: `You are a medical expert evaluating an AI model's answer to a clinical question.`
- User prompt fields: discharge summary, question, ground-truth answer, model answer.
- Required output: single digit only, `1` for correct and `0` for incorrect.

## Alignment Result

| Comparison | N | Agreement | Kappa | Label Counts |
|---|---:|---:|---:|---|
| Current GPT T=0 vs old GPT record | 100 | 94.0% (94/100) | 0.820 | old={'1': 82, '0': 18}, current={'1': 76, '0': 24} |
| Current GPT T=0 vs Caitlin | 100 | 76.0% (76/100) | 0.464 | Caitlin={'1': 60, '0': 40}, current={'1': 76, '0': 24} |
| Old GPT record vs Caitlin | 100 | 70.0% (70/100) | 0.312 | Caitlin={'1': 60, '0': 40}, old={'1': 82, '0': 18} |

## Decision

Use the current Stage-1 binary GPT-4o judge at temperature `0.0`, with sequential API calls. It reproduces the old GPT-4o record well on the full 100-case Caitlin overlap: 94% agreement and kappa 0.820.

The current T=0 run is slightly stricter than the old record on this set: current labels 76 correct / 24 incorrect, old labels 82 correct / 18 incorrect.

For later local model generation/evaluation, `c=8` applies only to local vLLM calls, not GPT API calls.


## Temperature Drift Check

A sequential `temperature=0.1` run was also tested on the same 100 cases.

| Current GPT setting | Agreement vs old GPT | Kappa | Current label counts | Output |
|---|---:|---:|---|---|
| T=0.0 | 94.0% (94/100) | 0.820 | {'1': 76, '0': 24} | `refactor/pre_atom_pipeline/output/quick_tests/judge_old_gpt_match_sequential/current_gpt_T0_vs_old_gpt_on_caitlin100_sequential.jsonl` |
| T=0.1 | 94.0% (94/100) | 0.820 | {'1': 76, '0': 24} | `refactor/pre_atom_pipeline/output/quick_tests/judge_old_gpt_match_sequential_T01/current_gpt_T01_vs_old_gpt_on_caitlin100_sequential.jsonl` |

T=0.1 did not move the current run closer to the old GPT record. It produced the same aggregate alignment as T=0.0 and only 2 label flips relative to T=0.0. The two flips cancel in the old-record agreement count.

Observed T=0.1 vs T=0.0 flips:

```json
[
  {
    "patient_id": 12788767,
    "t0": 0,
    "t01": 1,
    "old": 1,
    "caitlin": 0
  },
  {
    "patient_id": 16781251,
    "t0": 1,
    "t01": 0,
    "old": 1,
    "caitlin": 0
  }
]
```

Interpretation: the 6-case mismatch is not explained simply by using `temperature=0.1` instead of `temperature=0.0`. Model snapshot drift, borderline cases, and possible old-run implementation details remain more likely contributors.
