# GPT Judge Old Prompt Provenance Check

Status: provenance confirmed and formatting difference tested.

## Old Result Source

The external old GPT file is exactly derived from the old Stage 1 GPT output:

- External CSV: `datasets/external/all_users_openended_BioMistral-7B_GPT4o_evaluation.csv`
- Stage 1 CSV: `output/step3_evaluation/stage1_gpt4/biomistral-7b_stage1_accuracy.csv`
- Source script: `src/step3_evaluation/stage1_gpt4_accuracy.py`
- Label match across all 962 patients: 962/962.

## Old Stage 1 Settings

From `src/step3_evaluation/stage1_gpt4_accuracy.py`:

- Model: `gpt-4o`
- Temperature: `0.1`
- Max tokens for binary correctness: `10`
- Retry attempts: `10`
- Sleep after calls: `1` second in the main loop
- Prompt: Stage-1 binary prompt with system role:
  - `You are a medical expert evaluating an AI model's answer to a clinical question.`
- Output parse:
  - `1` if response contains `1` and not `0`
  - `0` if response contains `0`

## Important Data-Construction Difference

The old Stage 1 script did **not** use the exact data formatting from my first rerun.

Old Stage 1 used:

- Ground truth: choice text only, for example `The patient was fairly stabilized...`
- Note wrapper: `[note 1 start] ... [note 1 end]`
- Source answer: `openended_answer` from `output/ours_biomistral-7b_EHRNoteQA_processed.csv` / Stage 1 CSV

My first rerun used:

- Ground truth: answer letter plus text, for example `A. The patient was fairly stabilized...`
- Note wrapper: `[Note 1] ...`
- Source answer: same BioMistral open-ended answer content

## Exact Old-Style Rerun

I reran the Caitlin 100 overlap sequentially with the old Stage 1 construction and T=0.1.

| Rerun style | Agreement vs old GPT | Kappa | Current label counts |
|---|---:|---:|---|
| Prior T=0 rerun, lettered GT | 94/100 | 0.820 | 76 correct / 24 incorrect |
| Prior T=0.1 rerun, lettered GT | 94/100 | 0.820 | 76 correct / 24 incorrect |
| Old Stage 1 exact, T=0.1 | 97/100 | 0.901 | 81 correct / 19 incorrect |

Output: `refactor/pre_atom_pipeline/output/quick_tests/judge_old_stage1_exact_T01/current_gpt_T01_old_stage1_exact_on_caitlin100.jsonl`

## Interpretation

The earlier 6-case mismatch was partly caused by rerun formatting differences, especially ground-truth formatting and note wrapping. When using the old Stage 1 construction, agreement improves from 94/100 to 97/100.

There are still 3 remaining mismatches under the exact old-style rerun, so the residual drift is likely due to GPT-4o snapshot/model behavior changes or borderline cases, not temperature alone.

Remaining mismatches:

```json
[
  {
    "patient_id": 12788767,
    "old": 1,
    "current_old_exact": 0,
    "caitlin": 0,
    "raw": "0"
  },
  {
    "patient_id": 15679116,
    "old": 1,
    "current_old_exact": 0,
    "caitlin": 0,
    "raw": "0"
  },
  {
    "patient_id": 16652338,
    "old": 0,
    "current_old_exact": 1,
    "caitlin": 0,
    "raw": "1"
  }
]
```


## Exact Old Prompt: T=0.0 vs T=0.1

Using the same old Stage 1 construction on Caitlin's 100 cases:

| Setting | Agreement vs old GPT | Kappa | Current label counts |
|---|---:|---:|---|
| Exact old prompt/data, T=0.1 | 97/100 | 0.901 | {'1': 81, '0': 19} |
| Exact old prompt/data, T=0.0 | 96/100 | 0.864 | {'1': 82, '0': 18} |

T=0.1 matches the old record slightly better on this 100-case subset: 97/100 vs 96/100. T=0.0 has the same aggregate correct/incorrect count as the old record, but four case-level labels differ.

T=0.0 output: `refactor/pre_atom_pipeline/output/quick_tests/judge_old_stage1_exact_T0/current_gpt_T0_old_stage1_exact_on_caitlin100.jsonl`

T=0.0 vs T=0.1 flips under the exact old construction:

```json
[
  {
    "patient_id": 12788767,
    "t01": 0,
    "t0": 1,
    "old": 1,
    "caitlin": 0
  },
  {
    "patient_id": 16723233,
    "t01": 0,
    "t0": 1,
    "old": 0,
    "caitlin": 1
  },
  {
    "patient_id": 16740649,
    "t01": 1,
    "t0": 0,
    "old": 1,
    "caitlin": 0
  }
]
```

Remaining T=0.0 mismatches vs old record:

```json
[
  {
    "patient_id": 15679116,
    "old": 1,
    "current_T0": 0,
    "caitlin": 0,
    "raw": "0"
  },
  {
    "patient_id": 16652338,
    "old": 0,
    "current_T0": 1,
    "caitlin": 0,
    "raw": "1"
  },
  {
    "patient_id": 16723233,
    "old": 0,
    "current_T0": 1,
    "caitlin": 1,
    "raw": "1"
  },
  {
    "patient_id": 16740649,
    "old": 1,
    "current_T0": 0,
    "caitlin": 0,
    "raw": "0"
  }
]
```
