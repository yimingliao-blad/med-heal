# LLM Judge Prompt Confirmation

## Working Conclusion

The GPT-4o judge prompt that is documented as aligning best with the Sara/Jose human baseline is the **Stage 1 binary prompt**, not the Step 8 reason-producing prompt.

This is a working confirmation for the judge-prompt stage, not a final human-ground-truth conclusion.

## Confirmed GPT-4o Prompt

System message:

```text
You are a medical expert evaluating an AI model's answer to a clinical question.
```

User message template:

```text
DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect
```

Documented API settings:

- Main Stage 1 report: `gpt-4o`, `max_tokens=10`, `temperature=0.1`
- Step 9 v2 revalidation wrapper: same prompt, usually `temperature=0.0` for deterministic reruns

## Evidence

- `output/final/methods_technical.md` reports the Stage 1 binary GPT-4o prompt at 92.0% agreement and kappa 0.75 against the Sara/Jose A∩B N=112 human baseline.
- `output/final/PLAN_48hr_completion.md` states that the Step 8 reason-producing prompt gave 72.3% agreement and kappa 0.36, and should be replaced by the Stage 1 binary prompt.
- `src/step3_evaluation/stage1_gpt4_eval_combined.py` contains the validated binary prompt.
- `src/step8_multimodel_icl/evaluate_step8.py` contains the older reason-producing prompt: `CORRECT: yes/no` plus `REASON`, which is the prompt identified as lower agreement.

## Soft / Charitable Judge Variants

The softer/charitable language appears in later local judge experiments, not in the confirmed GPT-4o Stage 1 prompt. Existing Qwen3-235B gold112 outputs show:

| Local judge variant | N | Agreement with gold112 |
|---|---:|---:|
| C3E | 112 | 86.6% |
| M2_charitable_oneshot | 112 | 92.0% |
| M4_gpt_critique | 112 | 92.0% |

These are useful candidates if we later choose a local judge, but they should not be confused with the GPT-4o prompt used for the main judge labels.

## Current Refactor Policy

- Use the Stage 1 binary GPT-4o prompt as the GPT judge prompt to confirm/reuse.
- Do not use the Step 8 `CORRECT + REASON` prompt for new labels.
- Keep M2/M4 charitable variants as local-judge comparison candidates only.
- Use the Sara/Jose A∩B N=112 subset as the prompt-alignment baseline while the broader human set remains provisional.
