# Qwen2.5 Self-Detection Prompt Bakeoff

Status: prompt-screening result for Qwen2.5 self-detection. The goal is not just detecting errors, but producing feedback that can drive downstream retrieval and correction.

## Setup

- Target model: `Qwen/Qwen2.5-7B-Instruct` via local vLLM on port `8003`.
- vLLM generation concurrency: `8`.
- Sample: 50 Qwen2.5 zero-shot wrong answers + 50 zero-shot correct answers, seed 42.
- Parsers compared: deterministic regex/template parser and `gpt-4o-mini` JSON extractor.
- GPT parser was sequential. For payload prompts, full mini parsing was stopped after slow API behavior; regex scores are reported from saved raw outputs, while parser comparison is from the completed p1-p4 runs.

## Completed Parser Comparison

For the first four structured prompts, regex and `gpt-4o-mini` produced essentially identical verdict/type parses.

### T=0.0, p1-p4

| Prompt | Mini TP | Mini FP | Precision | Recall | F1 | Regex/Mini Verdict Agreement |
|---|---:|---:|---:|---:|---:|---:|
| `p1_three_axis_freeform` | 4 | 2 | 0.667 | 0.080 | 0.143 | 100.0% |
| `p2_claim_quote_check` | 5 | 3 | 0.625 | 0.100 | 0.172 | 100.0% |
| `p3_question_focus_first` | 5 | 2 | 0.714 | 0.100 | 0.175 | 100.0% |
| `p4_fewshot_conservative` | 3 | 2 | 0.600 | 0.060 | 0.109 | 100.0% |

### T=0.7, p1-p4

| Prompt | Mini TP | Mini FP | Precision | Recall | F1 | Regex/Mini Verdict Agreement |
|---|---:|---:|---:|---:|---:|---:|
| `p1_three_axis_freeform` | 3 | 2 | 0.600 | 0.060 | 0.109 | 99.0% |
| `p2_claim_quote_check` | 4 | 8 | 0.333 | 0.080 | 0.129 | 100.0% |
| `p3_question_focus_first` | 7 | 3 | 0.700 | 0.140 | 0.233 | 100.0% |
| `p4_fewshot_conservative` | 2 | 2 | 0.500 | 0.040 | 0.074 | 100.0% |

Finding: parser choice was not the limiting factor for these template-style prompts. Regex matched `gpt-4o-mini` on 99-100% of verdicts/error types. The bottleneck is the detector itself.

## Correction-Payload Prompts

These prompts were designed to emit downstream-usable fields: `WRONG_CLAIM`, `CORRECT_OR_MISSING_INFO`, `EVIDENCE_NEEDED`, `RETRIEVAL_QUERY_1..3`, and `CORRECTION_HINT`.

| Prompt | TP | FP | Precision | Recall | F1 | Usable feedback | Retrieval-ready |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p5_retrieval_payload` | 14 | 5 | 0.737 | 0.280 | 0.406 | 19 | 19 |
| `p6_claims_to_queries` | 8 | 6 | 0.571 | 0.160 | 0.250 | 14 | 14 |
| `p7_error_gate_payload` | 4 | 2 | 0.667 | 0.080 | 0.143 | 6 | 6 |

Best aligned candidate: `p5_retrieval_payload`.

- It had the best recall/F1 among tested prompts: TP 14/50, FP 5/50, precision 0.737, recall 0.28, F1 0.406.
- Every detection from `p5` had usable correction feedback and retrieval-ready fields under regex parsing: 19/19 detected outputs.
- It is less conservative than `p7_error_gate_payload` and less noisy than `p6_claims_to_queries`.

## Interpretation

The older prompt family showed that Qwen2.5 self-detection can be very conservative or very noisy depending on prompt structure. The new p5 result is useful because it improves recall while keeping FP moderate and, more importantly, produces fields that can directly construct retrieval queries.

Recommended detector payload for the next correction run:

```text
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE
QUESTION_FOCUS: ...
ANSWER_FOCUS: ...
WRONG_CLAIM: ...
CORRECT_OR_MISSING_INFO: ...
EVIDENCE_NEEDED: ...
RETRIEVAL_QUERY_1: ...
RETRIEVAL_QUERY_2: ...
RETRIEVAL_QUERY_3: ...
CORRECTION_HINT: ...
WHY: ...
```

## Next Step

Use `p5_retrieval_payload` as the detection-feedback source for a small downstream correction test. The evaluation should not stop at detection TP/FP: measure whether `RETRIEVAL_QUERY_*` retrieves sufficient same-patient evidence and whether the final correction improves fix/break/net.

Before larger runs, resume full `gpt-4o-mini` parsing for p5-p7 from saved raw outputs, or make the parser resumable with per-row writes and a shorter request timeout.

## Outputs

- T=0 p1-p4 summary: `refactor/pre_atom_pipeline/output/detection_bakeoff/qwen25_detection_nw50_nc50_seed42_t0p0/summary.json`
- T=0.7 p1-p4 summary: `refactor/pre_atom_pipeline/output/detection_bakeoff/qwen25_detection_nw50_nc50_seed42_t0p7/summary.json`
- Payload regex summary: `refactor/pre_atom_pipeline/output/detection_bakeoff/qwen25_detection_payload_t0p7_regex_summary.json`
- Script: `refactor/pre_atom_pipeline/scripts/qwen25_detection_prompt_bakeoff.py`
