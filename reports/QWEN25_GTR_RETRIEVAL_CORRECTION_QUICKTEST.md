# Qwen2.5 GTR Retrieval Correction Quick Test

Status: completed quick test. This is a method-screening result, not the final retrieval decision.

## Setup

- Served model: `Qwen/Qwen2.5-7B-Instruct` on vLLM port `8003`.
- Local generation concurrency: `8`.
- Retrieval workers: `4`.
- Retrieval: `gtr_q_answer`, GTR-T5 note-span top-`5`.
- Generation temperature: `0.0`.
- GPT judge: fixed old Stage 1 GPT-4o prompt, `temperature=0.1`, sequential calls.
- Sample: all 109 Qwen2.5 zero-shot wrong cases plus 109 seed-42 correct cases.

## Result

| Arm | N judged | Fix | Break | Net | Still wrong | Still correct |
|---|---:|---:|---:|---:|---:|---:|
| `evidence_only` | 218 | 26 | 6 | 20 | 83 | 103 |
| `taxonomy_evidence` | 218 | 32 | 5 | 27 | 77 | 104 |

Compared with the existing Qwen2.5 full-scale `regen+count` result (`fix=27`, `break=33`, `net=-6`), both retrieval-evidence arms are much more promising on this matched 109-wrong/109-correct quick test. The taxonomy-aware arm is currently the better candidate: `fix=32`, `break=5`, `net=+27`.

## By Error Type

### `evidence_only`

| Type | Fix | Still wrong | Break | Still correct |
|---|---:|---:|---:|---:|
| `CORRECT_OR_UNKNOWN` | 0 | 0 | 6 | 103 |
| `FABRICATION` | 3 | 4 | 0 | 0 |
| `HEDGING` | 0 | 1 | 0 | 0 |
| `MISREADING` | 14 | 49 | 0 | 0 |
| `OMISSION` | 2 | 15 | 0 | 0 |
| `QUESTION_MISALIGNMENT` | 7 | 14 | 0 | 0 |

### `taxonomy_evidence`

| Type | Fix | Still wrong | Break | Still correct |
|---|---:|---:|---:|---:|
| `CORRECT_OR_UNKNOWN` | 0 | 0 | 5 | 104 |
| `FABRICATION` | 2 | 5 | 0 | 0 |
| `HEDGING` | 1 | 0 | 0 | 0 |
| `MISREADING` | 16 | 47 | 0 | 0 |
| `OMISSION` | 3 | 14 | 0 | 0 |
| `QUESTION_MISALIGNMENT` | 10 | 11 | 0 | 0 |

## Bottleneck And Patch

Initial full run was inefficient because retrieval was recomputed once per arm-item row. With two arms, that meant 436 GTR note-span embedding passes for 218 unique cases. The runner now precomputes retrieval once per case and reuses spans across arms, with `--retrieval-workers` controlling the retrieval precompute stage.

Remaining bottleneck: CPU GTR sentence encoding. For the next embedding comparison, persist per-case sentence embeddings or retrieval JSON first, then run generation from the cached retrieval file. That will make vLLM C=8 the dominant runtime instead of embedding.

## Outputs

- Generated answers: `/home/ra/Projects/llm-ehr-hallucination/refactor/pre_atom_pipeline/output/retrieval_correction/qwen25_gtr_q_answer_nw-1_nc109_seed42/generated.jsonl`
- Judged answers: `/home/ra/Projects/llm-ehr-hallucination/refactor/pre_atom_pipeline/output/retrieval_correction/qwen25_gtr_q_answer_nw-1_nc109_seed42/judged.jsonl`
- Summary JSON: `refactor/pre_atom_pipeline/output/retrieval_correction/qwen25_gtr_q_answer_nw-1_nc109_seed42/summary.json`
