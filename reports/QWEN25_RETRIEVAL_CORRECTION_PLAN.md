# Qwen2.5 Retrieval And Correction Plan

Status: planning note for the next retrieval/correction validation. This is not a final method choice.

## Current Best Baseline To Beat

The strongest completed correction family is Step 9 V2 `regen+count`: regenerate the answer from scratch, compare original vs regenerated answer against the note, parse the A/B verdict, and keep the selected answer.

Full-scale result for Qwen2.5 is not positive:

| Method | N | Baseline | Final | Fix | Break | Net |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 `regen+count` | 962 | 88.67% | 88.05% | 27 | 33 | -6 |

So for Qwen2.5, `regen+count` is the best completed self-correction baseline but not a final candidate by itself. The immediate target is to beat both zero-shot and this regen baseline without increasing breaks.

## Qwen2.5 Mistake Profile

The newer canonical Qwen2.5 taxonomy covers all 109 wrong zero-shot answers.

| Primary error | Count | Share |
|---|---:|---:|
| `MISREADING` | 63 | 57.8% |
| `QUESTION_MISALIGNMENT` | 21 | 19.3% |
| `OMISSION` | 17 | 15.6% |
| `FABRICATION` | 7 | 6.4% |
| `HEDGING` | 1 | 0.9% |

Collapsed correction view:

| Correction type | Share |
|---|---:|
| Contradiction: `MISREADING` + `FABRICATION` | 64.2% |
| Question/context alignment: `QUESTION_MISALIGNMENT` + `HEDGING` | 20.2% |
| Missing information: `OMISSION` | 15.6% |

The old taxonomy is multi-label and broader. Across `output/step8/error_classification/all_errors_by_patient.json`, labels include:

| Old label | Count |
|---|---:|
| `hallucination` | 752 |
| `omission` | 557 |
| `reasoning_failure` | 235 |
| `context_confusion` | 195 |
| `specificity` | 121 |
| `temporal_error` | 94 |

Mapping for planning:

| New taxonomy | Related old labels | Retrieval implication |
|---|---|---|
| `MISREADING` | `reasoning_failure`, `context_confusion`, `temporal_error`, some `hallucination` | Retrieve contradictory evidence spans from the same note/time context. |
| `FABRICATION` | `hallucination` | Retrieve spans showing the answer claim is unsupported or contradicted; model should remove invented details. |
| `OMISSION` | `omission`, `specificity` | Retrieve answer-bearing spans and prompt for missing required components. |
| `QUESTION_MISALIGNMENT` | `context_confusion`, `temporal_error`, `reasoning_failure` | Retrieve spans anchored to the right visit/date/aspect and explicitly restate the question target. |

## What Regen Fixes Today

Joining Qwen2.5 `regen_fullscale.jsonl` to the 109 canonical taxonomy rows:

| Primary error | Wrong cases | Fixed by regen | Fix rate |
|---|---:|---:|---:|
| `MISREADING` | 63 | 15 | 23.8% |
| `QUESTION_MISALIGNMENT` | 21 | 6 | 28.6% |
| `OMISSION` | 17 | 4 | 23.5% |
| `FABRICATION` | 7 | 2 | 28.6% |
| `HEDGING` | 1 | 0 | 0.0% |

This says regen can sometimes fix every major error family, but the full-scale system loses overall because it also breaks 33 originally correct answers. The retrieval/correction design should focus on precision: only revise when retrieved evidence clearly supports the revision.

## Retrieval Questions To Separate

There are three different retrieval problems that should not be mixed:

1. **Evidence retrieval for correction**: given the question, note, and wrong answer, retrieve note spans that contain the information needed to correct the mistake.
2. **Positive example retrieval**: retrieve a similar question with a correct answer and show it as an in-context example.
3. **Error example retrieval**: retrieve a similar wrong answer plus corrected answer to teach the model what not to do.

The Stage 8 RA-ICL retrieval mostly tested examples. The Step 9 retrieval bakeoff tested evidence spans. For Qwen2.5 correction, evidence-span retrieval should be treated as the primary path.

## Evidence So Far

Historical Stage 8 / Pilot 12:

- `gtr_note_pos_k1` was the best simple example retrieval condition in older Qwen2.5 Pilot 12: 80.14 vs 77.13 zero-shot, +3.01pp.
- Later Step 8 judged results did not reproduce a stable Qwen2.5 gain: zero-shot 88.67 vs `gtr_note_pos_k1` 88.57.
- Negative examples were weak for Qwen2.5: `gtr_note_neg_k1` was below zero-shot in Pilot 12; larger negative k usually degraded or only had tiny gains.

Newer retrieval-quality study:

- `gte-large-en-v1.5` had the best embedding-quality metrics among `nomic`, `bge-m3`, and `gte-large-en-v1.5`.
- Multi-component scoring beat note-only retrieval for example selection.
- These retrieval-quality gains have not yet translated into stable RA-ICL downstream gains.

Step 9 correction retrieval:

| Retriever | Sufficient spans |
|---|---:|
| R1 single-query embedding on error statement, top-3 | 2/12 |
| R2 multi-query embedding + agreement scoring, top-5 | 5/12 |
| R3 Qwen cite-by-number K=5 | 4/12 |
| R4 union R3/R2 | 4/12 |

Working correction-retrieval baseline: R2 multi-query span retrieval, top-5.

## What Information May Help Qwen2.5 Correct Itself

For `MISREADING` / `FABRICATION`:

- Show the model its original wrong claim.
- Show top retrieved evidence spans that contradict or fail to support that claim.
- Ask for a minimal corrected answer grounded only in the spans and note.
- Do not rely on a generic similar QA example alone; most errors are note-specific attribution errors.

For `OMISSION`:

- Show retrieved answer-bearing spans.
- Add a compact checklist: answer every component asked by the question.
- A correct similar QA example may help if selected by question type and answer structure, but evidence spans should still be primary.

For `QUESTION_MISALIGNMENT`, `context_confusion`, and `temporal_error`:

- Parse and restate the target visit/date/aspect before answering.
- Retrieve spans from the right temporal context.
- Example retrieval should be type-matched by temporal/question structure, not just note similarity.

For `reasoning_failure` / `specificity`:

- Show expected answer components and evidence spans.
- Prefer concise final-answer revision over long chain-of-thought; long reasoning can introduce extra unsupported claims.

## Candidate Arms For The Next Test

Use Qwen2.5 wrong-answer set first, then test break rate on originally correct answers.

| Arm | Purpose | Retrieval |
|---|---|---|
| A. Zero-shot original | Baseline | none |
| B. Regen+count | Current completed correction baseline | none |
| C. Evidence-only GTR | Fast baseline with existing Stage 8 embedding | GTR note-span top-5 |
| D. Evidence-only R2 | Best current correction retrieval | Multi-query + agreement top-5 |
| E. Evidence + error-type instruction | Tests taxonomy-aware correction | R2 top-5 + primary error prompt |
| F. Positive example + evidence | Tests whether similar correct answer helps | GTR/gte positive example + R2 spans |
| G. Error example + evidence | Tests whether direct wrong/correct example helps | same-error example + R2 spans |

Do not use ground truth from the target item in retrieval. It is allowed to use correct answers from training-pool examples, because those are the retrieved demonstrations.

## Embedding/Recall Validation

Start with GTR for continuity, but compare it to newer candidates.

Embeddings/scorers to compare:

| Candidate | Role |
|---|---|
| `sentence-transformers/gtr-t5-base` note-only | Historical Stage 8 baseline |
| `Alibaba-NLP/gte-large-en-v1.5` note-only | Best newer embedding-quality candidate |
| `gte-large-en-v1.5` question+note scorer | Better example matching than note-only |
| R2 multi-query + agreement | Best current evidence-span retriever |
| BM25 question/note | Sparse lexical control |

Primary retrieval metric should be **evidence sufficiency recall**, not just embedding similarity:

- `sufficient@1`, `sufficient@3`, `sufficient@5`: does the retrieved content contain enough information to correct the known wrong answer?
- `same_error_type@k`: for error examples, does the retrieved example match the target taxonomy?
- `answer_structure_match@k`: for positive examples, does the retrieved example require the same answer shape?
- Downstream `fix`, `break`, and `net`: only after retrieval recall is acceptable.

Recommended first sample:

- All 109 Qwen2.5 wrong cases for sufficiency recall.
- A matched sample of 109 Qwen2.5 correct cases for break-risk prompts.
- GPT judge fixed configuration for final labels: old Stage 1 GPT-4o prompt, `temperature=0.1`, sequential calls.
- Local generation: Stage 8 policy `temperature=0` unless testing regen/multirun behavior.

## Working Decision

Use `gtr_note_pos_k1` as the historical RA-ICL baseline, not the final correction method.

For the correction path, start from R2 evidence-span retrieval and add taxonomy-aware prompting. The most important hypothesis is not “similar examples improve Qwen2.5”; it is “the model needs targeted evidence that exposes why its current answer is wrong, and the system must avoid revising correct answers unless evidence is strong.”
