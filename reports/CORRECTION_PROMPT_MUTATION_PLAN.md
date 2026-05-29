# Correction Prompt Mutation Plan

Status: exploratory branch only. These variants are not final method choices and should not be merged to `main` until tested.

Branch: `experiment/correction-prompt-mutations`

## Goal

Study how the correction prompt changes the fix/break tradeoff for Qwen2.5 zero-shot answers after retrieval. The real zero-shot distribution is mostly correct, so a useful correction prompt must avoid breaking correct answers, not merely maximize fixes on wrong answers.

## Current Retrieval Baseline

The runner uses retrieved note spans from the source project Step 9 v2 index code. The current working retrieval mode remains:

- `gtr_q_answer`: question + previous answer query with agreement scoring, top-k spans.

Other retrieval modes stay available for comparison:

- `gtr_question`
- `gtr_oracle_error`

## Correction Prompt Arms

The script now supports these arms in `scripts/qwen25_retrieval_correction_quicktest.py`:

| Arm | Purpose |
|---|---|
| `evidence_only` | Basic evidence-guided correction baseline. |
| `taxonomy_evidence` | Existing taxonomy-aware correction baseline. |
| `oracle_error_description` | Upper-bound/error-analysis hint arm; not final-test safe. |
| `conservative_keep_gate` | Reduce false correction by editing only when evidence proves material error. |
| `quote_then_revise` | Force evidence anchoring before revision. |
| `minimal_patch` | Preserve correct parts and make the smallest edit. |
| `answer_from_evidence_then_compare` | Draft from evidence, compare to prior answer, revise only if meaning changes. |
| `contradiction_first` | Target contradicted/fabricated claims; avoid completeness edits. |
| `omission_first` | Target missing required answer slot. |
| `focus_first` | Target wrong visit/date/aspect errors. |
| `claim_table_private` | Claim-by-claim audit but final answer only. |
| `error_type_router` | Uses detected/taxonomy error type as weak routing hint. |
| `no_new_entities` | Prevent hallucinated entities not present in evidence. |
| `abstain_if_uncertain` | Return previous answer exactly unless evidence proves a better answer. |

## Suggested Quick Screens

Small smoke screen without GPT judge:

```bash
python scripts/qwen25_retrieval_correction_quicktest.py   --port 8003   --concurrency 8   --retrieval-workers 4   --n-wrong 5   --n-correct 5   --arms conservative_keep_gate minimal_patch omission_first focus_first abstain_if_uncertain
```

Balanced judged screen:

```bash
python scripts/qwen25_retrieval_correction_quicktest.py   --port 8003   --concurrency 8   --retrieval-workers 4   --n-wrong 50   --n-correct 50   --arms evidence_only taxonomy_evidence conservative_keep_gate minimal_patch answer_from_evidence_then_compare contradiction_first omission_first focus_first no_new_entities abstain_if_uncertain   --judge
```

Full Qwen2.5 correction screen after smoke validation:

```bash
python scripts/qwen25_retrieval_correction_quicktest.py   --port 8003   --concurrency 8   --retrieval-workers 4   --n-wrong -1   --n-correct 109   --arms taxonomy_evidence conservative_keep_gate minimal_patch answer_from_evidence_then_compare contradiction_first omission_first focus_first no_new_entities abstain_if_uncertain   --judge
```

## Selection Rule

Rank arms by net gain `fixes - breaks`, but inspect false corrections first. A correction arm should advance only if it keeps break count low on originally correct cases and gives usable gains on wrong cases.


## Full Pipeline Lever Variants

`scripts/run_selfdetect_raicl_verdict.py` now exposes three independent prompt selectors. This is the test surface for the current hypothesis.

Detection prompt variants:

- `contradiction_first`: higher-precision detection by prioritizing direct contradiction, then wrong focus, then central omission.
- `claim_contradiction`: claim-level contradiction audit; intended to produce better retrieval payloads.
- `p5_retrieval_payload`: previous broader detection payload baseline.

Correction prompt variants:

- `accept_suggestion_if_supported`: makes the correction model more willing to follow detection feedback when evidence supports it.
- `direct_rewrite_from_feedback`: rewrites around the detected target instead of minimally preserving the old answer.
- `contradiction_repair`: specialized repair for contradicted/unsupported claims.
- `omission_repair`: specialized repair for missing answer-slot errors.
- `balanced`: previous correction baseline.

Verdict prompt variants:

- `false_correction_sensitive`: compares original and corrected answer against discharge note and question, accepting correction only if clearly better.
- `derive_then_compare`: privately derives a note-supported answer before choosing A/B.
- `contradiction_count`: chooses the answer with fewer material note contradictions, then answer-slot coverage.
- `balanced`: previous pairwise gate baseline.

Recommended narrow comparison:

```bash
python scripts/run_selfdetect_raicl_verdict.py \
  --port 8003 \
  --concurrency 8 \
  --n-wrong 20 \
  --n-correct 20 \
  --det-temperature 0.0 \
  --correction-temperature 0.0 \
  --verdict-temperature 0.0 \
  --det-prompt contradiction_first \
  --correction-prompt accept_suggestion_if_supported \
  --verdict-prompt false_correction_sensitive \
  --judge
```

Then swap one lever at a time:

```bash
# More correction willingness
--correction-prompt direct_rewrite_from_feedback

# Contradiction-specific correction
--correction-prompt contradiction_repair

# Verdict derives note-supported answer first
--verdict-prompt derive_then_compare

# Verdict focuses on contradictions
--verdict-prompt contradiction_count

# Broader claim-level detection
--det-prompt claim_contradiction
```

Interpretation rule: contradiction-focused detection should lower false detections; suggestion-following correction should increase fixes among detected wrong answers; false-correction-sensitive verdict should reduce breaks on originally correct answers. The final choice should be based on net gain and break count, not only detection F1.


## 20/20 Qwen2.5 Screen Results

Date: 2026-05-28. Sample: seed 42, 20 originally wrong + 20 originally correct Qwen2.5 zero-shot answers. Local generation used Qwen2.5 vLLM on port 8003 with `temperature=0.0`; final labels used the fixed GPT-4o judge with the old Stage 1 prompt. Verdict `k=1` for speed.

| Detection | Correction | Verdict | Detected | Accepted | Fix | Break | Net | Comment |
|---|---|---|---:|---:|---:|---:|---:|---|
| `contradiction_first` | `accept_suggestion_if_supported` | `false_correction_sensitive` | 3 | 3 | 1 | 1 | 0 | Too conservative, but still broke one accepted correction. |
| `claim_contradiction` | `accept_suggestion_if_supported` | `false_correction_sensitive` | 5 | 3 | 2 | 0 | 2 | Best detection setting in this screen. |
| `p5_retrieval_payload` | `accept_suggestion_if_supported` | `false_correction_sensitive` | 9 | 4 | 0 | 1 | -1 | Broader detection over-triggered without useful fixes. |
| `claim_contradiction` | `direct_rewrite_from_feedback` | `false_correction_sensitive` | 5 | 3 | 2 | 0 | 2 | More forceful rewrite did not change final net. |
| `claim_contradiction` | `contradiction_repair` | `false_correction_sensitive` | 6 | 3 | 2 | 0 | 2 | Contradiction-specialized correction did not improve beyond baseline. |
| `claim_contradiction` | `accept_suggestion_if_supported` | `derive_then_compare` | 6 | 3 | 2 | 0 | 2 | Verdict derive-first behaved like false-correction-sensitive gate. |
| `claim_contradiction` | `accept_suggestion_if_supported` | `contradiction_count` | 6 | 3 | 2 | 0 | 2 | Contradiction-count gate behaved like derive-first gate. |

Current working read: detection is the active bottleneck. `claim_contradiction` is less conservative than `contradiction_first` but avoids the over-triggering seen with p5. The correction and verdict mutations were not harmful, but did not create extra accepted fixes on this sample. Next screen should scale `claim_contradiction + accept_suggestion_if_supported + false_correction_sensitive` to a larger matched set, then inspect the detected-but-rejected cases to decide whether correction or verdict needs another mutation.


## Question-Type / Answer-Slot Screen

Added question-type-aware variants after observing that some EHRNoteQA questions require exact numbers, dates, medications/doses, time periods, or multi-item lists. The new prompt fields are `QUESTION_TYPE`, `REQUIRED_ANSWER_FORMAT`, and `SLOT_CHECK`; retrieval queries now include those fields.

New variants:

- Detection: `question_slot`, `claim_slot_conservative`
- Correction: `question_slot_repair`
- Verdict: `slot_sensitive`

20/20 Qwen2.5 screen, same seed/settings as above:

| Detection | Correction | Verdict | Detected | Accepted | Fix | Break | Net | Comment |
|---|---|---|---:|---:|---:|---:|---:|---|
| `question_slot` | `question_slot_repair` | `slot_sensitive` | 14 | 7 | 0 | 1 | -1 | Raw slot-aware path over-triggered; slot attention alone is too broad. |
| `claim_slot_conservative` | `question_slot_repair` | `slot_sensitive` | 4 | 2 | 2 | 1 | 1 | Conservative slot gate reduced over-triggering, but slot-aware repair/verdict still broke one correct case. |
| `claim_slot_conservative` | `accept_suggestion_if_supported` | `false_correction_sensitive` | 5 | 3 | 0 | 1 | -1 | Slot-aware detection alone did not improve over `claim_contradiction`. |

Current working read: the idea is technically useful as metadata, but not yet a better gate. Slot awareness should be used to improve retrieval/correction details after a reliable detection, not as the main reason to trigger correction. The current best 20/20 setting remains `claim_contradiction + accept_suggestion_if_supported + false_correction_sensitive` with net +2 and zero breaks.
