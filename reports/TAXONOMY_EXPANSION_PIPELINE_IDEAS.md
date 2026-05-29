# Taxonomy-Expanded Correction Pipeline Ideas

Status: exploratory. Branch: `experiment/correction-prompt-mutations`.

## Context Check

The current full-note prompt path uses `row['note'][:18000]` for detection/correction/verdict. On the 20 wrong / 20 correct seed-42 Qwen2.5 screen:

- no model/API errors occurred;
- max plan output: 2,980 chars;
- max confirmation output: 2,768 chars;
- max source note in sample: 19,500 chars;
- 4/40 sampled notes exceeded the 18k-character slice.

So the prompt generally fits the current vLLM server (`--max-model-len 16384`), but some notes are truncated. For final tests, long notes should use dynamic context construction rather than blindly taking the first 18k chars.

## Core Idea

The prompt should not only ask “is this wrong?” It should ask “what kind of answer is required, what kind of evidence can prove it, and what kind of failure is most plausible?” Then correction should perform the smallest provable operation.

A useful taxonomy has four layers:

| Layer | Labels | Purpose |
|---|---|---|
| Question intent | number/value, date/time, medication/dose, list, yes/no status, cause/reason, procedure/event, outcome, other | Determines what must be preserved in the answer. |
| Evidence relation | support, full contradiction, partial conflict, note-silent, wrong focus, missing central slot | Distinguishes provable errors from weak suspicions. |
| Error mechanism | value mismatch, temporal mismatch, polarity error, central omission, unsupported addition, wrong entity/visit, list incompleteness, reasoning overreach | Tells correction exactly what operation to do. |
| Correction operation | replace value, add central slot, remove unsupported claim, refocus answer, preserve original | Keeps correction essential and provable. |

## Proposed Pipeline

1. **Meta-plan from question + zero-shot answer**
   - Identify question type and required answer format.
   - Extract central claims from the zero-shot answer.
   - Predict likely risks: contradiction, omission, wrong focus, value/date/list mismatch.
   - Produce retrieval queries and a checklist.

2. **Dynamic evidence context build**
   - For short notes, include full note slice.
   - For long notes, retrieve spans using question + answer + plan queries.
   - Include section headers and neighboring sentences around retrieved spans.
   - Keep exact numeric/date/medication/list evidence verbatim.

3. **One-by-one confirmation**
   - Confirm each planned item against evidence.
   - Require `FULL_CONTRADICTION=YES` only when the note states an incompatible fact.
   - Treat note silence as unsupported, not contradiction.
   - Treat number/date/dose/list mismatches as critical only when central to the asked slot.

4. **Correction directive**
   - Convert confirmed error into an operation:
     - replace wrong value/date/dose/item;
     - add missing central slot;
     - remove unsupported claim;
     - refocus wrong visit/date/aspect;
     - keep original if no provable operation exists.

5. **Correction generation**
   - Give the model only the confirmed operation and decisive evidence.
   - Do not ask for broad rewriting unless the answer is wrong-focus or badly misleading.

6. **Verdict gate**
   - Compare original vs corrected against question + decisive evidence + full note context when available.
   - Reject correction if it adds unsupported facts, drops required slot elements, or changes a correct answer unnecessarily.

## Prompt Directions Worth Testing

### A. Plan-Then-Confirm Detector

Already added as `--det-prompt meta_plan_confirm`.

Current 20/20 result:

| Detection | Correction | Verdict | Detected | Accepted | Fix | Break | Net |
|---|---|---|---:|---:|---:|---:|---:|
| `meta_plan_confirm` | `accept_suggestion_if_supported` | `false_correction_sensitive` | 5 | 3 | 1 | 0 | +1 |

This is safer than raw slot prompting but still below `claim_contradiction` (+2). It may improve with better context construction and taxonomy-specific correction operations.

### B. Taxonomy-Routed Confirmation

After the plan, use different confirmation criteria:

- number/date/dose: exact value check;
- list: central required item check;
- status/yes-no: polarity check;
- cause/reason: evidence-supported rationale check;
- event/procedure/outcome: entity + time/focus check.

### C. Error-Operation Correction

Instead of giving general detection feedback, pass an operation field:

```text
CORRECTION_OPERATION: REPLACE_VALUE / ADD_MISSING_SLOT / REMOVE_UNSUPPORTED_CLAIM / REFOCUS_TIME_OR_VISIT / KEEP_ORIGINAL
DECISIVE_EVIDENCE: ...
DO_NOT_CHANGE: supported parts of original answer
```

This should reduce false correction and make the model more willing to change when the operation is specific and provable.

### D. Evidence-Sufficiency Gate Before Correction

Before generating correction, require:

```text
EVIDENCE_SUFFICIENT_FOR_CORRECTION: YES/NO
```

Only run correction when YES. Otherwise keep original or send to verdict-only comparison.

## Current Recommendation

Do not promote raw slot-aware detection yet. Use slot/type taxonomy as metadata and retrieval guidance, not as the primary detection trigger.

The strongest current 20/20 candidate remains:

```bash
--det-prompt claim_contradiction --correction-prompt accept_suggestion_if_supported --verdict-prompt false_correction_sensitive
```

Next experiment should add dynamic evidence context and an explicit `CORRECTION_OPERATION` field, then retest `meta_plan_confirm` and `claim_contradiction` on the same 20/20 sample.


## Dynamic Context and Operation-Guided Screens

Implemented context modes in `scripts/run_selfdetect_raicl_verdict.py`:

- `--note-context first18k`: previous behavior.
- `--note-context dynamic_spans`: if note length exceeds `--context-threshold`, retrieve question/answer/plan-focused spans and use them as the note context for detection, correction, and verdict.
- `--note-context dynamic_summary`: for long notes, summarize the retrieved spans while preserving source spans underneath. This is viable but slower because it adds a local model call.

The dynamic context run used the same 20/20 seed-42 sample. `dynamic_spans` replaced the full note context for 7/40 long-note cases and kept max context under 15,878 chars.

| Detection | Correction | Verdict | Context | Detected | Accepted | Fix | Break | Net | Comment |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `meta_plan_confirm` | `accept_suggestion_if_supported` | `false_correction_sensitive` | `first18k` | 5 | 3 | 1 | 0 | 1 | Previous meta-plan baseline. |
| `meta_plan_confirm` | `accept_suggestion_if_supported` | `false_correction_sensitive` | `dynamic_spans` | 5 | 3 | 1 | 0 | 1 | Same net, avoids first-18k truncation for long notes. |
| `meta_plan_confirm` | `operation_guided` | `false_correction_sensitive` | `dynamic_spans` | 6 | 5 | 2 | 1 | 1 | More willing to correct, but introduced one break. |
| `meta_plan_confirm` | `accept_suggestion_if_supported` | `false_correction_sensitive` | `dynamic_summary`, 5/5 smoke | 3 | 2 | 2 | 1 | 1 | Viable, but one break in small smoke; needs more testing before promotion. |

Interpretation: dynamic spans solve the long-note truncation issue without changing net performance on this screen. Operation-guided correction increases fixes but also break risk, so the next useful mutation is not more aggressive correction; it is a stricter verdict/evidence-sufficiency gate for `operation_guided`.

Recommended next test:

```bash
python scripts/run_selfdetect_raicl_verdict.py \
  --port 8003 \
  --concurrency 8 \
  --n-wrong 20 \
  --n-correct 20 \
  --det-temperature 0.0 \
  --correction-temperature 0.0 \
  --verdict-temperature 0.0 \
  --verdict-k 3 \
  --det-prompt meta_plan_confirm \
  --correction-prompt operation_guided \
  --verdict-prompt false_correction_sensitive \
  --note-context dynamic_spans \
  --context-threshold 16000 \
  --context-k 12 \
  --judge
```

The `k=3` verdict should test whether the one break is a single-sample gate instability or a systematic false correction.
