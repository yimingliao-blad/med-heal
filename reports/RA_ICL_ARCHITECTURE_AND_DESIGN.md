# RA-ICL Architecture and Design Discussion

Date: 2026-05-29
Status: pre-decision audit. No code changes proposed yet. The purpose of this document is to make every design knob explicit so the user can decide what to retrieve, what to present, and how to retrieve, for each of the two RA-ICL channels.

User direction this turn: *"the first is to confirm how RA-ICL should be configured. It has two parts. Direct few-shot or the RA during the correction. You shall check the architecture to understand it. The biggest thing is what to retrieve and what to present, and how to retrieve."*

## Two RA-ICL channels — different stages, different jobs

### Channel A — Direct few-shot at generation time (Step 8)

The model has not produced an answer yet. The retrieved example sits in the system prompt to shape the zero-shot output. Source: `src/step8_multimodel_icl/generate_step8.py` (source repo). Existing trained results from Step 8 full-scale.

### Channel B — RA during correction (Step 9 / med-heal runner)

The model has already produced an answer, detection has flagged it wrong, and retrieved spans + optionally an analogy example sit in the correction prompt. Source: `scripts/run_selfdetect_raicl_verdict.py` (med-heal). Current 200-case results were produced with Channel B at K=1.

The two channels are independent levers. They can be combined, but the design choices for each are separate.

## Channel A — current architecture

### What to retrieve

| Variant | Pool source | Pool size (per fold) | Object retrieved |
|---|---|---|---|
| `gtr_note_pos_k1` | `output/pilot_12_ra_icl/indices/fold_X/correct_pool.json` | ~600-700 | One reference case: question + ground-truth-style openended_answer. |
| `gtr_note_neg_k1` | `output/pilot_12_ra_icl/indices/fold_X/incorrect_pool.json` | ~100-200 | One reference case: question + wrong answer + ground truth. |
| `gtr_note_posneg_k1` | both pools | both | One positive + one negative. |
| `gtr_note_any_unlabeled_k1` | mixed correct + incorrect | combined | One reference case, label hidden. |
| `random_pos_k1` | correct pool | same | random positive (no retrieval). |
| `random_neg_k1` | incorrect pool | same | random negative. |

### How to retrieve

| Variant | Embedding | Query side | Pool side | Metric | Top |
|---|---|---|---|---|---|
| `gtr_note_*` | `sentence-transformers/gtr-t5-base` (CPU) | query NOTE text | pool NOTE embeddings | cosine | k=1 |
| `multiturn` | same | query NOTE text | correct_pool NOTE embeddings | cosine | k=1 |
| `random_*` | — | — | — | random shuffle | k=1 |

Key fact: Step 8 retrieves by NOTE similarity, not question similarity. The intuition is that similar discharge notes share similar clinical context.

Newer alternative (ICHL retrieval study, untrialed at full Step 8 scale): `Alibaba-NLP/gte-large-en-v1.5` multi-component scorer (note + question + GT alignment composite 0.637 vs note-only 0.452).

### What to present

| Variant | Fields shown | Section header |
|---|---|---|
| `gtr_note_pos_k1` | `[Question]`, `[Answer]` | "Here is an example of a good answer" |
| `gtr_note_neg_k1` | `[Question]`, `[Incorrect Answer]`, `[Correct Answer]` | "Here is an example of a common mistake to avoid" |
| `gtr_note_posneg_k1` | both blocks | "EXAMPLE OF A MISTAKE" + "EXAMPLE OF A GOOD ANSWER" |
| `multiturn` | `[Question]` + `[Answer]` as a prior turn pair | uses model's chat template |
| `gtr_note_any_unlabeled_k1` | `[Question]`, `[Answer]` (label hidden) | "Here is an example from a similar patient case" |

Field is `openended_answer` from the reference case — NOT the multiple-choice letter, NOT the discharge note from the reference case, NOT explicit evidence quotes.

### Existing evidence for Channel A

From `reports/RETRIEVAL_AUGMENT_EVIDENCE.md`:

| Model | Best Channel-A variant | Delta vs zeroshot |
|---|---|---:|
| BioMistral-7B | none — zeroshot best | retrieval hurt |
| Qwen2.5-7B | `multiturn` / `gtr_note_pos_k1` | +3.32 / +2.59 |
| Llama-3.1-8B | `gtr_note_pos_k1` | +1.04 |
| Qwen3-8B | `multiturn` / `gtr_note_neg_k1` / `gtr_note_pos_k1` | +3.96 / +3.64 / +3.54 |

Newer judged Step 8 summary (more conservative): Qwen2.5 essentially flat with retrieval; Qwen3 +0.73 on `gtr_note_pos_k1`; Llama small gain on `gtr_note_neg_k5`; BioMistral retrieval hurts.

## Channel B — current architecture

### What to retrieve

Two parallel pieces are retrieved in the current med-heal runner:

| Piece | Pool source | Pool size | Object |
|---|---|---|---|
| Same-patient note-span | the patient's own discharge note (per row) | varies (sentence-split, ≥10 chars) | top-5 sentences |
| Cross-patient analogy example | `workspace/self_critique/data/bm_contrast_pool/fold_X_pool.json` | 320/fold (fold-0) | one entry: question + wrong_answer + what_was_wrong + ground_truth + evidence_from_notes |

Pool quality flag: `bm_contrast_pool` audited pair coherence is 42% (most pairs have weak or mislabeled "what was wrong" / correct answer relationships). The newer correction prompts in scope (`operation_guided`, the natural-pipeline default) do NOT thread `{example_block}` into the prompt at all — only `question_slot_repair` consumes the analogy example. So the locked pipeline currently ignores Channel-B cross-patient retrieval.

### How to retrieve

| Piece | Embedding | Query | Pool side | Metric / scoring | Top |
|---|---|---|---|---|---|
| Same-patient note-span | `sentence-transformers/gtr-t5-base` (CPU) | multi-query list: detection's `question_focus`, `slot_check`, `key_evidence_reason`, `wrong_claim`, `correct_or_missing_info`, `evidence_needed`, plus its 3 `retrieval_queries` | per-row sentence embeddings | agreement-floor scoring (sum of `max(0, sim − 0.40)` over queries) | k=5 |
| Cross-patient analogy | (none — no embedding) | concat `[question, error_type, question_focus, wrong_claim, correct_or_missing_info]` | concat `[example.question, example.what_was_wrong, example.ground_truth]` | token-set intersection count | k=1 (argmax) |

The same-patient retriever IS the multi-query + agreement scoring from Step 9 (R2 in `step9_v2/retriever_bakeoff.md`; sufficient on 5/12 cases, the best of the four retrievers tried there).

The cross-patient analogy retrieval is plain lexical — no embedding model.

### What to present

| Piece | Rendered as | Inserted at |
|---|---|---|
| Same-patient note-span (top-5) | bullet list `[1] sentence\n[2] sentence ...` | `{spans_block}` in correction prompt |
| Cross-patient analogy (k=1) | block of `Question / Wrong answer / What was wrong / Correct answer pattern / Evidence style` (5 fields) | `{example_block}` — IGNORED by `operation_guided` and `evidence_only`, USED by `question_slot_repair` |

The locked natural pipeline uses `operation_guided`, so it presents same-patient spans only. Cross-patient analogy is dead code in the current configuration.

### Existing evidence for Channel B

From `reports/RETRIEVAL_AUGMENT_EVIDENCE.md` + `step9_v2/retriever_bakeoff.md`:

| Retriever (note-span) | Sufficient spans |
|---|---:|
| R1 single-query embedding on error statement, top-3 | 2/12 = 17% |
| R2 multi-query + agreement scoring, top-5 (current) | 5/12 = 42% |
| R3 Qwen cite-by-number, k=5, question-only | 4/12 = 33% |
| R4 union(R3, R2), top-5 | 4/12 = 33% |

No Channel-B equivalent of the Step 8 multi-model gain table — Channel B's effect lives inside the natural-pipeline net (current 200-case results), where the gain over zero-shot is mostly small/single-digit and the contribution of the spans vs the detection vs the correction prompt is not yet isolated.

## Where the design space is open

### Open for Channel A (direct few-shot at generation)

| Knob | Options |
|---|---|
| Pool | (a) keep `output/pilot_12_ra_icl/indices/fold_X/correct_pool.json` (most-tested); (b) rebuild from audited Step 8 high-confidence correct answers (smaller, higher quality); (c) per-model pool (each model's own correct cases). |
| Retrieval embedding | (a) GTR-T5-base note (tested, baseline winner); (b) `gte-large-en-v1.5` note (newer, no full-scale Step 8 evidence); (c) `gte-large-en-v1.5` multi-component scorer note+question+GT (newer, no Step 8 evidence). |
| Query side | (a) discharge note text (current); (b) question text; (c) note + question concat. |
| K | k=1 (Step 8 default) vs k=2-3 (Pilot 12 showed k>1 mostly degrades). |
| Prompt presentation | (a) `[Question]` + `[Answer]` (current `gtr_note_pos_k1`); (b) `[Question]` + `[Answer]` + `[Evidence quote from reference note]`; (c) Q + A in multiturn slot (Step 8's `multiturn`); (d) negative-only with `[Question]` + `[Incorrect Answer]` + `[Correct Answer]`. |
| Per-model on/off | which models include Channel A at all (BioMistral hurts from retrieval; everyone else neutral-to-positive). |

### Open for Channel B (RA during correction)

| Knob | Options |
|---|---|
| Same-patient note-span retrieval | keep R2 (current) vs rebuild with `gte-large-en-v1.5`. R2 is 42% sufficient — known ceiling. |
| Same-patient note-span K | k=5 (current) vs k=3 (cleaner context) vs k=8 (more recall). |
| Same-patient note-span query construction | current multi-query (8+ queries from detection payload) vs single-query of the wrong-claim text vs question-only. |
| Same-patient span presentation | bullet list (current) vs quoted block with source-line tag vs paraphrase-then-cite. |
| Cross-patient analogy on/off | currently OFF for `operation_guided`. ON if `question_slot_repair` is used. |
| Cross-patient analogy pool | (a) keep `bm_contrast_pool/fold_X_pool.json` (42% coherence); (b) rebuild from audited high-confidence correction pairs (smaller, expensive to label); (c) drop entirely and not use cross-patient analogy. |
| Cross-patient analogy retrieval | (a) keep lexical token overlap (current); (b) embedding-based (GTR / gte-large); (c) error-type bucket match first, then embedding within bucket. |
| Cross-patient analogy presentation | current 5-field block vs minimal 3-field (question + what_was_wrong + ground_truth) vs full reference note quote. |

### Open architectural choices that span both channels

| Question | Why it matters |
|---|---|
| Are Channel A + Channel B used together in the final pipeline, or do they compete? | If Channel A is the generation prompt, Channel B's detection signal may be on a DIFFERENT answer distribution than the 200-case results (which used Channel A zero-shot inputs). |
| Is the SAME embedding model used for both channels? | Sharing one embedder (GTR-T5 or gte-large) cuts model-load cost. Two embedders gives flexibility but adds CPU memory. |
| Is the SAME query construction used across both channels? | Step 8 retrieves by NOTE; Step 9 R2 retrieves by detection-derived multi-query. They serve different jobs. |
| Where does "oracle guide" sit? | The Substudy 7A "correction only with oracle guide" arm bypasses detection's retrieval queries entirely — it's a Channel-B-only variant where the retrieval is steered by a question-type + major-error oracle hint. Needs scope. |

## Existing facts that ground these choices

- **5-fold integrity** (Gate 2): the pools above must be fold-train-side only for the current fold's test data. Step 8's correct_pool.json/incorrect_pool.json folder structure already does this. `bm_contrast_pool` does too, by construction.
- **Same-patient note-span retrieval is NOT a leakage risk** — the test patient's own note is permitted as evidence; that's the whole point of the correction-time RA.
- **Cross-patient analogy IS a leakage risk** — must be fold-train-only.
- **Compute**: GTR-T5-base on CPU is the current model load. Adding gte-large would add ~1.5 GB CPU mem and ~3-5× embed cost. Both fit in current laptop budget.

## What this document leaves to the user

I will not pick. The decisions below need user direction before any code or pilot work begins.

1. Is Channel A in the final pipeline at all, or is the final answer always "zero-shot generation + Channel-B correction"?
2. If Channel A is in: which pool (existing pilot_12 vs rebuilt), which retrieval embedding (GTR vs gte-large), which presentation (pos-only vs neg-only vs posneg vs multiturn)?
3. For Channel B same-patient spans: keep GTR-T5 R2 multi-query at k=5, or test gte-large multi-component for spans?
4. For Channel B cross-patient analogy: drop entirely (current state for `operation_guided`), keep with current pool quality, or rebuild the pool first?
5. For Substudy 7A "correction-only with oracle guide": is this an INDEPENDENT correction prompt that bypasses detection and retrieval, or does it sit ON TOP OF Channel B's note-span retrieval?

Once these are decided, the bakeoffs in Gate 7 become concrete.
