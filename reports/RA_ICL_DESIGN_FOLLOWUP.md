# RA-ICL Design Follow-up

Date: 2026-05-29 (after user direction)
Status: pre-decision investigation. No code changes. The purpose of this document is to act on five specific instructions the user gave after reading `RA_ICL_ARCHITECTURE_AND_DESIGN.md`.

User direction this turn (verbatim):

1. Channel A direct few-shot at generation time: *"I don't think so. It was too rough at that time, and it doesn't provide any help for Channel B."* — Channel A is **OUT** of the final pipeline.
2. Channel A pool/embedding/presentation: *"need some plan to think how to confirm."*
3. Channel B same-patient spans: *"make sure the pilot test done yesterday have any more detail to tune."*
4. Channel B cross-patient analogy: *"use current BM based pool, but still concern with how to split and whether the GTR embedding recall is good enough."*
5. Substudy 7A oracle-guide: *"we need to gather the knowledge of A and the problem definition again to make the decision."*

This document answers each in order.

## 1. Channel A is OUT of the final pipeline

Locked. The final pipeline is **zero-shot generation → Channel-B correction**. Channel A's gtr_note_pos_k1 / multiturn / posneg / negative-only ICL conditions are not in the final method.

The Step 8 zero-shot CSVs at `output/step8/<model>/fold_*/zeroshot_evaluated_binary.csv` are the input to all downstream stages. The Channel-A-augmented Step 8 CSVs (gtr_note_pos_k1, multiturn, cot_evidence, etc.) are not used as inputs to detection or correction. They may still be reported as **baselines/comparators** for the paper's results table (see §2).

## 2. Channel A confirmation plan

Even with Channel A out of the final method, the paper benefits from a clean baseline table that shows zero-shot vs Channel A vs final. Step 8 already ran full-scale Channel A conditions; we don't need new generation, only re-judgment under the locked judge.

### What's already on disk

| Path | Status | Content |
|---|---|---|
| `output/step8/<model>/fold_*/zeroshot_evaluated_binary.csv` | judged (older Step 8 prompt) | zero-shot answers |
| `output/step8/<model>/fold_*/gtr_note_pos_k1_evaluated_binary.csv` | judged (older Step 8 prompt) | Channel A positive few-shot |
| `output/step8/<model>/fold_*/gtr_note_neg_k1_evaluated_binary.csv` | judged | Channel A negative few-shot |
| `output/step8/<model>/fold_*/gtr_note_posneg_k1_evaluated_binary.csv` | judged | Channel A pos+neg |
| `output/step8/<model>/fold_*/multiturn_evaluated_binary.csv` | judged | Channel A multiturn |
| `output/step8/<model>/fold_*/cot_evidence_evaluated_binary.csv` | judged | CoT evidence |
| `output/step8/<model>/fold_*/cot_conclusion_evaluated_binary.csv` | judged | CoT conclusion |

The labels were produced under the older Step 8 reason-producing GPT-4o prompt (per `reports/VARIANT_DECISION_MATRIX.md`), not the locked Stage 1 binary judge. For a paper-consistent baseline table we need to **re-judge** these with the locked judge.

### Re-judgment plan (no new generation)

1. Locked judge: `gpt4o_stage1_binary_T0.1` (`src/step9_self_correction/v2/judge.py` or `legacy/ichl/judges/gpt4o_stage1_binary_judge.py`).
2. Re-judge all five models × five folds × the baseline conditions: `zeroshot`, `gtr_note_pos_k1`, `gtr_note_neg_k1`, `gtr_note_posneg_k1`, `multiturn`, `cot_evidence`, `cot_conclusion`.
3. Estimated GPT-4o calls: 962 cases × 7 conditions × 5 models = ~33,670 judgments at ~$0.008 each → ~**$270 oracle budget**. Sequential per the GPT-4o-sequential rule (~10 hours wall time). Could be split overnight per model.
4. Output: `output/step8_rejudged_locked/<model>/<condition>_judged.csv` with binary labels under the locked judge.
5. Smoke first: re-judge fold 0 of Qwen2.5 zero-shot only (~193 calls, $1.50, 4 minutes) and confirm row-by-row agreement with the legacy label > 0.92. If it diverges materially, the judge change is itself a finding to report.

### Decision rule for §2 close-out

- Re-judgment is approved → run the smoke first, then the full re-judge overnight.
- Re-judgment is deferred → use the existing Step 8 evaluated CSVs as baselines and add a single sentence in the paper noting the legacy judge.

## 3. Channel B same-patient spans — what more is tunable from yesterday's pilot

Yesterday's pilot: `scripts/qwen25_retrieval_correction_quicktest.py`. Output: `~/Projects/llm-ehr-hallucination/refactor/pre_atom_pipeline/output/retrieval_correction/qwen25_gtr_q_answer_nw-1_nc109_seed42/`.

Sample: 109 wrong + 109 correct (218 judged). Single-stage pipeline: zero-shot answer → correction (no detection, no verdict). Retrieval: same-patient note spans via GTR-T5 multi-query + agreement scoring.

### Result recap

| Arm | N | Fix | Break | Net | Kept wrong | Kept correct |
|---|---:|---:|---:|---:|---:|---:|
| `evidence_only` | 218 | 26 | 6 | +20 | 83 | 103 |
| `taxonomy_evidence` | 218 | 32 | 5 | **+27** | 77 | 104 |

Compare with Qwen2.5 `regen+count` 962-case (net **-6**) and the meta_plan_confirm_natural 40-case Qwen2.5 best (net +5 / +4 in cross-context). The single-stage correction-only with taxonomy-evidence guide beats both by a large margin on a much larger Qwen2.5 sample (218 vs 40).

### Per-error-type breakdown (taxonomy_evidence arm)

| Error type | Fix | Still wrong | Recall (fix / wrong) |
|---|---:|---:|---:|
| MISREADING | 16 | 47 | 25% |
| QUESTION_MISALIGNMENT | 10 | 11 | 48% |
| OMISSION | 3 | 14 | 18% |
| FABRICATION | 2 | 5 | 29% |
| HEDGING | 1 | 0 | 100% (n=1) |
| CORRECT_OR_UNKNOWN (originally correct) | — | 104 still correct, 5 broken | break rate 4.6% |

Tuning opportunities (NOT exercised yesterday):

| Knob | Yesterday | Other values present in code | Untested |
|---|---|---|---|
| Retrieval mode | `gtr_q_answer` | `gtr_question`, `gtr_oracle_error` | both other modes |
| Top-K spans | 5 | adjustable | 3, 8 |
| Agreement floor | 0.40 (hardcoded in `note_span_index.py`) | configurable | 0.30, 0.50 |
| Correction arm | 2 of 14 | 12 more arms exist | `conservative_keep_gate`, `minimal_patch`, `quote_then_revise`, `answer_from_evidence_then_compare`, `contradiction_first`, `omission_first`, `focus_first`, `claim_table_private`, `error_type_router`, `no_new_entities`, `abstain_if_uncertain`, `oracle_error_description` |
| Span scoring | `agreement` | `max` is alternative | `max` mode |
| Sentence splitter | regex `(?<=[.!?])\s+` with ≥10 char filter | hardcoded in `note_span_index.py` | longer span unit (paragraph), shorter (clause) |
| Note slice limit | 18000 chars per arm | hardcoded in `build_correction_user` | unlimited / dynamic |
| Embedding model | GTR-T5-base | swap to gte-large-en-v1.5 | gte-large at same K |
| Query construction | `[question, original_answer[:800]]` | mode-specific | per-error-type query templates |

### High-value next pilots for Channel B

A focused micro-study before full-scale:

| Study | Variants | Purpose |
|---|---|---|
| 3B-1: Arm sweep | 14 correction arms × `gtr_q_answer` × k=5 | does `taxonomy_evidence` win consistently, or do `omission_first` / `quote_then_revise` win for specific error types? |
| 3B-2: Retrieval mode | `gtr_question` / `gtr_q_answer` / `gtr_oracle_error` × top arm | does adding error-taxonomy queries to retrieval (`gtr_oracle_error`) raise misreading recall above 25%? |
| 3B-3: K and agreement floor | k=3/5/8 × floor 0.30/0.40/0.50 × top arm | does narrower or broader span set change break rate on originally-correct cases? |
| 3B-4: Per-error-type best arm | union over arms per error type, hold-out by error type | could a per-error-type arm router give MISREADING > 25% recall without raising breaks? |
| 3B-5: Embedding swap | GTR-T5 vs gte-large-en-v1.5 × top arm | does gte-large-en-v1.5 (newer, untested at this stage) materially change span sufficiency? |

3B-1 closes the "any more detail to tune" ask cleanest — we know `taxonomy_evidence` is best of 2 tested, but 12 untested arms remain. ~218 cases × 12 new arms × 1 LLM call = ~2600 LLM calls; on Qwen2.5 at concurrency 8, ~1.5-2 hours.

### What the pilot did NOT test

- Multirun (K-stage) — single sample per case at T=0.
- Verdict gate — no acceptance filter; every corrected answer was kept.
- Wrong-case pool was 109 = ALL of Qwen2.5's wrong cases. Pool exhaustion, like Qwen3 200-case. The result is essentially full-scale for the wrong-class.
- 5-fold separation — sample mixed all 5 folds. Per-fold mean ± std (Gate 5) not yet produced for this pilot.

## 4. Channel B cross-patient analogy — fold split and GTR recall

### Current state of analogy retrieval

The natural-pipeline runner (`scripts/run_selfdetect_raicl_verdict.py:998`) defines `retrieve_example` but the locked `operation_guided` correction prompt does NOT consume `{example_block}`. Cross-patient analogy is dead code under the current locked prompt.

The retrieval mechanism in `retrieve_example`:

```python
def toks(s): return set(re.findall(r'[a-zA-Z0-9]+', (s or '').lower()))
def retrieve_example(row, det):
    pool = load_pool(row['fold'])
    query = ' '.join([row['question'], det.get('error_type',''), det.get('question_focus',''),
                      det.get('wrong_claim',''), det.get('correct_or_missing_info','')])
    qt = toks(query)
    def score(ex):
        text = ' '.join([ex.get('question',''), ex.get('what_was_wrong',''), ex.get('ground_truth','')])
        return len(qt & toks(text))
    return max(pool, key=score)
```

This is **plain lexical token-set intersection**, not GTR. The user's concern about "GTR embedding recall" is therefore about a retrieval mechanism that **is not yet in the runner**.

### What the bm_contrast_pool offers

| File | Content |
|---|---|
| `bm_contrast_pool/fold_X_pool.json` | 320 entries per fold (fold-0 sample). Each entry: question, ground_truth, wrong_answer, what_was_wrong, evidence_from_notes (verified quotes), evidence_verified (per-quote bool), n_verified_quotes, raw. |
| `bm_contrast_pool/fold_X_question_embeddings.npy` | Pre-computed GTR-T5 question embeddings for the pool entries. **EXISTS but UNUSED by the runner.** |

### Fold split convention (verified)

From `workspace/self_critique/scripts/build_atomic_pool.py:43`:
> "Using fold_0_pool gives us folds 1,2,3,4; fold_1_pool gives fold 0,2,3,4; etc."

So `fold_X_pool.json` EXCLUDES fold X. When processing a test row with `fold=X`, loading `fold_X_pool.json` gives the cross-patient analogy candidates from folds ≠ X. This is correct leave-one-out fold safety.

The runner does this correctly: `load_pool(row['fold'])` reads `fold_{row['fold']}_pool.json`. **No fold leakage exists in the current runner setup.**

This is auditable in Gate 2 — `scripts/audit_fold_integrity.py` should add an assertion that `set(pool_X.fold_id) ∩ {X} == ∅`.

### GTR recall question — what it would mean to test

The user's concern translates to: if we switch the analogy retrieval from lexical token-overlap to GTR cosine similarity on question embeddings, does GTR find materially-relevant analogy examples?

Test design (small, no new compute beyond CPU embedding):

1. For each test row in a 50-case Qwen2.5 sample (matched to yesterday's wrong-case set):
   - Build a query: same as the runner's `retrieve_example` query (question + detection payload).
   - Lexical retrieval: pick the lexical-top-1 entry from the fold-safe pool.
   - GTR retrieval: encode the query with GTR-T5-base, cosine top-1 against `fold_X_question_embeddings.npy`.
2. For each of the 50 cases, label each candidate analogy as "useful / weak / irrelevant" — manually, or via GPT-4o-mini with a fixed prompt.
3. Compute recall@1 and rank-correlation between lexical and GTR.

Recall@1 thresholds (user-discussed): we say GTR is "good enough" if useful-analogy rate at top-1 is ≥40% on Qwen2.5 wrong-cases. Threshold to be confirmed.

Even if GTR wins on recall, the downstream question stays open: does the analogy in the correction prompt actually raise net fix-rate? That requires running the correction with the analogy block included on Qwen2.5 wrong-cases, comparing to the existing 218-case baseline (no analogy). The analogy block is a prompt-engineering knob, not a free win.

### Suggested 4B-1: GTR vs lexical recall@1 on 50-case Qwen2.5 sample

- Build the comparison table. CPU only, ~5 min runtime.
- Output: `reports/CHANNEL_B_ANALOGY_RECALL_PILOT.md`.
- If GTR top-1 isn't materially better than lexical top-1, the question becomes "is analogy worth adding at all" — likely answer no.
- If GTR top-1 is better, then 4B-2: rerun the correction-only pilot with the analogy block added at top-1 GTR.

## 5. Channel A knowledge + problem definition — input to Substudy 7A scope

### What Channel A carries

Channel A (gtr_note_pos_k1 / multiturn) places a similar patient's `[Question]` + `[ground-truth-style Answer]` into the system prompt before zero-shot generation. The knowledge transferred is:

- Style — answer length, register, phrasing of medical facts.
- Slot — when the demonstration's answer matches the question's required slot (medication / date / list), the model is nudged toward the same shape.
- Calibration — for negative variants, the "this is wrong, here's the correction" frames the model's confidence.

The knowledge NOT transferred:

- Same-patient facts — the demonstration is from a different patient.
- Question-specific clinical content — the question text differs between demonstration and target.

### The problem the correction-only-with-oracle pipeline solves

INPUT to correction: discharge note + question + zero-shot wrong answer + (optional) error taxonomy primary_error label + (optional) retrieved same-patient spans.
TASK: rewrite the answer so it's correct for the exact question, grounded in the same-patient discharge note.
OUTPUT: corrected answer.

The taxonomy label is precomputed offline from the source repo's audited error-taxonomy data — it's not retrieval-derived, not LLM-judged at correction time, and not available for unseen new cases. **This is critical.** Yesterday's +27 result depends on having taxonomy.primary_error available. For a real deployment, taxonomy.primary_error must either be:

- (a) replaced by a live classifier that runs at correction time, OR
- (b) replaced by the detection step's `error_type` field (CONTRADICTION / OMISSION / QUESTION_MISALIGNMENT), OR
- (c) honestly reported as an oracle-aided result, not a deployable method.

Option (b) is the closest analogue to "live oracle" — the natural pipeline's detection step already produces `error_type` as one of {CONTRADICTION, OMISSION, QUESTION_MISALIGNMENT, NONE, UNCLEAR}, which maps cleanly onto the taxonomy primary_error categories used yesterday.

### Implication for Substudy 7A's scope

Per user's earlier direction, 7A should combine: question_type + major_error_for_correction (CONTRADICTION / OMISSION / REASONING) + error_taxonomy_entry.

Mapping to data sources:

| Field | Source for the pilot | Source for the real method |
|---|---|---|
| question_type | existing question-type classifier output (or GPT-4o-mini single-pass) | same |
| major_error_for_correction | offline error taxonomy primary_error | natural-pipeline detection's `error_type` |
| error_taxonomy_entry | offline error taxonomy `error_description` | GPT-4o-mini one-line interpretation of detection's `wrong_claim` + `correct_or_missing_info` |

The 7A pilot can run BOTH configurations side-by-side:
- 7A-oracle: uses the offline taxonomy (matches yesterday's +27 setup).
- 7A-live: uses the detection-step outputs at correction time (matches a deployable method).

Comparing the two head-to-head answers a key question: how much of yesterday's +27 came from the oracle's quality versus the bare presence of a hint? If 7A-live drops materially below 7A-oracle, the deployable method ceiling is somewhere between zero-shot and oracle.

Channel A knowledge does NOT contribute to either 7A configuration. Channel A would only matter at the zero-shot generation step before either of these arms runs; given the user's decision (§1), that step is left as plain zero-shot.

### Recommended 7A scope (for user approval)

- 7A-oracle (control): zero-shot wrong + same-patient gtr_q_answer spans + offline taxonomy primary_error + offline error_description + `taxonomy_evidence` prompt. Replicates yesterday's +27.
- 7A-live (deployable): zero-shot wrong + natural-pipeline detection (full meta_plan_confirm_natural + helper-v2) + same spans + detection's error_type + detection's wrong_claim/correct_or_missing_info as hint + `taxonomy_evidence` prompt. Same shape as 7A-oracle, but with detection-derived hints.
- 7A-detection-only-no-spans: control to show the spans are doing work.
- 7A-spans-only-no-hint: control to show the hint is doing work (the `evidence_only` arm essentially is this, net +20).

Sample: 218 Qwen2.5 (109w + 109c, same as yesterday) for direct comparability.

## Open decisions for the user

1. Approve re-judging Step 8 conditions under the locked judge (~$270 oracle budget)? Y/N + smoke first?
2. For 3B-1 arm sweep, accept the full 14 arms × 218 cases (~1.5-2 hours on Qwen2.5)? Or narrow to 4 arms?
3. For 4B-1 GTR-vs-lexical analogy recall, accept the 50-case CPU pilot? Or skip and report current lexical setup as-is?
4. For 7A scope, accept the 4-arm side-by-side (oracle / live / detection-only / spans-only)? Or pick a subset?
5. Across all of the above: any preference for which model gets the next pilot first (Qwen2.5 has the most cached data; others would require fresh generation under each model's locked setting)?
