# Pre-Full-Scale Confirmation Plan

Date: 2026-05-29 (revised after user answers)
Status: working plan with user-decided defaults. The 962-case full-scale rerun is BLOCKED until every gate below is closed.

User direction this session:

1. Multirun — *"3 steps, but we shall tune."* Working default: **K=3** for detection and verdict; tune exact stages in Gate 1.
2. Retrieval — *"it depends on how we set up the RA-ICL, which I think we need to do more. There is a test showing that correction only with oracle guide, it solves a lot of problems, but I think it hasn't [become] clear yet."* RA-ICL needs more work as a substudy. The "correction only with oracle guide" branch is promising but not conclusive — include it in Gate 7 alongside the embedding/pool comparison.
3. Stats — *"per-fold mean and standard deviations. Specifically, there is a calibration with ground truth answer, but we can do when all is done."* Per-fold mean ± std is the main quote. Ground-truth calibration is a follow-up after the rerun.
4. GPU budget — *"we can run overnight, and make sure it is clear first."* Overnight compute is available; spend it on clarity, not speed.
5. Prompt scope — **PROMPT TEXT IS GLOBAL ACROSS MODELS.** Only the SETTING varies per model: chat template, context length, instruction-following compliance, thinking on/off, temperature, multirun K, note-context mode.

That last point reshapes Gate 7 from "per-model prompt tuning" to "per-model setting tuning at a fixed global prompt".

## Current State — already known

### Per-model zero-shot baselines (962 cases, 5 folds)

| Model | Correct | Wrong | Accuracy |
|---|---:|---:|---:|
| BioMistral-7B | 518 | 444 | 0.538 |
| DeepSeek-R1-Distill-Llama-8B | 740 | 222 | 0.769 |
| Qwen2.5-7B-Instruct | 853 | 109 | 0.887 |
| Llama-3.1-8B-Instruct | 857 | 105 | 0.891 |
| Qwen3-8B | 889 | 73 | 0.924 |

### 200-case (100 wrong / 100 correct, seed 42) snapshot

Pipeline: `meta_plan_confirm_natural` + `gpt4o-mini-helper-v2` + `operation_guided` + `multi_dimension` + vk1.

| Model | Context | N | Detected | Accepted | Stored Net | Transition Net |
|---|---|---:|---:|---:|---:|---:|
| biomistral-7b | dynamic_spans | 200 | 12 | 12 | 0 | +4 |
| biomistral-7b | first18k | 200 | 10 | 10 | -3 | 0 |
| deepseek-r1-distill-llama-8b | dynamic_spans | 200 | 20 | 10 | +4 | +2 |
| deepseek-r1-distill-llama-8b | first18k | 200 | 23 | 12 | +1 | -2 |
| llama-3.1-8b-instruct | dynamic_spans | 200 | 156 | 134 | +6 | -2 |
| llama-3.1-8b-instruct | first18k | 200 | 150 | 140 | +17 | +8 |
| qwen3-8b | dynamic_spans | 173 | 3 | 3 | +8 | +2 |
| qwen3-8b | first18k | 173 | 1 | 1 | +9 | +2 |

Qwen3 N=173 is pool exhaustion (only 73 wrong cases exist). Qwen2.5 200-case is still missing — it will be produced as part of Gate 7.

### What this snapshot tells us, after the user's clarification

Since prompt text is now fixed globally, the Llama over-firing and Qwen3 under-firing are **setting-side** problems:

- **Llama over-firing**: candidate causes (in order to test in Gate 7) — chat-template handling, instruction-following compliance under K=1, no multirun majority vote suppressing noisy detections.
- **Qwen3 under-firing**: candidate causes — Qwen3 with `enable_thinking=False` for detection may collapse the natural-memo step; revisit Qwen3 with thinking ON for detection only (correction stays thinking-OFF per `sub_variants.py`).
- **Llama vs DeepSeek context split**: setting per-model, not prompt re-write.

## Gates (revised)

### Gate 1 — Multirun design

Status: **working default K=3, exact stages to tune**.

User decision: 3 steps as the starting plan, tune from there.

Closure test (Qwen2.5 50-case, balanced 25w/25c, seed 42):
- **K=1 baseline** — current behavior.
- **K=3 detection only, majority verdict pick** — same correction + verdict.
- **K=3 detection + K=3 verdict** — both stages voted.
- **K=3 detection + K=3 generation (correction)** — correction-side sampling.

Metrics:
- detected rate, accepted rate, fix count, break count, net, candidate fix/break, transition fix/break/net.
- runtime (cost of multirun is the constraint).

Decision rule:
- Pick the smallest K configuration that improves transition net by at least the K=1 baseline's standard error on the 50-case test (or that materially reduces Llama-style over-firing in a 25w/25c Llama smoke-out).
- If two configurations tie, prefer the one with lower accepted-break count (safety).

Output: `reports/MULTIRUN_DECISION.md` with the picked stages and K values. Frozen for all later gates.

### Gate 2 — 5-fold separation integrity

Status: **open**.

Closure test (script, no GPU):
```
scripts/audit_fold_integrity.py
```

Checks:
- Each `output/folds/fold_X/{train,test}.jsonl` has disjoint patient_ids.
- Each `output/step8/<model>/fold_X/zeroshot_evaluated_binary.csv` has patient_ids ⊆ fold-X test set.
- Each `workspace/self_critique/data/bm_contrast_pool/fold_X_pool.json` has patient_ids ⊆ fold-X train set.
- No patient_id appears across both train and test of any fold.

Pass: zero violations across all 5 folds and 5 models.

### Gate 3 — No training contamination

Status: **open**.

Checks (mostly code inspection):
- Confirm no prompt template substitutes data from train side into the test-time inference.
- Confirm the GPT-4o judge sees only the discharge note + question + ground_truth + candidate answer (no extra context).
- Confirm `meta_plan_confirm_natural` two-step uses the same fold-test note for plan AND confirm (intended).
- Document the MIMIC-IV / EHRNoteQA pretraining-exposure caveat in `reports/NO_CONTAMINATION_NOTE.md`.

### Gate 4 — Parser confidence

Status: **open**.

Audit script:
```
scripts/audit_parser_confidence.py --runs runs/selfdetect_raicl_verdict/*nw100*
```

Metrics per model:
- valid-JSON rate from `gpt4o-mini-helper-v2`.
- critical-field-fill rate (verdict, error_type, correction_operation, wrong_claim, correct_or_missing_info, decisive_evidence) on detected-INCORRECT rows.
- verdict agreement against a tier-2 LLM parser (Qwen3.5-27B on MLX :8803) on a stratified 50-case sample.

Pass: ≥95% valid JSON, ≥90% critical-field-fill, ≥95% verdict agreement.

If a model fails, switch its parser backend (e.g., to `qwen2.5` self-parse with a helper-v2-style prompt) and rerun ONLY that model's 200-case for confirmation.

### Gate 5 — Statistical analysis

Status: **open**.

Final report set per model:
- Accuracy (zero-shot, candidate, final) — per fold, mean ± std, and overall.
- F1 (binary correct) — per fold, mean ± std.
- F1 by question-type — macro, with per-type counts.
- McNemar paired test, zero-shot vs final.
- Fix / break / net.
- Detected rate, accepted rate, acceptance precision.
- Per-fold accuracy (5 values + mean ± std).

Closure test:
- Extend `scripts/run_stats.py` to consume per-model paired_outcomes.csv and emit `reports/final_stats/<model>/{summary.json, table.md}` with all of the above.
- Smoke-test on the existing 200-case data before the 962-case run.

Deferred to after the rerun: **ground-truth calibration** (per user). This is the calibration of judge output vs human gold; do not block the rerun on it.

### Gate 6 — Per-model setting (NOT prompt)

Status: **open**.

The prompt is `meta_plan_confirm_natural` + `gpt4o-mini-helper-v2` + `operation_guided` + `multi_dimension` for ALL models. What varies per model is the setting block.

Per-model setting matrix:

| Model | max-model-len | dtype | chat template | thinking (det / corr) | note-context | K-detect | K-verdict |
|---|---:|---|---|---|---|---:|---:|
| BioMistral-7B | 8192 | bfloat16 | llama2 [INST] | n/a | TBD by Gate 8 | 3 | 3 |
| Qwen2.5-7B-Instruct | 16384 | bfloat16 | ChatML | n/a | TBD by Gate 8 | 3 | 3 |
| Qwen3-8B | 16384 | bfloat16 | ChatML | retest ON for det, OFF for corr | TBD by Gate 8 | 3 | 3 |
| DeepSeek-R1-Distill-Llama-8B | 32768 | bfloat16 | Llama-3 | always ON, strip_think | TBD by Gate 8 | 3 | 3 |
| Llama-3.1-8B-Instruct | 8192 | bfloat16 | Llama-3 | n/a | TBD by Gate 8 | 3 | 3 |

Closure test:
- For each model: 1+1 smoke run (no compute waste) under the recorded setting, confirms vLLM startup is correct, no `<think>` leak, no truncation, valid JSON downstream.
- Then 50-case smoke (25w/25c) under the K-values chosen by Gate 1.

Output: `configs/per_model_runtime.json` plus `configs/final_project.json` per-model block.

### Gate 7 — Retrieval-augment tuning (UNSETTLED — needs more work)

Status: **open and EXPANDED**. User: *"RA-ICL needs more work; there is a test showing correction-only with oracle guide solves a lot of problems but it hasn't become clear yet."*

This is the gate with the most outstanding uncertainty. Three substudies live under it:

**Substudy 7A — Correction-only with composite oracle guide (most promising according to user; needs investigation)**

User direction: *"error taxonomy looks okay, but I think it should combine with both question type, and the major error that can be used for correct, such as contradiction, omission, reasoning."* The oracle hint is therefore a COMPOSITE field with three pieces:

| Field | Source | Example |
|---|---|---|
| question_type | Existing question-type classifier output, or GPT-4o-mini single-pass | "medication change" / "procedure history" / "etiology" |
| major_error_for_correction | One of {CONTRADICTION, OMISSION, REASONING} derived from comparing zero-shot vs ground-truth | "OMISSION" |
| error_taxonomy_entry | Existing audited error-taxonomy entry where available; else GPT-4o-mini one-line | "missed required medication continuation status" |

- Hypothesis: a composite oracle hint (no detection step) at correction time fixes a lot more than the full detection→retrieval→correction pipeline.
- Closure test: 50-case Qwen2.5 head-to-head between (a) full pipeline at chosen K, (b) correction-only with composite oracle guide.
- Decision rule: if (b) beats (a) by ≥+5 net on a balanced 25w/25c sample with break delta ≤+1, document and elevate to a comparator condition (NOT replace the main pipeline silently — needs explicit method-section rewrite).
- Investigation notes:
  - The CONTRADICTION / OMISSION / REASONING split should map cleanly onto operation_guided correction's existing operations (REPLACE_VALUE / ADD_MISSING_SLOT / REFOCUS_TIME_OR_VISIT) — confirm the mapping is 1:1 before locking the schema.
  - Test whether question_type alone (without error_taxonomy) is enough — the existing question_type-aware variants from `claim_slot_conservative` already showed some lift in 40-case screens. Adds an A/B/C comparison: (b1) question_type only, (b2) major_error only, (b3) full composite.

**Substudy 7B — RA-ICL example pool rebuild**
- Hypothesis: the current `bm_contrast_pool/fold_X_pool.json` has 42% audited pair coherence; rebuild from audited high-confidence fix pairs (no fold leakage) for fair RA-ICL.
- Closure test: 50-case head-to-head with current pool vs rebuilt pool on Qwen2.5 correction stage.
- Decision rule: rebuild only if it beats current pool by ≥+3 net with break delta ≤+1.

**Substudy 7C — Note-span retriever bakeoff**
- Hypothesis: `gte-large-en-v1.5` multi-component scorer beats Step-9 R2 multi-query + agreement scoring on span sufficiency.
- Closure test: rerun the Step-9 retriever bakeoff (12 cases per the existing protocol) with both retrievers; manually score sufficiency. If `gte-large` wins, swap.

These three substudies can run in parallel; they all use the same Qwen2.5 base.

### Gate 8 — Note context (`first18k` vs `dynamic_spans`)

Status: **open**.

Per-model decision. The choice falls out of Gate 6's 50-case smoke and Gate 7's correction-quality results. Set per model in `configs/final_project.json`.

### Gate 9 — Per-fold rerun protocol

Status: **open**.

For the full-scale rerun, each model produces:
- Per-fold predictions: `runs/<model>/fold_<F>/{pipeline_outputs.jsonl, judged_outputs.jsonl, summary.json}`.
- A merged 962-case file for paired-outcomes stats.

Closure test: the existing `run_selfdetect_raicl_verdict.py` mixes folds (it loads all 5 then samples). For per-fold output, either:
- Add a `--fold` flag and run 5×N per model (recommended), or
- Add a post-processor that splits the merged JSONL by `fold` field (cheap).

Pick the second (cheaper), but ensure the report shows per-fold accuracy explicitly.

## Order of work

1. **Gate 1 — Multirun (Qwen2.5 50-case bakeoff).** Output: chosen K per stage.
2. **Gate 2, 3, 4 — Fold integrity + parser confidence (no heavy GPU).** Output: pass/fail per model.
3. **Gate 6 — Per-model setting smoke (5 × 1+1, then 5 × 25w/25c at chosen K).** Output: per-model runtime config.
4. **Gate 7 — Retrieval-augment tuning (3 substudies in parallel on Qwen2.5).** Output: keep / swap / rebuild decisions.
5. **Gate 8 — Note context (resolved within Gate 6's 25w/25c run per model).**
6. **Per-model 100w/100c confirmation at the chosen setting** — this closes the Qwen2.5 200-case gap and verifies Llama / Qwen3 setting changes worked.
7. **Sign-off — combined approval before 962-case launch.**
8. **Gate 9 — 962-case full-scale rerun, one model overnight at a time.**
9. **Gate 5 — Statistical analysis, per-fold mean ± std, F1, McNemar.**
10. **Ground-truth calibration (post-completion follow-up per user).**

## Open questions remaining

These were not answered today:

1. Substudy 7A — is the "correction-only with oracle guide" a comparator in the paper, or a replacement candidate? Depends on the 50-case test result.
2. Substudy 7B — if pool rebuild wins, who builds the audited high-confidence pair set? Heuristic vs human curation?
3. McNemar test format — single-pass paired (one final per case) or multirun-averaged?
