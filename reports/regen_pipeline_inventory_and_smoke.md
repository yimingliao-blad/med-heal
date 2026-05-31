# Regen Pipeline Inventory and Smoke Tests

Date: 2026-05-29

This note records the existing regeneration/correction methods found in the repo, the cached full-scale/pilot results, and the smoke tests added to confirm that the methods are still wired correctly. The smoke tests are wiring tests only; they monkeypatch LLM, parser, and judge calls so they do not measure model quality.

## Regen Methods Found

| Method | Code Path | What It Does | Current Role |
|---|---|---|---|
| Step 9 `regen+count` | `src/pre_atom/legacy/step9_self_correction/v2/regen_pilot.py` | Regenerates a fresh zero-shot answer, then runs a blind A/B count-compare verdict. Qwen parser extracts the A/B choice. | Strong previous baseline, especially for DeepSeek full-scale. |
| Step 9 `regen_v3` / CoVe | `src/pre_atom/legacy/step9_self_correction/v2/regen_v3_pilot.py` | Regenerates with explicit evidence, critiques unsupported claims, optionally regenerates again, then uses a 3-count verdict with regex + Qwen fallback parsing. | Conservative follow-up; useful for parser/verdict engineering, but not clearly better in cached pilots. |
| Legacy T0 regen subvariants | `src/pre_atom/legacy/ichl/prompt_engineering/correction/runner.py` and `sub_variants.py` | Five single-call regeneration prompts: bare regen, CoT regen, packed CoT with A0, re-read, and challenge-and-regen. | Prompt family for correction-only/fullscale experiments. |

## Existing Cached Results

The previous Step 9 V2 output inventory is in `/home/ra/Projects/llm-ehr-hallucination/output/step9_v2/multi_model/`. All five model folders have `regen_fullscale.jsonl` with 962 rows.

### Full 962 `regen_fullscale.jsonl`

| Model | N | Corrected | Kept Original | Fixes | Breaks | Net |
|---|---:|---:|---:|---:|---:|---:|
| BioMistral-7B | 962 | 486 | 476 | 51 | 70 | -19 |
| DeepSeek-R1-Distill-Llama-8B | 962 | 509 | 453 | 76 | 31 | +45 |
| Llama-3.1-8B-Instruct | 962 | 497 | 465 | 30 | 20 | +10 |
| Qwen2.5-7B-Instruct | 962 | 504 | 458 | 27 | 33 | -6 |
| Qwen3-8B | 962 | 468 | 494 | 26 | 16 | +10 |

Interpretation: `regen+count` is not uniformly good, but it is a real signal. DeepSeek benefits strongly; Qwen2.5 and BioMistral regress; Llama and Qwen3 gain modestly. This supports keeping regen as a model-dependent comparator, not as a universal correction default.

### 40-Case `regen_audit.jsonl`

| Model | N | Fixes | Breaks | Net |
|---|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Llama-8B | 40 | 4 | 2 | +2 |
| Llama-3.1-8B-Instruct | 40 | 3 | 1 | +2 |
| Qwen2.5-7B-Instruct | 40 | 4 | 1 | +3 |
| Qwen3-8B | 40 | 5 | 2 | +3 |

### 40-Case `regen_v3_audit.jsonl`

| Model | N | Fixes | Breaks | Net |
|---|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Llama-8B | 40 | 3 | 0 | +3 |
| Llama-3.1-8B-Instruct | 40 | 2 | 2 | 0 |
| Qwen2.5-7B-Instruct | 40 | 1 | 0 | +1 |
| Qwen3-8B | 40 | 0 | 0 | 0 |

Interpretation: `regen_v3` is safer on this small screen, but it routes fewer useful changes. It may be useful for conservative gating ideas, but the cached evidence does not justify replacing the simpler `regen+count` comparator.

## Smoke Test Added

Added `scripts/smoke_regen_methods.py`.

It checks:

| Smoke Target | What Is Verified | Result |
|---|---|---|
| Step 9 `regen+count` | `run_one` builds a regen candidate, blind A/B verdict accepts the corrected slot, judge branch writes an audit record. | Passed |
| Step 9 `regen_v3` / CoVe | Evidence-format regen parses, critique branch runs, 3-count verdict parses via regex, audit record is written. | Passed |
| Legacy T0 subvariants | All five subvariant prompts run through `run_correction_one_item` with a fake client and non-truncated output. | Passed |
| Qwen3 legacy T0 thinking flag | `qwen3-8b` returns `enable_thinking=False` from the correction runner. | Passed |

Command used:

```bash
/home/ra/Projects/llm-ehr-hallucination/.venv/bin/python scripts/smoke_regen_methods.py
```

Output summary:

```json
{
  "ok": true,
  "results": {
    "step9_regen_count": {"method": "regen_zeroshot", "action": "corrected"},
    "step9_regen_v3": {"method": "cove_2round_regen_v3", "verdict_source": "regex", "action": "corrected"},
    "legacy_t0_subvariants": {
      "subvariants": ["a:bare_regen", "b:cot_regen", "c:packed_cot", "d:re_read", "e:challenge_and_regen"],
      "qwen3_enable_thinking": false
    }
  }
}
```

Also compiled successfully:

```bash
/home/ra/Projects/llm-ehr-hallucination/.venv/bin/python -m py_compile   scripts/smoke_regen_methods.py   src/pre_atom/legacy/ichl/prompt_engineering/correction/sub_variants.py   src/pre_atom/legacy/step9_self_correction/v2/regen_pilot.py   src/pre_atom/legacy/step9_self_correction/v2/regen_v3_pilot.py
```

## Qwen3 Thinking Handling

Two Qwen3 paths now explicitly avoid think-block contamination:

| Path | Handling |
|---|---|
| Current natural/self-detect script | Uses Qwen chat template prevention plus output `strip_think` cleanup. Fresh Qwen3 reruns contained no `<think>` tags. |
| Legacy T0 regen subvariants | `src/pre_atom/legacy/ichl/prompt_engineering/correction/sub_variants.py` now maps `qwen3-8b` subvariants to `enable_thinking=False`. |

Step 9 V2 regen pilots already use `MODELS["qwen3"]["chat_template_kwargs"] = {"enable_thinking": False}` through `set_default_chat_template_kwargs`.

## Recommendation

Keep regen as a comparator and fallback family, not as the single main path. The cached full-scale result shows strong model dependence: DeepSeek gains substantially, Qwen3/Llama gain modestly, and Qwen2.5/BioMistral regress. For the next larger validation, compare the natural detection/correction/verdict pipeline against `regen+count` per model. Do not assume the regen method transfers unchanged across model families.

For future runs, report these separately:

| Metric | Why |
|---|---|
| `regen_fullscale` final net | Captures the historical direct-regeneration baseline. |
| Candidate fix/break before verdict, when available | Separates regen candidate quality from the A/B gate. |
| Verdict route rate | Shows whether the method is over-accepting regenerated answers. |
| Qwen3 think-token audit | Confirms no hidden reasoning leaks into parsing or judging. |
| Per-question failure taxonomy | DeepSeek gains may come from different error types than Qwen/Llama. |
