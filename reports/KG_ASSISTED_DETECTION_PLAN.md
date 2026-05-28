# KG-Assisted Detection Study Plan

This note documents the correction-focused detection variants added after reviewing the legacy `src/self_atom_v6` and `src/self_atom_v7` folders in the source project.

## Legacy KG Code Checked

- `src/self_atom_v7/kg_build.py`: builds a question-level grammar KG with `build_kg`, newer `build_kg_reconstruct`, and `render`.
- `src/self_atom_v7/adapters.py`: converts a KG into question stems, target relation, and negation signals with `kg_to_qstems`.
- `src/self_atom_v6/v6_8/tests/pilot_grammar_kg.py`: older pilot comparing spaCy dependency triples and LLM grammar triples.

The v7 KG is designed around structured clinical question/vignette text. For EHRNoteQA notes, the safer first test is to build KG context from the question plus top lexical note spans, not from the entire discharge note.

## Hypothesis

KG context may help self-detection when the zero-shot answer has:

- contradiction: answer claim conflicts with a structured note fact;
- omission/ignorance: the required answer slot or question stem is present in the KG/evidence but missing from the answer;
- focus error: answer discusses the wrong visit, time period, treatment, or clinical aspect.

The KG must not become the judge. It is only a compact evidence index. The tested model still decides whether the answer is correct and what correction payload to generate.

## New Prompt Variants

Added to `scripts/qwen25_detection_prompt_bakeoff.py`:

- `p8_natural_blind_error_hypothesis`: natural blind error hypothesis before final verdict.
- `p9_claims_first_natural`: claim-by-claim natural audit without over-constrained formatting.
- `p10_omission_slot_check`: required-answer-slot and answered-slot comparison.
- `p11_kg_assisted_contradiction_omission`: KG/evidence-assisted contradiction and omission check.
- `p12_kg_plus_direct_evidence`: two-pass structured context plus direct evidence verification.

## Test Switches

The bakeoff script now supports:

```bash
python scripts/qwen25_detection_prompt_bakeoff.py   --port 8003   --concurrency 8   --temperature 0.0   --n-wrong 50   --n-correct 50   --prompts p7_error_gate_payload p8_natural_blind_error_hypothesis p9_claims_first_natural p10_omission_slot_check
```

Evidence-assisted comparison:

```bash
python scripts/qwen25_detection_prompt_bakeoff.py   --port 8003   --concurrency 8   --temperature 0.0   --n-wrong 50   --n-correct 50   --kg-source evidence   --prompts p11_kg_assisted_contradiction_omission p12_kg_plus_direct_evidence
```

Optional v7 KG-assisted comparison:

```bash
python scripts/qwen25_detection_prompt_bakeoff.py   --port 8003   --concurrency 8   --temperature 0.0   --n-wrong 20   --n-correct 20   --kg-source v7_kg   --prompts p11_kg_assisted_contradiction_omission p12_kg_plus_direct_evidence
```

`v7_kg` falls back to candidate evidence spans if KG extraction fails. This keeps tests executable even when spaCy/KG dependencies are unavailable.

## Decision Criteria

Use the same held-out sample and compare:

- detection F1 on wrong-vs-correct zero-shot answers;
- false-positive rate on originally correct answers, because the real distribution is heavily correct;
- retrieval-ready payload rate among detected errors;
- downstream net gain after correction plus verdict gate.

A prompt should advance only if it improves downstream net gain, not only detection recall.
