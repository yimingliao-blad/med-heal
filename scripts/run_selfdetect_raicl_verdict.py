#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, random, re, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get('MED_HEAL_SOURCE_REPO', PROJECT_ROOT.parent / 'llm-ehr-hallucination'))
OUT_ROOT = PROJECT_ROOT / 'runs' / 'selfdetect_raicl_verdict'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(SOURCE_REPO / 'src' / 'step9_self_correction' / 'v2'))
from note_span_index import topk_spans  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from llm_audit import set_ledger, log_call  # noqa: E402
from note_span_index import split_sentences, get_embedder  # noqa: E402
import note_chunker as _NC  # noqa: E402
import numpy as _np  # noqa: E402

DET_SYSTEM = "You are a careful clinical QA auditor. You must check whether an answer is supported by the discharge note."
DET_P5 = """Discharge note:
{note}

Question:
{question}

Answer to audit:
{answer}

Your job is not only to decide whether the answer is wrong. Your job is to create a correction payload that a downstream retrieval step can use.

Check in this order:
1. QUESTION_FOCUS: What exact visit/date/aspect/fact does the question ask for?
2. ANSWER_FOCUS: What does the answer actually focus on?
3. WRONG_OR_MISSING_TARGET: If wrong, identify the smallest wrong claim or missing required fact.
4. EVIDENCE_NEEDED: What note evidence would prove the correction?

Only mark INCORRECT if the issue changes the answer to the question. Do not flag minor extra details.

Return exactly this template:
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
WRONG_CLAIM: the smallest wrong claim, or NONE
CORRECT_OR_MISSING_INFO: the fact that should replace/add/refocus the answer, or NONE
EVIDENCE_NEEDED: what kind of note span should be retrieved
RETRIEVAL_QUERY_1: short query using the question focus
RETRIEVAL_QUERY_2: short query using the wrong/missing target
RETRIEVAL_QUERY_3: short query using key clinical entities
CORRECTION_HINT: one sentence telling the downstream corrector what to change
WHY: one short explanation"""

DET_CONTRADICTION_FIRST = """Discharge note:
{note}

Question:
{question}

Answer to audit:
{answer}

Focus first on contradiction, because contradiction is the most reliable error signal. Ask whether the answer makes a clinically important claim that the discharge note directly contradicts. If there is no direct contradiction, then check for wrong question focus. Only use omission when the answer misses the central required answer slot.

Do not mark incorrect for harmless extra wording or missing background detail. If you cannot name the contradicted claim or exact wrong focus, mark CORRECT.

Return exactly this template:
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or QUESTION_MISALIGNMENT or OMISSION or NONE
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
WRONG_CLAIM: exact contradicted/misaligned/missing claim, or NONE
CORRECT_OR_MISSING_INFO: exact note-supported replacement or required missing fact, or NONE
EVIDENCE_NEEDED: note span needed to verify the contradiction or focus
RETRIEVAL_QUERY_1: short query using the wrong claim
RETRIEVAL_QUERY_2: short query using the note-supported replacement
RETRIEVAL_QUERY_3: short query using the question focus
CORRECTION_HINT: one sentence telling the corrector what to change
WHY: one short explanation"""

DET_CLAIM_CONTRADICTION = """Discharge note:
{note}

Question:
{question}

Answer to audit:
{answer}

Extract the answer's main claims. For each claim, decide whether the note supports it, contradicts it, or does not address it. The correction pipeline should run only if at least one central claim is contradicted, the answer targets the wrong visit/date/aspect, or the required answer slot is absent.

Return exactly this template:
CLAIM_CHECK:
- claim: ... | status: supported/contradicted/not-addressed/wrong-focus | evidence target: ...
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or QUESTION_MISALIGNMENT or OMISSION or NONE
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
WRONG_CLAIM: exact claim to repair, or NONE
CORRECT_OR_MISSING_INFO: exact note-supported replacement or required missing fact, or NONE
EVIDENCE_NEEDED: exact evidence span needed
RETRIEVAL_QUERY_1: short query
RETRIEVAL_QUERY_2: short query
RETRIEVAL_QUERY_3: short query
CORRECTION_HINT: one sentence
WHY: one short explanation"""

DET_CLAIM_SLOT_CONSERVATIVE = """Discharge note:
{note}

Question:
{question}

Answer to audit:
{answer}

Audit the answer with two gates.

Gate 1: identify the exact answer slot only from what the question explicitly asks for. Slot types include NUMBER_OR_VALUE, DATE_OR_TIME, MEDICATION_OR_DOSE, LIST_OR_MULTI_PART, YES_NO_STATUS, CAUSE_REASON, PROCEDURE_EVENT, or OTHER.

Gate 2: check the answer's central claims against the discharge note. Mark INCORRECT only if one of these is true:
- the answer gives a central number/date/time/dose/list item that conflicts with the note;
- the question explicitly asks for a number/date/time/dose/list/status/reason/event and the answer omits that central required slot;
- the answer targets the wrong visit/date/aspect;
- a central clinical claim is contradicted by the note.

Be conservative. Do not flag missing background details, harmless wording differences, or extra non-central details. For list questions, require that a missing/extra item changes the clinical answer.

Return exactly this template:
QUESTION_TYPE: NUMBER_OR_VALUE or DATE_OR_TIME or MEDICATION_OR_DOSE or LIST_OR_MULTI_PART or YES_NO_STATUS or CAUSE_REASON or PROCEDURE_EVENT or OTHER
REQUIRED_ANSWER_FORMAT: what central answer slot must be present
CLAIM_CHECK:
- claim: ... | status: supported/contradicted/not-central/wrong-focus | evidence target: ...
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or QUESTION_MISALIGNMENT or OMISSION or NONE
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
SLOT_CHECK: whether the central slot is present and correct
WRONG_CLAIM: exact central claim/value/date/list item to repair, or NONE
CORRECT_OR_MISSING_INFO: exact note-supported replacement or required missing central slot, or NONE
EVIDENCE_NEEDED: exact note span needed
RETRIEVAL_QUERY_1: short query using the central required slot
RETRIEVAL_QUERY_2: short query using the wrong/missing value/date/item
RETRIEVAL_QUERY_3: short query using key clinical entities
CORRECTION_HINT: one sentence telling the corrector what to change
WHY: one short explanation"""

DET_SLOT_REASONED = """Discharge note:
{note}

Question:
{question}

Answer to audit:
{answer}

Use a short, staged audit. Do not write a long explanation; use the stages only to make the final fields reliable.

Stage 1 - question slot: identify the exact answer type explicitly requested by the question. Slot types include NUMBER_OR_VALUE, DATE_OR_TIME, MEDICATION_OR_DOSE, LIST_OR_MULTI_PART, YES_NO_STATUS, CAUSE_REASON, PROCEDURE_EVENT, or OTHER.

Stage 2 - required evidence: identify the specific note evidence needed to verify that slot. For numbers/dates/doses/lists, preserve exact values and units.

Stage 3 - answer comparison: compare the answer only on central claims and the required slot. A wrong or missing central number/date/dose/list item is a critical error. A non-central missing background detail is not.

Stage 4 - decision: mark INCORRECT only for a central contradiction, wrong focus, or missing central answer slot.

Return exactly this template:
QUESTION_TYPE: NUMBER_OR_VALUE or DATE_OR_TIME or MEDICATION_OR_DOSE or LIST_OR_MULTI_PART or YES_NO_STATUS or CAUSE_REASON or PROCEDURE_EVENT or OTHER
REQUIRED_ANSWER_FORMAT: exact format/facts the answer must contain
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
SLOT_CHECK: supported / contradicted / missing-central-slot / wrong-focus / sufficient
KEY_EVIDENCE_REASON: one short sentence naming the decisive note evidence or missing evidence
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or QUESTION_MISALIGNMENT or OMISSION or NONE
WRONG_CLAIM: exact central claim/value/date/list item to repair, or NONE
CORRECT_OR_MISSING_INFO: exact note-supported replacement or required missing central slot, or NONE
EVIDENCE_NEEDED: exact note span needed
RETRIEVAL_QUERY_1: short query using the required slot and entities
RETRIEVAL_QUERY_2: short query using the wrong/missing value/date/item
RETRIEVAL_QUERY_3: short query using the question focus
CORRECTION_OPERATION: REPLACE_VALUE or ADD_MISSING_SLOT or REMOVE_UNSUPPORTED_CLAIM or REFOCUS_TIME_OR_VISIT or KEEP_ORIGINAL
EVIDENCE_SUFFICIENT_FOR_CORRECTION: YES or NO
DECISIVE_EVIDENCE: exact note evidence that proves the operation, or NONE
DO_NOT_CHANGE: supported parts of the original answer that should be preserved, or NONE
CORRECTION_HINT: one sentence telling the corrector what to change
WHY: one short final rationale"""

DET_META_PLAN = """Question:
{question}

Zero-shot answer:
{answer}

Make an audit plan before reading the note. The plan should say what must be checked, not decide correctness.

Use the question to infer the required answer slot. Use the zero-shot answer to list central claims that could be contradicted, missing, or wrong-focus. Pay attention to numbers, dates/times, medication names/doses/frequencies, list completeness, yes/no status, causes/reasons, and procedures/events.

Return exactly this template:
QUESTION_TYPE: NUMBER_OR_VALUE or DATE_OR_TIME or MEDICATION_OR_DOSE or LIST_OR_MULTI_PART or YES_NO_STATUS or CAUSE_REASON or PROCEDURE_EVENT or OTHER
REQUIRED_ANSWER_FORMAT: exact format/facts the answer must contain
ANSWER_CLAIMS_TO_VERIFY:
- claim: ... | why it matters: ...
CONTRADICTION_CANDIDATES:
- claim/value/date/list item that could be contradicted, or NONE
OMISSION_CANDIDATES:
- required slot/list item that could be missing, or NONE
FOCUS_RISKS:
- visit/date/aspect/time-period risk, or NONE
CHECKLIST:
1. ...
2. ...
3. ...
LIKELY_ERROR_MODE: CONTRADICTION or OMISSION or QUESTION_MISALIGNMENT or NONE_OR_UNCLEAR
RETRIEVAL_QUERY_1: query for the required answer slot
RETRIEVAL_QUERY_2: query for possible contradiction evidence
RETRIEVAL_QUERY_3: query for focus/date/time/list evidence"""

DET_META_CONFIRM = """Discharge note:
{note}

Question:
{question}

Zero-shot answer to audit:
{answer}

Audit plan made before reading the note:
{plan}

Now verify the plan one item at a time against the discharge note. For each central claim or slot, decide whether the note supports it, directly contradicts it, only partially conflicts, does not address it, or shows the answer has the wrong focus.

A full contradiction requires that the answer makes a central claim and the note states an incompatible fact. Do not call something a contradiction when the note is merely silent or the answer is less detailed. For numbers, dates, doses, medication names, time periods, and central list items, exact mismatch can be a full contradiction if it changes the answer.

After verifying items one by one, decide what is most likely happening: full contradiction, omission of a central required slot, wrong focus, or no correction-worthy error.

Return exactly this template:
QUESTION_TYPE: NUMBER_OR_VALUE or DATE_OR_TIME or MEDICATION_OR_DOSE or LIST_OR_MULTI_PART or YES_NO_STATUS or CAUSE_REASON or PROCEDURE_EVENT or OTHER
REQUIRED_ANSWER_FORMAT: exact format/facts the answer must contain
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
PLAN_CHECK:
- item: ... | note status: supported/full-contradiction/partial-conflict/not-addressed/wrong-focus/missing-central-slot | decisive evidence: ...
FULL_CONTRADICTION: YES or NO
SLOT_CHECK: supported / full-contradiction / partial-conflict / missing-central-slot / wrong-focus / sufficient
KEY_EVIDENCE_REASON: one short sentence naming the decisive evidence
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or QUESTION_MISALIGNMENT or OMISSION or NONE
WRONG_CLAIM: exact central claim/value/date/list item to repair, or NONE
CORRECT_OR_MISSING_INFO: exact note-supported replacement or required missing central slot, or NONE
EVIDENCE_NEEDED: exact note span needed
RETRIEVAL_QUERY_1: short query using the required slot and entities
RETRIEVAL_QUERY_2: short query using the wrong/missing value/date/item
RETRIEVAL_QUERY_3: short query using the question focus
CORRECTION_OPERATION: REPLACE_VALUE or ADD_MISSING_SLOT or REMOVE_UNSUPPORTED_CLAIM or REFOCUS_TIME_OR_VISIT or KEEP_ORIGINAL
EVIDENCE_SUFFICIENT_FOR_CORRECTION: YES or NO
DECISIVE_EVIDENCE: exact note evidence that proves the operation, or NONE
DO_NOT_CHANGE: supported parts of the original answer that should be preserved, or NONE
CORRECTION_HINT: one sentence telling the corrector what to change
WHY: one short final rationale"""

DET_META_PLAN_NATURAL = """Question:
{question}

Zero-shot answer:
{answer}

Think naturally about what this question requires before reading the note. Write a concise audit plan in prose. Include the exact required answer slot, key entities, time window or visit focus, and the central claims in the zero-shot answer.

Separate three categories before seeing the note:
1. Must-fix items: facts that would make the answer wrong for the exact question if unsupported or contradicted.
2. Optional detail: extra specificity that may improve style but is not required to answer the question.
3. Do-not-disturb content: parts of the answer that should be preserved unless the note directly contradicts them.

Do not decide correctness yet."""

DET_META_CONFIRM_NATURAL = """Discharge note:
{note}

Question:
{question}

Zero-shot answer to audit:
{answer}

Audit plan made before reading the note:
{plan}

Now audit the answer in natural clinical prose. Compare the answer with the discharge note and the exact question. Your goal is to decide whether a correction is necessary, not whether the answer could be made more polished.

Use this standard:
- Recommend correction only for a central contradiction, wrong visit/time/aspect, or missing fact required by the question.
- Do not recommend correction for optional detail, harmless extra wording, or a supported answer that could merely be more complete.
- Do not remove an extra detail merely because it is not in the focused span. Remove it only if the discharge note contradicts it, it changes the answer, or it answers the wrong focus.
- Distinguish requested answer-slot content from background history/context. Past history, comorbidities, rationale, or ancillary treatment details should not be removed or added unless the question explicitly asks for that exact slot or the detail changes the answer.
- For removal/replacement, quote or name decisive evidence that proves the original claim is wrong or harmful. Silence is not enough.
- For additions, require that the missing fact is necessary to answer the question, not merely helpful context or a more exhaustive answer.
- If evidence is incomplete or the original answer is adequate, say KEEP ORIGINAL.

If correction is necessary, state the required operation, the exact wrong/missing central content, the note-supported target, decisive evidence, and what original content must be preserved. If no correction is necessary, say that clearly and explain why.

Write a concise audit memo in natural prose. Do not use a rigid field template; the parser will extract the structured fields afterward."""

DET_QUESTION_SLOT = """Discharge note:
{note}

Question:
{question}

Answer to audit:
{answer}

First classify what the question is asking for. Pay special attention to exact answer formats:
- NUMBER_OR_VALUE: counts, doses, lab values, vital signs, percentages, durations, scores.
- DATE_OR_TIME: date, time, admission/discharge timing, sequence, before/after.
- MEDICATION_OR_DOSE: drug names, starts/stops, dose, route, frequency.
- LIST_OR_MULTI_PART: multiple treatments, diagnoses, complications, reasons, follow-up items.
- YES_NO_STATUS: presence/absence, resolved/not resolved, performed/not performed.
- CAUSE_REASON: why something happened or rationale.
- PROCEDURE_EVENT: procedures, complications, clinical events, outcomes.

Then compare the answer against that required slot. For number/date/time/list questions, mark INCORRECT if the answer omits, changes, rounds incorrectly, or substitutes the required value/date/list. For list questions, mark INCORRECT only when a required central item is missing or an unsupported central item is added. For contradiction, identify the exact claim and the note-supported replacement.

Return exactly this template:
QUESTION_TYPE: NUMBER_OR_VALUE or DATE_OR_TIME or MEDICATION_OR_DOSE or LIST_OR_MULTI_PART or YES_NO_STATUS or CAUSE_REASON or PROCEDURE_EVENT or OTHER
REQUIRED_ANSWER_FORMAT: what the answer must contain, including number/date/list requirements
QUESTION_FOCUS: one sentence
ANSWER_FOCUS: one sentence
SLOT_CHECK: whether the answer provides the required value/date/list/status/reason/event
VERDICT: CORRECT or INCORRECT
ERROR_TYPE: CONTRADICTION or QUESTION_MISALIGNMENT or OMISSION or NONE
WRONG_CLAIM: exact claim/value/date/list item to repair, or NONE
CORRECT_OR_MISSING_INFO: exact note-supported replacement or required missing fact, or NONE
EVIDENCE_NEEDED: exact note span needed, naming numbers/dates/items if relevant
RETRIEVAL_QUERY_1: short query using the required answer slot
RETRIEVAL_QUERY_2: short query using the wrong/missing value/date/item
RETRIEVAL_QUERY_3: short query using key clinical entities
CORRECTION_HINT: one sentence telling the corrector what to change, preserving required numbers/dates/lists
WHY: one short explanation"""

DET_PROMPTS = {
    'p5_retrieval_payload': DET_P5,
    'contradiction_first': DET_CONTRADICTION_FIRST,
    'claim_contradiction': DET_CLAIM_CONTRADICTION,
    'question_slot': DET_QUESTION_SLOT,
    'claim_slot_conservative': DET_CLAIM_SLOT_CONSERVATIVE,
    'slot_reasoned': DET_SLOT_REASONED,
    'meta_plan_confirm': DET_META_CONFIRM,
    'meta_plan_confirm_natural': DET_META_CONFIRM_NATURAL,
}

PARSE_SYSTEM = "Extract structured fields from a clinical self-audit. Return JSON only."
PARSE_DET_USER = """Extract the detection payload from this text. Use only what the text says; do not re-judge the clinical case.

TEXT:
{raw}

Return JSON:
{{"verdict":"CORRECT|INCORRECT|UNCLEAR", "error_type":"CONTRADICTION|OMISSION|QUESTION_MISALIGNMENT|NONE|UNCLEAR", "question_type":"string", "required_answer_format":"string", "question_focus":"string", "answer_focus":"string", "slot_check":"string", "key_evidence_reason":"string", "full_contradiction":"string", "correction_operation":"string", "evidence_sufficient_for_correction":"string", "decisive_evidence":"string", "do_not_change":"string", "wrong_claim":"string", "correct_or_missing_info":"string", "evidence_needed":"string", "retrieval_queries":["string"], "correction_hint":"string", "why":"string"}}"""

PARSE_DET_HELPER_V2_USER = """Question:
{question}

Zero-shot answer:
{answer}

Natural audit plan:
{plan}

Natural audit memo to parse:
{raw}

Parsing rules:
- You are a faithful interpreter between pipeline stages, not an independent medical judge.
- Preserve correction signals from the tested model's memo, but classify only correction-worthy central issues as INCORRECT.
- If the memo says KEEP ORIGINAL, no change is necessary, answer is adequate, or the issue is optional detail/style, emit verdict CORRECT, error_type NONE, correction_operation KEEP_ORIGINAL.
- If the memo says a fact required by the exact question is missing, emit verdict INCORRECT with error_type OMISSION and operation ADD_MISSING_SLOT. Do not treat nice-to-have background detail, ancillary treatment context, or exhaustive-list detail as a required slot.
- If the memo identifies a wrong value/date/medication/fact that changes the answer, emit verdict INCORRECT with error_type CONTRADICTION and operation REPLACE_VALUE.
- If the memo says the answer addresses the wrong visit, date, time window, or aspect, emit verdict INCORRECT with error_type QUESTION_MISALIGNMENT and operation REFOCUS_TIME_OR_VISIT.
- Use REMOVE_UNSUPPORTED_CLAIM only when the memo says the extra claim is directly contradicted, harmful to the answer, part of the requested answer slot, or changes the answer focus. Do not remove background history/context merely because it is absent from the focused evidence.
- If the memo criticizes past history, comorbidities, rationale, or ancillary treatment details but the question does not require that slot, prefer KEEP_ORIGINAL unless there is direct contradiction.
- Use exactly one correction_operation enum: REPLACE_VALUE, ADD_MISSING_SLOT, REMOVE_UNSUPPORTED_CLAIM, REFOCUS_TIME_OR_VISIT, or KEEP_ORIGINAL.
- evidence_sufficient_for_correction should be YES only when the memo gives decisive note evidence for the exact operation. If the memo says evidence is unclear, incomplete, or silence-only, use NO and KEEP_ORIGINAL unless there is a required omission with direct support.
- Preserve supported original content in do_not_change, especially when the answer is mostly adequate.
- The downstream correction stage relies on wrong_claim, correct_or_missing_info, evidence_needed, decisive_evidence, and correction_hint. Fill them concretely when verdict is INCORRECT.

Return JSON only:
{{
  "verdict":"CORRECT|INCORRECT|UNCLEAR",
  "error_type":"CONTRADICTION|OMISSION|QUESTION_MISALIGNMENT|NONE|UNCLEAR",
  "question_type":"string",
  "required_answer_format":"string",
  "question_focus":"string",
  "answer_focus":"string",
  "slot_check":"supported|full-contradiction|partial-conflict|missing-central-slot|wrong-focus|sufficient|string",
  "key_evidence_reason":"string",
  "full_contradiction":"YES|NO|UNCLEAR",
  "correction_operation":"REPLACE_VALUE|ADD_MISSING_SLOT|REMOVE_UNSUPPORTED_CLAIM|REFOCUS_TIME_OR_VISIT|KEEP_ORIGINAL",
  "evidence_sufficient_for_correction":"YES|NO|UNCLEAR",
  "decisive_evidence":"string or NONE",
  "do_not_change":"string or NONE",
  "wrong_claim":"string or NONE",
  "correct_or_missing_info":"string or NONE",
  "evidence_needed":"string or NONE",
  "retrieval_queries":["string"],
  "correction_hint":"string or NONE",
  "why":"string"
}}"""

COR_SYSTEM = "You are a careful clinical QA assistant. Revise when same-patient evidence and detection feedback support the revision."

COR_PROMPTS: dict[str, str] = {
    'balanced': """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Detection feedback:
- error type: {error_type}
- question focus: {question_focus}
- wrong claim: {wrong_claim}
- correction target: {correct_or_missing_info}
- correction hint: {correction_hint}

Same-patient retrieved evidence. This is the factual source:
{spans_block}

Retrieved correction example from another patient. Use as style/pattern only, not factual content:
{example_block}

Write the best final answer to the question in 1-3 sentences. Use only facts supported by the discharge note and evidence spans.""",
    'accept_suggestion_if_supported': """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

The detection step found a likely problem:
- error type: {error_type}
- wrong or missing part: {wrong_claim}
- suggested correction target: {correct_or_missing_info}
- correction hint: {correction_hint}

Same-patient evidence:
{spans_block}

Correction example from another patient, for pattern only:
{example_block}

If the same-patient evidence supports the suggested correction target, apply it. Do not preserve the previous answer just because it is fluent. Keep only previous-answer content that is supported and still answers the question. Return only the corrected final answer.""",
    'direct_rewrite_from_feedback': """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Detected issue:
- type: {error_type}
- question focus: {question_focus}
- problematic claim: {wrong_claim}
- target fact: {correct_or_missing_info}
- hint: {correction_hint}

Evidence spans:
{spans_block}

Rewrite the answer around the target fact and the question focus. Prefer a clear corrected answer over a minimal edit when the previous answer is misleading. Do not include unsupported facts. Return only the final answer.""",
    'contradiction_repair': """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Detection feedback says the main risk is a contradicted or unsupported claim:
- wrong claim: {wrong_claim}
- note-supported replacement: {correct_or_missing_info}
- hint: {correction_hint}

Evidence spans:
{spans_block}

Replace the contradicted claim with the note-supported fact. Remove any part of the previous answer that conflicts with the evidence. Keep supported parts. Return only the final answer.""",
    'omission_repair': """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Detection feedback says the answer may be missing the required answer slot:
- question focus: {question_focus}
- missing target: {correct_or_missing_info}
- hint: {correction_hint}

Evidence spans:
{spans_block}

Add the required missing answer fact if it is present in the evidence. Keep the answer focused on the question. Return only the final answer.""",
    'operation_guided': """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Confirmed audit result:
- error type: {error_type}
- question type: {question_type}
- required answer format: {required_answer_format}
- full contradiction: {full_contradiction}
- correction operation: {correction_operation}
- evidence sufficient for correction: {evidence_sufficient_for_correction}
- decisive evidence: {decisive_evidence}
- wrong or missing part: {wrong_claim}
- target fact/value/date/list item: {correct_or_missing_info}
- do not change: {do_not_change}
- correction hint: {correction_hint}

Same-patient retrieved evidence:
{spans_block}

If evidence sufficient for correction is NO or the correction operation is KEEP_ORIGINAL, return the previous answer exactly. Otherwise perform only the named correction operation. Preserve supported parts listed in DO_NOT_CHANGE. Do not add facts beyond the decisive evidence and retrieved same-patient evidence. Return only the final answer.""",
    'question_slot_repair': """Discharge note:
{note}

Question:
{question}

Previous answer:
{answer}

Question-slot analysis from detection:
- question type: {question_type}
- required answer format: {required_answer_format}
- question focus: {question_focus}
- answer focus: {answer_focus}
- slot check: {slot_check}
- wrong or missing part: {wrong_claim}
- target fact/value/date/list item: {correct_or_missing_info}
- correction hint: {correction_hint}

Same-patient evidence:
{spans_block}

Correction example from another patient, for pattern only:
{example_block}

Write the final answer so it satisfies the required answer format. Preserve exact numbers, dates, time periods, medication names, doses, frequencies, and list items when the question asks for them. If the previous answer has the wrong value/date/item, replace it with the evidence-supported one. If a required central list item is missing, add it. Do not add background details that are not needed by the question. Return only the final answer.""",
}

VERDICT_SYSTEM = "You are a strict medical expert comparing two answers against the discharge note and question."
VERDICT_PROMPTS: dict[str, str] = {
    'balanced': """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Pick the answer that better answers the question using the discharge summary.
Consider: consistency with notes, completeness for all parts of the question, and directness.
If the corrected answer introduces unsupported facts or drops necessary information, prefer the original.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, give one short reason.""",
    'false_correction_sensitive': """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Compare both answers against the discharge summary and the exact question. A correction should be accepted only if it is clearly more faithful to the note and better answers the question. Reject a correction if it adds unsupported facts, changes a supported answer unnecessarily, drops a required detail, or answers a different focus.

Choose the safer answer for final reporting.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, give one short reason.""",
    'derive_then_compare': """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

First determine, privately, the answer that is supported by the discharge summary for this question. Then choose whether Answer A or Answer B is closer to that note-supported answer. Penalize unsupported additions and wrong-focus answers. Prefer the original if both are equivalent or the correction is not clearly better.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, give one short reason.""",
    'contradiction_count': """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count material contradictions with the discharge summary in each answer. Also check whether each answer directly answers the question. Pick the answer with fewer material contradictions. If contradiction counts tie, pick the answer that better covers the requested answer slot. If still tied, prefer the original answer.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, give one short reason.""",
    'slot_sensitive': """Discharge summary:
{note}

Question:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Identify what exact answer slot the question requires before choosing. Pay attention to whether the question asks for a number/value, date/time, medication/dose/frequency, multi-item list, yes/no status, cause/reason, procedure/event, or outcome. Choose the answer that best preserves the required slot from the discharge summary. Penalize wrong or missing numbers, dates, doses, time periods, central list items, and unsupported substitutions. If both answers satisfy the slot equally, prefer the original answer.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, give one short reason naming the slot.""",
    'detection_aware_direct': """Discharge summary:
{note}

Question:
{question}

Detection feedback from the previous step:
- error type: {error_type}
- question type: {question_type}
- required answer format: {required_answer_format}
- question focus: {question_focus}
- answer focus: {answer_focus}
- detected wrong or missing part: {wrong_claim}
- target fact/value/date/list item: {correct_or_missing_info}
- correction hint: {correction_hint}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Choose the better final answer. Use three gates:
1. Did the candidate remove the detected error or missing answer slot?
2. Did the candidate avoid introducing any new contradiction or unsupported fact?
3. Does the candidate completely answer the exact question and required answer format?

Accept a correction only if it passes all three gates and is clearly better than the original. If both answers are equivalent, or the correction fixes one issue but introduces another, prefer the original.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, state the decisive gate result in one short sentence.""",
    'detection_aware_cot': """Discharge summary:
{note}

Question:
{question}

Detection feedback from the previous step:
- error type: {error_type}
- question type: {question_type}
- required answer format: {required_answer_format}
- question focus: {question_focus}
- answer focus: {answer_focus}
- detected wrong or missing part: {wrong_claim}
- target fact/value/date/list item: {correct_or_missing_info}
- correction hint: {correction_hint}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Privately check each answer against the discharge summary and the detection feedback:
- whether the detected error remains, is removed, or was not real;
- whether any new unsupported or contradicted fact appears;
- whether the exact question is fully answered, including required numbers, dates, medications, list items, and time focus.

Then choose the safer final answer. Prefer the original unless the correction is clearly more faithful and complete.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, give one concise reason without showing the private checklist.""",
    'detection_aware_scorecard': """Discharge summary:
{note}

Question:
{question}

Detection feedback from the previous step:
- error type: {error_type}
- question type: {question_type}
- required answer format: {required_answer_format}
- question focus: {question_focus}
- answer focus: {answer_focus}
- detected wrong or missing part: {wrong_claim}
- target fact/value/date/list item: {correct_or_missing_info}
- correction hint: {correction_hint}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Score each answer on four dimensions using the discharge summary:
- ERROR_REMOVED: whether the detected problem is absent.
- NO_NEW_CONTRADICTION: whether the answer avoids unsupported or contradicted facts.
- QUESTION_COMPLETE: whether the answer fully covers the exact question and required format.
- PRESERVES_SUPPORTED_CONTENT: whether supported useful content was not unnecessarily dropped.

Pick the answer with the stronger overall score. A correction must improve ERROR_REMOVED without failing NO_NEW_CONTRADICTION or QUESTION_COMPLETE. If scores are tied, prefer the original.

Reply on the FIRST line with exactly one letter: A or B
On the SECOND line, name the decisive dimension.""",
    'multi_dimension': """Three-prompt verdict: separately checks detected-error removal, new contradiction, and question completeness before choosing original or corrected answer.""",
}
PARSE_VERDICT_USER = """The text below was supposed to pick answer A or B. Extract the pick. If unclear, return UNCLEAR.

TEXT:
{raw}

Return JSON: {{"pick":"A|B|UNCLEAR", "reason":"string"}}"""

CONTEXT_SUMMARY_SYSTEM = "You summarize discharge-note evidence for a clinical QA audit. Keep exact numbers, dates, medications, doses, and list items."
CONTEXT_SUMMARY_USER = """Question:
{question}

Zero-shot answer:
{answer}

Audit plan or detection focus:
{plan}

Retrieved source spans from the same discharge note:
{spans}

Summarize only the evidence relevant to answering the question and checking the zero-shot answer. Preserve exact values, dates, doses, medication names, and list items. Do not add medical knowledge. End with a short line naming any evidence that is missing or not found.

Return:
FOCUSED_NOTE_SUMMARY: ...
KEY_EVIDENCE:
- ...
MISSING_OR_NOT_FOUND: ..."""

JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

def judge_user(note: str, question: str, ground_truth: str, answer: str) -> str:
    return f"DISCHARGE SUMMARY:\n{note}\n\nQUESTION:\n{question}\n\nCORRECT ANSWER (Ground Truth):\n{ground_truth}\n\nMODEL'S ANSWER:\n{answer}\n\nTask: Evaluate if the model's answer is correct compared to the ground truth.\n\nRespond with ONLY a single digit:\n1 = Correct\n0 = Incorrect"


def load_api_key() -> str:
    for env in (PROJECT_ROOT/'.env', SOURCE_REPO/'.env'):
        if env.exists():
            for line in env.read_text().splitlines():
                line=line.strip()
                if line.startswith('OPENAI_API_KEY=') and not line.startswith('#'):
                    return line.split('=',1)[1].strip()
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    raise RuntimeError('OPENAI_API_KEY not found')

_client: OpenAI|None=None
def openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client=OpenAI(api_key=load_api_key())
    return _client


def served_model_id(port:int)->str:
    r=requests.get(f'http://localhost:{port}/v1/models',timeout=10); r.raise_for_status(); return r.json()['data'][0]['id']

def strip_think(text:str)->str:
    text=str(text or '')
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S|re.I).strip()
    if '</think>' in text.lower(): text=re.sub(r'^.*?</think>\s*','',text,flags=re.S|re.I).strip()
    if '<think>' in text.lower(): text=re.sub(r'<think>[\s\S]*$','',text,flags=re.I).strip()
    return text

def clean_answer(text:str)->str:
    return strip_think(text).strip()

def build_llama2_prompt(system:str,user:str)->str:
    return f'[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]'

def build_llama3_prompt(system:str,user:str)->str:
    return (
        '<|begin_of_text|>'
        '<|start_header_id|>system<|end_header_id|>\n\n'
        f'{system}<|eot_id|>'
        '<|start_header_id|>user<|end_header_id|>\n\n'
        f'{user}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    )

def build_chatml_prompt(system:str,user:str)->str:
    return (
        f'<|im_start|>system\n{system}<|im_end|>\n'
        f'<|im_start|>user\n{user}<|im_end|>\n'
        '<|im_start|>assistant\n'
    )

def build_qwen_prompt(system:str,user:str)->str:
    qwen_system = system + "\n\nDo not produce hidden reasoning or a thinking section. Give only the requested answer."
    qwen_user = user + "\n\n/no_think"
    return build_chatml_prompt(qwen_system, qwen_user)

def prompt_for_served_model(model:str, system:str, user:str)->str:
    m=model.lower()
    if 'biomistral' in m or 'mistral' in m:
        return build_llama2_prompt(system,user)
    if 'llama' in m or 'deepseek' in m:
        return build_llama3_prompt(system,user)
    if 'qwen' in m:
        return build_qwen_prompt(system,user)
    return build_chatml_prompt(system,user)

def vllm_chat(system:str,user:str,port:int,max_tokens:int,temperature:float)->str:
    model=served_model_id(port)
    prompt=prompt_for_served_model(model,system,user)
    payload={'model':model,'prompt':prompt,'max_tokens':max_tokens,'temperature':temperature}
    r=requests.post(f'http://localhost:{port}/v1/completions',json=payload,timeout=300)
    body=r.json()
    if 'choices' not in body: raise RuntimeError(str(body))
    out=strip_think((body['choices'][0].get('text') or '').strip())
    log_call('vllm',model,system,user,out,temperature=temperature,max_tokens=max_tokens)
    return out


def parse_binary(text:str|None)->int|None:
    if text is None: return None
    if '1' in text and '0' not in text: return 1
    if '0' in text: return 0
    return None


def gpt_json(user:str, max_tokens:int=300)->dict[str,Any]:
    for attempt in range(4):
        try:
            r=openai_client().chat.completions.create(model='gpt-4o-mini',messages=[{'role':'system','content':PARSE_SYSTEM},{'role':'user','content':user}],temperature=0.0,max_tokens=max_tokens,response_format={'type':'json_object'})
            txt=(r.choices[0].message.content or '{}').strip()
            log_call('parser',  'gpt-4o-mini', PARSE_SYSTEM, user, txt, max_tokens=max_tokens)
            return json.loads(txt)
        except Exception as e:
            if attempt==3: return {'error':str(e)}
            time.sleep(2*(attempt+1))
    return {}


def qwen_json(system:str,user:str,port:int,max_tokens:int=500,temperature:float=0.0)->dict[str,Any]:
    raw=vllm_chat(system,user,port,max_tokens,temperature)
    m=re.search(r'\{[\s\S]*\}', raw or '')
    if not m:
        return {'error':'no_json','raw':raw}
    try:
        return json.loads(m.group())
    except Exception as e:
        return {'error':str(e),'raw':raw}


def parse_detection_with_backend(raw:str, backend:str, port:int, row:dict[str,Any]|None=None, plan:str='')->dict[str,Any]:
    if backend == 'regex_then_gpt4o-mini':
        return parse_detection(raw)
    if backend == 'gpt4o-mini':
        obj=gpt_json(PARSE_DET_USER.format(raw=(raw or '')[:5000]))
    elif backend == 'gpt4o-mini-helper-v2':
        if row is None:
            raise ValueError('gpt4o-mini-helper-v2 requires row context')
        obj=gpt_json(PARSE_DET_HELPER_V2_USER.format(question=row['question'], answer=row['answer'][:2000], plan=plan[:3500], raw=(raw or '')[:7000]), max_tokens=650)
    elif backend == 'qwen2.5':
        obj=qwen_json(PARSE_SYSTEM, PARSE_DET_USER.format(raw=(raw or '')[:5000]), port, max_tokens=500, temperature=0.0)
    else:
        raise ValueError(f'unknown detection parser backend: {backend}')
    out={'verdict':str(obj.get('verdict','UNCLEAR')).upper(),'error_type':str(obj.get('error_type','UNCLEAR')).upper(),'question_type':str(obj.get('question_type','')),'required_answer_format':str(obj.get('required_answer_format','')),'question_focus':str(obj.get('question_focus','')),'answer_focus':str(obj.get('answer_focus','')),'slot_check':str(obj.get('slot_check','')),'key_evidence_reason':str(obj.get('key_evidence_reason','')),'full_contradiction':str(obj.get('full_contradiction','')),'correction_operation':str(obj.get('correction_operation','')),'evidence_sufficient_for_correction':str(obj.get('evidence_sufficient_for_correction','')),'decisive_evidence':str(obj.get('decisive_evidence','')),'do_not_change':str(obj.get('do_not_change','')),'wrong_claim':str(obj.get('wrong_claim','')),'correct_or_missing_info':str(obj.get('correct_or_missing_info','')),'evidence_needed':str(obj.get('evidence_needed','')),'retrieval_queries':obj.get('retrieval_queries',[]) if isinstance(obj.get('retrieval_queries',[]),list) else [],'correction_hint':str(obj.get('correction_hint','')),'why':str(obj.get('why','')),'parse_path':backend}
    if valid_detection(out):
        return out
    out['verdict']='UNCLEAR'; out['valid']=False
    return out


def gpt_judge(note:str,question:str,gt:str,answer:str)->dict[str,Any]:
    for attempt in range(5):
        try:
            ju=judge_user(note,question,gt,answer)
            r=openai_client().chat.completions.create(model='gpt-4o',messages=[{'role':'system','content':JUDGE_SYSTEM},{'role':'user','content':ju}],temperature=0.1,max_tokens=10)
            raw=(r.choices[0].message.content or '').strip()
            log_call('judge','gpt-4o',JUDGE_SYSTEM,ju,raw,temperature=0.1)
            return {'label':parse_binary(raw),'raw':raw,'temperature':0.1,'model':'gpt-4o'}
        except Exception as e:
            if attempt==4: return {'label':None,'raw':'','error':str(e)}
            time.sleep(2*(attempt+1))
    return {'label':None}


def field(text:str,name:str)->str:
    m=re.search(rf'^\s*\*?\*?{re.escape(name)}\*?\*?\s*:\s*(.+)$',text or '',re.I|re.M)
    return m.group(1).strip() if m else ''

def parse_detection_regex(raw:str)->dict[str,Any]:
    verdict_s=field(raw,'VERDICT').upper()
    verdict='INCORRECT' if 'INCORRECT' in verdict_s else ('CORRECT' if 'CORRECT' in verdict_s else 'UNCLEAR')
    et_s=field(raw,'ERROR_TYPE').upper(); et='UNCLEAR'
    for t in ['QUESTION_MISALIGNMENT','CONTRADICTION','OMISSION','NONE']:
        if t in et_s: et=t; break
    qs=[field(raw,f'RETRIEVAL_QUERY_{i}') for i in (1,2,3)]
    return {'verdict':verdict,'error_type':et,'question_type':field(raw,'QUESTION_TYPE'),'required_answer_format':field(raw,'REQUIRED_ANSWER_FORMAT'),'question_focus':field(raw,'QUESTION_FOCUS'),'answer_focus':field(raw,'ANSWER_FOCUS'),'slot_check':field(raw,'SLOT_CHECK'),'key_evidence_reason':field(raw,'KEY_EVIDENCE_REASON'),'full_contradiction':field(raw,'FULL_CONTRADICTION'),'correction_operation':field(raw,'CORRECTION_OPERATION'),'evidence_sufficient_for_correction':field(raw,'EVIDENCE_SUFFICIENT_FOR_CORRECTION'),'decisive_evidence':field(raw,'DECISIVE_EVIDENCE'),'do_not_change':field(raw,'DO_NOT_CHANGE'),'wrong_claim':field(raw,'WRONG_CLAIM'),'correct_or_missing_info':field(raw,'CORRECT_OR_MISSING_INFO'),'evidence_needed':field(raw,'EVIDENCE_NEEDED'),'retrieval_queries':[q for q in qs if q and q.upper()!='NONE'],'correction_hint':field(raw,'CORRECTION_HINT'),'why':field(raw,'WHY'),'parse_path':'regex'}


def valid_detection(p:dict[str,Any])->bool:
    if p.get('verdict')=='CORRECT': return True
    if p.get('verdict')!='INCORRECT': return False
    if p.get('error_type') not in {'CONTRADICTION','OMISSION','QUESTION_MISALIGNMENT'}: return False
    if not (p.get('wrong_claim') or p.get('correct_or_missing_info')): return False
    if not (p.get('retrieval_queries') or p.get('evidence_needed') or p.get('question_focus')): return False
    return True


def parse_detection(raw:str)->dict[str,Any]:
    p=parse_detection_regex(raw)
    if valid_detection(p): return p
    obj=gpt_json(PARSE_DET_USER.format(raw=(raw or '')[:5000]))
    out={'verdict':str(obj.get('verdict','UNCLEAR')).upper(),'error_type':str(obj.get('error_type','UNCLEAR')).upper(),'question_type':str(obj.get('question_type','')),'required_answer_format':str(obj.get('required_answer_format','')),'question_focus':str(obj.get('question_focus','')),'answer_focus':str(obj.get('answer_focus','')),'slot_check':str(obj.get('slot_check','')),'key_evidence_reason':str(obj.get('key_evidence_reason','')),'full_contradiction':str(obj.get('full_contradiction','')),'correction_operation':str(obj.get('correction_operation','')),'evidence_sufficient_for_correction':str(obj.get('evidence_sufficient_for_correction','')),'decisive_evidence':str(obj.get('decisive_evidence','')),'do_not_change':str(obj.get('do_not_change','')),'wrong_claim':str(obj.get('wrong_claim','')),'correct_or_missing_info':str(obj.get('correct_or_missing_info','')),'evidence_needed':str(obj.get('evidence_needed','')),'retrieval_queries':obj.get('retrieval_queries',[]) if isinstance(obj.get('retrieval_queries',[]),list) else [],'correction_hint':str(obj.get('correction_hint','')),'why':str(obj.get('why','')),'parse_path':'gpt4o-mini'}
    if valid_detection(out): return out
    out['verdict']='UNCLEAR'; out['valid']=False
    return out


def parse_verdict(raw:str)->dict[str,Any]:
    first=(raw or '').strip().splitlines()[0] if (raw or '').strip() else ''
    m=re.match(r'^[\s\-*#>]*([AaBb])\b',first)
    if m: return {'pick':m.group(1).upper(),'parse_path':'regex','reason':' '.join((raw or '').splitlines()[1:])[:200]}
    obj=gpt_json(PARSE_VERDICT_USER.format(raw=(raw or '')[:2000]),max_tokens=120)
    pick=str(obj.get('pick','UNCLEAR')).upper()
    if pick not in {'A','B'}: pick='UNCLEAR'
    return {'pick':pick,'parse_path':'gpt4o-mini','reason':str(obj.get('reason',''))[:200]}


def load_notes()->dict[str,str]:
    df=pd.read_json(SOURCE_REPO/'output'/'EHRNoteQA_processed.jsonl',lines=True); out={}
    for _,r in df.iterrows():
        parts=[]
        for i in (1,2,3):
            t=r.get(f'note_{i}')
            if pd.notna(t) and str(t).strip() and str(t).strip().lower()!='nan': parts.append(f'[Note {i}]\n{str(t).strip()}')
        out[str(int(r['patient_id']))]='\n\n'.join(parts)
    return out


def load_rows(n_wrong:int,n_correct:int,seed:int,input_model:str)->list[dict[str,Any]]:
    notes=load_notes(); rows=[]
    for fold in range(5):
        df=pd.read_csv(SOURCE_REPO/'output'/'step8'/input_model/f'fold_{fold}'/'zeroshot_evaluated_binary.csv')
        for _,r in df.iterrows():
            pid=int(r['patient_id']); answer_raw=str(r['model_answer']); rows.append({'fold':fold,'idx':int(r['idx']),'patient_id':pid,'question':str(r['question']),'ground_truth':str(r['ground_truth']),'answer':clean_answer(answer_raw),'answer_raw':answer_raw,'orig_label':int(r['binary_correct']),'note':notes[str(pid)]})
    wrong=[r for r in rows if r['orig_label']==0]; correct=[r for r in rows if r['orig_label']==1]
    rng=random.Random(seed); rng.shuffle(wrong); rng.shuffle(correct)
    if n_wrong<0: n_wrong=len(wrong)
    if n_correct<0: n_correct=len(correct)
    sample=wrong[:min(n_wrong,len(wrong))]+correct[:min(n_correct,len(correct))]; rng.shuffle(sample); return sample

_pool_cache={}
def load_pool(fold:int)->list[dict[str,Any]]:
    if fold not in _pool_cache:
        p=SOURCE_REPO/'workspace'/'self_critique'/'data'/'bm_contrast_pool'/f'fold_{fold}_pool.json'
        _pool_cache[fold]=json.loads(p.read_text()) if p.exists() else []
    return _pool_cache[fold]

def toks(s:str)->set[str]: return set(re.findall(r'[a-zA-Z0-9]+',(s or '').lower()))
def retrieve_example(row:dict[str,Any], det:dict[str,Any])->dict[str,Any]|None:
    pool=load_pool(row['fold']);
    if not pool: return None
    query=' '.join([row['question'],det.get('error_type',''),det.get('question_focus',''),det.get('wrong_claim',''),det.get('correct_or_missing_info','')])
    qt=toks(query)
    def score(ex):
        text=' '.join([ex.get('question',''),ex.get('what_was_wrong',''),ex.get('ground_truth','')])
        return len(qt & toks(text))
    return max(pool,key=score)

def retrieve_spans(row:dict[str,Any], det:dict[str,Any], k:int)->list[dict[str,Any]]:
    queries=[row['question'],det.get('question_type',''),det.get('required_answer_format',''),det.get('question_focus',''),det.get('slot_check',''),det.get('key_evidence_reason',''),det.get('wrong_claim',''),det.get('correct_or_missing_info',''),det.get('evidence_needed','')]+list(det.get('retrieval_queries') or [])
    return topk_spans(row['note'],queries,k=k,scoring='agreement')

def render_spans(spans:list[dict[str,Any]])->str:
    return '\n'.join(f'[{i+1}] {s["sentence"]}' for i,s in enumerate(spans)) if spans else '(none)'

def render_example(ex:dict[str,Any]|None)->str:
    if not ex: return '(none)'
    ev='; '.join(ex.get('evidence_from_notes') or [])[:600]
    return f"Question: {ex.get('question','')}\nWrong answer: {ex.get('wrong_answer','')}\nWhat was wrong: {ex.get('what_was_wrong','')}\nCorrect answer pattern: {ex.get('ground_truth','')}\nEvidence style: {ev}"


def base_note_context(row:dict[str,Any], limit:int=18000)->dict[str,Any]:
    note=row['note']
    return {'text': note[:limit], 'mode':'first18k', 'note_chars':len(note), 'truncated':len(note)>limit, 'spans':[], 'summary_raw':''}


def focused_context_queries(row:dict[str,Any], plan:str='')->list[str]:
    return [row['question'], row['answer'][:800], plan[:1200]]


def build_note_context(row:dict[str,Any], args, port:int, plan:str='')->dict[str,Any]:
    note=row['note']
    if args.note_context == 'first18k' or len(note) <= args.context_threshold:
        ctx=base_note_context(row)
        ctx['mode']=args.note_context if args.note_context != 'first18k' else 'first18k'
        return ctx
    queries=focused_context_queries(row, plan)
    spans=topk_spans(note, queries, k=args.context_k, scoring='agreement')
    span_text=render_spans(spans)
    header=(
        f"QUESTION-FOCUSED NOTE CONTEXT (source note length={len(note)} chars; "
        f"using top {len(spans)} retrieved spans because full note exceeds context threshold).\n"
        "Use this as same-patient discharge-note evidence. Exact values/dates/medications in spans are source evidence.\n\n"
    )
    if args.note_context == 'dynamic_spans':
        return {'text': header + span_text, 'mode':'dynamic_spans', 'note_chars':len(note), 'truncated':True, 'spans':spans, 'summary_raw':''}
    if args.note_context == 'dynamic_summary':
        summary=vllm_chat(
            CONTEXT_SUMMARY_SYSTEM,
            CONTEXT_SUMMARY_USER.format(question=row['question'], answer=row['answer'][:1200], plan=plan[:2500] or '(none)', spans=span_text[:6000]),
            port,
            700,
            args.context_summary_temperature,
        )
        text=header + summary + "\n\nSOURCE SPANS:\n" + span_text
        return {'text':text, 'mode':'dynamic_summary', 'note_chars':len(note), 'truncated':True, 'spans':spans, 'summary_raw':summary}
    raise ValueError(f"unknown note context mode: {args.note_context}")


def _detect_temp(args)->float:
    return args.detect_temperature_multirun if args.detect_k>1 else args.det_temperature

def _detect_sample(row,port,args,prompt_id,note_context,temp)->dict[str,Any]:
    if prompt_id in {'meta_plan_confirm','meta_plan_confirm_natural'}:
        plan_template = DET_META_PLAN_NATURAL if prompt_id == 'meta_plan_confirm_natural' else DET_META_PLAN
        confirm_template = DET_META_CONFIRM_NATURAL if prompt_id == 'meta_plan_confirm_natural' else DET_META_CONFIRM
        plan_raw = vllm_chat(DET_SYSTEM, plan_template.format(question=row['question'], answer=row['answer'][:2000]), port, 700, temp)
        if note_context is None:
            note_context=build_note_context(row,args,port,plan_raw)
        raw = vllm_chat(DET_SYSTEM, confirm_template.format(note=note_context['text'], question=row['question'], answer=row['answer'][:2000], plan=plan_raw[:3500]), port, 1200, temp)
        parsed=parse_detection_with_backend(raw,args.det_parse_backend,args.port,row,plan_raw); parsed['valid']=valid_detection(parsed)
        return {'raw':raw,'plan_raw':plan_raw,'parsed':parsed,'temperature':temp,'note_context':note_context}
    if note_context is None:
        note_context=build_note_context(row,args,port)
    template=DET_PROMPTS[prompt_id]
    raw=vllm_chat(DET_SYSTEM,template.format(note=note_context['text'],question=row['question'],answer=row['answer'][:2000]),port,1000,temp)
    parsed=parse_detection_with_backend(raw,args.det_parse_backend,args.port,row); parsed['valid']=valid_detection(parsed)
    return {'raw':raw,'parsed':parsed,'temperature':temp,'note_context':note_context}

def run_detect(row,port,args)->dict[str,Any]:
    prompt_id=args.det_prompt
    K=max(1,args.detect_k)
    temp=_detect_temp(args)
    if K==1:
        s=_detect_sample(row,port,args,prompt_id,None,temp)
        return {'raw':s['raw'],'plan_raw':s.get('plan_raw',''),'parsed':s['parsed'],'prompt':prompt_id,'temperature':temp,'note_context':s['note_context'],'parse_backend':args.det_parse_backend,'k':1,'vote':None,'samples':None}
    # K>1: build note_context once from a single plan, then sample the confirm step K times
    note_context=None
    samples=[]
    for i in range(K):
        s=_detect_sample(row,port,args,prompt_id,note_context,temp)
        if note_context is None:
            note_context=s['note_context']
        samples.append(s)
    incorrect_idx=[i for i,s in enumerate(samples) if s['parsed'].get('verdict')=='INCORRECT' and s['parsed'].get('valid')]
    vote='INCORRECT' if len(incorrect_idx)>K/2 else 'CORRECT'
    if vote=='INCORRECT':
        chosen=samples[incorrect_idx[0]]
        parsed=chosen['parsed']
    else:
        chosen=samples[0]
        parsed=dict(chosen['parsed']); parsed['verdict']='CORRECT'; parsed['valid']=False
    return {'raw':chosen['raw'],'plan_raw':chosen.get('plan_raw',''),'parsed':parsed,'prompt':prompt_id,'temperature':temp,'note_context':note_context,'parse_backend':args.det_parse_backend,'k':K,'vote':{'incorrect_count':len(incorrect_idx),'samples':[{'verdict':s['parsed'].get('verdict'),'valid':s['parsed'].get('valid')} for s in samples]},'samples':[{'raw':s['raw'],'plan_raw':s.get('plan_raw',''),'parsed':s['parsed']} for s in samples]}

def _correction_temp(args)->float:
    return args.correction_temperature_multirun if args.gen_k>1 else args.correction_temperature

def _correction_sample(row,note_context,det,spans,example,port,temp,prompt_id)->str:
    template=COR_PROMPTS[prompt_id]
    user=template.format(note=note_context,question=row['question'],answer=row['answer'][:1800],error_type=det.get('error_type',''),question_type=det.get('question_type',''),required_answer_format=det.get('required_answer_format',''),question_focus=det.get('question_focus',''),answer_focus=det.get('answer_focus',''),slot_check=det.get('slot_check',''),full_contradiction=det.get('full_contradiction',''),correction_operation=det.get('correction_operation',''),evidence_sufficient_for_correction=det.get('evidence_sufficient_for_correction',''),decisive_evidence=det.get('decisive_evidence',''),do_not_change=det.get('do_not_change',''),wrong_claim=det.get('wrong_claim',''),correct_or_missing_info=det.get('correct_or_missing_info',''),correction_hint=det.get('correction_hint',''),spans_block=render_spans(spans),example_block=render_example(example))
    return vllm_chat(COR_SYSTEM,user,port,700,temp)

def run_correction(row,note_context,det,spans,example,port,temp,prompt_id,gen_k:int=1)->dict[str,Any]:
    K=max(1,gen_k)
    if K==1:
        ans=_correction_sample(row,note_context,det,spans,example,port,temp,prompt_id)
        return {'answer':ans,'candidates':[ans],'temperature':temp,'prompt':prompt_id,'raicl_example':example,'spans':spans,'gen_k':1}
    cands=[_correction_sample(row,note_context,det,spans,example,port,temp,prompt_id) for _ in range(K)]
    # The first candidate is the default; the verdict stage will pick the best when gen_k>1.
    return {'answer':cands[0],'candidates':cands,'temperature':temp,'prompt':prompt_id,'raicl_example':example,'spans':spans,'gen_k':K}

def verdict_feedback(det:dict[str,Any])->dict[str,str]:
    return {
        'error_type': det.get('error_type',''),
        'question_type': det.get('question_type',''),
        'required_answer_format': det.get('required_answer_format',''),
        'question_focus': det.get('question_focus',''),
        'answer_focus': det.get('answer_focus',''),
        'wrong_claim': det.get('wrong_claim',''),
        'correct_or_missing_info': det.get('correct_or_missing_info',''),
        'correction_hint': det.get('correction_hint',''),
    }


def yes_no_from_text(text:str)->str:
    first=(text or '').strip().splitlines()[0].upper() if (text or '').strip() else ''
    if re.match(r'^[\s\-*#>]*(YES|Y)\b', first): return 'YES'
    if re.match(r'^[\s\-*#>]*(NO|N)\b', first): return 'NO'
    if 'YES' in first and 'NO' not in first: return 'YES'
    if 'NO' in first: return 'NO'
    return 'UNCLEAR'


def multi_dimension_verdict_sample(row,note_context,orig_answer,corr_answer,det,port,temp)->dict[str,Any]:
    fb=verdict_feedback(det)
    common=(
        f"Discharge summary:\n{note_context}\n\n"
        f"Question:\n{row['question']}\n\n"
        "Detection feedback from the previous step:\n"
        f"- error type: {fb['error_type']}\n"
        f"- question type: {fb['question_type']}\n"
        f"- required answer format: {fb['required_answer_format']}\n"
        f"- question focus: {fb['question_focus']}\n"
        f"- detected wrong or missing part: {fb['wrong_claim']}\n"
        f"- target fact/value/date/list item: {fb['correct_or_missing_info']}\n"
        f"- correction hint: {fb['correction_hint']}\n\n"
        f"Original answer:\n{orig_answer[:1500]}\n\n"
        f"Corrected answer:\n{corr_answer[:1500]}\n\n"
    )
    checks=[
        ('error_removed', 'Does the corrected answer remove the detected error or missing required slot, compared with the original? Reply first line YES or NO.'),
        ('no_new_contradiction', 'Does the corrected answer avoid introducing any new contradiction, unsupported fact, or wrong focus? Reply first line YES or NO.'),
        ('question_complete', 'Does the corrected answer fully answer the exact question and required answer format at least as well as the original? Reply first line YES or NO.'),
    ]
    results={}
    raw_parts=[]
    for name, prompt in checks:
        raw=vllm_chat(VERDICT_SYSTEM, common+prompt, port, 220, temp)
        val=yes_no_from_text(raw)
        results[name]=val
        raw_parts.append(f"[{name}] {raw}")
    accept=(results.get('error_removed')=='YES' and results.get('no_new_contradiction')=='YES' and results.get('question_complete')=='YES')
    pick='CORRECTED' if accept else 'ORIGINAL'
    return {'raw':'\n\n'.join(raw_parts),'pick':pick,'checks':results,'parse_path':'multi_dimension'}


def _verdict_one_candidate(row,note_context,corr_answer,det,port,k,temp,prompt_id)->dict[str,Any]:
    rng=random.Random(42+(row['fold']<<16)+row['idx']); orig_a=rng.random()>0.5
    ans_a=row['answer'] if orig_a else corr_answer; ans_b=corr_answer if orig_a else row['answer']
    samples=[]
    for _ in range(k):
        if prompt_id == 'multi_dimension':
            multi=multi_dimension_verdict_sample(row,note_context,row['answer'],corr_answer,det,port,temp)
            pick=('A' if orig_a else 'B') if multi['pick']=='ORIGINAL' else ('B' if orig_a else 'A')
            samples.append({**multi,'pick':pick})
            continue
        raw=vllm_chat(VERDICT_SYSTEM,VERDICT_PROMPTS[prompt_id].format(note=note_context,question=row['question'],answer_a=ans_a[:1500],answer_b=ans_b[:1500],**verdict_feedback(det)),port,320,temp)
        parsed=parse_verdict(raw); samples.append({'raw':raw,**parsed})
    counts=Counter(s['pick'] for s in samples); corrected_slot='B' if orig_a else 'A'; corrected_votes=counts.get(corrected_slot,0)
    a,b=counts.get('A',0),counts.get('B',0)
    majority='TIE' if a==b else ('A' if a>b else 'B')
    accept=(majority==corrected_slot and corrected_votes>k/2)
    return {'variant':prompt_id,'k':k,'temperature':temp,'orig_in_slot_A':orig_a,'corrected_slot':corrected_slot,'votes':dict(counts),'majority_pick':majority,'accept_correction':accept,'corrected_votes':corrected_votes,'samples':samples}

def run_verdict(row,note_context,corr_payload,det,port,k,temp,prompt_id)->dict[str,Any]:
    candidates=corr_payload.get('candidates') or [corr_payload['answer']]
    if len(candidates)==1:
        v=_verdict_one_candidate(row,note_context,candidates[0],det,port,k,temp,prompt_id)
        return {**v,'gen_k':1,'chosen_candidate_idx':0,'per_candidate':None,'accept_correction':v['accept_correction'],'final_answer':candidates[0] if v['accept_correction'] else row['answer']}
    # gen_k > 1: run verdict for each candidate vs original; pick the candidate with most corrected_votes
    per_cand=[]
    for i,cand in enumerate(candidates):
        v=_verdict_one_candidate(row,note_context,cand,det,port,k,temp,prompt_id)
        per_cand.append(v)
    accepted=[(i,v) for i,v in enumerate(per_cand) if v['accept_correction']]
    if accepted:
        best_idx,best_v=max(accepted,key=lambda t: t[1]['corrected_votes'])
        return {**best_v,'gen_k':len(candidates),'chosen_candidate_idx':best_idx,'per_candidate':per_cand,'accept_correction':True,'final_answer':candidates[best_idx]}
    # none accepted: keep original, but report the closest contender
    best_idx,best_v=max(enumerate(per_cand),key=lambda t: t[1]['corrected_votes'])
    return {**best_v,'gen_k':len(candidates),'chosen_candidate_idx':best_idx,'per_candidate':per_cand,'accept_correction':False,'final_answer':row['answer']}

# ===== regen-ra multirun correction (additive; --correction-method regen_ra) =====
# Detection -> regen K from section-index RA spans (GTR) -> regen's OWN count-compare verdict.
_RA_EMB=None
def _ra_emb():
    global _RA_EMB
    if _RA_EMB is None: _RA_EMB=get_embedder()
    return _RA_EMB

REGEN_RA_SYS="You are a medical expert."
REGEN_RA_USER="Discharge note:\n{src}\n\nQuestion: {question}\n\nAnswer the question using only information from the discharge note. Be specific and complete. If the question asks about multiple visits, conditions, or events, cover all of them."
CMP_RA_SYS="You are a strict medical expert."
CMP_RA_USER="Discharge summary:\n{src}\n\nQuestion: {question}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\nCount how many factual claims in each answer contradict the discharge notes. Different wording for the same fact is NOT a contradiction.\n\nA_ERRORS: <number>\nB_ERRORS: <number>"

def build_ra_context(row,det,n_sec=6,k_sent=6)->str:
    """Section-index RA retrieval: admission split + whitelisted sections as index, expand to sentences, dedup, provenance-tagged."""
    note=row['note']; q=row['question']
    queries=[q]+[x for x in (det.get('retrieval_queries') or []) if isinstance(x,str) and x.strip()][:8]
    adms=[a for a in _NC.split_admissions(note) if a.get('chartdate')!='9999-99-99']
    secs=[]; sent_pool=[]
    for a in adms:
        for h,c in _NC.split_sections_clean(a['text']):
            if not c.strip(): continue
            secs.append((a['n'],a['chartdate'],h,c))
            for s in split_sentences(c): sent_pool.append((a['n'],a['chartdate'],h,s))
    if not secs:
        sp=topk_spans(note,queries,k=10,scoring='agreement')
        return '\n'.join(f'[note] {s["sentence"][:160]}' for s in sp)
    emb=_ra_emb()
    qv=emb.encode(queries,normalize_embeddings=True,show_progress_bar=False)
    out=[]; seen=set()
    sv=emb.encode([f'{h}: {c}'[:500] for _,_,h,c in secs],normalize_embeddings=True,show_progress_bar=False)
    for i in _np.argsort(-(sv@qv.T).max(axis=1))[:n_sec]:
        n,d,h,c=secs[int(i)]
        for s in split_sentences(c):
            if s[:60] not in seen: seen.add(s[:60]); out.append(f'[Adm#{n} {d} | {h}] {s[:160]}')
    if sent_pool:
        ev=emb.encode([x[3] for x in sent_pool],normalize_embeddings=True,show_progress_bar=False)
        for i in _np.argsort(-(ev@qv.T).max(axis=1))[:k_sent]:
            n,d,h,s=sent_pool[int(i)]
            if s[:60] not in seen: seen.add(s[:60]); out.append(f'[Adm#{n} {d} | {h}] {s[:160]}')
    return '\n'.join(out)

def _ra_err(raw):
    a=re.search(r'A_ERRORS\s*:\s*(\d+)',raw or '',re.I); b=re.search(r'B_ERRORS\s*:\s*(\d+)',raw or '',re.I)
    return (int(a.group(1)) if a else 99),(int(b.group(1)) if b else 99)

def run_regen_ra(row,det,port,gen_k)->dict[str,Any]:
    ctx=build_ra_context(row,det)
    K=max(1,gen_k)
    cands=[vllm_chat(REGEN_RA_SYS,REGEN_RA_USER.format(src=ctx[:18000],question=row['question']),port,700,1.0) for _ in range(K)]
    # regen's OWN verdict: count-compare tournament (original vs each candidate; keep fewer contradictions)
    champ=row['answer']; samples=[]; chosen=None
    for i,c in enumerate(cands):
        raw=vllm_chat(CMP_RA_SYS,CMP_RA_USER.format(src=ctx[:18000],question=row['question'],a=champ[:1500],b=c[:1500]),port,256,0.0)
        ae,be=_ra_err(raw); sw=be<ae; samples.append({'cand':i,'a_err':ae,'b_err':be,'switched':sw})
        if sw: champ=c; chosen=i
    return {'method':'regen_ra','context_chars':len(ctx),'candidates':cands,'gen_k':K,'verdict_samples':samples,'accept_correction':chosen is not None,'chosen_candidate_idx':chosen,'final_answer':champ}


def process_one(row,port,args)->dict[str,Any]:
    out={k:row[k] for k in ['fold','idx','patient_id','question','ground_truth','answer','orig_label']}
    if 'answer_raw' in row and row['answer_raw'] != row['answer']:
        out['answer_sanitized_from_raw']=True
        out['answer_raw_chars']=len(row['answer_raw'])
    try:
        det=run_detect(row,port,args); out['detection']=det
        p=det['parsed']
        if p.get('verdict')!='INCORRECT' or not p.get('valid'):
            out['action']='kept_original_no_detection'; out['final_answer']=clean_answer(row['answer']); return out
        if getattr(args,'correction_method','raicl')=='regen_ra':
            rr=run_regen_ra(row,p,port,args.gen_k); out['correction']=rr
            out['verdict']={'method':'count_compare','accept_correction':rr['accept_correction'],'chosen_candidate_idx':rr['chosen_candidate_idx']}
            out['action']='accepted_correction' if rr['accept_correction'] else 'rejected_by_verdict'
            out['final_answer']=clean_answer(rr['final_answer']); return out
        spans=retrieve_spans(row,p,args.k_spans); ex=retrieve_example(row,p)
        note_context=(det.get('note_context') or {}).get('text') or row['note'][:18000]
        corr=run_correction(row,note_context,p,spans,ex,port,_correction_temp(args),args.correction_prompt,gen_k=args.gen_k); out['correction']=corr
        verdict=run_verdict(row,note_context,corr,p,port,args.verdict_k,args.verdict_temperature,args.verdict_prompt); out['verdict']=verdict
        chosen_idx=verdict.get('chosen_candidate_idx',0) or 0
        candidates=corr.get('candidates') or [corr['answer']]
        chosen_candidate=candidates[chosen_idx] if 0<=chosen_idx<len(candidates) else corr['answer']
        if verdict['accept_correction']:
            out['action']='accepted_correction'; out['final_answer']=clean_answer(chosen_candidate)
        else:
            out['action']='rejected_by_verdict'; out['final_answer']=clean_answer(row['answer'])
        return out
    except Exception as e:
        out['error']=str(e); out['action']='error_keep_original'; out['final_answer']=clean_answer(row['answer']); return out


def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
    judged=[r for r in rows if (r.get('judge_final') or {}).get('label') is not None]
    fixes=sum(1 for r in judged if r['orig_label']==0 and r['judge_final']['label']==1)
    breaks=sum(1 for r in judged if r['orig_label']==1 and r['judge_final']['label']==0)
    transition_judged=[
        r for r in rows
        if (r.get('judge_original') or {}).get('label') is not None
        and (r.get('judge_final') or {}).get('label') is not None
    ]
    transition_fixes=sum(
        1 for r in transition_judged
        if r['judge_original']['label']==0 and r['judge_final']['label']==1
    )
    transition_breaks=sum(
        1 for r in transition_judged
        if r['judge_original']['label']==1 and r['judge_final']['label']==0
    )
    corr_judged=[r for r in rows if (r.get('judge_correction_candidate') or {}).get('label') is not None]
    corr_fixes=sum(1 for r in corr_judged if r['orig_label']==0 and r['judge_correction_candidate']['label']==1)
    corr_breaks=sum(1 for r in corr_judged if r['orig_label']==1 and r['judge_correction_candidate']['label']==0)
    corr_transition_judged=[
        r for r in rows
        if (r.get('judge_original') or {}).get('label') is not None
        and (r.get('judge_correction_candidate') or {}).get('label') is not None
    ]
    corr_transition_fixes=sum(
        1 for r in corr_transition_judged
        if r['judge_original']['label']==0 and r['judge_correction_candidate']['label']==1
    )
    corr_transition_breaks=sum(
        1 for r in corr_transition_judged
        if r['judge_original']['label']==1 and r['judge_correction_candidate']['label']==0
    )
    return {
        'n':len(rows),
        'n_judged':len(judged),
        'actions':dict(Counter(r.get('action') for r in rows)),
        'detected':sum(1 for r in rows if ((r.get('detection') or {}).get('parsed') or {}).get('verdict')=='INCORRECT'),
        'accepted':sum(1 for r in rows if r.get('action')=='accepted_correction'),
        'fixes':fixes,
        'breaks':breaks,
        'net':fixes-breaks,
        'transition_judged':len(transition_judged),
        'transition_fixes':transition_fixes,
        'transition_breaks':transition_breaks,
        'transition_net':transition_fixes-transition_breaks,
        'correction_candidates_judged':len(corr_judged),
        'correction_candidate_fixes':corr_fixes,
        'correction_candidate_breaks':corr_breaks,
        'correction_candidate_net':corr_fixes-corr_breaks,
        'correction_candidate_transition_judged':len(corr_transition_judged),
        'correction_candidate_transition_fixes':corr_transition_fixes,
        'correction_candidate_transition_breaks':corr_transition_breaks,
        'correction_candidate_transition_net':corr_transition_fixes-corr_transition_breaks,
        'parse_paths':dict(Counter(((r.get('detection') or {}).get('parsed') or {}).get('parse_path','none') for r in rows)),
        'errors':sum(1 for r in rows if r.get('error')),
        'detect_k':dict(Counter((r.get('detection') or {}).get('k') or 1 for r in rows)),
        'detect_vote_incorrect_counts':dict(Counter(((r.get('detection') or {}).get('vote') or {}).get('incorrect_count') for r in rows if (r.get('detection') or {}).get('vote'))),
        'gen_k':dict(Counter((r.get('correction') or {}).get('gen_k') or 1 for r in rows if r.get('correction'))),
        'chosen_candidate_idx':dict(Counter((r.get('verdict') or {}).get('chosen_candidate_idx') for r in rows if r.get('verdict'))),
    }

def write_jsonl(path:Path,rows:list[dict[str,Any]]):
    with path.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--port',type=int,default=8003); ap.add_argument('--concurrency',type=int,default=8)
    ap.add_argument('--n-wrong',type=int,default=2); ap.add_argument('--n-correct',type=int,default=2); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--det-temperature',type=float,default=0.7); ap.add_argument('--correction-temperature',type=float,default=0.0); ap.add_argument('--verdict-temperature',type=float,default=0.7)
    ap.add_argument('--verdict-k',type=int,default=3); ap.add_argument('--k-spans',type=int,default=5); ap.add_argument('--judge',action='store_true')
    ap.add_argument('--detect-k',type=int,default=1,help='Samples drawn for the natural detection confirm step. K>1 uses majority vote on verdict.')
    ap.add_argument('--gen-k',type=int,default=1,help='Correction-candidate samples. K>1 lets the verdict stage pick the best candidate.')
    ap.add_argument('--detect-temperature-multirun',type=float,default=0.3,help='Detection sampling temperature when detect-k>1. Ignored if detect-k==1.')
    ap.add_argument('--correction-temperature-multirun',type=float,default=0.3,help='Correction sampling temperature when gen-k>1. Ignored if gen-k==1.')
    ap.add_argument('--note-context',choices=['first18k','dynamic_spans','dynamic_summary'],default='first18k')
    ap.add_argument('--context-threshold',type=int,default=16000)
    ap.add_argument('--context-k',type=int,default=12)
    ap.add_argument('--context-summary-temperature',type=float,default=0.0)
    ap.add_argument('--det-prompt',choices=sorted(DET_PROMPTS),default='contradiction_first')
    ap.add_argument('--det-parse-backend',choices=['regex_then_gpt4o-mini','gpt4o-mini','gpt4o-mini-helper-v2','qwen2.5'],default='regex_then_gpt4o-mini')
    ap.add_argument('--correction-prompt',choices=sorted(COR_PROMPTS),default='accept_suggestion_if_supported')
    ap.add_argument('--verdict-prompt',choices=sorted(VERDICT_PROMPTS),default='false_correction_sensitive')
    ap.add_argument('--correction-method',choices=['raicl','regen_ra'],default='raicl',help='raicl=existing detect->retrieve->correct->verdict path; regen_ra=K regens from section-index RA spans (GTR) + count-compare verdict (the multirun).')
    ap.add_argument('--input-model',default='qwen2.5-7b-instruct',help='Step8 answer directory to audit/correct')
    ap.add_argument('--run-model-tag',default='',help='Short tag for the local model doing detection/correction/verdict')
    ap.add_argument('--served-model-contains',default='',help='Optional case-insensitive substring required in /v1/models served id')
    args=ap.parse_args(); served=served_model_id(args.port)
    if args.served_model_contains and args.served_model_contains.lower() not in served.lower():
        raise RuntimeError(f'Expected served model containing {args.served_model_contains!r}, found {served}')
    if args.correction_method=='regen_ra':
        _ra_emb()  # pre-load GTR in main thread (avoid concurrent lazy-init meta-tensor race)
    sample=load_rows(args.n_wrong,args.n_correct,args.seed,args.input_model)
    def slug(s:str)->str:
        return re.sub(r'[^A-Za-z0-9_.-]+','-',s).strip('-').lower()[:80] or 'model'
    model_tag=slug(args.run_model_tag or served)
    input_tag=slug(args.input_model)
    k_tag=f"vk{args.verdict_k}"
    if args.detect_k>1 or args.gen_k>1:
        k_tag=f"dk{args.detect_k}_gk{args.gen_k}_vk{args.verdict_k}"
    run_id=(
        f"{model_tag}_input-{input_tag}_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}_"
        f"{args.det_prompt}_{args.det_parse_backend}_{args.correction_prompt}_{args.verdict_prompt}_"
        f"{k_tag}_{args.note_context}"
    )
    out_dir=OUT_ROOT/run_id; out_dir.mkdir(parents=True,exist_ok=True)
    set_ledger(out_dir/'llm_calls.jsonl', script='run_selfdetect_raicl_verdict', served=served, args=vars(args))
    print(f'served_model={served} sample={len(sample)} c={args.concurrency}',flush=True)
    # Warm GTR on first item if needed later.
    if sample: topk_spans(sample[0]['note'],[sample[0]['question']],k=1,scoring='agreement')
    rows=[]
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs=[ex.submit(process_one,row,args.port,args) for row in sample]
        for i,fut in enumerate(as_completed(futs),1):
            rows.append(fut.result()); print(f'pipeline {i}/{len(futs)}',flush=True)
    write_jsonl(out_dir/'pipeline_outputs.jsonl',rows)
    if args.judge:
        note_by_key={(r['fold'],r['idx']):r['note'] for r in sample}
        for i,r in enumerate(rows,1):
            note=note_by_key[(r['fold'],r['idx'])]
            r['judge_original']=gpt_judge(note,r['question'],r['ground_truth'],r['answer'])
            corr=(r.get('correction') or {}).get('answer')
            if corr:
                r['judge_correction_candidate']=gpt_judge(note,r['question'],r['ground_truth'],corr)
            else:
                r['judge_correction_candidate']={'label':None,'raw':'','skipped':'no_correction_candidate'}
            r['judge_final']=gpt_judge(note,r['question'],r['ground_truth'],r['final_answer'])
            if i%10==0 or i==len(rows): print(f'judged {i}/{len(rows)}',flush=True)
        write_jsonl(out_dir/'judged_outputs.jsonl',rows)
    summary={'task':'self_detection_raicl_correction_verdict','served_model':served,'settings':vars(args),'prompt_texts':{'detection':DET_PROMPTS[args.det_prompt],'correction':COR_PROMPTS[args.correction_prompt],'verdict':VERDICT_PROMPTS[args.verdict_prompt]},'summary':summarize(rows),'outputs':{'pipeline':str(out_dir/'pipeline_outputs.jsonl'),'judged':str(out_dir/'judged_outputs.jsonl') if args.judge else None}}
    (out_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
