"""Rule-based structural mutator — the 'bone'.

Takes a candidate prompt and a list of variation primitives, produces
structural skeletons. Deterministic and cheap.

Rules are registered via @register_rule(name). Each callable:
    (template_str, primitive) -> list[str]
returns zero or more transformed strings (the skeleton variants). A single
primitive can produce multiple variants (e.g. different insertion points).

Current rule set focuses on ANTI-ANCHORING mutations for the detection
verdict-only family (per Round 0 finding: anchoring bias was the dominant
failure mode). See `variations/detection.yaml` for the primitives that
reference these rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ichl.prompt_engineering.pool import SeedPrompt, VariationPrimitive


@dataclass
class Skeleton:
    """A rule-produced structural variant (pre-LLM-polish)."""
    name: str
    template: str
    parent_seed: str
    rule_name: str
    rule_kind: str


# Rule registry — concrete transforms below.
_RULE_REGISTRY: dict[str, Callable[[str, VariationPrimitive], list[str]]] = {}


def register_rule(name: str):
    """Decorator to register a concrete rule implementation."""
    def _decorator(fn: Callable[[str, VariationPrimitive], list[str]]):
        _RULE_REGISTRY[name] = fn
        return fn
    return _decorator


# ─────────────────────── helpers ───────────────────────

_FINAL_OUTPUT_RE = re.compile(
    r"(Reply with|Your final output|Output format).*$",
    re.IGNORECASE | re.DOTALL,
)


def _split_header_body_tail(template: str) -> tuple[str, str, str]:
    """Split a seed template into (header, body, tail).

    header = text before 'Discharge summary:' (instructions/preamble)
    body   = from 'Discharge summary:' through 'Model's answer: {model_answer}'
    tail   = output-format rule (e.g. 'Reply with exactly one word: CORRECT or INCORRECT.')

    Best-effort string surgery; returns the original template as body if it can't
    find the split points.
    """
    # Find tail (output rule).
    m = _FINAL_OUTPUT_RE.search(template)
    if m:
        tail_start = m.start()
        tail = template[tail_start:].strip()
        pre_tail = template[:tail_start].rstrip()
    else:
        tail = ""
        pre_tail = template

    # Find body start.
    idx = pre_tail.find("Discharge summary:")
    if idx < 0:
        return "", pre_tail, tail
    header = pre_tail[:idx].rstrip()
    body = pre_tail[idx:].rstrip()
    return header, body, tail


# ─────────────────────── structural (anti-anchoring) rules ───────────────────────

@register_rule("write_own_answer_first")
def _write_own_answer_first(template: str, primitive: VariationPrimitive) -> list[str]:
    """Inject P9-style instruction: read notes, write own answer, THEN compare."""
    header, body, tail = _split_header_body_tail(template)
    insert = (
        "Before looking at the model's answer, read only the discharge summary and the "
        "question. Write in 2-3 sentences what the correct answer SHOULD be based on the "
        "notes alone. Then compare your own answer to the model's answer.\n"
    )
    new = ((header + "\n\n") if header else "") + insert + "\n" + body + ("\n\n" + tail if tail else "")
    return [new]


@register_rule("pre_commitment")
def _pre_commitment(template: str, primitive: VariationPrimitive) -> list[str]:
    """Inject: predict answer shape before being anchored."""
    header, body, tail = _split_header_body_tail(template)
    insert = (
        "Step 1: Read only the discharge summary and the question. Predict in one "
        "sentence what kind of answer the notes support.\n"
        "Step 2: Read the model's answer. Does it match your prediction? "
        "Any factual claim that is NOT supported by the notes counts as INCORRECT.\n"
    )
    new = ((header + "\n\n") if header else "") + insert + "\n" + body + ("\n\n" + tail if tail else "")
    return [new]


@register_rule("claim_by_claim")
def _claim_by_claim(template: str, primitive: VariationPrimitive) -> list[str]:
    """Inject: list 2-4 claims, verify each, then verdict."""
    header, body, tail = _split_header_body_tail(template)
    insert = (
        "Checklist approach:\n"
        "1. List 2-4 factual claims from the model's answer (medications, diagnoses, dates, "
        "lab values, procedures).\n"
        "2. For each claim, find the matching fact in the discharge summary. Mark each as "
        "SUPPORTED, CONTRADICTED, or UNSUPPORTED.\n"
        "3. If every claim is SUPPORTED, the verdict is CORRECT. If ANY claim is "
        "CONTRADICTED or UNSUPPORTED by the notes, the verdict is INCORRECT.\n"
    )
    new = ((header + "\n\n") if header else "") + body + "\n\n" + insert + ("\n" + tail if tail else "")
    return [new]


@register_rule("notes_first")
def _notes_first(template: str, primitive: VariationPrimitive) -> list[str]:
    """Put discharge summary at the very top, above the instructions."""
    header, body, tail = _split_header_body_tail(template)
    # Move the body (which starts with Discharge summary:) ABOVE header.
    new = body + ("\n\n" + header if header else "") + ("\n\n" + tail if tail else "")
    return [new]


@register_rule("counter_anchor")
def _counter_anchor(template: str, primitive: VariationPrimitive) -> list[str]:
    """Add explicit anti-anchoring warning."""
    warning = (
        "IMPORTANT: models frequently make subtle factual errors in discharge QA. "
        "Assume the model's answer may be wrong; actively search for claims not supported "
        "by the notes. Do not default to accepting the answer."
    )
    header, body, tail = _split_header_body_tail(template)
    new = ((header + "\n\n") if header else "") + warning + "\n\n" + body + ("\n\n" + tail if tail else "")
    return [new]


# ─────────────────────── role rules ───────────────────────

@register_rule("role_skeptical_auditor")
def _role_skeptical_auditor(template: str, primitive: VariationPrimitive) -> list[str]:
    """Replace medical-expert framing with skeptical-auditor framing."""
    header, body, tail = _split_header_body_tail(template)
    role = (
        "You are a skeptical medical auditor. Your task is to find factual errors, not to "
        "affirm correctness. Err on the side of flagging problems if a claim is not "
        "explicitly supported by the discharge summary."
    )
    new = role + "\n\n" + body + ("\n\n" + tail if tail else "")
    return [new]


@register_rule("role_adversarial")
def _role_adversarial(template: str, primitive: VariationPrimitive) -> list[str]:
    """Adversarial hallucination-hunter framing."""
    header, body, tail = _split_header_body_tail(template)
    role = (
        "You are a hallucination hunter. Your only job is to find unsupported claims in "
        "the model's answer. If any claim is not backed by the discharge summary, the "
        "verdict is INCORRECT. If every claim is grounded, the verdict is CORRECT."
    )
    new = role + "\n\n" + body + ("\n\n" + tail if tail else "")
    return [new]


# ─────────────────────── output-format rules ───────────────────────

@register_rule("repeat_output_rule")
def _repeat_output_rule(template: str, primitive: VariationPrimitive) -> list[str]:
    """Repeat the single-word output requirement at the top AND bottom."""
    header, body, tail = _split_header_body_tail(template)
    top_rule = (
        "Output format: your entire response must be exactly one word — CORRECT or "
        "INCORRECT. No reasoning, no explanation, no preamble."
    )
    new = ((header + "\n\n") if header else "") + top_rule + "\n\n" + body + (
        "\n\n" + tail if tail else "\n\nReply with exactly one word: CORRECT or INCORRECT."
    )
    return [new]


@register_rule("reason_then_verdict")
def _reason_then_verdict(template: str, primitive: VariationPrimitive) -> list[str]:
    """Allow a 1-sentence reason, then a new-line verdict token."""
    header, body, tail = _split_header_body_tail(template)
    new_tail = (
        "Output:\n"
        "Line 1: a ONE-SENTENCE reason.\n"
        "Line 2: exactly one word — CORRECT or INCORRECT.\n"
        "Nothing else."
    )
    new = ((header + "\n\n") if header else "") + body + "\n\n" + new_tail
    return [new]


@register_rule("chain_of_verification")
def _chain_of_verification(template: str, primitive: VariationPrimitive) -> list[str]:
    """Inject 2-3 yes/no verification questions, then the verdict."""
    header, body, tail = _split_header_body_tail(template)
    insert = (
        "Verification:\n"
        "Q1: Does the model's answer contain any factual claim that is contradicted by "
        "the discharge summary? (yes / no)\n"
        "Q2: Does the model's answer contain any claim not present in the discharge "
        "summary? (yes / no)\n"
        "Q3: If you were the clinician, would you accept this answer as-is? (yes / no)\n"
        "Based on these answers, your final output must be exactly one word: CORRECT "
        "(if all Q1=no, Q2=no, Q3=yes) or INCORRECT (otherwise)."
    )
    new = ((header + "\n\n") if header else "") + body + "\n\n" + insert
    return [new]


# ─────────────────────── orchestration ───────────────────────

@register_rule("noop")
def _noop(template: str, primitive: VariationPrimitive) -> list[str]:
    """Pass-through; no change. Used as a sanity default for unregistered rules."""
    return [template]


def rule_based_mutate(
    seed: SeedPrompt,
    primitives: list[VariationPrimitive],
    max_per_primitive: int = 1,
) -> list[Skeleton]:
    """Apply each primitive to the seed, return all produced skeletons.

    A primitive without a registered rule falls back to _noop (no change, logged
    so the caller sees it). Ideal usage: primitives passed here already have
    matching rule implementations.
    """
    skeletons: list[Skeleton] = []
    for p in primitives:
        fn = _RULE_REGISTRY.get(p.name, _noop)
        variants = fn(seed.template, p)[:max_per_primitive]
        for i, t in enumerate(variants):
            if not t or t.strip() == seed.template.strip():
                continue  # skip noop-identical outputs (unregistered rule fallback)
            skeletons.append(Skeleton(
                name=f"{seed.name}__{p.kind}__{p.name}__{i:02d}",
                template=t,
                parent_seed=seed.name,
                rule_name=p.name,
                rule_kind=p.kind,
            ))
    return skeletons


def list_registered_rules() -> list[str]:
    """Return names of all rules with concrete implementations (excludes noop)."""
    return [name for name in _RULE_REGISTRY if name != "noop"]
