"""LLM polish mutator — the 'flesh'.

Takes a rule-produced skeleton and rephrases it at temperature=1 via the
tool model (GPT-4o default, MLX fallback). Preserves structure; improves
phrasing / naturalness / instruction clarity.

See Notion "Claude: Module: Prompt Engineering Iteration" — Variation strategy.

STUB — polish prompt templates and calling convention fill in as we
exercise the first optimization run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ichl.clients import make_client
from ichl.prompt_engineering.mutator.rules import Skeleton


POLISH_SYSTEM_PROMPT = """You are a prompt engineer. Your job is to take a
draft LLM prompt and rephrase it for better clarity and natural language
flow WITHOUT changing the structure or the task semantics. Output ONLY the
rephrased prompt text; no meta-commentary."""


POLISH_USER_TEMPLATE = """Rephrase the following prompt draft. Preserve:
- All sections and their order
- All constraints and requirements
- All variable placeholders (like {{note}}, {{question}}, {{model_answer}})

Improve:
- Naturalness of phrasing
- Instruction clarity
- Remove redundancy

Draft prompt:
---
{draft}
---

Return ONLY the rephrased prompt, nothing else."""


@dataclass
class PolishedVariant:
    name: str
    template: str
    parent_skeleton: str
    tool_model: str
    llm_raw: str                  # raw response for audit


def llm_polish(
    skeleton: Skeleton,
    tool_model: str = "gpt-4o",
    n_variants: int = 1,
    temperature: float = 1.0,
    log_dir: Path | None = None,
) -> list[PolishedVariant]:
    """Call the tool model at temperature=1 to polish the skeleton.

    Returns n_variants polished variants.
    """
    client = make_client(tool_model)
    out: list[PolishedVariant] = []
    for i in range(n_variants):
        resp = client.call(
            system=POLISH_SYSTEM_PROMPT,
            user=POLISH_USER_TEMPLATE.format(draft=skeleton.template),
            temperature=temperature,
            max_tokens=2048,
            log_dir=log_dir,
        )
        if not resp.success or not resp.text.strip():
            continue
        out.append(PolishedVariant(
            name=f"{skeleton.name}__polish_{i:02d}",
            template=resp.text.strip(),
            parent_skeleton=skeleton.name,
            tool_model=tool_model,
            llm_raw=resp.raw_text,
        ))
    return out
