"""LLM verdict parser — delegates to Qwen3.5-27B-6bit on the Mac Studio MLX server.

Reads the LLM's original response and asks the parser model to classify the
final verdict. Always strips `<think>...</think>` blocks from the parser's
own output before pattern-matching (Qwen3.5 may think briefly even in
non-think mode if the input contains leaked CoT tokens).

Default system + user prompt are drafts — they get replaced by the
`LLMParser` constructor in Step 1 with whatever version the sub-pilot
finalises (saved to `sub_pilot/llm_parser_prompt.txt`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ichl.clients.base import LLMClient
from ichl.clients.factory import make_client
from ichl.prompt_engineering.parsers.base import Parser, ParserResult


DEFAULT_SYSTEM = (
    "You extract a final binary verdict from an LLM evaluator's response. "
    "Return exactly one token: CORRECT, INCORRECT, or UNKNOWN."
)

DEFAULT_USER_TEMPLATE = """\
The text below is an evaluator LLM's response about whether another model's \
answer was correct or incorrect.

Text:
\"\"\"
{text}
\"\"\"

What was the evaluator's final verdict? Answer with EXACTLY ONE TOKEN:
- CORRECT  (evaluator said the answer was correct)
- INCORRECT (evaluator said the answer was incorrect)
- UNKNOWN (evaluator did not commit to a verdict, or you cannot tell)

Output only the token, no explanation.
"""


# ─────────────────────── post-processing ───────────────────────

_TOKEN_RE = re.compile(r"\b(CORRECT|INCORRECT|UNKNOWN)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_token(text: str) -> str:
    """Pick the parser model's verdict from its short response."""
    clean = _strip_think(text)
    m = _TOKEN_RE.search(clean)
    if m is None:
        return "UNKNOWN"
    return m.group(1).upper()


# ─────────────────────── parser class ───────────────────────

@dataclass
class LLMParser:
    """LLM-backed verdict extractor using MLX (Qwen3.5-27B-6bit by default).

    The Notion parser principle requires this parser to run in every pilot
    alongside a regex parser so agreement can be measured.

    Args:
        client_name:     entry in `configs/tool_models.yaml` — defaults to 'mlx-qwen35'
        system_prompt:   override DEFAULT_SYSTEM (normally from sub_pilot/llm_parser_prompt.txt)
        user_template:   must contain '{text}'
        max_tokens:      cap on parser's response (small — verdict only)
        log_dir:         optional, per-call JSONL audit
        client:          inject a pre-built client (for testing)
    """

    client_name: str = "mlx-qwen35"
    system_prompt: str = DEFAULT_SYSTEM
    user_template: str = DEFAULT_USER_TEMPLATE
    max_tokens: int = 32
    name: str = "llm"
    log_dir: Path | None = None
    client: LLMClient | None = None

    def __post_init__(self) -> None:
        if "{text}" not in self.user_template:
            raise ValueError("user_template must contain '{text}' placeholder")
        if self.client is None:
            self.client = make_client(self.client_name)

    def parse(self, text: str) -> ParserResult:
        if not text:
            return ParserResult(
                verdict="UNKNOWN", parser_name=self.name, notes="empty input",
            )
        user_msg = self.user_template.format(text=text)
        resp = self.client.call(  # type: ignore[union-attr]
            system=self.system_prompt,
            user=user_msg,
            temperature=0.0,
            max_tokens=self.max_tokens,
            log_dir=self.log_dir,
        )
        if not resp.success:
            return ParserResult(
                verdict="UNKNOWN", parser_name=self.name,
                latency_s=max(resp.latency, 0.0),
                raw_response=resp.raw_text,
                notes=f"client error: {resp.error}",
                extra={"finish_reason": resp.finish_reason},
            )
        verdict = _extract_token(resp.text)
        return ParserResult(
            verdict=verdict,  # type: ignore[arg-type]
            parser_name=self.name,
            match_text=verdict,
            latency_s=max(resp.latency, 0.0),
            raw_response=resp.raw_text,
            notes=(f"finish_reason={resp.finish_reason}" if resp.finish_reason != "stop" else ""),
            extra={"client": resp.client, "finish_reason": resp.finish_reason},
        )
