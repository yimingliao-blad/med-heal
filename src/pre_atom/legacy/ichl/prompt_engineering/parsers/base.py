"""Parser protocol + ParserResult dataclass.

A parser takes an LLM's raw response text and returns a verdict in
{'CORRECT', 'INCORRECT', 'UNKNOWN'} plus audit metadata. All parsers share
this interface so the runner can run multiple parsers side-by-side and
compute agreement.

'UNKNOWN' is the legal way to say "I couldn't decide". Never return
'CORRECT' as a fallback — that silently biases accuracy upward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


Verdict = Literal["CORRECT", "INCORRECT", "UNKNOWN"]


@dataclass
class ParserResult:
    """Verdict + minimal metadata for parser comparison and audit.

    Attributes:
        verdict:        decision in {'CORRECT', 'INCORRECT', 'UNKNOWN'}
        parser_name:    the parser that produced this result ('regex' | 'llm')
        match_text:     the substring matched (regex) or a summary (llm)
        match_pos:      0-based offset of `match_text` in the input (regex only; -1 otherwise)
        latency_s:      wall-clock seconds spent parsing (llm parsers only, else 0.0)
        raw_response:   the LLM parser's own raw response (llm only; empty for regex)
        notes:          freeform — e.g. 'multiple matches', 'negation detected'
        extra:          any parser-specific metadata (do NOT rely on its shape downstream)
    """
    verdict: Verdict
    parser_name: str
    match_text: str = ""
    match_pos: int = -1
    latency_s: float = 0.0
    raw_response: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Parser(Protocol):
    """Protocol every parser implements."""
    name: str

    def parse(self, text: str) -> ParserResult:
        ...
