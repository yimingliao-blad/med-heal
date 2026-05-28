"""Regex verdict parser.

Default pattern: `\\b(CORRECT|INCORRECT)\\b`, first-match-wins.

Pattern is configurable via constructor so Step 1 of the parser-design sub-pilot
can overwrite it (from `sub_pilot/regex_pattern.txt`) without code edits.

Caveats (per Notion 'Claude: Principle: Regex Parser Unreliability'):
  - Regex is NOT reliable for free-form LLM output. Always compare against
    an LLM parser in pilots before trusting.
  - INCORRECT matches before CORRECT because it's a substring-superset —
    the \\b word-boundary wins this, but double-check on every candidate
    because thinking-mode leaks ('this claim is not incorrect...') can
    invert the verdict.
  - Negation-aware handling will be added ONLY if the sub-pilot shows a
    pattern like '...is NOT incorrect...' (currently: none observed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ichl.prompt_engineering.parsers.base import Parser, ParserResult


DEFAULT_PATTERN = r"\b(CORRECT|INCORRECT)\b"


@dataclass
class RegexParser:
    """First-match-wins regex over {CORRECT, INCORRECT}."""

    name: str = "regex"
    pattern: str = DEFAULT_PATTERN
    case_insensitive: bool = False

    def __post_init__(self) -> None:
        flags = re.IGNORECASE if self.case_insensitive else 0
        self._compiled = re.compile(self.pattern, flags)

    def parse(self, text: str) -> ParserResult:
        if not text:
            return ParserResult(
                verdict="UNKNOWN", parser_name=self.name,
                notes="empty input",
            )
        m = self._compiled.search(text)
        if m is None:
            return ParserResult(
                verdict="UNKNOWN", parser_name=self.name,
                notes="no match",
                extra={"pattern": self.pattern},
            )
        matched = m.group(1).upper()
        # Count multiple matches as an audit signal (useful when the model
        # reasons aloud — "this is not INCORRECT because ..." — first-match-wins
        # can be wrong then).
        n_matches = len(self._compiled.findall(text))
        notes = f"n_matches={n_matches}" if n_matches > 1 else ""
        return ParserResult(
            verdict=matched,  # type: ignore[arg-type]
            parser_name=self.name,
            match_text=m.group(0),
            match_pos=m.start(),
            notes=notes,
            extra={"pattern": self.pattern, "n_matches": n_matches},
        )
