"""Parsers for verdict extraction from LLM detection output.

Public surface:
    ParserResult   — verdict + metadata dataclass
    Parser         — protocol all parsers implement
    RegexParser    — regex-based (fast, deterministic)
    LLMParser      — MLX-backed fallback (Qwen3.5-27B-6bit)

See Notion 'Claude: Principle: Regex Parser Unreliability' for the pilot-first
rule: in every pilot, run BOTH parsers and log agreement before trusting regex.
"""
from ichl.prompt_engineering.parsers.base import Parser, ParserResult
from ichl.prompt_engineering.parsers.regex import RegexParser
from ichl.prompt_engineering.parsers.llm import LLMParser

__all__ = ["Parser", "ParserResult", "RegexParser", "LLMParser"]
