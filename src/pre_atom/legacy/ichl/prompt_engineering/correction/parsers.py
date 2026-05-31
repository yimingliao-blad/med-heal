"""Correction-response parsers for the T0 parser sub-pilot.

Two parsers with a common interface:
    parse(raw_response: str, target: str, sub_variant: str) -> ExtractedAnswer

- `RegexCorrectionParser` — model/sub-variant aware final-answer extraction.
- `MLXCorrectionParser` — Qwen3.5-27B-6bit via MLX server at localhost:8800.

Per Notion "Claude: Principle: Regex Parser Unreliability", the T0 parser
sub-pilot runs BOTH parsers on every Step-0 raw response, sends both
extracted answers through the Stage-1 binary GPT-4o judge, and flags any
(target, sub-variant-run) with <95 % agreement as "MLX-as-primary" for the
pilot / full-scale stages.

Input contract — the regex parser receives the *already-think-stripped*
`text` field from Step-0 JSON for Qwen3 think-on (runner already called
`strip_think`). For DS the stored `text` still contains the trailing
`</think>` boundary because DS never emits an opening `<think>`; we handle
that explicitly below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────── result dataclass ───────────────────────


@dataclass
class ExtractedAnswer:
    """Output of any correction parser.

    Attributes:
        answer_text : the final answer extracted from the raw response.
                      "UNKNOWN" sentinel if extraction failed.
        match_span  : (start, end) character offsets into the input, or None
                      if the parser could not point at a precise region.
        notes       : human-readable diagnostic for audit logs.
        parser_name : 'regex' or 'mlx'.
        latency_s   : wall-clock seconds (MLX parser only; 0.0 for regex).
        extra       : any additional debug fields.
    """

    answer_text: str
    match_span: tuple[int, int] | None = None
    notes: str = ""
    parser_name: str = "unknown"
    latency_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────── helpers ───────────────────────


_THINK_CLOSE_RE = re.compile(r"</think>\s*", re.IGNORECASE)
_THINK_FULL_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# "**Answer:**" or "Answer:" at the very start (possibly bolded), case-insensitive
_LEADING_ANSWER_PREAMBLE_RE = re.compile(
    r"^\s*(?:\*\*)?answer\s*:?(?:\*\*)?\s*[:\-–—]?\s*",
    re.IGNORECASE,
)

# Markdown header like "### Step 3" or "### Analysis of ..."
_HEADER_LINE_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

# Qwen2.5 "### Step N" / "### Analysis of the Previous Answer" / "### Corrected ..."
_QWEN25_SECTION_HEADER_RE = re.compile(
    r"^#{1,6}\s*(?:Step\s*\d+|Analysis of the Previous Answer|"
    r"Corrected\s+Final\s+Answer|Corrected\s+Answer|Final\s+Answer).*?$",
    re.IGNORECASE | re.MULTILINE,
)

# Llama "Step 1:" / "The problem with the previous answer is" style
_LLAMA_STEP_LINE_RE = re.compile(
    r"^(?:Step\s*\d+\s*[:\)]\s*|"
    r"The problem with the previous answer is|"
    r"The corrected (?:final )?answer is)",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize(s: str) -> str:
    return s.strip()


def _last_non_empty_block(blocks: list[str]) -> str | None:
    """Return the last block whose stripped form is non-empty, else None."""
    for b in reversed(blocks):
        if b.strip():
            return b.strip()
    return None


def _split_by_markdown_headers(text: str) -> list[str]:
    """Split on markdown header lines; return the non-header paragraphs."""
    parts = _HEADER_LINE_RE.split(text)
    # .split returns chunks in-order with header lines removed; each chunk
    # is the body that follows the previous header.
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────── regex parser ───────────────────────


class RegexCorrectionParser:
    """Model-aware final-answer regex extractor.

    Per T0 plan Parser Sub-Pilot:
      - DS (always-think): text after the LAST </think>; DS never emits the
        opening <think>, so we treat start-of-string as the virtual open.
      - Qwen3 think-ON: strip <think>...</think>; take what follows.
      - Qwen3 think-OFF: strip a leading "**Answer:**" / "Answer:" preamble.
      - Qwen2.5 c/e: if response leads with "### Step 1" / "### Analysis of
        the Previous Answer", take the last non-header clinical paragraph.
      - Llama c/e: if response leads with "Step 1:" / "The problem with the
        previous answer is…", take the final non-step paragraph.
      - All others: return raw_response.strip().
      - If a heuristic fails: return "UNKNOWN" + a diagnostic note.

    `target` values (from sub_variants.THINK_MODE / run dir names):
        'deepseek-r1-distill-llama-8b', 'qwen3-8b', 'qwen2.5-7b-instruct',
        'llama-3.1-8b-instruct', 'biomistral-7b'.
    `sub_variant` is the run-dir folder name:
        - DS/Q2.5/Llama: 'a'..'e'
        - Qwen3:         'a_think', 'a_nothink', ..., 'e_think', 'e_nothink'
    """

    name = "regex"

    def parse(self, raw_response: str, target: str, sub_variant: str) -> ExtractedAnswer:
        if not raw_response or not raw_response.strip():
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=None,
                notes="empty input", parser_name=self.name,
            )

        # Route by target.
        if target == "deepseek-r1-distill-llama-8b":
            return self._parse_ds(raw_response)
        if target == "qwen3-8b":
            if sub_variant.endswith("_think"):
                return self._parse_qwen3_think(raw_response, sv_id=sub_variant[0])
            if sub_variant.endswith("_nothink"):
                return self._parse_qwen3_nothink(raw_response, sv_id=sub_variant[0])
            # Unknown Qwen3 variant — treat as default.
            return self._default(raw_response, note=f"qwen3 unknown sub_variant '{sub_variant}'")
        if target == "qwen2.5-7b-instruct":
            if sub_variant in {"c", "e"}:
                return self._parse_qwen25_ce(raw_response, sv_id=sub_variant)
            return self._default(raw_response)
        if target == "llama-3.1-8b-instruct":
            if sub_variant in {"c", "e"}:
                return self._parse_llama_ce(raw_response, sv_id=sub_variant)
            return self._default(raw_response)

        # Fallback.
        return self._default(raw_response)

    # --- per-target branches ---------------------------------------------

    def _parse_ds(self, text: str) -> ExtractedAnswer:
        # Always-think; take text after the LAST </think>.
        matches = list(_THINK_CLOSE_RE.finditer(text))
        if not matches:
            # DS produced no </think>. Either think-off (rare) or truncated.
            # Fall back to the whole text.
            return ExtractedAnswer(
                answer_text=_normalize(text),
                match_span=(0, len(text)),
                notes="DS: no </think> found; returning full text",
                parser_name=self.name,
            )
        last = matches[-1]
        tail = text[last.end():]
        tail_clean = _normalize(tail)
        if not tail_clean:
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=(last.end(), len(text)),
                notes="DS: empty text after last </think>",
                parser_name=self.name,
            )
        return ExtractedAnswer(
            answer_text=tail_clean,
            match_span=(last.end(), len(text)),
            notes="DS: took tail after last </think>",
            parser_name=self.name,
        )

    def _parse_qwen3_think(self, text: str, sv_id: str) -> ExtractedAnswer:
        # Strip any full <think>...</think> block if one is present. Then
        # strip a leading "Answer:" preamble (the think-on path sometimes
        # reintroduces one).
        stripped = _THINK_FULL_RE.sub("", text)
        # Also handle a dangling </think> tag (runner may have already
        # stripped the opening; strip_think catches that case but belt-and-
        # braces here too).
        close_m = list(_THINK_CLOSE_RE.finditer(stripped))
        if close_m:
            stripped = stripped[close_m[-1].end():]
        stripped = _LEADING_ANSWER_PREAMBLE_RE.sub("", stripped, count=1)
        stripped_clean = _normalize(stripped)
        if not stripped_clean:
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=None,
                notes=f"qwen3 think-on sv={sv_id}: empty after <think> strip",
                parser_name=self.name,
            )
        return ExtractedAnswer(
            answer_text=stripped_clean,
            match_span=None,
            notes=f"qwen3 think-on sv={sv_id}: stripped <think> + leading Answer:",
            parser_name=self.name,
        )

    def _parse_qwen3_nothink(self, text: str, sv_id: str) -> ExtractedAnswer:
        # No <think> block expected. Strip leading "**Answer:**" / "Answer:".
        stripped = _LEADING_ANSWER_PREAMBLE_RE.sub("", text, count=1)
        stripped_clean = _normalize(stripped)
        if not stripped_clean:
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=None,
                notes=f"qwen3 think-off sv={sv_id}: empty after preamble strip",
                parser_name=self.name,
            )
        return ExtractedAnswer(
            answer_text=stripped_clean,
            match_span=None,
            notes=f"qwen3 think-off sv={sv_id}: stripped leading Answer preamble",
            parser_name=self.name,
        )

    def _parse_qwen25_ce(self, text: str, sv_id: str) -> ExtractedAnswer:
        # Heuristic: if response leads with "### Step 1" or "### Analysis of
        # the Previous Answer", take the last non-header clinical paragraph.
        lead = text.lstrip()[:200]
        has_section_header = bool(_QWEN25_SECTION_HEADER_RE.search(lead))
        if not has_section_header:
            # Not in the step-structured form — return as-is.
            return ExtractedAnswer(
                answer_text=_normalize(text), match_span=(0, len(text)),
                notes=f"qwen2.5 sv={sv_id}: no section header; raw",
                parser_name=self.name,
            )
        blocks = _split_by_markdown_headers(text)
        if not blocks:
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=None,
                notes=f"qwen2.5 sv={sv_id}: section headers but no bodies",
                parser_name=self.name,
            )
        last = _last_non_empty_block(blocks)
        if last is None:
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=None,
                notes=f"qwen2.5 sv={sv_id}: all blocks empty",
                parser_name=self.name,
            )
        return ExtractedAnswer(
            answer_text=last,
            match_span=None,
            notes=f"qwen2.5 sv={sv_id}: took last non-header block",
            parser_name=self.name,
        )

    def _parse_llama_ce(self, text: str, sv_id: str) -> ExtractedAnswer:
        # Heuristic: if response leads with "Step 1:" / "The problem with
        # the previous answer is", take the final non-step paragraph.
        lead = text.lstrip()[:200]
        has_step_lead = bool(_LLAMA_STEP_LINE_RE.search(lead))
        if not has_step_lead:
            return ExtractedAnswer(
                answer_text=_normalize(text), match_span=(0, len(text)),
                notes=f"llama sv={sv_id}: no step lead; raw",
                parser_name=self.name,
            )
        # Split on Step-N / "The problem..." line boundaries. Return the
        # final non-empty chunk.
        parts = _LLAMA_STEP_LINE_RE.split(text)
        blocks = [p.strip() for p in parts if p.strip()]
        last = _last_non_empty_block(blocks)
        if last is None:
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=None,
                notes=f"llama sv={sv_id}: all blocks empty after step split",
                parser_name=self.name,
            )
        # If the "last block" itself is short and looks like another step
        # marker line (e.g. a trailing "Step 3:" introducing the fix), we
        # want the paragraph following it. _LLAMA_STEP_LINE_RE.split already
        # removed the marker, so `last` is post-marker text.
        return ExtractedAnswer(
            answer_text=last,
            match_span=None,
            notes=f"llama sv={sv_id}: last non-step block",
            parser_name=self.name,
        )

    def _default(self, text: str, note: str = "default passthrough") -> ExtractedAnswer:
        return ExtractedAnswer(
            answer_text=_normalize(text),
            match_span=(0, len(text)),
            notes=note, parser_name=self.name,
        )


# ─────────────────────── MLX parser ───────────────────────


_MLX_SYSTEM_PROMPT = (
    "You are a careful extraction assistant. Given an LLM response to a "
    "medical question, your job is to return ONLY the final answer text "
    "with all reasoning, step markers, preambles, and evidence enumeration "
    "stripped out. Do not add or remove clinical information. Return the "
    "final answer verbatim from the response."
)

_MLX_USER_TEMPLATE = """\
Given this model response to a medical question, extract ONLY the final \
answer text, stripping any reasoning, step markers, preambles, or evidence \
enumeration. Do not add or remove information. Return the final answer \
verbatim.

RESPONSE:
{raw_response}

EXTRACTED FINAL ANSWER:
"""


class MLXCorrectionParser:
    """MLX-backed final-answer extractor (Qwen3.5-27B-6bit on localhost:8800).

    The MEMORY rule "Claude: Principle: Use MLX as External Validator §
    Concurrency rules" mandates ThreadPoolExecutor(max_workers=2) when
    batching calls — that concurrency is applied by the sub-pilot caller,
    not by this class. This class performs a single synchronous call per
    .parse().
    """

    name = "mlx"

    def __init__(
        self,
        *,
        url: str = "http://localhost:8800/v1/chat/completions",
        model_name: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_s: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ):
        self.url = url
        # MLX dispatch: server-side chooses model when model_name is a
        # loaded-model ID. `default_model` works with the mlx-qwen35 config
        # (see configs/tool_models.yaml). For local 8800 probe, we pass the
        # Qwen3.5-27B id the server reports.
        self.model_name = model_name or "/Users/madblade/Projects/local-llm/models/mlx/Qwen3.5-27B-6bit-NexVeridian"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def parse(self, raw_response: str, target: str, sub_variant: str) -> ExtractedAnswer:
        import time

        import requests

        if not raw_response or not raw_response.strip():
            return ExtractedAnswer(
                answer_text="UNKNOWN", match_span=None,
                notes="empty input", parser_name=self.name,
            )

        user_msg = _MLX_USER_TEMPLATE.format(raw_response=raw_response)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _MLX_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # keep think off; extraction task doesn't need CoT
            "chat_template_kwargs": {"enable_thinking": False},
        }

        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                r = requests.post(self.url, json=payload, timeout=self.timeout_s)
                lat = time.monotonic() - t0
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    if attempt < self.max_retries:
                        time.sleep(self.retry_sleep)
                        continue
                    return ExtractedAnswer(
                        answer_text="UNKNOWN", match_span=None,
                        notes=f"mlx error: {last_err}",
                        parser_name=self.name, latency_s=lat,
                    )
                body = r.json()
                choice = body["choices"][0]
                msg = choice["message"]
                text = (msg.get("content") or "").strip()
                if not text:
                    # Some MLX servers put output in `reasoning`
                    text = (msg.get("reasoning") or "").strip()
                # strip any <think>...</think> Qwen3.5 may emit anyway
                text = _THINK_FULL_RE.sub("", text).strip()
                if not text:
                    return ExtractedAnswer(
                        answer_text="UNKNOWN", match_span=None,
                        notes=f"mlx empty after strip (finish_reason={choice.get('finish_reason')})",
                        parser_name=self.name, latency_s=lat,
                        extra={"finish_reason": choice.get("finish_reason")},
                    )
                return ExtractedAnswer(
                    answer_text=text,
                    match_span=None,
                    notes=f"mlx ok (finish_reason={choice.get('finish_reason')})",
                    parser_name=self.name, latency_s=lat,
                    extra={
                        "finish_reason": choice.get("finish_reason"),
                        "usage": body.get("usage", {}),
                    },
                )
            except Exception as e:  # noqa: BLE001
                last_err = str(e)[:200]
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
                    continue
                return ExtractedAnswer(
                    answer_text="UNKNOWN", match_span=None,
                    notes=f"mlx exception: {last_err}",
                    parser_name=self.name, latency_s=0.0,
                )
        return ExtractedAnswer(
            answer_text="UNKNOWN", match_span=None,
            notes=f"mlx exhausted retries: {last_err}",
            parser_name=self.name, latency_s=0.0,
        )
