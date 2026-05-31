"""Uniform LLM client interface.

Every client (OpenAI, MLX, vLLM) implements `LLMClient.call(...)` returning
a `ClientResponse`. This lets downstream code stay client-agnostic.

Key design rules (from Notion principles):
  - Strip <think>...</think> blocks from `.text` but keep them in `.raw_text`
  - Content/reasoning-field fallback (some servers put CoT in `reasoning`)
  - Retries handled inside the client; final failure returns success=False
  - Optional per-call audit logging (one JSONL line per call)
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ClientResponse:
    """Standard response object returned by every client.

    Attributes:
        text:           content with <think>...</think> blocks stripped
        raw_text:       original content (including think blocks if present)
        latency:        wall-clock seconds for the HTTP/SDK round-trip
        usage:          server-reported token usage dict (may be empty for some servers)
        finish_reason:  server-reported stop reason ('stop'|'length'|'tool_calls'|None)
                        — 'length' means `max_tokens` was hit (truncation).
        success:        True if the call returned a non-empty response after retries
        error:          short error string if success=False, else None
        client:         client short name (e.g., 'gpt-4o', 'mlx-qwen35', 'vllm-qwen3-8b')
    """
    text: str
    raw_text: str = ""
    latency: float = -1.0
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    success: bool = True
    error: str | None = None
    client: str = ""


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 / DeepSeek CoT leakage)."""
    if not text or "<think>" not in text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class LLMClient(ABC):
    """Abstract base. Subclasses: OpenAIClient, MLXClient, VLLMClient."""

    name: str = "<base>"

    @abstractmethod
    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        log_dir: Path | None = None,
        **kwargs: Any,
    ) -> ClientResponse:
        """Make a single call. Returns ClientResponse (never raises for server errors)."""
        ...

    def _log(self, log_dir: Path | None, payload: dict, response: ClientResponse) -> None:
        """Append one JSONL line to log_dir/<client>_calls.jsonl."""
        if log_dir is None:
            return
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "client": self.name,
            "payload": payload,
            "response": {
                "text": response.text[:2000],     # cap per-line size
                "raw_text": response.raw_text[:4000],
                "latency": response.latency,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "success": response.success,
                "error": response.error,
            },
        }
        with open(log_dir / f"{self.name}_calls.jsonl", "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
