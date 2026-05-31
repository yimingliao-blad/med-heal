"""OpenAI client (GPT-4o and friends).

Reuses the call pattern from src/step2_benchmarking/generate_gpt4.py with
retry + backoff. API key loaded from env (default: OPENAI_API_KEY).

Config: see configs/tool_models.yaml — entries with `type: openai`.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ichl.clients.base import ClientResponse, LLMClient


@dataclass
class OpenAIConfig:
    name: str
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    max_retries: int = 10
    retry_sleep: float = 5.0


class OpenAIClient(LLMClient):
    """OpenAI chat-completion client.

    Example:
        client = OpenAIClient(OpenAIConfig(name='gpt-4o', model='gpt-4o'))
        resp = client.call(system='...', user='...', temperature=0.0, max_tokens=400)
    """

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.name = config.name
        # Late-bind the OpenAI SDK so import-time failures don't poison the package.
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed. Run: uv pip install openai"
            ) from e
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env} is not set. "
                f"Load it from .env (see Notion 'Claude: Principle: Misc' entry for "
                f"OPENAI_API_KEY shell pattern)."
            )
        self._client = OpenAI(api_key=api_key)

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        log_dir: Path | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> ClientResponse:
        temperature = temperature if temperature is not None else self.config.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.default_max_tokens

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p

        for attempt in range(1, self.config.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = self._client.chat.completions.create(**payload)
                lat = time.monotonic() - t0
                choice0 = resp.choices[0]
                text = (choice0.message.content or "").strip()
                usage = {}
                if getattr(resp, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                finish_reason = getattr(choice0, "finish_reason", None)
                result = ClientResponse(
                    text=text,
                    raw_text=text,
                    latency=lat,
                    usage=usage,
                    finish_reason=finish_reason,
                    success=bool(text),
                    client=self.name,
                )
                self._log(log_dir, payload, result)
                return result
            except Exception as e:
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_sleep)
                    continue
                result = ClientResponse(
                    text="", raw_text="", latency=-1.0,
                    usage={}, success=False, error=str(e)[:500],
                    client=self.name,
                )
                self._log(log_dir, payload, result)
                return result
        return ClientResponse(text="", raw_text="", success=False, client=self.name)
