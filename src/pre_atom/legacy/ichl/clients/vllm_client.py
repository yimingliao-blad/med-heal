"""Local vLLM client (Qwen3-8B judge, and the 5 target models).

Logic preserved from src/external_judge_benchmark/crossmodel_detection.py:call_vllm:
  - System-role injection for models that reject it (BioMistral / Llama-2 template)
  - `chat_template_kwargs={"enable_thinking": ...}` when set
  - HTTP-body sniff for "maximum context length" → returns success=False with
    error='context_too_long' so callers can decide to skip
  - <think>...</think> stripping
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from ichl.clients.base import ClientResponse, LLMClient, strip_think


@dataclass
class VLLMConfig:
    name: str
    url: str = "http://localhost:8003/v1/chat/completions"
    model: str = ""                              # exact HF id sent to server
    default_temperature: float = 0.0
    default_max_tokens: int = 2048
    enable_thinking: bool | None = None          # None = don't send kwarg
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    no_system_role: bool = False                 # merge system into user if True
    max_model_len: int = 8192                    # informational only
    timeout: int = 600
    max_retries: int = 3
    retry_sleep: float = 5.0


class VLLMClient(LLMClient):
    """vLLM OpenAI-compatible chat-completions client.

    Example:
        cfg = VLLMConfig(name='vllm-qwen3-8b', model='Qwen/Qwen3-8B', enable_thinking=True,
                         chat_template_kwargs={'enable_thinking': True})
        client = VLLMClient(cfg)
        resp = client.call(system='...', user='...')
    """

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.name = config.name

    def call(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        log_dir: Path | None = None,
        **kwargs: Any,
    ) -> ClientResponse:
        cfg = self.config
        temperature = temperature if temperature is not None else cfg.default_temperature
        max_tokens = max_tokens if max_tokens is not None else cfg.default_max_tokens

        # BioMistral / Llama-2 template: no system role → merge to user
        if cfg.no_system_role:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # chat_template_kwargs — merged with per-call override
        ctk = dict(cfg.chat_template_kwargs) if cfg.chat_template_kwargs else {}
        if enable_thinking is not None:
            ctk["enable_thinking"] = enable_thinking
        elif cfg.enable_thinking is not None:
            ctk.setdefault("enable_thinking", cfg.enable_thinking)
        if ctk:
            payload["chat_template_kwargs"] = ctk

        for attempt in range(1, cfg.max_retries + 1):
            try:
                t0 = time.monotonic()
                r = requests.post(cfg.url, json=payload, timeout=cfg.timeout)
                lat = time.monotonic() - t0
                if r.status_code != 200:
                    err_body = (r.text or "").lower()
                    if "maximum context length" in err_body or "too many tokens" in err_body:
                        result = ClientResponse(
                            text="", raw_text="", latency=lat,
                            usage={}, success=False, error="context_too_long",
                            client=self.name,
                        )
                        self._log(log_dir, payload, result)
                        return result
                    if attempt < cfg.max_retries:
                        time.sleep(cfg.retry_sleep)
                        continue
                    result = ClientResponse(
                        text="", raw_text="", latency=lat,
                        usage={}, success=False,
                        error=f"HTTP {r.status_code}: {r.text[:300]}",
                        client=self.name,
                    )
                    self._log(log_dir, payload, result)
                    return result
                body = r.json()
                choice = body["choices"][0]
                raw_text = choice["message"]["content"] or ""
                text = strip_think(raw_text)
                usage = body.get("usage", {}) or {}
                finish_reason = choice.get("finish_reason")
                result = ClientResponse(
                    text=text, raw_text=raw_text, latency=lat,
                    usage=usage, finish_reason=finish_reason,
                    success=bool(text), client=self.name,
                )
                self._log(log_dir, payload, result)
                return result
            except Exception as e:
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_sleep)
                    continue
                result = ClientResponse(
                    text="", raw_text="", latency=-1.0,
                    usage={}, success=False, error=str(e)[:500],
                    client=self.name,
                )
                self._log(log_dir, payload, result)
                return result
        return ClientResponse(text="", raw_text="", success=False, client=self.name)
