"""Mac Studio MLX client (Qwen3.5-27B-6bit by default).

Logic preserved from src/external_judge_benchmark/common.py:call_external:
  - content/reasoning-field fallback (some MLX servers put CoT in `reasoning`)
  - <think>...</think> stripping
  - `chat_template_kwargs={"enable_thinking": ...}` per call
  - Env-var URL/model override: LOCAL_QWEN35_URL, LOCAL_QWEN35_MODEL (legacy)

Concurrency: never run two MLX servers on one Mac; within one server, use
ThreadPoolExecutor(max_workers=2) for short extraction tasks (2.31x speedup).
See Notion principle "Claude: Principle: Use MLX as External Validator".
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from ichl.clients.base import ClientResponse, LLMClient, strip_think


@dataclass
class MLXConfig:
    name: str
    url: str = "http://192.168.68.107:8800/v1/chat/completions"
    model_name: str = "default_model"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    enable_thinking: bool = False
    max_tokens_think: int = 12288
    max_tokens_plain: int = 4096
    url_env: str | None = "LOCAL_QWEN35_URL"
    model_env: str | None = "LOCAL_QWEN35_MODEL"
    timeout: int = 600
    max_retries: int = 3
    retry_sleep: float = 4.0


class MLXClient(LLMClient):
    """MLX server client. Supports Qwen3.5 thinking toggle per call.

    Example:
        client = MLXClient(MLXConfig(name='mlx-qwen35'))
        resp = client.call(system='...', user='...', max_tokens=400)
    """

    def __init__(self, config: MLXConfig):
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
        # URL / model-name env-var overrides for back-compat
        url = (cfg.url_env and os.environ.get(cfg.url_env)) or cfg.url
        model_name = (cfg.model_env and os.environ.get(cfg.model_env)) or cfg.model_name

        think = enable_thinking if enable_thinking is not None else cfg.enable_thinking
        temperature = temperature if temperature is not None else cfg.default_temperature
        if max_tokens is None:
            max_tokens = cfg.max_tokens_think if think else cfg.max_tokens_plain

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": think},
        }

        for attempt in range(1, cfg.max_retries + 1):
            try:
                t0 = time.monotonic()
                r = requests.post(url, json=payload, timeout=cfg.timeout)
                lat = time.monotonic() - t0
                if r.status_code != 200:
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
                msg = choice["message"]
                raw_text = (msg.get("content") or "").strip()
                if not raw_text:
                    raw_text = (msg.get("reasoning") or "").strip()
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
