"""Reusable MLX-server client using the OpenAI Python SDK.

Why this exists alongside `mlx_client.py`:
  - `mlx_client.py` uses raw `requests.post`. Works, but doesn't share the
    OpenAI SDK's retry/streaming machinery and is harder to swap with vLLM.
  - This module wraps the same MLX server (mlx_lm.server) using the OpenAI
    Python SDK with `extra_body={"chat_template_kwargs": ...}`, which the SDK
    forwards to the request body's top-level field — matching the curl example:

    curl http://192.168.68.107:8800/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role":"user","content":"..."}],
        "chat_template_kwargs": {"enable_thinking": true}
      }'

Verified-working endpoint
-------------------------
  Live endpoint:  http://192.168.68.107:8800/v1   (verified 2026-04-25)
  Loaded models:
    - Qwen/Qwen3-32B
    - /Users/madblade/Projects/local-llm/models/mlx/Qwen3.5-27B-6bit-NexVeridian

NOTE on port: an earlier user note referenced port 8090 — that port is NOT
responding on this machine. Use 8800. If you maintain multiple MLX servers,
expose the port via env var MLX_SERVER_URL.

LM Studio is a DIFFERENT server type
-----------------------------------
  LM Studio at http://192.168.68.107:1234/v1 is *also* OpenAI-compatible but
  does NOT respect `chat_template_kwargs`. It toggles thinking via its own
  UI/loaded-model-config, not the API. For LM Studio use the standard
  `OpenAI(base_url=...)` without `chat_template_kwargs`.

How thinking output is split (mlx_lm.server.py:1517-1549, verified 2026-04-25)
-----------------------------------------------------------------------------
  When the model loads, the tokenizer chat template is inspected. If it
  defines `<think>` / `</think>` tokens (Qwen3, DeepSeek-R1, etc.), the
  server sets has_thinking=True. During generation an `in_reasoning` flag
  routes each token to either `reasoning_text` or `content`; on `</think>`
  it flips off. In the final JSON:

    - `message.reasoning`  ← the chain-of-thought text (THIS naming, NOT
                             `reasoning_content` like vLLM/DeepSeek)
    - `message.content`    ← the user-facing answer
    - the `<think>` / `</think>` delimiters are stripped from both fields

  Cross-server compatibility: vLLM and the DeepSeek API put CoT in
  `message.reasoning_content`; OpenAI o1/o3 hide reasoning entirely. This
  client reads `reasoning` first, falls back to `reasoning_content`, and
  populates a single `MLXResponse.reasoning_content` field for callers.

  If the server was started with thinking-disabled at startup (default on
  port 8800 currently), pass `enable_thinking=True` to override per-request.

  Per-model behavior verified on this server (2026-04-25):
    - Qwen3.5-27B-6bit-NexVeridian (true MLX-loaded) → server splits
      properly: content has clean answer ('\\n\\n17 × 23 = 391'),
      reasoning_content has CoT. **Use this model with MLXOpenAIClient.**

  Important — what is NOT served by mlx_lm.server here:
    - Qwen/Qwen3-32B is NOT actually MLX-served on this Mac. It runs on
      LM Studio at port 1234 (GGUF / llama.cpp backend). The id may appear
      in port 8800's /v1/models listing for legacy/proxy reasons, but
      requests for it behave like LM Studio (think-tags inline in content,
      no chat_template_kwargs split). For Qwen3-32B and any other LM-Studio
      model, **use LMStudioClient** below, which does not send
      chat_template_kwargs and connects to port 1234 directly.

    User intent (2026-04-25): all judge models will eventually move to the
    real MLX server — at that point, MLXOpenAIClient becomes the single
    judge entry point. Until then, the two clients stay split because their
    response semantics differ.

Usage
-----
    from ichl.clients.mlx_openai_client import MLXOpenAIClient
    client = MLXOpenAIClient()
    resp = client.chat(
        system="You are a medical expert...",
        user="...",
        enable_thinking=True,
        max_tokens=4096,
    )
    print(resp.content, resp.completion_tokens, resp.latency_s)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

# Verified live MLX server (mlx_lm.server)
DEFAULT_MLX_URL = "http://192.168.68.107:8800/v1"
DEFAULT_MLX_MODEL = "/Users/madblade/Projects/local-llm/models/mlx/Qwen3.5-27B-6bit-NexVeridian"
# NOTE: Qwen3-32B is on LM Studio (port 1234), NOT MLX. Use LMStudioClient for it.


@dataclass
class MLXResponse:
    content: str
    reasoning_content: str
    finish_reason: str | None
    completion_tokens: int | None
    prompt_tokens: int | None
    latency_s: float
    raw: Any | None = None


class MLXOpenAIClient:
    """OpenAI-SDK wrapper for the MLX server.

    Sends `chat_template_kwargs={"enable_thinking": ...}` via OpenAI SDK's
    `extra_body`, which forwards it to the top-level of the request body.
    The MLX server (mlx_lm.server) reads it from there.

    Args:
        base_url: defaults to the live MLX server. Override via env MLX_SERVER_URL
                  or constructor arg.
        model:    defaults to Qwen3.5-27B-6bit MLX. Override via env MLX_MODEL.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: int = 600):
        self.base_url = base_url or os.environ.get("MLX_SERVER_URL", DEFAULT_MLX_URL)
        self.model = model or os.environ.get("MLX_MODEL", DEFAULT_MLX_MODEL)
        self.timeout = timeout
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed", timeout=timeout)

    def chat(self, *, system: str, user: str,
             enable_thinking: bool = False,
             temperature: float = 0.0,
             max_tokens: int = 4096,
             max_retries: int = 3) -> MLXResponse:
        """Send a chat-completion to the MLX server.

        Args:
            enable_thinking: forwarded as `chat_template_kwargs.enable_thinking`
                via the OpenAI SDK's `extra_body`. The mlx_lm server default
                (port 8800 currently) is thinking-OFF at startup; this flag
                overrides per-request. With thinking ON and a template that
                defines `<think>` tokens, reasoning is captured in
                `MLXResponse.reasoning_content` (read from `message.reasoning`,
                with fallback to `message.reasoning_content` for cross-server
                compat). With thinking OFF, the model emits the answer
                immediately into `content`.
        """
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        extra = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        last_err: Exception | None = None
        for attempt in range(max_retries):
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                    extra_body=extra,
                )
                lat = time.monotonic() - t0
                msg = resp.choices[0].message
                # mlx_lm.server uses `reasoning`; vLLM/DeepSeek use `reasoning_content`.
                # Read both, prefer mlx_lm's convention; expose unified field on response.
                reasoning = (getattr(msg, "reasoning", None)
                             or getattr(msg, "reasoning_content", None)
                             or "")
                return MLXResponse(
                    content=getattr(msg, "content", None) or "",
                    reasoning_content=reasoning,
                    finish_reason=resp.choices[0].finish_reason,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else None,
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
                    latency_s=round(lat, 2),
                    raw=resp,
                )
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 + attempt * 3)
        return MLXResponse(content="", reasoning_content="", finish_reason="ERROR",
                           completion_tokens=None, prompt_tokens=None,
                           latency_s=-1.0, raw=str(last_err))


class LMStudioClient:
    """OpenAI-SDK wrapper for LM Studio (GGUF / llama.cpp backend, port 1234).

    LM Studio is OpenAI-compatible but does NOT respect chat_template_kwargs.
    Thinking is toggled per-model at LM Studio UI load time, not per-call.

    When the loaded model has a thinking template (e.g. Qwen3-32B, Qwen3.5-27B,
    DeepSeek-R1), reasoning is emitted inline in `message.content` wrapped in
    literal `<think>...</think>` tags — LM Studio does NOT split it into a
    separate `reasoning` / `reasoning_content` field. Callers needing the
    cleaned answer should run `ichl.clients.base.strip_think()` on the content.

    Models verified here on port 1234 (2026-04-25):
        qwen3-30b-a3b-instruct-2507-mlx, qwen/qwen3-235b-a22b-2507,
        mistralai/magistral-small-2509, qwen/qwen3-vl-8b, qwen/qwen3.5-9b,
        qwen3.5-27b, qwen/qwen3-32b, meta-llama-3-8b-instruct,
        text-embedding-nomic-embed-text-v1.5, mistralai/mistral-small-3.2.
    """
    DEFAULT_URL = "http://192.168.68.107:1234/v1"

    def __init__(self, base_url: str | None = None, model: str = "",
                 timeout: int = 600):
        self.base_url = base_url or os.environ.get("LMS_SERVER_URL", self.DEFAULT_URL)
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(base_url=self.base_url, api_key="lm-studio", timeout=timeout)

    def chat(self, *, system: str, user: str,
             temperature: float = 0.0,
             max_tokens: int = 4096,
             max_retries: int = 3) -> MLXResponse:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        last_err: Exception | None = None
        for attempt in range(max_retries):
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                lat = time.monotonic() - t0
                msg = resp.choices[0].message
                return MLXResponse(
                    content=getattr(msg, "content", None) or "",
                    reasoning_content=getattr(msg, "reasoning_content", None) or "",
                    finish_reason=resp.choices[0].finish_reason,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else None,
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
                    latency_s=round(lat, 2),
                    raw=resp,
                )
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 + attempt * 3)
        return MLXResponse(content="", reasoning_content="", finish_reason="ERROR",
                           completion_tokens=None, prompt_tokens=None,
                           latency_s=-1.0, raw=str(last_err))
