"""LLM client factory + implementations.

Uniform interface (see `base.LLMClient`) for:
  - OpenAI (gpt-4o and friends)
  - Mac Studio MLX (Qwen3.5-27B via HTTP)
  - Local vLLM (Qwen3-8B, target models)

Usage:
    from ichl.clients import make_client
    client = make_client('gpt-4o')
    resp = client.call(system='...', user='...', temperature=0.0, max_tokens=400)
    # resp.text, resp.raw_text, resp.latency, resp.usage, resp.success

Each call can be audit-logged by passing `log_dir=Path(...)`; every request +
response is saved as a JSONL line per the Audit Guidelines principle.
"""
from ichl.clients.base import LLMClient, ClientResponse
from ichl.clients.factory import make_client

__all__ = ["LLMClient", "ClientResponse", "make_client"]
