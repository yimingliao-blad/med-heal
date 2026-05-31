"""Factory: load client configs from YAML, return the right LLMClient.

Default config search order:
    1. explicit path passed to make_client(name, config_path=...)
    2. configs/tool_models.yaml
    3. configs/models.yaml
First match wins. This lets the same factory hand back either a tool model
(GPT-4o, MLX) or a target model (BM, Q2.5, Qwen3-8B, Llama, DS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ichl.clients.base import LLMClient
from ichl.clients.mlx_client import MLXClient, MLXConfig
from ichl.clients.openai_client import OpenAIClient, OpenAIConfig
from ichl.clients.vllm_client import VLLMClient, VLLMConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATHS = [
    PROJECT_ROOT / "configs" / "tool_models.yaml",
    PROJECT_ROOT / "configs" / "models.yaml",
]


def _load_entry(name: str, config_path: Path | None = None) -> dict[str, Any]:
    paths = [config_path] if config_path else DEFAULT_CONFIG_PATHS
    errors: list[str] = []
    for p in paths:
        if p is None or not p.exists():
            continue
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except Exception as e:
            errors.append(f"{p}: {e}")
            continue
        if name in data:
            entry = dict(data[name])
            entry.setdefault("name", name)
            return entry
    hint = "; ".join(errors) if errors else "(no errors)"
    raise KeyError(
        f"Client '{name}' not found in config files: "
        f"{[str(p) for p in paths if p is not None]}. {hint}"
    )


def make_client(name: str, config_path: Path | None = None) -> LLMClient:
    """Return an LLMClient for the given config name.

    Examples:
        client = make_client('gpt-4o')
        client = make_client('mlx-qwen35')
        client = make_client('qwen3-8b')                    # from models.yaml
        client = make_client('vllm-qwen3-8b-judge')         # from tool_models.yaml
    """
    entry = _load_entry(name, config_path=config_path)
    ctype = entry.pop("type", None)
    if ctype is None:
        raise ValueError(f"Config entry '{name}' missing 'type' (openai|mlx|vllm)")

    if ctype == "openai":
        cfg = OpenAIConfig(
            name=entry["name"],
            model=entry.get("model", "gpt-4o"),
            api_key_env=entry.get("api_key_env", "OPENAI_API_KEY"),
            default_temperature=entry.get("default_temperature", 0.0),
            default_max_tokens=entry.get("default_max_tokens", 1024),
            max_retries=entry.get("max_retries", 10),
            retry_sleep=entry.get("retry_sleep", 5.0),
        )
        return OpenAIClient(cfg)

    if ctype == "mlx":
        cfg = MLXConfig(
            name=entry["name"],
            url=entry.get("url", "http://192.168.68.107:8800/v1/chat/completions"),
            model_name=entry.get("model_name", entry.get("model", "default_model")),
            default_temperature=entry.get("default_temperature", 0.0),
            default_max_tokens=entry.get("default_max_tokens", 1024),
            enable_thinking=entry.get("enable_thinking", False),
            max_tokens_think=entry.get("max_tokens_think", 12288),
            max_tokens_plain=entry.get("max_tokens_plain", 4096),
            url_env=entry.get("url_env", "LOCAL_QWEN35_URL"),
            model_env=entry.get("model_env", "LOCAL_QWEN35_MODEL"),
            timeout=entry.get("timeout", 600),
            max_retries=entry.get("max_retries", 3),
            retry_sleep=entry.get("retry_sleep", 4.0),
        )
        return MLXClient(cfg)

    if ctype == "vllm":
        # Accept either `model` (used in tool_models.yaml) or `hf_name` (models.yaml)
        model = entry.get("model") or entry.get("hf_name") or ""
        cfg = VLLMConfig(
            name=entry["name"],
            url=entry.get("url", "http://localhost:8003/v1/chat/completions"),
            model=model,
            default_temperature=entry.get("default_temperature", 0.0),
            default_max_tokens=entry.get("default_max_tokens", entry.get("max_tokens", 2048)),
            enable_thinking=entry.get("enable_thinking"),
            chat_template_kwargs=entry.get("chat_template_kwargs", {}) or {},
            no_system_role=entry.get("no_system_role", False),
            max_model_len=entry.get("max_model_len", 8192),
            timeout=entry.get("timeout", 600),
            max_retries=entry.get("max_retries", 3),
            retry_sleep=entry.get("retry_sleep", 5.0),
        )
        return VLLMClient(cfg)

    raise ValueError(f"Unknown client type '{ctype}' in config entry '{name}'")
