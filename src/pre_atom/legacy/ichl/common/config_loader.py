"""Config loader with project/step/run hierarchy.

See Notion page "Claude: Module: Prompt Engineering Iteration" — Config
hierarchy section.

Project-level:  configs/models.yaml, configs/tool_models.yaml, configs/parsers.yaml
Step-level:     configs/steps/<step>.yaml
Run-level:      <run_dir>/config.yaml  (inherits from step, with overrides)

Each level is a YAML dict. Merge is deep (nested dicts merge recursively,
lists replace wholesale).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge configs. Later configs override earlier ones.

    Dicts merge recursively. Lists and scalars replace.
    """
    result: dict[str, Any] = {}
    for cfg in configs:
        if cfg is None:
            continue
        result = _deep_merge(result, cfg)
    return result


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_config(
    step: str | None = None,
    run_config_path: Path | None = None,
) -> dict[str, Any]:
    """Load and merge project + step + run configs for a given step.

    Args:
        step: name like 'detection', 'verdict', 'regeneration', etc.
              If provided, configs/steps/<step>.yaml is loaded.
        run_config_path: optional path to a run-level config.yaml.

    Returns:
        Merged config dict. Use dict access to read values.
    """
    # Project-level: all of them merged; step-level can override any.
    project_cfg = {
        "models": load_yaml(CONFIGS_DIR / "models.yaml"),
        "tool_models": load_yaml(CONFIGS_DIR / "tool_models.yaml"),
        "parsers": load_yaml(CONFIGS_DIR / "parsers.yaml"),
    }

    step_cfg: dict[str, Any] = {}
    if step:
        step_cfg = load_yaml(CONFIGS_DIR / "steps" / f"{step}.yaml")

    run_cfg: dict[str, Any] = {}
    if run_config_path:
        run_cfg = load_yaml(Path(run_config_path))

    return merge_configs(project_cfg, step_cfg, run_cfg)
