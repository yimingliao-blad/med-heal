from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.json"
    with config_path.open() as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return ((base or PROJECT_ROOT) / p).resolve()


def env_for_legacy(cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            str(PROJECT_ROOT / "src" / "pre_atom" / "legacy"),
            str(PROJECT_ROOT / "src" / "pre_atom" / "legacy" / "step9_self_correction" / "v2"),
            str(PROJECT_ROOT / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PRE_ATOM_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["PRE_ATOM_SOURCE_REPO_ROOT"] = str(resolve_path(cfg["paths"]["source_repo_root"]))
    return env


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("output_dir", "reports_dir"):
        resolve_path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

