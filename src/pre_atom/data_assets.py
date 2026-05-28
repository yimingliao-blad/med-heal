from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def file_info(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def copy_asset(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": file_info(src), "copy": file_info(dst)}


def prepare_data_assets(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = resolve_project_path(config["local_copies"]["raw_dir"])
    processed_dir = resolve_project_path(config["local_copies"]["processed_dir"])
    manifest: dict[str, Any] = {"raw": {}, "processed": {}}

    raw_ehr = resolve_project_path(config["raw_data"]["ehrnoteqa_processed_jsonl"])
    raw_human = resolve_project_path(config["raw_data"]["human_judgment_csv"])
    gold100 = resolve_project_path(config["processed_data"]["human_gold_100_from_112_sara_jose_agreed"])
    gold200 = resolve_project_path(config["processed_data"]["human_gold_200_extended"])

    manifest["raw"]["ehrnoteqa_processed_jsonl"] = copy_asset(raw_ehr, raw_dir / "EHRNoteQA_processed.jsonl")
    manifest["raw"]["human_judgment_csv"] = copy_asset(raw_human, raw_dir / raw_human.name)
    manifest["processed"]["human_gold_100"] = copy_asset(gold100, processed_dir / "human_gold_100_from_112_sara_jose_agreed.csv")
    manifest["processed"]["human_gold_200"] = copy_asset(gold200, processed_dir / "human_gold_200_extended.csv")

    notes_df = pd.read_json(raw_ehr, lines=True)
    human_df = pd.read_csv(raw_human)
    gold100_df = pd.read_csv(gold100)
    manifest["summary"] = {
        "ehrnoteqa_rows": int(len(notes_df)),
        "human_judgment_rows": int(len(human_df)),
        "human_gold_100_rows": int(len(gold100_df)),
        "human_reviewers": sorted([str(x) for x in human_df.get("User Name", pd.Series(dtype=str)).dropna().unique()]),
    }
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path
