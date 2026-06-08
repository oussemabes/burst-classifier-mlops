from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def resolve_data_root(config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    root = Path(config["data"]["root"])
    if root.is_absolute():
        return root
    if config_path is None:
        return root.resolve()
    return (Path(config_path).resolve().parent.parent / root).resolve()


def latest_run_dir(base: str | Path = "artifacts") -> Path:
    latest = Path(base) / "latest"
    if latest.exists():
        return latest.resolve()
    latest_file = Path(base) / "latest_run.txt"
    if latest_file.exists():
        return Path(latest_file.read_text(encoding="utf-8").strip()).resolve()
    runs = sorted(
        (p for p in Path(base).glob("*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run directory found under {base}")
    return runs[0].resolve()
