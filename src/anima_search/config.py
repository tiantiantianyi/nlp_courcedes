from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["project_root"] = str(config_path.parent.parent)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["project_root"]) / path
