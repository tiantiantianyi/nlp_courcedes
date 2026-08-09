from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

INDEX_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_ids_digest(image_ids: list[str]) -> str:
    payload = "\n".join(image_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_index_manifest(path: Path, *, split: str, image_ids: list[str], annotation_path: Path,
                         annotation_version: str, branches: dict[str, dict[str, Any]],
                         config_digest: str = "") -> dict[str, Any]:
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "split": split,
        "record_count": len(image_ids),
        "image_ids_sha256": image_ids_digest(image_ids),
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "annotation_version": annotation_version,
        "active_branches": sorted(branches),
        "branches": branches,
        "config_digest": config_digest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_index_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported index manifest version: {payload.get('schema_version')}; "
            f"expected {INDEX_SCHEMA_VERSION}"
        )
    return payload


def validate_index_manifest(payload: dict[str, Any], image_ids: list[str],
                            branch_image_ids: dict[str, list[str]]) -> None:
    if payload.get("record_count") != len(image_ids):
        raise ValueError("index manifest record_count does not match annotations")
    if payload.get("image_ids_sha256") != image_ids_digest(image_ids):
        raise ValueError("index manifest image IDs do not match annotations")
    expected = set(payload.get("active_branches", []))
    if not set(branch_image_ids).issubset(expected):
        raise ValueError("loaded index contains branches not declared by the index manifest")
    for name, ids in branch_image_ids.items():
        if ids != image_ids:
            raise ValueError(f"{name} index image IDs do not match annotations")
