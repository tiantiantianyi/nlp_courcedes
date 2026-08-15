from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from anima_search.config import load_config, resolve_path
from anima_search.data.manifest import sha256_file
from anima_search.schemas import ImageAnnotation, ManifestItem

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_image_id(relative_path: Path) -> str:
    digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"val-{digest}"


def scan_input_directory(
    input_dir: Path,
    project_root: Path,
    *,
    limit: int | None = None,
) -> list[ManifestItem]:
    """Scan an arbitrary directory recursively into the existing Val data contract."""
    input_dir = input_dir.resolve()
    project_root = project_root.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {input_dir}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    paths = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        suffixes = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"no supported images found under {input_dir}; expected: {suffixes}")

    first_by_hash: dict[str, str] = {}
    items: list[ManifestItem] = []
    for path in paths:
        source_relative = path.relative_to(input_dir)
        digest = sha256_file(path)
        image_id = _stable_image_id(source_relative)
        metadata: dict[str, Any] = {}
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                metadata.update(width=image.width, height=image.height, mode=image.mode)
        except Exception as exc:
            metadata.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        duplicate_of = first_by_hash.get(digest)
        first_by_hash.setdefault(digest, image_id)
        items.append(
            ManifestItem(
                image_id=image_id,
                split="Val",
                relative_path=Path(os.path.relpath(path, project_root)).as_posix(),
                sha256=digest,
                size_bytes=path.stat().st_size,
                duplicate_of=duplicate_of,
                **metadata,
            )
        )
    return items


def write_manifest_snapshot(items: list[ManifestItem], artifacts_dir: Path) -> dict[str, Any]:
    manifest_dir = artifacts_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "val.jsonl"
    manifest_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in items),
        encoding="utf-8",
    )
    quality = {
        "source": "arbitrary-input-directory",
        "val_count": len(items),
        "valid_count": sum(item.valid for item in items),
        "invalid": [item.image_id for item in items if not item.valid],
        "duplicates": {
            item.image_id: item.duplicate_of for item in items if item.duplicate_of
        },
        "manifest_sha256": sha256_file(manifest_path),
    }
    (manifest_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return quality


def _absolutize_runtime_paths(payload: dict[str, Any], base_config: dict[str, Any]) -> None:
    for key, value in list(payload.get("models", {}).items()):
        if isinstance(value, str):
            payload["models"][key] = str(resolve_path(base_config, value).resolve())
    annotation = payload.get("annotation", {})
    if isinstance(annotation.get("prompt"), str):
        annotation["prompt"] = str(resolve_path(base_config, annotation["prompt"]).resolve())
    retrieval = payload.get("retrieval", {})
    if isinstance(retrieval.get("aliases"), str):
        retrieval["aliases"] = str(resolve_path(base_config, retrieval["aliases"]).resolve())


def materialize_runtime_config(
    base_config_path: Path,
    workspace: Path,
    input_dir: Path,
    *,
    mode: str,
) -> Path:
    if mode not in {"image-only", "full"}:
        raise ValueError(f"unsupported pipeline mode: {mode}")
    base_config_path = base_config_path.resolve()
    workspace = workspace.resolve()
    input_dir = input_dir.resolve()
    base_config = load_config(base_config_path)
    payload = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    _absolutize_runtime_paths(payload, base_config)

    artifacts_dir = workspace / "artifacts"
    payload.setdefault("data", {})
    payload["data"].update(
        train_dir=str(input_dir),
        val_dir=str(input_dir),
        artifacts_dir=str(artifacts_dir),
    )
    payload.setdefault("retrieval", {})["enabled_branches"] = (
        ["image"] if mode == "image-only" else ["image", "text", "bm25"]
    )

    config_dir = workspace / "configs"
    prompt_source = base_config_path.parent / "prompts"
    prompt_target = config_dir / "prompts"
    config_dir.mkdir(parents=True, exist_ok=True)
    if prompt_source.is_dir():
        shutil.copytree(prompt_source, prompt_target, dirs_exist_ok=True)
    runtime_config = config_dir / "runtime.yaml"
    runtime_config.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_config


def validate_annotation_snapshot(
    manifest_path: Path,
    annotation_path: Path,
    prompt_version: str,
) -> int:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {annotation_path}")
    manifests = [
        ManifestItem.model_validate_json(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotations = [
        ImageAnnotation.model_validate_json(line)
        for line in annotation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {item.image_id: item for item in manifests if item.valid}
    actual = {item.image_id: item for item in annotations}
    if len(actual) != len(annotations):
        raise ValueError("annotations contain duplicate image IDs")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(
        image_id
        for image_id in set(expected).intersection(actual)
        if (
            expected[image_id].sha256 != actual[image_id].sha256
            or expected[image_id].relative_path != actual[image_id].relative_path
            or actual[image_id].prompt_version != prompt_version
        )
    )
    if missing or extra or mismatched:
        raise ValueError(
            "annotation validation failed: "
            f"missing={missing[:10]}, extra={extra[:10]}, mismatched={mismatched[:10]}"
        )
    return len(annotations)


class PipelineState:
    def __init__(self, path: Path, identity: dict[str, Any], *, resume: bool) -> None:
        self.path = path
        if path.is_file():
            if not resume:
                raise FileExistsError(
                    f"pipeline state already exists: {path}; use --resume or a new --workspace"
                )
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data.get("identity") != identity:
                raise ValueError("resume arguments do not match the existing pipeline identity")
        else:
            self.data = {
                "schema_version": 1,
                "status": "running",
                "identity": identity,
                "created_at": _utc_now(),
                "stages": {},
            }
            self.write()

    def completed(self, stage: str) -> bool:
        payload = self.data.get("stages", {}).get(stage, {})
        return payload.get("status") == "completed"

    def update(self, stage: str, status: str, **details: Any) -> None:
        self.data.setdefault("stages", {})[stage] = {
            "status": status,
            "updated_at": _utc_now(),
            **details,
        }
        self.data["status"] = "running" if status != "failed" else "failed"
        self.data["current_stage"] = stage
        self.write()

    def finish(self, **details: Any) -> None:
        self.data.update(status="completed", finished_at=_utc_now(), **details)
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
