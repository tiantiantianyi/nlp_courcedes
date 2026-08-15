from __future__ import annotations

import json
from pathlib import Path

import pytest

from anima_search.schemas import ImageAnnotation, ManifestItem


def _load_pipeline_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_m3_m5_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_m3_m5_pipeline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[dict, dict, dict]:
    manifest = ManifestItem(
        image_id="val-1",
        split="Val",
        relative_path="../Val/1.jpg",
        sha256="a" * 64,
        size_bytes=10,
    ).model_dump()
    annotation = ImageAnnotation(
        image_id="val-1",
        split="Val",
        relative_path="../Val/1.jpg",
        sha256="a" * 64,
        summary="测试图片",
        scene="室外",
        search_queries=["测试", "室外", "图片"],
        generation_prompt="测试图片",
        model_version="model",
        prompt_version="v4",
    ).model_dump()
    config = {
        "project_root": str(tmp_path),
        "data": {"artifacts_dir": "artifacts"},
        "annotation": {"prompt_version": "v4"},
    }
    return config, manifest, annotation


def test_validate_annotations_accepts_complete_matching_data(tmp_path):
    module = _load_pipeline_module()
    config, manifest, annotation = _fixture(tmp_path)
    _write_jsonl(tmp_path / "artifacts/manifests/val.jsonl", [manifest])
    _write_jsonl(tmp_path / "artifacts/annotations/val.v4.jsonl", [annotation])
    assert module.validate_annotations(config, "Val") == 1


def test_validate_annotations_rejects_stale_paths(tmp_path):
    module = _load_pipeline_module()
    config, manifest, annotation = _fixture(tmp_path)
    annotation["relative_path"] = "Val/1.jpg"
    _write_jsonl(tmp_path / "artifacts/manifests/val.jsonl", [manifest])
    _write_jsonl(tmp_path / "artifacts/annotations/val.v4.jsonl", [annotation])
    with pytest.raises(ValueError, match="mismatched"):
        module.validate_annotations(config, "Val")
