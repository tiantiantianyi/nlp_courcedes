from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from anima_search.pipeline.directory import (
    PipelineState,
    materialize_runtime_config,
    scan_input_directory,
    validate_annotation_snapshot,
    write_manifest_snapshot,
)
from anima_search.schemas import ImageAnnotation


REPOSITORY = Path(__file__).resolve().parents[2]


def _load_annotation_script():
    path = REPOSITORY / "scripts/annotate_images.py"
    spec = importlib.util.spec_from_file_location("annotate_images", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arbitrary_directory_scanner_is_recursive_and_marks_duplicates(tmp_path):
    source = tmp_path / "相册"
    nested = source / "旅行"
    nested.mkdir(parents=True)
    Image.new("RGB", (12, 8), "red").save(source / "a.JPG")
    shutil.copyfile(source / "a.JPG", nested / "copy.jpg")
    (source / "ignore.txt").write_text("not an image", encoding="utf-8")

    items = scan_input_directory(source, tmp_path / "workspace")

    assert len(items) == 2
    assert all(item.split == "Val" and item.valid for item in items)
    assert len({item.image_id for item in items}) == 2
    assert items[0].relative_path.startswith("../相册/")
    assert items[1].duplicate_of == items[0].image_id


def test_manifest_and_annotation_snapshot_validation(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (10, 10), "blue").save(source / "one.jpg")
    workspace = tmp_path / "workspace"
    items = scan_input_directory(source, workspace)
    quality = write_manifest_snapshot(items, workspace / "artifacts")
    item = items[0]
    annotation = ImageAnnotation(
        image_id=item.image_id,
        split="Val",
        relative_path=item.relative_path,
        sha256=item.sha256,
        summary="蓝色测试图片",
        scene="测试",
        search_queries=["蓝色", "测试", "图片"],
        generation_prompt="blue test image",
        model_version="fake",
        prompt_version="v1",
    )
    annotation_path = workspace / "artifacts/annotations/val.v1.jsonl"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(annotation.model_dump_json() + "\n", encoding="utf-8")

    assert quality["valid_count"] == 1
    assert validate_annotation_snapshot(
        workspace / "artifacts/manifests/val.jsonl", annotation_path, "v1"
    ) == 1


def test_runtime_config_is_isolated_and_absolutizes_model_paths(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    prompt_dir = config_dir / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "caption.txt").write_text("prompt", encoding="utf-8")
    (prompt_dir / "query_parser.txt").write_text("prompt", encoding="utf-8")
    base = config_dir / "default.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "data": {"artifacts_dir": "artifacts"},
                "models": {"image_embedder": "models/clip"},
                "annotation": {"prompt": "configs/prompts/caption.txt"},
                "retrieval": {"aliases": "configs/aliases.yaml"},
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "images"
    source.mkdir()
    workspace = tmp_path / "run"

    runtime_path = materialize_runtime_config(base, workspace, source, mode="image-only")
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))

    assert runtime["retrieval"]["enabled_branches"] == ["image"]
    assert Path(runtime["models"]["image_embedder"]).is_absolute()
    assert runtime["data"]["artifacts_dir"] == str(workspace / "artifacts")
    assert (workspace / "configs/prompts/query_parser.txt").is_file()


def test_pipeline_state_requires_explicit_matching_resume(tmp_path):
    state_path = tmp_path / "pipeline_state.json"
    identity = {"input_dir": "/images", "mode": "image-only"}
    state = PipelineState(state_path, identity, resume=False)
    state.update("manifest", "completed", val_count=2)

    with pytest.raises(FileExistsError, match="--resume"):
        PipelineState(state_path, identity, resume=False)
    resumed = PipelineState(state_path, identity, resume=True)
    assert resumed.completed("manifest")
    with pytest.raises(ValueError, match="identity"):
        PipelineState(state_path, {**identity, "mode": "full"}, resume=True)


def test_scene_route_prompts_are_image_specific(tmp_path):
    module = _load_annotation_script()
    routes = tmp_path / "routes.jsonl"
    routes.write_text(
        json.dumps(
            {
                "image_id": "val-1",
                "label": "夜景",
                "category": "night",
                "prompt_suffix": "重点核对灯光。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prompts = module.load_scene_prompts(routes, "基础提示")
    assert "场景路由：夜景（night）" in prompts["val-1"]
    assert prompts["val-1"].endswith("重点核对灯光。")


def test_run_cli_dry_run_lists_full_pipeline_without_writing(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "run.py"),
            "--input_dir",
            str(source),
            "--workspace",
            str(tmp_path / "workspace"),
            "--dry-run",
            "--launch",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["stages"] == [
        "manifest",
        "image_index",
        "scene_routing",
        "annotation",
        "full_indexes",
        "launch",
    ]
    assert not (tmp_path / "workspace").exists()
