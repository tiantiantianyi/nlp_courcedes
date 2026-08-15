from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from anima_search.app.factory import create_service
from anima_search.indexing.bm25_index import BM25Index
from anima_search.schemas import ImageAnnotation


def test_build_indexes_cli_creates_bm25_snapshot_and_manifest(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    prompt_dir = config_dir / "prompts"
    annotation_dir = project / "artifacts" / "annotations"
    prompt_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    for name in ("query_parser.txt", "reranker.txt", "content_writer.txt", "sd_prompt.txt"):
        (prompt_dir / name).write_text("只输出 JSON", encoding="utf-8")
    config_path = config_dir / "default.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {"artifacts_dir": "artifacts"},
        "models": {"qwen_vl": "unused-qwen", "stable_diffusion": "unused-sd"},
        "annotation": {"prompt_version": "v1"},
        "retrieval": {"enabled_branches": ["bm25"], "candidate_count": 10,
                      "result_count": 2, "rerank_count": 2, "rrf_k": 60,
                      "query_parser_use_llm": False, "rrf_weight": 0.35, "vlm_weight": 0.65},
        "runtime": {"device": "cpu", "dtype": "float32", "max_image_pixels": 1024},
        "generation": {"seed": 1},
    }, allow_unicode=True), encoding="utf-8")

    records = [
        ImageAnnotation(
            image_id="train-1", split="Train", relative_path="Train/1.jpg", sha256="1",
            summary="雨夜城市", scene="城市", colors=["冷色"], search_queries=["a", "b", "c"],
            generation_prompt="city", model_version="qwen", prompt_version="v1",
        ),
        ImageAnnotation(
            image_id="train-2", split="Train", relative_path="Train/2.jpg", sha256="2",
            summary="晴天公园", scene="公园", colors=["绿色"], search_queries=["a", "b", "c"],
            generation_prompt="park", model_version="qwen", prompt_version="v1",
        ),
    ]
    annotation_path = annotation_dir / "train.v1.jsonl"
    annotation_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )

    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(repository / "scripts" / "build_indexes.py"),
         "--config", str(config_path), "--split", "Train", "--branches", "bm25"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    output = project / "artifacts" / "indexes" / "train"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_branches"] == ["bm25"]
    assert manifest["record_count"] == 2
    assert (output / "annotations.json").is_file()
    assert BM25Index.load(output / "bm25.pkl").search("雨夜", 1)[0][0] == "train-1"

    service = create_service(str(config_path), "train")
    results = service.search("雨夜城市", use_reranker=False)
    assert results[0].image_id == "train-1"
    assert results[0].active_branches == ["bm25"]
