from __future__ import annotations

from pathlib import Path

import pytest

from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.app.service import SearchService
from anima_search.schemas import ImageAnnotation, SearchQuery, SearchResult


class Parser:
    def parse(self, query: str, client=None):
        assert client is None
        return SearchQuery(raw_text=query)


class Searcher:
    def search(self, query, candidate_count, result_count):
        return [SearchResult(image_id="val-1", relative_path="Val/1.jpg", fused_score=1.0)]


class Manager:
    def qwen_session(self):
        raise AssertionError("default search must not load Qwen")


def _service(tmp_path: Path) -> SearchService:
    annotation = ImageAnnotation(
        image_id="val-1",
        split="Val",
        relative_path="Val/missing.jpg",
        sha256="x",
        summary="城市",
        scene="城市",
        search_queries=["a", "b", "c"],
        generation_prompt="city",
        model_version="qwen",
        prompt_version="v1",
    )
    config = {
        "project_root": str(tmp_path),
        "retrieval": {
            "candidate_count": 10,
            "result_count": 5,
            "rerank_count": 3,
            "query_parser_use_llm": False,
        },
        "models": {"qwen_vl": "missing-qwen"},
    }
    return SearchService(config, Parser(), Searcher(), Manager(), {"val-1": annotation}, "", "", "")


def test_search_defaults_to_no_reranker_and_no_qwen(tmp_path: Path):
    result = _service(tmp_path).search("城市")
    assert result[0].image_id == "val-1"


def test_missing_image_error_names_id_and_path(tmp_path: Path):
    service = _service(tmp_path)
    with pytest.raises(FileNotFoundError, match="val-1"):
        service.answer_about_image("val-1", "在哪里")


def test_qwen_load_reports_missing_local_model(tmp_path: Path):
    client = QwenVLClient(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="models.qwen_vl"):
        client.load()
