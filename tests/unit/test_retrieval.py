import pytest

from anima_search.retrieval.search import HybridSearcher
from anima_search.schemas import ImageAnnotation, SearchQuery


class FakeIndex:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error

    def search(self, query, limit):
        if self.error:
            raise self.error
        return self.results[:limit]


ALIASES = {
    "aliases": {"人物": ["人", "行人"]},
    "fields": {"objects": ["人物"]},
}


def make_annotation(image_id: str, objects=None) -> ImageAnnotation:
    return ImageAnnotation(
        image_id=image_id,
        split="Val",
        relative_path=f"Val/{image_id}.jpg",
        sha256=image_id,
        summary=f"图片 {image_id}",
        objects=objects or [],
        scene="城市",
        search_queries=["a", "b", "c"],
        generation_prompt="city",
        model_version="qwen",
        prompt_version="v1",
    )


def test_three_branch_rrf_returns_provenance():
    annotations = {key: make_annotation(key) for key in ("a", "b", "c")}
    searcher = HybridSearcher(
        annotations,
        indexes={
            "image": FakeIndex([("a", 0.9), ("b", 0.8)]),
            "text": FakeIndex([("b", 0.95), ("a", 0.7)]),
            "bm25": FakeIndex([("c", 5.0), ("b", 4.0)]),
        },
    )
    results = searcher.search(SearchQuery(raw_text="城市"), candidate_count=3, result_count=3)
    assert results[0].image_id == "b"
    assert results[0].active_branches == ["image", "text", "bm25"]
    assert results[0].branch_ranks == {"image": 2, "text": 1, "bm25": 2}


def test_filter_runs_before_rrf_and_compacts_branch_rank():
    annotations = {
        "person": make_annotation("person", ["行人"]),
        "empty": make_annotation("empty", ["汽车"]),
    }
    searcher = HybridSearcher(
        annotations,
        indexes={"image": FakeIndex([("person", 1.0), ("empty", 0.5)])},
        aliases=ALIASES,
    )
    results = searcher.search(
        SearchQuery(raw_text="不要人物", excluded_terms=["人物"]),
        candidate_count=2,
        result_count=2,
    )
    assert [item.image_id for item in results] == ["empty"]
    assert results[0].branch_ranks == {"image": 1}


def test_failed_branch_degrades_to_remaining_branches():
    annotations = {"a": make_annotation("a")}
    searcher = HybridSearcher(
        annotations,
        indexes={
            "image": FakeIndex(error=RuntimeError("model unavailable")),
            "text": FakeIndex([("a", 0.8)]),
        },
    )
    results = searcher.search(SearchQuery(raw_text="城市"))
    assert results[0].active_branches == ["text"]
    assert "image" in searcher.last_branch_errors


def test_all_failed_branches_raise_clear_error():
    searcher = HybridSearcher(
        {"a": make_annotation("a")},
        indexes={"image": FakeIndex(error=RuntimeError("offline"))},
    )
    with pytest.raises(RuntimeError, match="all retrieval branches failed"):
        searcher.search(SearchQuery(raw_text="城市"))
