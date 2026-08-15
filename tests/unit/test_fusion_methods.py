from __future__ import annotations

import pytest

from anima_search.evaluation.fusion_comparison import compare_fusion_methods
from anima_search.retrieval.fusion import (
    normalized_weighted_fusion_with_ranks,
    reciprocal_rank_fusion_with_ranks,
)
from anima_search.retrieval.search import HybridSearcher
from anima_search.schemas import ImageAnnotation, SearchQuery, SearchResult


RANKINGS = {
    "image": [("a", 0.9), ("b", 0.8)],
    "text": [("b", 0.9), ("a", 0.8)],
    "bm25": [("b", 10.0), ("a", 1.0)],
}


def annotation(image_id: str) -> ImageAnnotation:
    return ImageAnnotation(
        image_id=image_id,
        split="Val",
        relative_path=f"Val/{image_id}.jpg",
        sha256=image_id,
        summary=image_id,
        scene="测试",
        search_queries=["a", "b", "c"],
        generation_prompt="test",
        model_version="fake",
        prompt_version="v1",
    )


class Index:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, limit):
        return self.rows[:limit]


def test_weighted_fusion_can_differ_from_rrf_and_keeps_provenance():
    rrf = reciprocal_rank_fusion_with_ranks(RANKINGS)
    weighted = normalized_weighted_fusion_with_ranks(
        RANKINGS, {"image": 0.8, "text": 0.1, "bm25": 0.1}
    )
    assert rrf[0][0] == "b"
    assert weighted[0][0] == "a"
    assert weighted[0][2] == {"image": 0.9, "text": 0.8, "bm25": 1.0}
    assert weighted[0][3] == {"image": 1, "text": 2, "bm25": 2}


def test_weighted_fusion_rejects_invalid_weights():
    with pytest.raises(ValueError, match="non-negative"):
        normalized_weighted_fusion_with_ranks(RANKINGS, {"image": -1})
    with pytest.raises(ValueError, match="inactive"):
        normalized_weighted_fusion_with_ranks(RANKINGS, {"unknown": 1})


def test_hybrid_searcher_uses_configured_weighted_method():
    searcher = HybridSearcher(
        {key: annotation(key) for key in ("a", "b")},
        indexes={name: Index(rows) for name, rows in RANKINGS.items()},
        fusion_method="weighted",
        fusion_weights={"image": 0.8, "text": 0.1, "bm25": 0.1},
    )
    results = searcher.search(SearchQuery(raw_text="测试"), result_count=2)
    assert [item.image_id for item in results] == ["a", "b"]


class ComparisonService:
    def __init__(self):
        self.searcher = type(
            "Searcher",
            (),
            {"fusion_method": "rrf", "fusion_weights": {"image": 0.4}},
        )()

    def search(self, query: str, use_reranker: bool):
        order = ["a", "b"] if self.searcher.fusion_method == "rrf" else ["b", "c"]
        return [
            SearchResult(image_id=image_id, relative_path=f"Val/{image_id}.jpg", fused_score=1)
            for image_id in order
        ]


def test_comparison_reports_overlap_without_quality_claim():
    ticks = iter([0.0, 0.1, 0.1, 0.3])
    rows, summary = compare_fusion_methods(
        ComparisonService(),
        [{"query_id": "q1", "text": "城市", "category": "simple"}],
        top_k=2,
        clock=lambda: next(ticks),
    )
    assert rows[0]["top_k_overlap_rate"] == 0.5
    assert summary["quality_claim"] == "none_without_reviewed_relevance"
    assert summary["overall"]["mean_weighted_latency_seconds"] == pytest.approx(0.2)
