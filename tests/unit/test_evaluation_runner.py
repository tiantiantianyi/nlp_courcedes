from __future__ import annotations

import json
from pathlib import Path

import pytest

from anima_search.evaluation.runner import (
    evaluate_queries,
    validate_formal_queries,
    write_evaluation,
)
from anima_search.schemas import SearchResult


QUERIES = [
    {"query_id": "q1", "text": "城市", "category": "simple", "reviewed": True},
    {"query_id": "q2", "text": "雨夜", "category": "hard", "reviewed": True},
]
RELEVANCE = {"q1": {"a": 2}, "q2": {"b": 2}}


class FakeService:
    def search(self, query: str, use_reranker: bool):
        assert use_reranker is False
        if query == "雨夜":
            raise RuntimeError("index unavailable")
        return [SearchResult(image_id="a", relative_path="Val/a.jpg", fused_score=1.0)]


def test_evaluation_reports_percentiles_failures_and_categories():
    details, summary = evaluate_queries(FakeService(), QUERIES, RELEVANCE)
    assert len(details) == 2
    assert summary["overall"]["failure_rate"] == 0.5
    assert "latency_p50_seconds" in summary["overall"]
    assert "latency_p95_seconds" in summary["overall"]
    assert set(summary["by_category"]) == {"hard", "simple"}
    assert summary["by_category"]["simple"]["recall@1"] == 1.0
    assert summary["by_category"]["hard"]["failure_rate"] == 1.0


def test_evaluation_refuses_unreviewed_and_auto_seed_queries():
    with pytest.raises(ValueError, match="Evaluation refused"):
        validate_formal_queries(
            [{"query_id": "q1", "text": "x", "category": "auto_seed", "reviewed": True}]
        )
    with pytest.raises(ValueError, match="Evaluation refused"):
        validate_formal_queries(
            [{"query_id": "q2", "text": "x", "category": "simple", "reviewed": False}]
        )


def test_evaluation_requires_relevance_for_every_query():
    with pytest.raises(ValueError, match="no relevance judgments"):
        evaluate_queries(FakeService(), QUERIES, {"q1": {"a": 2}})


def test_evaluation_writes_json_csv_latex_and_failures(tmp_path: Path):
    details, summary = evaluate_queries(FakeService(), QUERIES, RELEVANCE)
    paths = write_evaluation(tmp_path, details, summary)
    assert set(paths) == {"json", "details_csv", "summary_csv", "latex", "failures"}
    assert all(path.is_file() for path in paths.values())
    assert "by_category" in json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "\\begin{tabular}" in paths["latex"].read_text(encoding="utf-8")
    assert "index unavailable" in paths["failures"].read_text(encoding="utf-8")
