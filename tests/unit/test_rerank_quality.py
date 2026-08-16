from __future__ import annotations

import pytest

from anima_search.evaluation.rerank_quality import (
    aggregate_rerank_quality,
    build_rerank_quality,
    evaluate_rerank_orders,
    rank_pointwise_scores,
)


def test_evaluate_rerank_orders_scores_each_order() -> None:
    metrics = evaluate_rerank_orders(
        ["a", "b", "c"],
        ["b", "a", "c"],
        ["c", "b", "a"],
        {"b": 2, "c": 1},
    )

    assert metrics["baseline"]["mrr"] == 0.5
    assert metrics["pointwise"]["mrr"] == 1.0
    assert metrics["listwise"]["mrr"] == 1.0
    assert metrics["baseline"]["ndcg@10"] < metrics["pointwise"]["ndcg@10"]


def test_evaluate_rerank_orders_rejects_different_candidate_sets() -> None:
    with pytest.raises(
        ValueError,
        match="all reranker variants must contain the same candidate IDs",
    ):
        evaluate_rerank_orders(
            ["a", "b", "c"],
            ["b", "a", "c"],
            ["c", "b", "different"],
            {"b": 2},
        )


def test_rank_pointwise_scores_uses_baseline_order_for_ties_and_failures() -> None:
    ranked = rank_pointwise_scores(
        ["a", "b", "c", "d"],
        {"a": 1.0, "b": 2.0, "c": 2.0, "d": float("nan")},
    )

    assert ranked == ["b", "c", "a", "d"]


def test_aggregate_rerank_quality_means_query_repeat_metrics() -> None:
    summary = aggregate_rerank_quality(
        [
            {
                "baseline": {"mrr": 0.5, "ndcg@10": 0.4},
                "pointwise": {"mrr": 1.0, "ndcg@10": 0.8},
                "listwise": {"mrr": 1.0, "ndcg@10": 0.9},
            },
            {
                "baseline": {"mrr": 1.0, "ndcg@10": 0.6},
                "pointwise": {"mrr": 0.5, "ndcg@10": 0.7},
                "listwise": {"mrr": 0.5, "ndcg@10": 0.5},
            },
        ]
    )

    assert summary == {
        "baseline": {"mrr": 0.75, "ndcg@10": 0.5},
        "pointwise": {"mrr": 0.75, "ndcg@10": 0.75},
        "listwise": {"mrr": 0.75, "ndcg@10": 0.7},
    }


def test_build_rerank_quality_groups_repeats_and_scores_only_qrels_queries() -> None:
    payload = build_rerank_quality(
        baseline_by_query={"q1": ["a", "b", "c"], "q2": ["d", "e", "f"]},
        pointwise_records=[
            {"query_id": "q1", "repeat": 1, "image_id": "a", "rerank_score": 1.0, "success": True},
            {"query_id": "q1", "repeat": 1, "image_id": "b", "rerank_score": 2.0, "success": True},
            {"query_id": "q1", "repeat": 1, "image_id": "c", "rerank_score": 99.0, "success": False},
            {"query_id": "q2", "repeat": 1, "image_id": "d", "rerank_score": 3.0, "success": True},
            {"query_id": "q2", "repeat": 1, "image_id": "e", "rerank_score": 2.0, "success": True},
            {"query_id": "q2", "repeat": 1, "image_id": "f", "rerank_score": 1.0, "success": True},
        ],
        listwise_records=[
            {"query_id": "q1", "repeat": 1, "ranked_image_ids": ["c", "b", "a"]},
            {"query_id": "q2", "repeat": 1, "ranked_image_ids": ["f", "e", "d"]},
        ],
        relevance={"q1": {"b": 2, "c": 1}},
    )

    assert payload["scored_query_ids"] == ["q1"]
    assert payload["row_count"] == 1
    assert payload["rows"][0]["pointwise_image_ids"] == ["b", "a", "c"]
    assert set(payload["summary"]) == {"baseline", "pointwise", "listwise"}


def test_build_rerank_quality_rejects_listwise_candidate_set_change() -> None:
    with pytest.raises(ValueError, match="same candidate IDs"):
        build_rerank_quality(
            baseline_by_query={"q1": ["a", "b"]},
            pointwise_records=[
                {"query_id": "q1", "repeat": 1, "image_id": "a", "rerank_score": 1.0, "success": True},
                {"query_id": "q1", "repeat": 1, "image_id": "b", "rerank_score": 0.0, "success": True},
            ],
            listwise_records=[
                {
                    "query_id": "q1",
                    "repeat": 1,
                    "ranked_image_ids": ["a", "other"],
                }
            ],
            relevance={"q1": {"a": 2}},
        )
