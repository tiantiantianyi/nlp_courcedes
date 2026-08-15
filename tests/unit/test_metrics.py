import pytest

from anima_search.evaluation.metrics import (
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics_on_known_ranking():
    ranked = ["a", "b", "c"]
    relevance = {"b": 2, "c": 1}
    assert recall_at_k(ranked, relevance, 2) == 0.5
    assert reciprocal_rank(ranked, relevance) == 0.5
    assert 0 < ndcg_at_k(ranked, relevance, 3) < 1


def test_percentile_interpolates_and_handles_single_value():
    assert percentile([1.0], 95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([0.0, 10.0], 95) == pytest.approx(9.5)


def test_percentile_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 50)
