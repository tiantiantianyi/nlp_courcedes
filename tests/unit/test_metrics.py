from anima_search.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_on_known_ranking():
    ranked = ["a", "b", "c"]; relevance = {"b": 2, "c": 1}
    assert recall_at_k(ranked, relevance, 2) == 0.5
    assert reciprocal_rank(ranked, relevance) == 0.5
    assert 0 < ndcg_at_k(ranked, relevance, 3) < 1
