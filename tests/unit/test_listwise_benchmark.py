from __future__ import annotations

from anima_search.evaluation.listwise_benchmark import benchmark_listwise_candidates
from anima_search.schemas import SearchResult


def _candidate(index: int) -> SearchResult:
    return SearchResult(
        image_id=f"val-{index}",
        relative_path=f"Val/{index}.jpg",
        fused_score=1.0 / index,
    )


class Reranker:
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        assert query == "城市"
        ranked = list(reversed(candidates))
        for rank, item in enumerate(ranked):
            item.rerank_score = 100 - rank
        return ranked


class FailingReranker:
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        for item in candidates:
            item.mismatch = ["视觉重排不可用：Listwise invalid JSON"]
        return candidates


class PartialFallbackReranker:
    last_degraded_reason = "appended missing IDs: ['val-2']"

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return candidates


class Probe:
    def start(self) -> None:
        pass

    def stop(self) -> int:
        return 4321


def test_listwise_benchmark_records_one_call_per_repeat():
    records, summary = benchmark_listwise_candidates(
        "城市",
        "val",
        [_candidate(1), _candidate(2), _candidate(3)],
        Reranker(),
        repeats=2,
        memory_probe=Probe(),
    )
    assert len(records) == 2
    assert records[0]["ranked_image_ids"] == ["val-3", "val-2", "val-1"]
    assert all(record["success"] for record in records)
    assert summary["model_calls"] == 2
    assert summary["top_k"] == 3
    assert summary["peak_cuda_memory_bytes"] == 4321
    assert summary["quality_claim"] == "not_evaluated_without_relevance_judgments"


def test_listwise_benchmark_detects_explicit_degradation():
    records, summary = benchmark_listwise_candidates(
        "公路",
        "val",
        [_candidate(1), _candidate(2)],
        FailingReranker(),
        repeats=1,
        memory_probe=Probe(),
    )
    assert not records[0]["success"]
    assert summary["failure_rate"] == 1.0


def test_listwise_benchmark_separates_partial_fallback_from_failure():
    records, summary = benchmark_listwise_candidates(
        "公路",
        "val",
        [_candidate(1), _candidate(2)],
        PartialFallbackReranker(),
        repeats=1,
        memory_probe=Probe(),
    )
    assert records[0]["success"]
    assert records[0]["degraded"]
    assert summary["failure_rate"] == 0.0
    assert summary["degraded_rate"] == 1.0
