from __future__ import annotations

import json
from pathlib import Path

import pytest

from anima_search.evaluation.rerank_benchmark import (
    benchmark_candidates,
    collect_candidates,
    write_benchmark,
)
from anima_search.schemas import SearchResult


def _candidate(index: int) -> SearchResult:
    return SearchResult(
        image_id=f"val-{index}",
        relative_path=f"Val/{index}.jpg",
        fused_score=1.0 / index,
    )


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def search(self, query: str, use_reranker: bool) -> list[SearchResult]:
        self.calls.append((query, use_reranker))
        return [_candidate(index) for index in range(1, 6)]


class FakeReranker:
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        assert len(candidates) == 1
        candidates[0].rerank_score = 80.0
        return candidates


class FailingReranker:
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        raise RuntimeError("offline")


class FakeProbe:
    def start(self) -> None:
        pass

    def stop(self) -> int:
        return 1234


def test_collect_candidates_explicitly_disables_reranking():
    service = FakeService()
    candidates = collect_candidates(service, "雨夜街道", 3)
    assert [item.image_id for item in candidates] == ["val-1", "val-2", "val-3"]
    assert service.calls == [("雨夜街道", False)]


def test_pointwise_benchmark_records_every_candidate_and_repeat():
    candidates = [_candidate(index) for index in range(1, 4)]
    records, summary = benchmark_candidates(
        "公路",
        "val",
        candidates,
        FakeReranker(),
        repeats=2,
        memory_probe=FakeProbe(),
    )
    assert len(records) == 6
    assert all(record["success"] for record in records)
    assert all(record["rerank_score"] == 80.0 for record in records)
    assert summary["top_k"] == 3
    assert summary["failure_rate"] == 0.0
    assert summary["peak_cuda_memory_bytes"] == 1234
    assert summary["quality_claim"] == "not_evaluated_without_relevance_judgments"


def test_pointwise_benchmark_keeps_running_after_failure():
    records, summary = benchmark_candidates(
        "公路",
        "val",
        [_candidate(1), _candidate(2)],
        FailingReranker(),
        repeats=1,
        memory_probe=FakeProbe(),
    )
    assert len(records) == 2
    assert summary["failure_count"] == 2
    assert summary["failure_rate"] == 1.0
    assert records[0]["error"] == "RuntimeError: offline"


def test_write_benchmark_outputs_jsonl_and_summary(tmp_path: Path):
    output = tmp_path / "benchmark.jsonl"
    records = [{"image_id": "val-1", "success": True}]
    summary_path = write_benchmark(output, records, {"candidate_runs": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == records[0]
    assert summary_path.name == "benchmark.summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"candidate_runs": 1}


def test_collect_candidates_rejects_too_small_result_set():
    service = FakeService()
    with pytest.raises(ValueError, match="returned only 5"):
        collect_candidates(service, "公路", 6)
