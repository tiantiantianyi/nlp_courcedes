from __future__ import annotations

import json
import sys
from contextlib import contextmanager


from pathlib import Path

import pytest

from anima_search.evaluation.listwise_benchmark import benchmark_listwise_candidates
from anima_search.evaluation.manual_set import write_relevance, write_tasks
from anima_search.schemas import SearchResult
from scripts import benchmark_listwise_top20


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


def test_quality_output_and_graded_only_require_relevance() -> None:
    with pytest.raises(ValueError, match="--relevance is required"):
        benchmark_listwise_top20._validate_quality_options(
            relevance=None,
            quality_output=Path("quality.json"),
            graded_only=False,
        )
    with pytest.raises(ValueError, match="--relevance is required"):
        benchmark_listwise_top20._validate_quality_options(
            relevance=None,
            quality_output=None,
            graded_only=True,
        )


def test_graded_only_selects_explicit_validation_query_ids() -> None:
    queries = [
        {"query_id": "q1", "text": "城市", "category": "simple"},
        {"query_id": "q2", "text": "文字", "category": "ocr"},
        {"query_id": "q3", "text": "公园", "category": "simple"},
    ]

    selected = benchmark_listwise_top20._select_benchmark_queries(
        queries,
        relevance={"q1": {"a": 2}, "q2": {"b": 2}, "q3": {"c": 2}},
        graded_only=True,
        qrels_validation={"graded_query_ids": ["q2", "q3"]},
        query_limit=50,
    )

    assert [row["query_id"] for row in selected] == ["q2", "q3"]

    with pytest.raises(ValueError, match="graded_query_ids"):
        benchmark_listwise_top20._select_benchmark_queries(
            queries,
            relevance={"q1": {"a": 2}},
            graded_only=True,
            qrels_validation={},
            query_limit=50,
        )


def test_top20_cli_writes_operational_and_quality_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_path = tmp_path / "queries.jsonl"
    relevance_path = tmp_path / "relevance.csv"
    output_path = tmp_path / "operational.json"
    quality_path = tmp_path / "quality.json"
    prompt_path = tmp_path / "configs" / "prompts" / "reranker_listwise.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("rank candidates", encoding="utf-8")
    write_tasks(
        query_path,
        [{"query_id": "q1", "text": "城市", "category": "simple"}],
    )
    write_relevance(
        relevance_path,
        [
            {"query_id": "q1", "image_id": "val-1", "relevance": 2, "annotator": "甲", "note": ""},
            {"query_id": "q1", "image_id": "val-2", "relevance": 0, "annotator": "甲", "note": ""},
        ],
    )

    class Manager:
        @contextmanager
        def qwen_session(self):
            yield object()

    class Service:
        config = {
            "project_root": str(tmp_path),
            "retrieval": {
                "candidate_count": 2,
                "result_count": 2,
                "rrf_weight": 0.35,
                "vlm_weight": 0.65,
                "rerank_max_new_tokens": 32,
                "rerank_listwise_prompt": "configs/prompts/reranker_listwise.txt",
                "rerank_listwise_max_new_tokens": 32,
                "rerank_listwise_columns": 2,
                "rerank_listwise_tile_size": 64,
            },
        }
        manager = Manager()
        reranker_prompt = "score"

        def search(self, query: str, use_reranker: bool = False):
            return [_candidate(1), _candidate(2)]

        def release_retrieval_encoders(self) -> list[str]:
            return ["image"]

    class Pointwise:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def rerank(self, _query: str, candidates: list[SearchResult]):
            for candidate in candidates:
                candidate.rerank_score = 2.0 if candidate.image_id == "val-2" else 1.0
            return candidates

    class Listwise:
        last_contact_sheet_size = (128, 64)
        last_degraded_reason = None

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def rerank(self, _query: str, candidates: list[SearchResult]):
            return list(reversed(candidates))

    monkeypatch.setattr(benchmark_listwise_top20, "create_service", lambda *_: Service())
    monkeypatch.setattr(benchmark_listwise_top20, "VisualReranker", Pointwise)
    monkeypatch.setattr(benchmark_listwise_top20, "ListwiseVisualReranker", Listwise)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_listwise_top20.py",
            "--queries",
            str(query_path),
            "--relevance",
            str(relevance_path),
            "--quality-output",
            str(quality_path),
            "--top-k",
            "2",
            "--query-limit",
            "1",
            "--repeats",
            "1",
            "--output",
            str(output_path),
        ],
    )

    benchmark_listwise_top20.main()

    operational = json.loads(output_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert operational["summary"]["quality_claim"] == "evaluated_with_relevance_judgments"
    assert operational["per_query"][0]["baseline_image_ids"] == ["val-1", "val-2"]
    assert quality["row_count"] == 1
    assert quality["rows"][0]["pointwise_image_ids"] == ["val-2", "val-1"]
