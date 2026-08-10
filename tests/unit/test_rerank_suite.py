from __future__ import annotations

import json
from pathlib import Path

import pytest

from anima_search.evaluation.rerank_suite import (
    annotate_suite_records,
    load_operational_queries,
    summarize_suite,
)


def test_load_operational_queries_checks_required_fields_and_duplicates(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "text": "城市", "category": "simple"}) + "\n",
        encoding="utf-8",
    )
    assert load_operational_queries(path)[0]["text"] == "城市"

    path.write_text(
        "\n".join(
            json.dumps({"query_id": "q1", "text": "城市", "category": "simple"})
            for _ in range(2)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_operational_queries(path)


def test_suite_marks_only_global_first_candidate_as_cold():
    source = [
        {"latency_ms": 100.0, "success": True, "peak_cuda_memory_bytes": 10},
        {"latency_ms": 20.0, "success": True, "peak_cuda_memory_bytes": 12},
    ]
    first = annotate_suite_records(
        source,
        query_id="q1",
        category="simple",
        top_k=3,
        starting_run=1,
    )
    second = annotate_suite_records(
        source,
        query_id="q2",
        category="negative",
        top_k=5,
        starting_run=3,
    )
    assert [row["startup_phase"] for row in first] == ["cold", "warm"]
    assert all(row["startup_phase"] == "warm" for row in second)


def test_suite_summary_separates_cold_warm_topk_and_failures():
    records = [
        {
            "latency_ms": 100.0,
            "success": True,
            "peak_cuda_memory_bytes": 10,
            "startup_phase": "cold",
            "top_k": 3,
            "category": "simple",
        },
        {
            "latency_ms": 20.0,
            "success": False,
            "peak_cuda_memory_bytes": 12,
            "startup_phase": "warm",
            "top_k": 5,
            "category": "negative",
        },
    ]
    summary = summarize_suite(
        records,
        [{"query_id": "q3", "error": "no candidates"}],
        query_count=3,
        top_k_values=[3, 5],
        repeats=1,
    )
    assert summary["cold_start_latency_ms"] == 100.0
    assert summary["warm"]["latency_mean_ms"] == 20.0
    assert summary["failure_rate"] == 0.5
    assert summary["collection_failure_count"] == 1
    assert summary["by_top_k"]["3"]["candidate_runs"] == 1
    assert summary["peak_cuda_memory_bytes"] == 12
