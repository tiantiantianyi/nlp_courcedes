from __future__ import annotations

import statistics
import time
from collections.abc import Callable

from anima_search.evaluation.rerank_benchmark import CudaMemoryProbe
from anima_search.schemas import SearchResult


def _failure_message(results: list[SearchResult]) -> str | None:
    for result in results:
        for message in result.mismatch:
            if message.startswith("视觉重排不可用："):
                return message
    return None


def benchmark_listwise_candidates(
    query: str,
    split: str,
    candidates: list[SearchResult],
    reranker: object,
    repeats: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
    memory_probe: CudaMemoryProbe | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Benchmark one listwise call per candidate set without inferring quality."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len(candidates) > 20:
        raise ValueError("listwise benchmark supports at most 20 candidates")

    expected_ids = {item.image_id for item in candidates}
    probe = memory_probe or CudaMemoryProbe()
    records: list[dict[str, object]] = []

    for repeat_index in range(repeats):
        working = [item.model_copy(deep=True) for item in candidates]
        probe.start()
        started = clock()
        error: str | None = None
        try:
            ranked = reranker.rerank(query, working)
            ranked_ids = [item.image_id for item in ranked]
            if len(ranked_ids) != len(candidates) or set(ranked_ids) != expected_ids:
                raise RuntimeError(
                    "listwise reranker must return every input candidate exactly once"
                )
            error = _failure_message(ranked)
        except Exception as exc:
            ranked = working
            ranked_ids = [item.image_id for item in ranked]
            error = f"{type(exc).__name__}: {exc}"
        degraded_reason = getattr(
            reranker, "last_degraded_reason", None
        )
        peak_memory = probe.stop()
        latency_ms = (clock() - started) * 1000.0
        records.append(
            {
                "method": "listwise",
                "query": query,
                "split": split,
                "repeat": repeat_index + 1,
                "top_k": len(candidates),
                "latency_ms": round(latency_ms, 3),
                "success": error is None,
                "error": error,
                "degraded": degraded_reason is not None,
                "degraded_reason": degraded_reason,
                "ranked_image_ids": ranked_ids,
                "rerank_scores": {
                    item.image_id: item.rerank_score for item in ranked
                },
                "peak_cuda_memory_bytes": peak_memory,
            }
        )

    latencies = [float(record["latency_ms"]) for record in records]
    failures = sum(not bool(record["success"]) for record in records)
    degradations = sum(bool(record["degraded"]) for record in records)
    peaks = [
        int(record["peak_cuda_memory_bytes"])
        for record in records
        if record["peak_cuda_memory_bytes"] is not None
    ]
    summary: dict[str, object] = {
        "method": "listwise",
        "query": query,
        "split": split,
        "top_k": len(candidates),
        "repeats": repeats,
        "model_calls": len(records),
        "failure_count": failures,
        "failure_rate": failures / len(records),
        "mean_query_latency_ms": statistics.fmean(latencies),
        "degraded_count": degradations,
        "degraded_rate": degradations / len(records),
        "total_query_latency_ms": sum(latencies),
        "peak_cuda_memory_bytes": max(peaks) if peaks else None,
        "quality_claim": "not_evaluated_without_relevance_judgments",
    }
    return records, summary
