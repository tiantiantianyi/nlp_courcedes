from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Callable, Protocol

from anima_search.schemas import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]: ...


class CudaMemoryProbe:
    """Best-effort CUDA peak-memory measurement without making torch mandatory."""

    def __init__(self) -> None:
        try:
            import torch

            self._cuda = torch.cuda if torch.cuda.is_available() else None
        except (ImportError, RuntimeError):
            self._cuda = None

    def start(self) -> None:
        if self._cuda is not None:
            self._cuda.synchronize()
            self._cuda.reset_peak_memory_stats()

    def stop(self) -> int | None:
        if self._cuda is None:
            return None
        self._cuda.synchronize()
        return int(self._cuda.max_memory_allocated())


def collect_candidates(service: object, query: str, top_k: int) -> list[SearchResult]:
    """Retrieve a fixed candidate set without invoking the visual reranker."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    results = service.search(query, use_reranker=False)
    candidates = list(results[:top_k])
    if len(candidates) < top_k:
        raise ValueError(
            f"retrieval returned only {len(candidates)} candidates; top_k={top_k} was requested"
        )
    return candidates


def _is_failure(result: SearchResult) -> bool:
    return any(message.startswith("视觉重排不可用：") for message in result.mismatch)


def benchmark_candidates(
    query: str,
    split: str,
    candidates: list[SearchResult],
    reranker: Reranker,
    repeats: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
    memory_probe: CudaMemoryProbe | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Benchmark pointwise reranking; relevance quality is intentionally not inferred."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not candidates:
        raise ValueError("at least one candidate is required")

    probe = memory_probe or CudaMemoryProbe()
    records: list[dict[str, object]] = []
    repeat_totals_ms: list[float] = []

    for repeat_index in range(repeats):
        repeat_started = clock()
        for candidate_rank, source in enumerate(candidates, start=1):
            candidate = source.model_copy(deep=True)
            probe.start()
            started = clock()
            error: str | None = None
            try:
                reranked = reranker.rerank(query, [candidate])
                if len(reranked) != 1:
                    raise RuntimeError(
                        f"reranker returned {len(reranked)} results for one candidate"
                    )
                result = reranked[0]
            except Exception as exc:  # benchmark must preserve the remaining runs
                result = candidate
                result.rerank_score = 0.0
                error = f"{type(exc).__name__}: {exc}"
            peak_memory = probe.stop()
            latency_ms = (clock() - started) * 1000.0
            failed = error is not None or _is_failure(result)
            if error is None and failed:
                error = next(
                    message for message in result.mismatch
                    if message.startswith("视觉重排不可用：")
                )
            records.append(
                {
                    "query": query,
                    "split": split,
                    "repeat": repeat_index + 1,
                    "candidate_rank": candidate_rank,
                    "image_id": source.image_id,
                    "relative_path": source.relative_path,
                    "latency_ms": round(latency_ms, 3),
                    "success": not failed,
                    "error": error,
                    "rerank_score": result.rerank_score,
                    "peak_cuda_memory_bytes": peak_memory,
                }
            )
        repeat_totals_ms.append((clock() - repeat_started) * 1000.0)

    latencies = [float(record["latency_ms"]) for record in records]
    failures = sum(not bool(record["success"]) for record in records)
    peaks = [
        int(record["peak_cuda_memory_bytes"])
        for record in records
        if record["peak_cuda_memory_bytes"] is not None
    ]
    summary: dict[str, object] = {
        "query": query,
        "split": split,
        "top_k": len(candidates),
        "repeats": repeats,
        "candidate_runs": len(records),
        "failure_count": failures,
        "failure_rate": failures / len(records),
        "mean_candidate_latency_ms": statistics.fmean(latencies),
        "total_candidate_latency_ms": sum(latencies),
        "repeat_total_latency_ms": [round(value, 3) for value in repeat_totals_ms],
        "peak_cuda_memory_bytes": max(peaks) if peaks else None,
        "quality_claim": "not_evaluated_without_relevance_judgments",
    }
    return records, summary


def write_benchmark(
    output_path: Path,
    records: list[dict[str, object]],
    summary: dict[str, object],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path
