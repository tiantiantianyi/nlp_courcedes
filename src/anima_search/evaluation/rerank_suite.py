from __future__ import annotations

import json
import statistics
from pathlib import Path

from anima_search.evaluation.metrics import percentile


def load_operational_queries(path: Path) -> list[dict[str, str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("M6 benchmark query file is empty")
    required = {"query_id", "text", "category"}
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"query row {index} is missing fields: {sorted(missing)}")
        normalized.append({name: str(row[name]).strip() for name in required})
    ids = [row["query_id"] for row in normalized]
    duplicates = sorted({query_id for query_id in ids if ids.count(query_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate M6 query IDs: {duplicates}")
    if any(not row["text"] for row in normalized):
        raise ValueError("M6 benchmark queries must not be empty")
    return normalized


def annotate_suite_records(
    records: list[dict[str, object]],
    *,
    query_id: str,
    category: str,
    top_k: int,
    starting_run: int,
) -> list[dict[str, object]]:
    enriched = []
    for offset, record in enumerate(records):
        global_run = starting_run + offset
        enriched.append(
            {
                **record,
                "query_id": query_id,
                "category": category,
                "top_k": top_k,
                "global_run": global_run,
                "startup_phase": "cold" if global_run == 1 else "warm",
            }
        )
    return enriched


def _latency_summary(records: list[dict[str, object]]) -> dict[str, float | int | None]:
    if not records:
        return {
            "candidate_runs": 0,
            "failure_count": 0,
            "failure_rate": 0.0,
            "latency_mean_ms": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
        }
    latencies = [float(row["latency_ms"]) for row in records]
    failures = sum(not bool(row["success"]) for row in records)
    return {
        "candidate_runs": len(records),
        "failure_count": failures,
        "failure_rate": failures / len(records),
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }


def summarize_suite(
    records: list[dict[str, object]],
    collection_failures: list[dict[str, object]],
    *,
    query_count: int,
    top_k_values: list[int],
    repeats: int,
) -> dict[str, object]:
    cold = [row for row in records if row["startup_phase"] == "cold"]
    warm = [row for row in records if row["startup_phase"] == "warm"]
    peaks = [
        int(row["peak_cuda_memory_bytes"])
        for row in records
        if row.get("peak_cuda_memory_bytes") is not None
    ]
    categories = sorted({str(row["category"]) for row in records})
    return {
        "query_count": query_count,
        "top_k_values": top_k_values,
        "repeats": repeats,
        **_latency_summary(records),
        "cold_start_latency_ms": float(cold[0]["latency_ms"]) if cold else None,
        "warm": _latency_summary(warm),
        "peak_cuda_memory_bytes": max(peaks) if peaks else None,
        "collection_failure_count": len(collection_failures),
        "collection_failures": collection_failures,
        "by_top_k": {
            str(top_k): _latency_summary(
                [row for row in records if int(row["top_k"]) == top_k]
            )
            for top_k in top_k_values
        },
        "by_category": {
            category: _latency_summary(
                [row for row in records if str(row["category"]) == category]
            )
            for category in categories
        },
        "quality_claim": "not_evaluated_without_relevance_judgments",
    }
