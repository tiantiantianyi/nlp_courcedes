from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable


def _rank_shift(left: list[str], right: list[str]) -> float:
    common = set(left).intersection(right)
    if not common:
        return float(max(len(left), len(right)))
    left_rank = {image_id: rank for rank, image_id in enumerate(left, start=1)}
    right_rank = {image_id: rank for rank, image_id in enumerate(right, start=1)}
    return sum(abs(left_rank[item] - right_rank[item]) for item in common) / len(common)


def compare_fusion_methods(
    service: object,
    queries: list[dict[str, object]],
    *,
    top_k: int = 8,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not queries:
        raise ValueError("fusion comparison requires at least one query")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    searcher = service.searcher
    original_method = searcher.fusion_method
    rows: list[dict[str, object]] = []
    try:
        for query in queries:
            text = str(query.get("text", "")).strip()
            if not text:
                raise ValueError("fusion comparison query text must not be empty")
            rankings: dict[str, list[str]] = {}
            latencies: dict[str, float] = {}
            for method in ("rrf", "weighted"):
                searcher.fusion_method = method
                started = clock()
                results = service.search(text, use_reranker=False)
                latencies[method] = clock() - started
                rankings[method] = [item.image_id for item in results[:top_k]]
            denominator = max(1, min(top_k, len(set(rankings["rrf"] + rankings["weighted"]))))
            overlap_count = len(set(rankings["rrf"]).intersection(rankings["weighted"]))
            rows.append(
                {
                    "query_id": str(query.get("query_id", len(rows) + 1)),
                    "query": text,
                    "category": str(query.get("category", "unknown")),
                    "rrf_ids": rankings["rrf"],
                    "weighted_ids": rankings["weighted"],
                    "top_k_overlap_count": overlap_count,
                    "top_k_overlap_rate": overlap_count / denominator,
                    "mean_common_rank_shift": _rank_shift(
                        rankings["rrf"], rankings["weighted"]
                    ),
                    "rrf_latency_seconds": latencies["rrf"],
                    "weighted_latency_seconds": latencies["weighted"],
                }
            )
    finally:
        searcher.fusion_method = original_method

    categories: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        categories[str(row["category"])].append(row)

    def aggregate(selected: list[dict[str, object]]) -> dict[str, float | int]:
        return {
            "query_count": len(selected),
            "mean_top_k_overlap_rate": sum(
                float(row["top_k_overlap_rate"]) for row in selected
            ) / len(selected),
            "mean_common_rank_shift": sum(
                float(row["mean_common_rank_shift"]) for row in selected
            ) / len(selected),
            "mean_rrf_latency_seconds": sum(
                float(row["rrf_latency_seconds"]) for row in selected
            ) / len(selected),
            "mean_weighted_latency_seconds": sum(
                float(row["weighted_latency_seconds"]) for row in selected
            ) / len(selected),
        }

    summary = {
        "top_k": top_k,
        "weights": dict(searcher.fusion_weights),
        "quality_claim": "none_without_reviewed_relevance",
        "overall": aggregate(rows),
        "by_category": {
            category: aggregate(selected)
            for category, selected in sorted(categories.items())
        },
    }
    return rows, summary
