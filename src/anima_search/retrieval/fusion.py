from __future__ import annotations

from collections.abc import Mapping
from collections import defaultdict


def reciprocal_rank_fusion_with_ranks(rankings: dict[str, list[tuple[str, float]]], k: int = 60):
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    fused: dict[str, float] = defaultdict(float)
    branches: dict[str, dict[str, float]] = defaultdict(dict)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, results in rankings.items():
        seen: set[str] = set()
        for rank, (image_id, original_score) in enumerate(results, start=1):
            if image_id in seen:
                continue
            seen.add(image_id)
            fused[image_id] += 1.0 / (k + rank)
            branches[image_id][name] = original_score
            ranks[image_id][name] = rank
    return [(image_id, fused[image_id], branches[image_id], ranks[image_id])
            for image_id in sorted(fused, key=lambda key: (-fused[key], key))]


def reciprocal_rank_fusion(rankings: dict[str, list[tuple[str, float]]], k: int = 60):
    return [(image_id, score, branches)
            for image_id, score, branches, _ in reciprocal_rank_fusion_with_ranks(rankings, k)]


def _minmax_scores(results: list[tuple[str, float]]) -> dict[str, float]:
    deduplicated: dict[str, float] = {}
    for image_id, score in results:
        deduplicated.setdefault(image_id, float(score))
    if not deduplicated:
        return {}
    minimum = min(deduplicated.values())
    maximum = max(deduplicated.values())
    if maximum == minimum:
        return {image_id: 1.0 for image_id in deduplicated}
    scale = maximum - minimum
    return {
        image_id: (score - minimum) / scale
        for image_id, score in deduplicated.items()
    }


def normalized_weighted_fusion_with_ranks(
    rankings: dict[str, list[tuple[str, float]]],
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float, dict[str, float], dict[str, int]]]:
    """Fuse heterogeneous branch scores after per-query min-max normalization."""
    configured = dict(weights or {})
    unknown = sorted(set(configured) - set(rankings))
    if unknown:
        raise ValueError(f"fusion weights reference inactive branches: {unknown}")
    active_weights = {
        name: float(configured.get(name, 1.0))
        for name in rankings
    }
    if any(weight < 0 for weight in active_weights.values()):
        raise ValueError("fusion weights must be non-negative")
    weight_sum = sum(active_weights.values())
    if weight_sum <= 0:
        raise ValueError("at least one active fusion weight must be positive")

    fused: dict[str, float] = defaultdict(float)
    branches: dict[str, dict[str, float]] = defaultdict(dict)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, results in rankings.items():
        normalized = _minmax_scores(results)
        seen: set[str] = set()
        for rank, (image_id, original_score) in enumerate(results, start=1):
            if image_id in seen:
                continue
            seen.add(image_id)
            fused[image_id] += active_weights[name] * normalized[image_id] / weight_sum
            branches[image_id][name] = float(original_score)
            ranks[image_id][name] = rank
    return [
        (image_id, fused[image_id], branches[image_id], ranks[image_id])
        for image_id in sorted(fused, key=lambda key: (-fused[key], key))
    ]


def normalized_weighted_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    weights: Mapping[str, float] | None = None,
):
    return [
        (image_id, score, branches)
        for image_id, score, branches, _ in normalized_weighted_fusion_with_ranks(
            rankings, weights
        )
    ]
