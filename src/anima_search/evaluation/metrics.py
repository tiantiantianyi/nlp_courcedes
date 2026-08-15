from __future__ import annotations

import math
from statistics import mean


def recall_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = {image_id for image_id, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevance: dict[str, int]) -> float:
    for rank, image_id in enumerate(ranked_ids, start=1):
        if relevance.get(image_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_ids: list[str], relevance: dict[str, int]) -> float:
    relevant_count = sum(grade > 0 for grade in relevance.values())
    if relevant_count == 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, image_id in enumerate(ranked_ids, start=1):
        if relevance.get(image_id, 0) > 0:
            hits += 1
            score += hits / rank
    return score / relevant_count


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    gains = [relevance.get(image_id, 0) for image_id in ranked_ids[:k]]
    dcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], percentage: float) -> float:
    """Return a linearly interpolated percentile, including one-value samples."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_query_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: mean(row[key] for row in rows) for key in rows[0]}
