from __future__ import annotations

import math
from statistics import fmean

from anima_search.evaluation.metrics import ndcg_at_k, reciprocal_rank


def evaluate_rerank_orders(
    baseline_ids: list[str],
    pointwise_ids: list[str],
    listwise_ids: list[str],
    relevance: dict[str, int],
) -> dict[str, dict[str, float]]:
    expected = set(baseline_ids)
    if (
        len(expected) != len(baseline_ids)
        or set(pointwise_ids) != expected
        or set(listwise_ids) != expected
    ):
        raise ValueError(
            "all reranker variants must contain the same candidate IDs"
        )
    orders = {
        "baseline": baseline_ids,
        "pointwise": pointwise_ids,
        "listwise": listwise_ids,
    }
    return {
        method: {
            "mrr": reciprocal_rank(ids, relevance),
            "ndcg@10": ndcg_at_k(ids, relevance, 10),
        }
        for method, ids in orders.items()
    }


def rank_pointwise_scores(
    baseline_ids: list[str],
    scores: dict[str, float | None],
) -> list[str]:
    expected = set(baseline_ids)
    if len(expected) != len(baseline_ids) or set(scores) != expected:
        raise ValueError(
            "pointwise scores must contain every baseline candidate exactly once"
        )
    baseline_rank = {
        image_id: rank for rank, image_id in enumerate(baseline_ids)
    }

    def sort_key(image_id: str) -> tuple[int, float, int]:
        value = scores[image_id]
        if value is not None and math.isfinite(float(value)):
            return (0, -float(value), baseline_rank[image_id])
        return (1, 0.0, baseline_rank[image_id])

    return sorted(baseline_ids, key=sort_key)


def aggregate_rerank_quality(
    rows: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    return {
        method: {
            metric: fmean(row[method][metric] for row in rows)
            for metric in ("mrr", "ndcg@10")
        }
        for method in ("baseline", "pointwise", "listwise")
    }


def build_rerank_quality(
    *,
    baseline_by_query: dict[str, list[str]],
    pointwise_records: list[dict[str, object]],
    listwise_records: list[dict[str, object]],
    relevance: dict[str, dict[str, int]],
) -> dict[str, object]:
    """Align fixed-candidate rerank records and score only queries with qrels."""
    scored_query_ids = [
        query_id for query_id in baseline_by_query if query_id in relevance
    ]
    if not scored_query_ids:
        raise ValueError("no benchmark query IDs are present in relevance judgments")

    pointwise_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in pointwise_records:
        key = (str(record.get("query_id", "")), int(record.get("repeat", 0)))
        pointwise_by_key.setdefault(key, []).append(record)
    listwise_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for record in listwise_records:
        key = (str(record.get("query_id", "")), int(record.get("repeat", 0)))
        if key in listwise_by_key:
            raise ValueError(f"duplicate listwise quality record: {key}")
        listwise_by_key[key] = record

    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, dict[str, float]]] = []
    for query_id in scored_query_ids:
        baseline_ids = list(baseline_by_query[query_id])
        repeat_keys = sorted(
            [key for key in listwise_by_key if key[0] == query_id],
            key=lambda key: key[1],
        )
        if not repeat_keys:
            raise ValueError(f"no listwise quality records for {query_id}")
        for key in repeat_keys:
            point_records = pointwise_by_key.get(key, [])
            if len(point_records) != len(baseline_ids):
                raise ValueError(
                    f"pointwise quality records for {key} must contain "
                    f"{len(baseline_ids)} candidates"
                )
            scores: dict[str, float | None] = {}
            for record in point_records:
                image_id = str(record.get("image_id", ""))
                if image_id in scores:
                    raise ValueError(
                        f"duplicate pointwise quality record: {key}/{image_id}"
                    )
                raw_score = record.get("rerank_score")
                scores[image_id] = (
                    float(raw_score)
                    if bool(record.get("success")) and raw_score is not None
                    else None
                )
            pointwise_ids = rank_pointwise_scores(baseline_ids, scores)
            listwise_ids = [
                str(image_id)
                for image_id in listwise_by_key[key].get("ranked_image_ids", [])
            ]
            metrics = evaluate_rerank_orders(
                baseline_ids,
                pointwise_ids,
                listwise_ids,
                relevance[query_id],
            )
            metric_rows.append(metrics)
            rows.append(
                {
                    "query_id": query_id,
                    "repeat": key[1],
                    "baseline_image_ids": baseline_ids,
                    "pointwise_image_ids": pointwise_ids,
                    "listwise_image_ids": listwise_ids,
                    "metrics": metrics,
                }
            )
    return {
        "schema_version": "formal-a6-quality-v1.0",
        "scored_query_ids": scored_query_ids,
        "row_count": len(rows),
        "rows": rows,
        "summary": aggregate_rerank_quality(metric_rows),
    }
