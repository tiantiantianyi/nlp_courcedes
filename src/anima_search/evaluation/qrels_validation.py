from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from anima_search.evaluation.manual_set import (
    QUERY_CATEGORIES,
    validate_manual_set,
    write_relevance,
    write_tasks,
)


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _candidate_ids(pool_row: Mapping[str, object]) -> list[str]:
    return [
        str(candidate.get("image_id", ""))
        for candidate in list(pool_row.get("candidates", []))
    ]


def _normalize_relevance(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "query_id": str(row.get("query_id", "")),
        "image_id": str(row.get("image_id", "")),
        "relevance": int(row.get("relevance", -1)),
        "annotator": str(row.get("annotator", "")).strip(),
        "note": str(row.get("note", "")).strip(),
    }


def finalize_qrels(
    queries: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    pool: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Validate complete candidate qrels and combine them with source-only queries."""
    validate_manual_set(
        queries,
        source_rows,
        expected_count=len(queries),
    )
    query_by_id = {str(row["query_id"]): row for row in queries}
    query_order = {query_id: index for index, query_id in enumerate(query_by_id)}

    source_by_query: dict[str, list[dict[str, object]]] = {}
    for row in source_rows:
        source_by_query.setdefault(str(row.get("query_id", "")), []).append(row)
    for query_id, query in query_by_id.items():
        judgments = source_by_query.get(query_id, [])
        source_id = str(query["source_image_id"])
        if len(judgments) != 1 or str(judgments[0].get("image_id")) != source_id:
            raise ValueError(
                f"source relevance for {query_id} must contain only source image {source_id}"
            )
        if int(judgments[0].get("relevance", -1)) != 2:
            raise ValueError(f"source image {source_id} must retain grade 2")

    pool_by_query: dict[str, dict[str, object]] = {}
    for pool_row in pool:
        query_id = str(pool_row.get("query_id", ""))
        if query_id in pool_by_query:
            raise ValueError(f"duplicate candidate pool query: {query_id}")
        if query_id not in query_by_id:
            raise ValueError(f"candidate pool references unknown query: {query_id}")
        query = query_by_id[query_id]
        if str(pool_row.get("category")) != str(query.get("category")):
            raise ValueError(f"candidate pool category mismatch for {query_id}")
        if str(pool_row.get("source_image_id")) != str(query.get("source_image_id")):
            raise ValueError(f"candidate pool source image mismatch for {query_id}")
        candidate_ids = _candidate_ids(pool_row)
        if not candidate_ids or any(not image_id for image_id in candidate_ids):
            raise ValueError(f"candidate pool {query_id} has blank or no candidate IDs")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"candidate pool {query_id} contains duplicate candidate IDs")
        if str(query["source_image_id"]) not in candidate_ids:
            raise ValueError(f"candidate pool {query_id} omits its source image")
        pool_by_query[query_id] = pool_row

    graded_categories = {
        str(query_by_id[query_id]["category"]) for query_id in pool_by_query
    }
    missing_categories = sorted(set(QUERY_CATEGORIES) - graded_categories)
    if missing_categories:
        raise ValueError(
            f"graded subset category coverage is incomplete; missing: {missing_categories}"
        )

    candidate_by_query: dict[str, list[dict[str, object]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in candidate_rows:
        query_id = str(row.get("query_id", ""))
        image_id = str(row.get("image_id", ""))
        pair = (query_id, image_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate candidate judgment: {query_id}/{image_id}")
        seen_pairs.add(pair)
        if query_id not in pool_by_query:
            raise ValueError(f"extra candidate judgments for unknown pool query: {query_id}")
        try:
            grade = int(row.get("relevance", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid candidate relevance for {query_id}/{image_id}"
            ) from exc
        if grade not in {0, 1, 2}:
            raise ValueError(
                f"invalid candidate relevance for {query_id}/{image_id}: {grade}"
            )
        if not str(row.get("annotator", "")).strip():
            raise ValueError(f"candidate judgment {query_id}/{image_id} is missing annotator")
        if not _is_true(row.get("reviewed")):
            raise ValueError(f"candidate judgment {query_id}/{image_id} is not reviewed")
        candidate_by_query.setdefault(query_id, []).append(row)

    final_rows: list[dict[str, object]] = []
    for query_id, query in query_by_id.items():
        if query_id not in pool_by_query:
            final_rows.append(_normalize_relevance(source_by_query[query_id][0]))
            continue
        expected_ids = _candidate_ids(pool_by_query[query_id])
        judgments = candidate_by_query.get(query_id, [])
        actual_ids = {str(row.get("image_id", "")) for row in judgments}
        missing = sorted(set(expected_ids) - actual_ids)
        extra = sorted(actual_ids - set(expected_ids))
        if missing:
            raise ValueError(f"missing candidate judgments for {query_id}: {missing}")
        if extra:
            raise ValueError(f"extra candidate judgments for {query_id}: {extra}")
        by_image = {str(row["image_id"]): row for row in judgments}
        source_id = str(query["source_image_id"])
        if int(by_image[source_id].get("relevance", -1)) != 2:
            raise ValueError(f"source image {source_id} must retain grade 2")
        final_rows.extend(
            _normalize_relevance(by_image[image_id]) for image_id in expected_ids
        )

    final_rows.sort(
        key=lambda row: (
            query_order[str(row["query_id"])],
            _candidate_ids(pool_by_query[str(row["query_id"])]).index(
                str(row["image_id"])
            )
            if str(row["query_id"]) in pool_by_query
            else 0,
        )
    )
    grade_counts = Counter(int(row["relevance"]) for row in final_rows)
    summary: dict[str, object] = {
        "schema_version": "formal-qrels-validation-v1.0",
        "valid": True,
        "query_count": len(queries),
        "relevance_row_count": len(final_rows),
        "single_positive_query_count": len(queries) - len(pool_by_query),
        "graded_pool_query_count": len(pool_by_query),
        "graded_query_ids": [
            query_id for query_id in query_by_id if query_id in pool_by_query
        ],
        "single_positive_query_ids": [
            query_id
            for query_id in query_by_id
            if query_id not in pool_by_query
        ],
        "graded_category_counts": {
            category: sum(
                str(query_by_id[query_id]["category"]) == category
                for query_id in pool_by_query
            )
            for category in QUERY_CATEGORIES
        },
        "grade_counts": {str(grade): grade_counts.get(grade, 0) for grade in (0, 1, 2)},
        "annotators": sorted(
            {
                str(row["annotator"])
                for row in final_rows
                if str(row.get("annotator", "")).strip()
            }
        ),
    }
    return [dict(row) for row in queries], final_rows, summary


def write_finalized_qrels(
    output_dir: Path,
    queries: list[dict[str, object]],
    relevance_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> dict[str, Path]:
    paths = {
        "queries": output_dir / "val_queries.jsonl",
        "relevance": output_dir / "val_relevance.csv",
        "validation": output_dir / "qrels_validation.json",
    }
    write_tasks(paths["queries"], queries)
    write_relevance(paths["relevance"], relevance_rows)
    temporary = paths["validation"].with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(paths["validation"])
    return paths
