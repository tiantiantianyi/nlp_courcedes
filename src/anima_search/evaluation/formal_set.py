from __future__ import annotations

import re
from pathlib import Path

from anima_search.evaluation.manual_set import (
    load_relevance_rows,
    load_tasks,
    validate_manual_set,
)


_QUERY_NUMBER = re.compile(r"^(.*?)(\d+)$")


def _query_sort_key(query_id: str) -> tuple[str, int, str]:
    match = _QUERY_NUMBER.fullmatch(query_id)
    if match is None:
        return query_id, -1, query_id
    return match.group(1), int(match.group(2)), query_id


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates, key=_query_sort_key)


def merge_reviewed_sets(
    query_paths: list[Path],
    relevance_paths: list[Path],
    *,
    expected_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Load, validate, and merge reviewed query sets without modifying inputs."""
    if len(query_paths) != len(relevance_paths):
        raise ValueError("query and relevance path counts must match")
    if not query_paths:
        raise ValueError("at least one reviewed set is required")

    tasks: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for query_path, relevance_path in zip(query_paths, relevance_paths, strict=True):
        batch_tasks = load_tasks(query_path)
        batch_rows = load_relevance_rows(relevance_path)
        validate_manual_set(
            batch_tasks,
            batch_rows,
            expected_count=len(batch_tasks),
        )
        tasks.extend(batch_tasks)
        rows.extend(batch_rows)

    duplicate_queries = _duplicates(
        [str(task.get("query_id", "")) for task in tasks]
    )
    if duplicate_queries:
        raise ValueError(f"duplicate query IDs across reviewed sets: {duplicate_queries}")

    duplicate_sources = _duplicates(
        [str(task.get("source_image_id", "")) for task in tasks]
    )
    if duplicate_sources:
        raise ValueError(
            f"duplicate source image IDs across reviewed sets: {duplicate_sources}"
        )

    summary = validate_manual_set(tasks, rows, expected_count=expected_count)
    tasks.sort(key=lambda task: _query_sort_key(str(task["query_id"])))
    query_order = {
        str(task["query_id"]): index for index, task in enumerate(tasks)
    }
    rows.sort(
        key=lambda row: (
            query_order.get(str(row.get("query_id", "")), len(query_order)),
            str(row.get("image_id", "")),
        )
    )
    summary = {
        **summary,
        "query_paths": [str(path) for path in query_paths],
        "relevance_paths": [str(path) for path in relevance_paths],
    }
    return tasks, rows, summary
