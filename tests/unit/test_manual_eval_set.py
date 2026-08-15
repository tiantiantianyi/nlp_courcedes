from __future__ import annotations

from pathlib import Path

import pytest

from anima_search.evaluation.manual_set import (
    load_relevance_rows,
    load_tasks,
    parse_judgments,
    sample_manual_tasks,
    save_review,
    validate_manual_set,
    write_relevance,
    write_tasks,
)
from anima_search.schemas import ManifestItem


def manifest(count: int) -> list[ManifestItem]:
    return [
        ManifestItem(
            image_id=f"val-{index}",
            split="Val",
            relative_path=f"../Val/{index}.jpg",
            sha256=str(index),
            size_bytes=1,
        )
        for index in range(count)
    ]


def test_manual_sampling_is_deterministic_and_blank():
    first = sample_manual_tasks(manifest(20), count=10, seed=7)
    second = sample_manual_tasks(manifest(20), count=10, seed=7)
    assert first == second
    assert len({row["source_image_id"] for row in first}) == 10
    assert all(row["text"] == "" and row["reviewed"] is False for row in first)


def test_parse_judgments_rejects_duplicates_and_ignores_zero():
    rows = parse_judgments(
        "val-1:2\nval-2:0\nval-3：1",
        query_id="q001",
        annotator="human-a",
    )
    assert [(row["image_id"], row["relevance"]) for row in rows] == [
        ("val-1", 2),
        ("val-3", 1),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        parse_judgments(
            "val-1:2\nval-1:1",
            query_id="q001",
            annotator="human-a",
        )


def test_save_and_validate_completed_manual_set(tmp_path: Path):
    tasks = sample_manual_tasks(manifest(2), count=2, seed=1)
    task_path = tmp_path / "queries.jsonl"
    relevance_path = tmp_path / "relevance.csv"
    write_tasks(task_path, tasks)
    write_relevance(relevance_path, [])

    for index, task in enumerate(tasks):
        save_review(
            task_path,
            relevance_path,
            index=index,
            text=f"人工查询 {index}",
            category="simple",
            annotator="human-a",
            note="",
            judgments=f"{task['source_image_id']}:2",
            reviewed=True,
        )

    summary = validate_manual_set(
        load_tasks(task_path),
        load_relevance_rows(relevance_path),
        expected_count=2,
        valid_image_ids={"val-0", "val-1"},
    )
    assert summary["reviewed_count"] == 2
    assert summary["relevance_row_count"] == 2


def test_incomplete_manual_set_is_refused():
    tasks = sample_manual_tasks(manifest(1), count=1)
    with pytest.raises(ValueError, match="not reviewed"):
        validate_manual_set(tasks, [], expected_count=1)
