from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from anima_search.evaluation.formal_set import merge_reviewed_sets
from anima_search.evaluation.manual_set import write_relevance, write_tasks
from scripts import prepare_formal_eval


def _write_set(
    root: Path,
    name: str,
    *,
    query_id: str,
    source_id: str,
    relevance_query_id: str | None = None,
) -> tuple[Path, Path]:
    query_path = root / name / "queries.jsonl"
    relevance_path = root / name / "relevance.csv"
    write_tasks(
        query_path,
        [
            {
                "query_id": query_id,
                "text": f"人工查询 {query_id}",
                "category": "simple",
                "source_image_id": source_id,
                "source_relative_path": f"../Val/{source_id.removeprefix('val-')}.jpg",
                "reviewed": True,
                "annotator": "张添翼",
                "note": "",
            }
        ],
    )
    write_relevance(
        relevance_path,
        [
            {
                "query_id": relevance_query_id or query_id,
                "image_id": source_id,
                "relevance": 2,
                "annotator": "张添翼",
                "note": "",
            }
        ],
    )
    return query_path, relevance_path


def test_merge_reviewed_sets_is_sorted_and_does_not_modify_sources(tmp_path: Path):
    right_queries, right_relevance = _write_set(
        tmp_path, "right", query_id="q002", source_id="val-2"
    )
    left_queries, left_relevance = _write_set(
        tmp_path, "left", query_id="q001", source_id="val-1"
    )
    before = {
        path: path.read_bytes()
        for path in (left_queries, left_relevance, right_queries, right_relevance)
    }

    tasks, rows, summary = merge_reviewed_sets(
        [right_queries, left_queries],
        [right_relevance, left_relevance],
        expected_count=2,
    )

    assert [row["query_id"] for row in tasks] == ["q001", "q002"]
    assert [row["query_id"] for row in rows] == ["q001", "q002"]
    assert summary["query_count"] == 2
    assert summary["relevance_row_count"] == 2
    assert all(path.read_bytes() == content for path, content in before.items())


def test_merge_reviewed_sets_rejects_duplicate_query_id(tmp_path: Path):
    first = _write_set(tmp_path, "first", query_id="q001", source_id="val-1")
    second = _write_set(tmp_path, "second", query_id="q001", source_id="val-2")

    with pytest.raises(ValueError, match="duplicate query IDs"):
        merge_reviewed_sets(
            [first[0], second[0]], [first[1], second[1]], expected_count=2
        )


def test_merge_reviewed_sets_rejects_duplicate_source_image(tmp_path: Path):
    first = _write_set(tmp_path, "first", query_id="q001", source_id="val-1")
    second = _write_set(tmp_path, "second", query_id="q002", source_id="val-1")

    with pytest.raises(ValueError, match="duplicate source image IDs"):
        merge_reviewed_sets(
            [first[0], second[0]], [first[1], second[1]], expected_count=2
        )


def test_merge_reviewed_sets_rejects_relevance_for_unknown_query(tmp_path: Path):
    query_path, relevance_path = _write_set(
        tmp_path,
        "broken",
        query_id="q001",
        source_id="val-1",
        relevance_query_id="q999",
    )

    with pytest.raises(ValueError, match="relevance references unknown query"):
        merge_reviewed_sets([query_path], [relevance_path], expected_count=1)


def test_prepare_formal_eval_writes_merged_data_and_input_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = _write_set(tmp_path, "first", query_id="q001", source_id="val-1")
    second = _write_set(tmp_path, "second", query_id="q002", source_id="val-2")
    inputs = [*first, *second]
    before = {path: path.read_bytes() for path in inputs}
    output_dir = tmp_path / "formal"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_formal_eval.py",
            "--queries",
            str(first[0]),
            str(second[0]),
            "--relevance",
            str(first[1]),
            str(second[1]),
            "--output-dir",
            str(output_dir),
            "--expected-count",
            "2",
        ],
    )

    prepare_formal_eval.main()

    assert len((output_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    report = json.loads((output_dir / "merge_report.json").read_text(encoding="utf-8"))
    assert report["query_count"] == 2
    assert report["input_sha256"][str(first[0])] == hashlib.sha256(
        before[first[0]]
    ).hexdigest()
