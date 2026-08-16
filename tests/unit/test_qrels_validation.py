from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from anima_search.evaluation.manual_set import (
    load_relevance_rows,
    write_relevance,
    write_tasks,
)
from anima_search.evaluation.qrels_validation import (
    finalize_qrels,
    write_finalized_qrels,
)
from scripts import finalize_formal_qrels


CATEGORIES = ["simple", "compositional", "negative", "count", "ocr"]


def _inputs():
    queries = [
        {
            "query_id": f"q{index:03d}",
            "text": f"查询 {index}",
            "category": CATEGORIES[(index - 1) % len(CATEGORIES)],
            "source_image_id": f"val-{index}",
            "source_relative_path": f"../Val/{index}.jpg",
            "reviewed": True,
            "annotator": "张添翼",
            "note": "",
        }
        for index in range(1, 11)
    ]
    source_rows = [
        {
            "query_id": row["query_id"],
            "image_id": row["source_image_id"],
            "relevance": 2,
            "annotator": "张添翼",
            "note": "",
        }
        for row in queries
    ]
    pool = []
    candidate_rows = []
    for index, query in enumerate(queries[:5], start=1):
        distractor = f"val-{100 + index}"
        pool.append(
            {
                "schema_version": "formal-relevance-pool-v1.0",
                "query_id": query["query_id"],
                "text": query["text"],
                "category": query["category"],
                "source_image_id": query["source_image_id"],
                "source_relative_path": query["source_relative_path"],
                "candidates": [
                    {"image_id": query["source_image_id"]},
                    {"image_id": distractor},
                ],
            }
        )
        candidate_rows.extend(
            [
                {
                    "query_id": query["query_id"],
                    "image_id": query["source_image_id"],
                    "relevance": 2,
                    "annotator": "张添翼",
                    "note": "",
                    "reviewed": True,
                },
                {
                    "query_id": query["query_id"],
                    "image_id": distractor,
                    "relevance": 0,
                    "annotator": "张添翼",
                    "note": "",
                    "reviewed": True,
                },
            ]
        )
    return queries, source_rows, pool, candidate_rows


def test_finalize_qrels_replaces_graded_subset_and_preserves_zero(tmp_path: Path):
    queries, source_rows, pool, candidate_rows = _inputs()

    final_queries, final_rows, summary = finalize_qrels(
        queries, source_rows, pool, candidate_rows
    )

    assert len(final_queries) == 10
    assert len(final_rows) == 15
    assert summary["valid"] is True
    assert summary["single_positive_query_count"] == 5
    assert summary["graded_pool_query_count"] == 5
    assert summary["graded_query_ids"] == ["q001", "q002", "q003", "q004", "q005"]
    assert summary["single_positive_query_ids"] == [
        "q006",
        "q007",
        "q008",
        "q009",
        "q010",
    ]

    assert summary["grade_counts"] == {"0": 5, "1": 0, "2": 10}

    paths = write_finalized_qrels(tmp_path, final_queries, final_rows, summary)
    loaded = load_relevance_rows(paths["relevance"])
    assert sum(int(row["relevance"]) == 0 for row in loaded) == 5


def test_finalize_qrels_rejects_missing_and_extra_candidate_judgments():
    queries, source_rows, pool, candidate_rows = _inputs()
    with pytest.raises(ValueError, match="missing candidate judgments"):
        finalize_qrels(queries, source_rows, pool, candidate_rows[:-1])

    extra = [
        *candidate_rows,
        {
            "query_id": "q001",
            "image_id": "val-999",
            "relevance": 0,
            "annotator": "张添翼",
            "note": "",
            "reviewed": True,
        },
    ]
    with pytest.raises(ValueError, match="extra candidate judgments"):
        finalize_qrels(queries, source_rows, pool, extra)


def test_finalize_qrels_rejects_duplicate_rows_and_invalid_source_grade():
    queries, source_rows, pool, candidate_rows = _inputs()
    with pytest.raises(ValueError, match="duplicate candidate judgment"):
        finalize_qrels(
            queries, source_rows, pool, [*candidate_rows, dict(candidate_rows[0])]
        )

    candidate_rows[0]["relevance"] = 1
    with pytest.raises(ValueError, match="source image .* grade 2"):
        finalize_qrels(queries, source_rows, pool, candidate_rows)


def test_finalize_qrels_rejects_invalid_annotator_or_unreviewed_row():
    queries, source_rows, pool, candidate_rows = _inputs()
    candidate_rows[0]["annotator"] = ""
    with pytest.raises(ValueError, match="missing annotator"):
        finalize_qrels(queries, source_rows, pool, candidate_rows)

    queries, source_rows, pool, candidate_rows = _inputs()
    candidate_rows[0]["reviewed"] = False
    with pytest.raises(ValueError, match="not reviewed"):
        finalize_qrels(queries, source_rows, pool, candidate_rows)


def test_finalize_qrels_requires_all_five_categories_in_graded_subset():
    queries, source_rows, pool, candidate_rows = _inputs()
    removed_query_id = str(pool[-1]["query_id"])
    pool = pool[:-1]
    candidate_rows = [
        row for row in candidate_rows if str(row["query_id"]) != removed_query_id
    ]

    with pytest.raises(ValueError, match="graded subset category coverage"):
        finalize_qrels(queries, source_rows, pool, candidate_rows)


def test_finalize_formal_qrels_cli_writes_validation_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    queries, source_rows, pool, candidate_rows = _inputs()
    queries_path = tmp_path / "queries.jsonl"
    source_path = tmp_path / "source.csv"
    pool_path = tmp_path / "pool.jsonl"
    candidate_path = tmp_path / "candidate.csv"
    output_dir = tmp_path / "formal"
    write_tasks(queries_path, queries)
    write_relevance(source_path, source_rows)
    pool_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool),
        encoding="utf-8",
    )
    with candidate_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_formal_qrels.py",
            "--queries",
            str(queries_path),
            "--source-relevance",
            str(source_path),
            "--pool",
            str(pool_path),
            "--candidate-relevance",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    finalize_formal_qrels.main()

    validation = json.loads(
        (output_dir / "qrels_validation.json").read_text(encoding="utf-8")
    )
    assert validation["valid"] is True
    assert validation["graded_pool_query_count"] == 5
    assert sum(
        int(row["relevance"]) == 0
        for row in load_relevance_rows(output_dir / "val_relevance.csv")
    ) == 5
