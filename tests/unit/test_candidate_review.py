from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from anima_search.evaluation.candidate_review import (
    load_candidate_relevance,
    parse_candidate_grades,
    render_candidate_contact_sheet,
    review_progress,
    save_candidate_review,
)
from scripts import launch_candidate_annotator


def _pool_row(query_id: str = "q001") -> dict[str, object]:
    return {
        "schema_version": "formal-relevance-pool-v1.0",
        "query_id": query_id,
        "text": "一只猫",
        "category": "simple",
        "source_image_id": "val-1",
        "source_relative_path": "../Val/1.jpg",
        "candidates": [
            {
                "image_id": "val-1",
                "relative_path": "../Val/1.jpg",
                "is_source": True,
                "retrieved_by": ["clip_only"],
                "best_rank": 1,
                "grade": None,
                "annotator": "",
                "reviewed": False,
            },
            {
                "image_id": "val-2",
                "relative_path": "../Val/2.jpg",
                "is_source": False,
                "retrieved_by": ["text_only"],
                "best_rank": 2,
                "grade": None,
                "annotator": "",
                "reviewed": False,
            },
        ],
    }


def test_parse_candidate_grades_requires_exact_candidate_coverage():
    grades = parse_candidate_grades(
        "val-1:2\nval-2：0", expected_ids={"val-1", "val-2"}
    )
    assert grades == {"val-1": 2, "val-2": 0}

    with pytest.raises(ValueError, match="missing candidate IDs"):
        parse_candidate_grades("val-1:2", expected_ids={"val-1", "val-2"})
    with pytest.raises(ValueError, match="unexpected candidate IDs"):
        parse_candidate_grades(
            "val-1:2\nval-2:0\nval-9:1",
            expected_ids={"val-1", "val-2"},
        )


def test_parse_candidate_grades_rejects_duplicates_and_invalid_grades():
    with pytest.raises(ValueError, match="duplicate candidate ID"):
        parse_candidate_grades(
            "val-1:2\nval-1:0", expected_ids={"val-1"}
        )
    with pytest.raises(ValueError, match="invalid candidate grade"):
        parse_candidate_grades("val-1:3", expected_ids={"val-1"})


def test_save_candidate_review_requires_source_grade_two_and_annotator(tmp_path: Path):
    output = tmp_path / "candidate_relevance.csv"
    with pytest.raises(ValueError, match="source image .* grade 2"):
        save_candidate_review(
            [_pool_row()],
            output,
            index=0,
            grades_text="val-1:1\nval-2:0",
            annotator="张添翼",
            note="",
            reviewed=True,
        )
    with pytest.raises(ValueError, match="annotator"):
        save_candidate_review(
            [_pool_row()],
            output,
            index=0,
            grades_text="val-1:2\nval-2:0",
            annotator="",
            note="",
            reviewed=True,
        )


def test_save_candidate_review_is_atomic_and_progress_requires_reviewed_rows(
    tmp_path: Path,
):
    pool = [_pool_row()]
    output = tmp_path / "candidate_relevance.csv"

    save_candidate_review(
        pool,
        output,
        index=0,
        grades_text="val-1:2\nval-2:0",
        annotator="张添翼",
        note="已逐图检查",
        reviewed=False,
    )
    assert review_progress(pool, load_candidate_relevance(output)) == (0, 1)

    save_candidate_review(
        pool,
        output,
        index=0,
        grades_text="val-1:2\nval-2:0",
        annotator="张添翼",
        note="已逐图检查",
        reviewed=True,
    )
    rows = load_candidate_relevance(output)
    assert review_progress(pool, rows) == (1, 1)
    assert [(row["image_id"], int(row["relevance"])) for row in rows] == [
        ("val-1", 2),
        ("val-2", 0),
    ]
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_render_candidate_contact_sheet_contains_all_candidates(tmp_path: Path):
    val_dir = tmp_path.parent / "Val"
    val_dir.mkdir(exist_ok=True)
    Image.new("RGB", (32, 20), "red").save(val_dir / "1.jpg")
    Image.new("RGB", (20, 32), "blue").save(val_dir / "2.jpg")

    sheet = render_candidate_contact_sheet(
        _pool_row(), tmp_path, columns=2, tile_size=100
    )

    assert sheet.size == (200, 100)


def test_render_candidate_contact_sheet_rejects_more_than_twenty_five():
    row = _pool_row()
    row["candidates"] = [
        {
            "image_id": f"val-{index}",
            "relative_path": f"../Val/{index}.jpg",
            "is_source": index == 1,
            "retrieved_by": [],
            "best_rank": None,
        }
        for index in range(1, 27)
    ]

    with pytest.raises(ValueError, match="at most 25"):
        render_candidate_contact_sheet(row, Path("."))


def test_candidate_form_defaults_only_source_grade_and_annotator(tmp_path: Path):
    val_dir = tmp_path.parent / "Val"
    val_dir.mkdir(exist_ok=True)
    Image.new("RGB", (32, 20), "red").save(val_dir / "1.jpg")
    Image.new("RGB", (20, 32), "blue").save(val_dir / "2.jpg")

    values = launch_candidate_annotator._form_values(_pool_row(), [], tmp_path)

    assert values[1] == "q001"
    assert values[4] == "val-1"
    assert values[5] == "张添翼"
    assert values[7] == "val-1:2\nval-2:"
    assert values[8] is False


def test_candidate_form_preserves_saved_grades_and_reviewer(tmp_path: Path):
    val_dir = tmp_path.parent / "Val"
    val_dir.mkdir(exist_ok=True)
    Image.new("RGB", (32, 20), "red").save(val_dir / "1.jpg")
    Image.new("RGB", (20, 32), "blue").save(val_dir / "2.jpg")
    rows = [
        {
            "query_id": "q001",
            "image_id": "val-1",
            "relevance": "2",
            "annotator": "复核者",
            "note": "逐图完成",
            "reviewed": "true",
        },
        {
            "query_id": "q001",
            "image_id": "val-2",
            "relevance": "0",
            "annotator": "复核者",
            "note": "逐图完成",
            "reviewed": "true",
        },
    ]

    values = launch_candidate_annotator._form_values(_pool_row(), rows, tmp_path)

    assert values[5] == "复核者"
    assert values[6] == "逐图完成"
    assert values[7] == "val-1:2\nval-2:0"
    assert values[8] is True
