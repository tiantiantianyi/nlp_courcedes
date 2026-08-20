import csv
from pathlib import Path

from scripts import launch_a9_annotator


def _write_review(path: Path) -> None:
    rows = [
        {
            "query_id": "a9-art-1",
            "query": "印象派绘画",
            "domain": "wikiart",
            "image_id": "a9-art-000001",
            "relative_path": "images/wikiart/a9-art-000001.jpg",
            "reference_text": "style=Impressionism",
            "auto_relevance": "2",
            "human_relevance": "",
            "annotator": "张添翼",
            "review_status": "待复核",
            "review_note": "",
        }
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_save_row_persists_grade_and_note_atomically(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    _write_review(path)
    launch_a9_annotator.save_row(path, 0, "1", "", "边界案例")
    rows = launch_a9_annotator.load_rows(path)
    assert rows[0]["human_relevance"] == "1"
    assert rows[0]["annotator"] == "张添翼"
    assert rows[0]["review_status"] == "已复核"
    assert rows[0]["review_note"] == "边界案例"


def test_progress_text_counts_only_completed_reviews(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    _write_review(path)
    rows = launch_a9_annotator.load_rows(path)
    assert launch_a9_annotator.progress_text(rows) == "**复核进度：0/1 条**"
    launch_a9_annotator.save_row(path, 0, "2", "张添翼", "")
    assert launch_a9_annotator.progress_text(
        launch_a9_annotator.load_rows(path)
    ) == "**复核进度：1/1 条**"
