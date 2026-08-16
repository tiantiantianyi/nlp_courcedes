from __future__ import annotations

import sys
from pathlib import Path

from anima_search.evaluation.manual_set import write_relevance, write_tasks
from scripts import launch_eval_annotator


def _task(*, annotator: str = "", category: str = "") -> dict[str, object]:
    return {
        "query_id": "q001",
        "source_image_id": "val-2007",
        "source_relative_path": "../Val/2007.jpg",
        "text": "",
        "category": category,
        "annotator": annotator,
        "note": "",
        "reviewed": False,
    }


def test_task_form_values_exposes_source_id_and_defaults_annotator(
    tmp_path: Path,
) -> None:
    values = launch_eval_annotator._task_form_values(_task(), [], tmp_path)

    assert values[2] == "val-2007"
    assert values[5] == "张添翼"


def test_task_form_values_preserves_saved_annotator(tmp_path: Path) -> None:
    values = launch_eval_annotator._task_form_values(
        _task(annotator="已有标注者", category="simple"),
        [],
        tmp_path,
    )

    assert values[5] == "已有标注者"


def test_category_choices_have_chinese_labels_and_stable_values() -> None:
    assert launch_eval_annotator.CATEGORY_CHOICES == [
        ("简单查询", "simple"),
        ("组合查询", "compositional"),
        ("否定查询", "negative"),
        ("数量查询", "count"),
        ("文字识别查询", "ocr"),
    ]


def test_save_callback_returns_progress_after_source_id_output(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "queries.jsonl"
    relevance_path = tmp_path / "relevance.csv"
    second = _task()
    second.update(query_id="q002", source_image_id="val-2009")
    write_tasks(task_path, [_task(), second])
    write_relevance(relevance_path, [])
    app = launch_eval_annotator.build_app(task_path, relevance_path, tmp_path)
    save_callback = next(
        block.fn
        for block in app.fns.values()
        if getattr(block.fn, "__name__", "") == "save_position"
    )

    status, progress = save_callback(
        1,
        "一只灰色长毛猫",
        "simple",
        "张添翼",
        "",
        "val-2007:2",
        True,
    )

    assert status == "已保存 q001"
    assert progress == "**进度：1/2 已审核**"


def test_main_allows_only_resolved_configured_image_directories(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "repository"
    config = {
        "project_root": str(project_root),
        "data": {"train_dir": "../Train", "val_dir": "../Val"},
    }
    launch_arguments: dict[str, object] = {}

    class App:
        def launch(self, **kwargs: object) -> None:
            launch_arguments.update(kwargs)

    monkeypatch.setattr(launch_eval_annotator, "load_config", lambda _: config)
    monkeypatch.setattr(launch_eval_annotator, "build_app", lambda *_: App())
    monkeypatch.setattr(sys, "argv", ["launch_eval_annotator.py"])

    launch_eval_annotator.main()

    assert launch_arguments["allowed_paths"] == [
        str((project_root / "../Train").resolve()),
        str((project_root / "../Val").resolve()),
    ]
