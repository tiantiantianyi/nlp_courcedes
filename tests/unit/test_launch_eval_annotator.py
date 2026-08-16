from __future__ import annotations

import sys
from pathlib import Path

from scripts import launch_eval_annotator


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
