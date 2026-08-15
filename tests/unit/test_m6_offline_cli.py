from __future__ import annotations

import json
from pathlib import Path

import pytest

from anima_search.m6.contract import M5QueryBatch
from anima_search.m6.interface_validation import InterfaceValidationReport
from scripts.run_m6_from_m5 import main


def _batch() -> M5QueryBatch:
    return M5QueryBatch.model_validate(
        {
            "schema_version": "m5-to-m6-v1.0",
            "query_id": "dry-q001",
            "query": "夜晚街道",
            "category": "simple",
            "split": "val",
            "fusion_method": "rrf",
            "top_k": 20,
            "annotation_version": "qwen35-canonical-v1.3",
            "index_manifest_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "candidates": [
                {
                    "rank": rank,
                    "image_id": f"val-{2001 + rank}",
                    "relative_path": f"../Val/{2001 + rank}.jpg",
                    "fused_score": 1.0 / rank,
                    "branch_scores": {"image": 1.0 / rank},
                    "branch_ranks": {"image": rank},
                    "matched_fields": [],
                }
                for rank in range(1, 21)
            ],
        }
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    project_root = tmp_path / "project"
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True)
    config_path = configs_dir / "test.yaml"
    config_path.write_text(
        "project_root: .\n"
        "models: {}\n"
        "retrieval: {}\n"
        "runtime: {}\n",
        encoding="utf-8",
    )
    input_path = project_root / "m5.jsonl"
    input_path.write_text('{"untouched": true}\n', encoding="utf-8")
    return {
        "project_root": project_root,
        "config": config_path,
        "input": input_path,
        "output": project_root / "m6.jsonl",
        "report": project_root / "validation.json",
        "manifest": project_root / "manifest.json",
        "train": tmp_path / "Train",
        "val": tmp_path / "Val",
    }


def _argv(paths: dict[str, Path]) -> list[str]:
    return [
        "--input",
        str(paths["input"]),
        "--output",
        str(paths["output"]),
        "--validation-report",
        str(paths["report"]),
        "--config",
        str(paths["config"]),
        "--index-manifest",
        str(paths["manifest"]),
        "--train-dir",
        str(paths["train"]),
        "--val-dir",
        str(paths["val"]),
        "--method",
        "listwise",
        "--dry-run",
    ]


def test_cli_refuses_to_overwrite_m5_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths["output"] = paths["input"]

    with pytest.raises(ValueError, match="must differ from input"):
        main(_argv(paths))


def test_dry_run_writes_auditable_degraded_result_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    original_input = paths["input"].read_bytes()

    def valid_delivery(**_: object) -> tuple[
        list[M5QueryBatch],
        InterfaceValidationReport,
    ]:
        return [
            _batch()
        ], InterfaceValidationReport(
            valid=True,
            query_count=1,
            candidate_count=20,
            issues=[],
        )

    monkeypatch.setattr(
        "scripts.run_m6_from_m5.validate_interface_file",
        valid_delivery,
    )

    assert main(_argv(paths)) == 0

    payload = json.loads(
        paths["output"].read_text(encoding="utf-8").splitlines()[0]
    )
    assert payload["schema_version"] == "m6-rerank-v1.0"
    assert payload["degraded"] is True
    assert payload["mismatch"] == ["dry-run: Qwen3-VL was not invoked"]
    assert [item["image_id"] for item in payload["candidates"]] == [
        f"val-{number}" for number in range(2002, 2022)
    ]
    assert paths["input"].read_bytes() == original_input
    assert json.loads(paths["report"].read_text(encoding="utf-8"))["valid"]
