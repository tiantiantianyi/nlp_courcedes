from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from anima_search.indexing.index_manifest import image_ids_digest, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "validate_m5_m6_interface.py"


def _delivery(tmp_path: Path, *, candidate_count: int = 20) -> dict[str, Path]:
    project_root = tmp_path / "project"
    index_dir = project_root / "artifacts" / "indexes" / "val"
    val_dir = tmp_path / "Val"
    train_dir = tmp_path / "Train"
    index_dir.mkdir(parents=True)
    val_dir.mkdir()
    train_dir.mkdir()

    candidates: list[dict[str, object]] = []
    annotations: list[dict[str, str]] = []
    image_ids: list[str] = []
    for rank in range(1, 21):
        number = 2001 + rank
        image_id = f"val-{number}"
        relative_path = f"../Val/{number}.jpg"
        Image.new("RGB", (4, 4), "navy").save(val_dir / f"{number}.jpg")
        image_ids.append(image_id)
        annotations.append(
            {"image_id": image_id, "relative_path": relative_path}
        )
        candidates.append(
            {
                "rank": rank,
                "image_id": image_id,
                "relative_path": relative_path,
                "fused_score": 1.0 / rank,
                "branch_scores": {"image": 1.0 / rank},
                "branch_ranks": {"image": rank},
                "matched_fields": [],
            }
        )

    annotations_path = index_dir / "annotations.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "split": "val",
                "record_count": 20,
                "image_ids_sha256": image_ids_digest(image_ids),
                "annotation_path": str(annotations_path),
                "annotation_sha256": sha256_file(annotations_path),
                "annotation_version": "qwen35-canonical-v1.3",
                "config_digest": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = project_root / "m5_retrieval_config.snapshot.json"
    snapshot_path.write_text(
        '{"schema_version":"m5-retrieval-config-v1"}\n',
        encoding="utf-8",
    )
    payload = {
        "schema_version": "m5-to-m6-v1.0",
        "query_id": "cli-q001",
        "query": "城市夜景",
        "category": "simple",
        "split": "val",
        "fusion_method": "rrf",
        "top_k": 20,
        "annotation_version": "qwen35-canonical-v1.3",
        "index_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(snapshot_path),
        "candidates": candidates[:candidate_count],
    }
    input_path = project_root / "delivery.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return {
        "input": input_path,
        "project_root": project_root,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "manifest": manifest_path,
        "snapshot": snapshot_path,
    }


def _run(paths: dict[str, Path], report_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(paths["input"]),
            "--project-root",
            str(paths["project_root"]),
            "--train-dir",
            str(paths["train_dir"]),
            "--val-dir",
            str(paths["val_dir"]),
            "--index-manifest",
            str(paths["manifest"]),
            "--m5-config-snapshot",
            str(paths["snapshot"]),
            "--report",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_writes_valid_report_and_returns_zero(tmp_path: Path) -> None:
    paths = _delivery(tmp_path)
    report_path = tmp_path / "valid-report.json"

    completed = _run(paths, report_path)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"] is True


def test_cli_writes_complete_invalid_report_and_returns_one(
    tmp_path: Path,
) -> None:
    paths = _delivery(tmp_path, candidate_count=19)
    report_path = tmp_path / "invalid-report.json"

    completed = _run(paths, report_path)

    assert completed.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "E_CANDIDATE_COUNT"
    }


@pytest.mark.parametrize("protected_key", ["input", "manifest", "snapshot"])
def test_cli_refuses_report_aliasing_read_only_input(
    tmp_path: Path,
    protected_key: str,
) -> None:
    paths = _delivery(tmp_path)
    protected = paths[protected_key]
    original = protected.read_bytes()

    completed = _run(paths, protected)

    assert completed.returncode != 0
    assert "output path alias" in completed.stderr
    assert protected.read_bytes() == original


def test_cli_refuses_report_symlink_aliasing_input(tmp_path: Path) -> None:
    paths = _delivery(tmp_path)
    original = paths["input"].read_bytes()
    report_path = tmp_path / "report-link.json"
    report_path.symlink_to(paths["input"])

    completed = _run(paths, report_path)

    assert completed.returncode != 0
    assert "output path alias" in completed.stderr
    assert paths["input"].read_bytes() == original
