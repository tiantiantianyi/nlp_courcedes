import json
from pathlib import Path

from anima_search.schemas import ManifestItem
from scripts.import_m1_qwen35 import import_annotations, require_complete_from_config


def _canonical_record(numeric_id: str, digest: str) -> dict[str, object]:
    return {
        "image_id": numeric_id,
        "processed_sha256": digest,
        "source_model_id": "Qwen/Qwen3.5-9B",
        "annotation": {
            "scene": {"primary_type": "general"},
            "capture_visual": {},
            "entities": [],
            "ocr": [],
            "relations": [],
            "event": {},
            "subjective": {},
            "captions": {"short_zh": f"测试图像 {numeric_id}"},
            "uncertainties": [],
        },
    }


def test_import_preserves_manifest_identity_and_reports_missing(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    manifests = artifacts / "manifests"
    manifests.mkdir(parents=True)
    train = ManifestItem(
        image_id="train-1",
        split="Train",
        relative_path="../Train/1.jpg",
        sha256="a" * 64,
        size_bytes=10,
    )
    val = ManifestItem(
        image_id="val-2",
        split="Val",
        relative_path="../Val/2.jpg",
        sha256="b" * 64,
        size_bytes=20,
    )
    (manifests / "train.jsonl").write_text(
        train.model_dump_json() + "\n", encoding="utf-8"
    )
    (manifests / "val.jsonl").write_text(
        val.model_dump_json() + "\n", encoding="utf-8"
    )
    source = tmp_path / "qwen35.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                _canonical_record("1", "a" * 64),
                _canonical_record("2", "b" * 64),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = import_annotations(source, artifacts)

    assert report["annotation_version"] == "qwen35-canonical-v1.3"
    assert report["imported"] == {"train": 1, "val": 1}
    assert report["missing_image_ids"] == []
    assert report["failures"] == []


def test_require_complete_combines_config_and_cli_override() -> None:
    assert not require_complete_from_config({"allow_missing": True}, cli_required=False)
    assert require_complete_from_config({"allow_missing": False}, cli_required=False)
    assert require_complete_from_config({"allow_missing": True}, cli_required=True)


def test_import_report_counts_every_source_record_and_classifies_rejections(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    manifests = artifacts / "manifests"
    manifests.mkdir(parents=True)
    train = ManifestItem(
        image_id="train-1",
        split="Train",
        relative_path="../Train/1.jpg",
        sha256="a" * 64,
        size_bytes=10,
    )
    val = ManifestItem(
        image_id="val-2",
        split="Val",
        relative_path="../Val/2.jpg",
        sha256="b" * 64,
        size_bytes=20,
    )
    (manifests / "train.jsonl").write_text(
        train.model_dump_json() + "\n", encoding="utf-8"
    )
    (manifests / "val.jsonl").write_text(
        val.model_dump_json() + "\n", encoding="utf-8"
    )
    source_rows = [
        _canonical_record("1", "a" * 64),
        _canonical_record("1", "a" * 64),
        _canonical_record("99", "c" * 64),
        _canonical_record("2", "wrong"),
    ]
    source = tmp_path / "qwen35.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    report = import_annotations(source, artifacts)

    assert report["source_record_count"] == 4
    assert report["imported"] == {"train": 1, "val": 0}
    assert [failure["error"] for failure in report["failures"][:2]] == [
        "duplicate ID",
        "not in manifest",
    ]
    assert "processed_sha256 mismatch" in report["failures"][2]["error"]
