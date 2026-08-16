from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from anima_search.indexing.index_manifest import (
    image_ids_digest,
    sha256_file,
)
from anima_search.m6.interface_validation import validate_interface_file


def _candidate(rank: int) -> dict[str, object]:
    number = 2001 + rank
    return {
        "rank": rank,
        "image_id": f"val-{number}",
        "relative_path": f"../Val/{number}.jpg",
        "fused_score": 1.0 / rank,
        "branch_scores": {"image": 0.9 / rank, "text": 0.8 / rank},
        "branch_ranks": {"image": rank, "text": rank + 1},
        "matched_fields": ["scene"],
    }


def _batch(query_id: str = "m6-q001") -> dict[str, object]:
    return {
        "schema_version": "m5-to-m6-v1.0",
        "query_id": query_id,
        "query": "夜晚的城市街道",
        "category": "simple",
        "split": "val",
        "fusion_method": "rrf",
        "top_k": 20,
        "annotation_version": "qwen35-canonical-v1.3",
        "index_manifest_sha256": "",
        "config_sha256": "b" * 64,
        "candidates": [_candidate(rank) for rank in range(1, 21)],
    }


def _fixture_paths(
    tmp_path: Path,
    *,
    payloads: list[dict[str, object]] | None = None,
    corrupt_image: bool = False,
    catalog_path_mismatch: bool = False,
    declared_jsonl_elsewhere: bool = False,
) -> dict[str, Path]:
    project_root = tmp_path / "project"
    index_dir = project_root / "artifacts" / "indexes" / "val"
    val_dir = tmp_path / "Val"
    train_dir = tmp_path / "Train"
    index_dir.mkdir(parents=True)
    val_dir.mkdir()
    train_dir.mkdir()

    image_ids = [f"val-{number}" for number in range(2002, 2022)]
    annotations: list[dict[str, object]] = []
    for number, image_id in zip(range(2002, 2022), image_ids, strict=True):
        image_path = val_dir / f"{number}.jpg"
        if corrupt_image and number == 2002:
            image_path.write_bytes(b"not-jpeg")
        else:
            Image.new("RGB", (8, 8), (number % 255, 20, 40)).save(
                image_path,
                format="JPEG",
            )
        relative_path = f"../Val/{number}.jpg"
        if catalog_path_mismatch and number == 2002:
            relative_path = "../Val/9999.jpg"
        annotations.append(
            {
                "image_id": image_id,
                "relative_path": relative_path,
            }
        )

    if declared_jsonl_elsewhere:
        annotations_path = (
            project_root / "artifacts" / "manifests" / "val.jsonl"
        )
        annotations_path.parent.mkdir(parents=True)
        annotations_path.write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False) for row in annotations
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        annotations_path = index_dir / "annotations.json"
        annotations_path.write_text(
            json.dumps(annotations, ensure_ascii=False),
            encoding="utf-8",
        )
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
                "active_branches": ["image", "text"],
                "branches": {},
                "config_digest": "c" * 64,
            }
        ),
        encoding="utf-8",
    )

    config_snapshot_path = project_root / "m5_retrieval_config.snapshot.json"
    config_snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": "m5-retrieval-config-v1",
                "split": "val",
                "retrieval": {"fusion_method": "rrf"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config_snapshot_hash = sha256_file(config_snapshot_path)
    rows = payloads or [_batch()]
    manifest_hash = sha256_file(manifest_path)
    for row in rows:
        if not row["index_manifest_sha256"]:
            row["index_manifest_sha256"] = manifest_hash
        row["config_sha256"] = config_snapshot_hash
    input_path = project_root / "m5.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "input_path": input_path,
        "project_root": project_root,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "index_manifest_path": manifest_path,
        "config_snapshot_path": config_snapshot_path,
        "annotation_path": annotations_path,
    }


def test_valid_file_returns_batches_and_zero_issues(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    annotation_path = paths.pop("annotation_path")

    batches, report = validate_interface_file(**paths)

    assert annotation_path.is_file()
    assert report.valid
    assert report.query_count == 1
    assert report.candidate_count == 20
    assert report.issues == []
    assert len(batches) == 1


def test_validator_rejects_tampered_m5_config_snapshot(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    paths.pop("annotation_path")
    paths["config_snapshot_path"].write_text(
        '{"schema_version":"tampered"}\n',
        encoding="utf-8",
    )

    batches, report = validate_interface_file(**paths)

    assert batches == []
    assert "E_MANIFEST_MISMATCH" in {issue.code for issue in report.issues}


def test_validator_collects_duplicate_query_path_and_branch_errors(
    tmp_path: Path,
) -> None:
    valid = _batch("same-query")
    wrong_path = _batch("same-query")
    wrong_path["candidates"][0]["relative_path"] = "../Train/2002.jpg"  # type: ignore[index]
    branch_error = _batch("branch-error")
    branch_error["candidates"][0]["branch_ranks"] = {"image": 1}  # type: ignore[index]
    paths = _fixture_paths(
        tmp_path,
        payloads=[valid, wrong_path, branch_error],
    )
    paths.pop("annotation_path")

    batches, report = validate_interface_file(**paths)

    assert batches == []
    assert not report.valid
    assert {issue.code for issue in report.issues} >= {
        "E_DUPLICATE_QUERY_ID",
        "E_PATH_SPLIT",
        "E_BRANCH_KEYS",
    }


def test_validator_rejects_manifest_hash_and_candidate_path_mismatch(
    tmp_path: Path,
) -> None:
    payload = _batch()
    payload["index_manifest_sha256"] = "c" * 64
    paths = _fixture_paths(
        tmp_path,
        payloads=[payload],
        catalog_path_mismatch=True,
    )
    paths.pop("annotation_path")

    _, report = validate_interface_file(**paths)

    assert {issue.code for issue in report.issues} >= {"E_MANIFEST_MISMATCH"}


def test_validator_rejects_nonfinite_score(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    paths.pop("annotation_path")
    raw = paths["input_path"].read_text(encoding="utf-8").replace(
        '"fused_score": 1.0',
        '"fused_score": NaN',
        1,
    )
    paths["input_path"].write_text(raw, encoding="utf-8")

    _, report = validate_interface_file(**paths)

    assert "E_NONFINITE_SCORE" in {issue.code for issue in report.issues}


def test_validator_rejects_corrupt_image(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path, corrupt_image=True)
    paths.pop("annotation_path")

    _, report = validate_interface_file(**paths)

    assert "E_IMAGE_DECODE" in {issue.code for issue in report.issues}


def test_validator_rejects_nineteen_candidates(tmp_path: Path) -> None:
    payload = _batch()
    payload["candidates"] = payload["candidates"][:-1]  # type: ignore[index]
    paths = _fixture_paths(tmp_path, payloads=[payload])
    paths.pop("annotation_path")

    _, report = validate_interface_file(**paths)

    assert "E_CANDIDATE_COUNT" in {issue.code for issue in report.issues}


def test_validator_rejects_tampered_declared_annotation_artifact(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    annotation_path = paths.pop("annotation_path")
    annotation_path.write_text(
        annotation_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    _, report = validate_interface_file(**paths)

    assert "E_MANIFEST_MISMATCH" in {
        issue.code for issue in report.issues
    }
    assert any(
        "annotation_sha256" in issue.message for issue in report.issues
    )


def test_validator_loads_declared_jsonl_annotation_artifact(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path, declared_jsonl_elsewhere=True)
    annotation_path = paths.pop("annotation_path")

    batches, report = validate_interface_file(**paths)

    assert annotation_path.name == "val.jsonl"
    assert report.valid
    assert len(batches) == 1


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("rank", 0, "E_RANK_SEQUENCE"),
        ("fused_score", "not-a-number", "E_NONFINITE_SCORE"),
    ],
)
def test_validator_uses_stable_candidate_error_codes(
    tmp_path: Path,
    field: str,
    value: object,
    expected_code: str,
) -> None:
    payload = _batch()
    payload["candidates"][0][field] = value  # type: ignore[index]
    paths = _fixture_paths(tmp_path, payloads=[payload])
    paths.pop("annotation_path")

    _, report = validate_interface_file(**paths)

    assert expected_code in {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    ("branch_field", "value", "expected_code"),
    [
        ("branch_scores", "not-a-number", "E_NONFINITE_SCORE"),
        ("branch_ranks", 0, "E_BRANCH_KEYS"),
    ],
)
def test_validator_uses_stable_branch_value_error_codes(
    tmp_path: Path,
    branch_field: str,
    value: object,
    expected_code: str,
) -> None:
    payload = _batch()
    payload["candidates"][0][branch_field]["image"] = value  # type: ignore[index]
    paths = _fixture_paths(tmp_path, payloads=[payload])
    paths.pop("annotation_path")

    _, report = validate_interface_file(**paths)

    assert expected_code in {issue.code for issue in report.issues}
