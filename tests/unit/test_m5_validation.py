from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from anima_search.delivery.m5_validation import validate_m5_m6_file
from anima_search.indexing.index_manifest import sha256_file


def fixture_payload(tmp_path: Path) -> tuple[dict, Path, Path]:
    project_root = tmp_path / "repo"
    val_dir = tmp_path / "Val"
    project_root.mkdir()
    val_dir.mkdir()
    image_records = []
    candidates = []
    for rank in range(1, 21):
        image_id = f"val-{2001 + rank}"
        relative_path = f"../Val/{2001 + rank}.jpg"
        image_path = val_dir / f"{2001 + rank}.jpg"
        Image.new("RGB", (2, 2), (rank, rank, rank)).save(image_path)
        image_records.append(
            {
                "image_id": image_id,
                "relative_path": relative_path,
                "sha256": sha256_file(image_path),
            }
        )
        candidates.append(
            {
                "rank": rank,
                "image_id": image_id,
                "relative_path": relative_path,
                "fused_score": 1.0 / rank,
                "branch_scores": {"image": 1.0 / rank, "text": 0.5 / rank},
                "branch_ranks": {"image": rank, "text": 21 - rank},
                "matched_fields": [],
            }
        )
    manifest = {
        "schema_version": 2,
        "split": "val",
        "record_count": 20,
        "annotation_version": "qwen35-canonical-v1.3",
        "image_records": image_records,
    }
    manifest_path = project_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = {
        "schema_version": "m5-to-m6-v1.0",
        "query_id": "m6-q001",
        "query": "night street",
        "category": "simple",
        "split": "val",
        "fusion_method": "rrf",
        "top_k": 20,
        "annotation_version": "qwen35-canonical-v1.3",
        "index_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": "b" * 64,
        "candidates": candidates,
    }
    return payload, project_root, manifest_path


def run_validation(payloads: list[dict], tmp_path: Path, project_root: Path, manifest: Path):
    input_path = project_root / "candidates.jsonl"
    input_path.write_text(
        "".join(json.dumps(payload, allow_nan=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    return validate_m5_m6_file(
        input_path,
        project_root=project_root,
        train_dir=Path("../Train"),
        val_dir=Path("../Val"),
        index_manifest=manifest,
    )


def test_validate_m5_m6_file_accepts_complete_batch(tmp_path: Path):
    payload, project_root, manifest = fixture_payload(tmp_path)
    report = run_validation([payload], tmp_path, project_root, manifest)
    assert report["valid"] is True
    assert report["query_count"] == 1
    assert report["candidate_count"] == 20
    assert report["decoded_unique_images"] == 20
    assert report["hashed_unique_images"] == 20
    assert report["errors"] == []


def test_validate_m5_m6_file_collects_stable_error_codes(tmp_path: Path):
    payload, project_root, manifest = fixture_payload(tmp_path)
    payload["unexpected"] = True
    payload["candidates"][0]["rank"] = 2
    payload["candidates"][0]["branch_ranks"] = {"image": 1}
    payload["candidates"][1]["image_id"] = payload["candidates"][0]["image_id"]
    payload["candidates"][2]["relative_path"] = "../Train/0.jpg"
    report = run_validation([payload, payload], tmp_path, project_root, manifest)
    codes = {issue["code"] for issue in report["errors"]}
    assert report["valid"] is False
    assert {
        "E_UNKNOWN_FIELD",
        "E_RANK_SEQUENCE",
        "E_BRANCH_KEYS",
        "E_DUPLICATE_IMAGE_ID",
        "E_PATH_SPLIT",
        "E_DUPLICATE_QUERY_ID",
    }.issubset(codes)
