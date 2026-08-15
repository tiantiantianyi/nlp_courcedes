from __future__ import annotations

import json
from pathlib import Path

import pytest

from anima_search.m7.canonical_annotations import (
    load_canonical_m7_annotations,
)


def _record() -> dict[str, object]:
    return {
        "image_id": "2002",
        "processed_sha256": "a" * 64,
        "source_model_id": "Qwen/Qwen3.5-9B",
        "normalizer_version": "m1-normalize-v1.0.0",
        "repairs_applied": 1,
        "lossy_repairs": False,
        "annotation": {
            "scene": {
                "primary_type": "street_urban",
                "secondary_types": [],
                "media_type": "natural_image",
                "sub_type_zh": "城市街道",
                "environment": "outdoor",
            },
            "capture_visual": {
                "time_of_day": "night",
                "weather": "clear",
                "lighting": "artificial",
                "viewpoint": "eye_level",
                "shot_scale": "wide",
                "blur_level": "none",
            },
            "entities": [
                {
                    "entity_id": "e1",
                    "entity_type": "vehicle",
                    "name_zh": "汽车",
                    "count": 2,
                    "count_exact": True,
                    "attributes": {
                        "colors_zh": ["黑色"],
                        "materials_zh": ["金属"],
                        "states_zh": ["行驶中"],
                        "action_zh": "行驶",
                        "attire_zh": [],
                    },
                }
            ],
            "ocr": [{"text_raw": "便利店"}],
            "relations": [],
            "event": {"summary_zh": "汽车在夜间街道行驶"},
            "subjective": {
                "mood_terms_zh": ["安静"],
                "palette_terms_zh": ["冷色"],
            },
            "captions": {
                "short_zh": "夜间街道。",
                "dense_zh": "黑色汽车驶过夜间城市街道。",
            },
            "uncertainties": [
                {
                    "field_path": "/entities/e1/count",
                    "reason": "occlusion",
                    "note_zh": "远处车辆可能被遮挡",
                }
            ],
        },
    }


def _paths(tmp_path: Path, record: dict[str, object]) -> tuple[Path, Path]:
    annotation_path = tmp_path / "qwen35.jsonl"
    annotation_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "val.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image_id": "val-2002",
                "split": "Val",
                "relative_path": "../Val/2002.jpg",
                "sha256": "a" * 64,
                "size_bytes": 100,
                "valid": True,
                "error": None,
                "duplicate_of": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return annotation_path, manifest_path


def test_canonical_record_maps_to_m7_annotation_without_loss(
    tmp_path: Path,
) -> None:
    annotation_path, manifest_path = _paths(tmp_path, _record())

    loaded = load_canonical_m7_annotations(
        annotation_path,
        [manifest_path],
        split="Val",
    )

    item = loaded["val-2002"]
    assert item.summary == "黑色汽车驶过夜间城市街道。"
    assert item.scene == "城市街道"
    assert item.objects == ["汽车"]
    assert item.object_counts == {"汽车": 2}
    assert item.actions == ["行驶"]
    assert item.colors == ["冷色", "黑色"]
    assert item.mood == ["安静"]
    assert item.ocr_text == ["便利店"]
    assert item.uncertainty == ["远处车辆可能被遮挡"]
    assert "time_of_day:night" in item.attributes
    assert "state:汽车=行驶中" in item.attributes
    assert item.search_queries == [
        "黑色汽车驶过夜间城市街道。",
        "夜间街道。",
        "城市街道",
    ]
    assert item.model_version == "Qwen/Qwen3.5-9B"
    assert item.prompt_version == "canonical-v1.3"
    assert item.generation_parameters == {
        "normalizer_version": "m1-normalize-v1.0.0",
        "repairs_applied": 1,
        "lossy_repairs": False,
    }


def test_loader_rejects_processed_hash_mismatch(tmp_path: Path) -> None:
    record = _record()
    record["processed_sha256"] = "c" * 64
    annotation_path, manifest_path = _paths(tmp_path, record)

    with pytest.raises(ValueError, match="processed_sha256"):
        load_canonical_m7_annotations(
            annotation_path,
            [manifest_path],
            split="Val",
        )


def test_unreliable_same_name_count_is_not_exposed(tmp_path: Path) -> None:
    record = _record()
    annotation = record["annotation"]
    annotation["entities"].append(  # type: ignore[index]
        {
            "entity_id": "e2",
            "entity_type": "vehicle",
            "name_zh": "汽车",
            "count": None,
            "count_exact": False,
            "attributes": {
                "colors_zh": ["白色"],
                "materials_zh": [],
                "states_zh": [],
                "action_zh": None,
                "attire_zh": [],
            },
        }
    )
    annotation_path, manifest_path = _paths(tmp_path, record)

    loaded = load_canonical_m7_annotations(
        annotation_path,
        [manifest_path],
    )

    assert "汽车" not in loaded["val-2002"].object_counts
