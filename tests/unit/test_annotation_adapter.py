import pytest

from anima_search.adapters.annotation import CANONICAL_QWEN35_VERSION, adapt_annotation
from anima_search.retrieval.filters import AnnotationFilter
from anima_search.retrieval.query_parser import QueryParser
from anima_search.schemas import ImageAnnotation, ManifestItem


def manifest() -> ManifestItem:
    return ManifestItem(
        image_id="val-2244", split="Val", relative_path="../Val/2244.jpg",
        sha256="abc", size_bytes=123,
    )


def test_adapts_nested_proposal_schema_and_preserves_counts():
    annotation = adapt_annotation({
        "image_id": "val-2244",
        "scene": {"type": "街景", "sub_type": "商业街", "indoor_outdoor": "outdoor"},
        "capture": {"time_of_day": "夜晚", "weather": "雨"},
        "objects": [{"name": "汽车", "count": 2, "attrs": {"color": "红色"}}],
        "persons": {"count": 1, "activity": "行走"},
        "ocr": [{"text": "便利店"}],
        "relations": ["汽车位于商店前"],
        "affect": {"mood": ["安静"], "palette": ["蓝色"]},
        "caption_dense": "雨夜商业街上有两辆汽车和一名行人。",
        "keywords_zh": ["雨夜", "汽车"],
        "uncertain_fields": ["具体地点"],
    }, manifest())
    assert annotation.scene == "街景"
    assert set(annotation.objects) == {"汽车", "人物"}
    assert "count:汽车=2" in annotation.attributes
    assert "count:人物=1" in annotation.attributes
    assert "time_of_day:夜晚" in annotation.attributes
    assert annotation.ocr_text == ["便利店"]


def test_accepts_existing_flat_schema_and_normalizes_windows_path():
    flat = ImageAnnotation(
        image_id="train-1", split="Train", relative_path="..\\Train\\1.jpg", sha256="1",
        summary="城市", scene="城市", search_queries=["a", "b", "c"],
        generation_prompt="city", model_version="qwen", prompt_version="v4",
    ).model_dump()
    adapted = adapt_annotation(flat)
    assert adapted.relative_path == "../Train/1.jpg"


def test_nested_schema_requires_caption():
    try:
        adapt_annotation({"image_id": "val-1", "scene": {"type": "城市"}}, manifest())
    except ValueError as exc:
        assert "caption" in str(exc)
    else:
        raise AssertionError("missing caption must fail")


def test_adapts_qwen35_canonical_v13_and_checks_manifest_identity():
    payload = {
        "image_id": "2244",
        "processed_sha256": "abc",
        "source_model_id": "Qwen/Qwen3.5-9B",
        "annotation": {
            "scene": {
                "primary_type": "street_urban",
                "secondary_types": ["transport"],
                "media_type": "natural_image",
                "sub_type_zh": "雨夜街道",
                "environment": "outdoor",
            },
            "capture_visual": {"time_of_day": "night", "weather": "rain"},
            "entities": [
                {
                    "entity_id": "e1",
                    "entity_type": "vehicle",
                    "name_zh": "汽车",
                    "count": 2,
                    "count_exact": True,
                    "position_zone": "center",
                    "salience": "primary",
                    "visibility": "clear",
                    "attributes": {
                        "colors_zh": ["红色"],
                        "materials_zh": ["金属"],
                        "states_zh": ["行驶中"],
                        "action_zh": "行驶",
                        "attire_zh": [],
                    },
                },
                {
                    "entity_id": "e2",
                    "entity_type": "object",
                    "name_zh": "商店",
                    "count": 1,
                    "count_exact": True,
                    "attributes": {
                        "colors_zh": [],
                        "materials_zh": [],
                        "states_zh": [],
                        "action_zh": None,
                        "attire_zh": [],
                    },
                },
            ],
            "ocr": [{"text_raw": "便利店"}],
            "relations": [
                {
                    "subject_id": "e1",
                    "predicate": "in_front_of",
                    "object_id": "e2",
                    "predicate_other_zh": None,
                }
            ],
            "event": {"summary_zh": "汽车在雨中行驶。"},
            "subjective": {
                "mood_terms_zh": ["安静"],
                "palette_terms_zh": ["冷色"],
            },
            "captions": {
                "short_zh": "雨夜街道",
                "dense_zh": "雨夜街道上有两辆红色汽车，附近有便利店。",
            },
            "uncertainties": [],
        },
    }
    annotation = adapt_annotation(payload, manifest())
    assert annotation.image_id == "val-2244"
    assert annotation.prompt_version == CANONICAL_QWEN35_VERSION
    assert annotation.object_counts == {"汽车": 2, "商店": 1}
    assert "time_of_day:夜晚" in annotation.attributes
    assert "time_of_day_code:night" in annotation.attributes
    assert "weather:雨天" in annotation.attributes
    assert annotation.ocr_text == ["便利店"]
    assert annotation.spatial_relations == ["汽车位于前方商店"]

    aliases = {
        "positive_filter_mode": "hybrid",
        "aliases": {
            "街道": ["街景"],
            "夜晚": ["夜间", "雨夜"],
            "雨天": ["下雨", "雨夜"],
        },
        "fields": {
            "scene": ["街道"],
            "time_of_day": ["夜晚"],
            "weather": ["雨天"],
        },
    }
    query = QueryParser(aliases=aliases).parse("雨夜街景")
    assert AnnotationFilter(aliases).evaluate(annotation, query).allowed

    payload["processed_sha256"] = "wrong"
    with pytest.raises(ValueError, match="processed_sha256 mismatch"):
        adapt_annotation(payload, manifest())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("processed_sha256", None, "processed_sha256 is required"),
        ("source_model_id", None, "source_model_id is required"),
        ("source_model_id", "OpenGVLab/InternVL3-8B", "source_model_id mismatch"),
    ],
)
def test_canonical_qwen35_rejects_missing_or_wrong_provenance(
    field: str,
    value: str | None,
    message: str,
) -> None:
    payload = {
        "image_id": "2244",
        "processed_sha256": "abc",
        "source_model_id": "Qwen/Qwen3.5-9B",
        "annotation": {
            "scene": {"primary_type": "general"},
            "captions": {"short_zh": "测试图像"},
        },
    }
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value

    with pytest.raises(ValueError, match=message):
        adapt_annotation(payload, manifest())
