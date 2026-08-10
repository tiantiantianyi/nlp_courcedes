from anima_search.adapters.annotation import adapt_annotation
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
