from anima_search.retrieval.filters import AnnotationFilter
from anima_search.schemas import ImageAnnotation, SearchQuery

ALIASES = {
    "aliases": {"人物": ["人", "行人", "游客"]},
    "fields": {"objects": ["人物"]},
}


def annotation(image_id: str, *, objects=None, ocr=None) -> ImageAnnotation:
    return ImageAnnotation(
        image_id=image_id,
        split="Val",
        relative_path=f"Val/{image_id}.jpg",
        sha256=image_id,
        summary="城市街景",
        objects=objects or [],
        scene="城市",
        ocr_text=ocr or [],
        search_queries=["a", "b", "c"],
        generation_prompt="city",
        model_version="qwen",
        prompt_version="v1",
    )


def test_negative_object_filter_uses_aliases():
    decision = AnnotationFilter(ALIASES).evaluate(
        annotation("person", objects=["行人"]),
        SearchQuery(raw_text="不要人物", excluded_terms=["人物"]),
    )
    assert not decision.allowed
    assert "objects" in decision.mismatch[0]


def test_absent_object_is_not_rejected_by_negative_filter():
    decision = AnnotationFilter(ALIASES).evaluate(
        annotation("empty", objects=["汽车"]),
        SearchQuery(raw_text="不要人物", excluded_terms=["人物"]),
    )
    assert decision.allowed


def test_single_character_alias_does_not_match_drone_object():
    decision = AnnotationFilter(ALIASES).evaluate(
        annotation("drone", objects=["无人机"]),
        SearchQuery(raw_text="不要人物", excluded_terms=["人物"]),
    )
    assert decision.allowed


def test_ocr_term_is_a_hard_filter_and_evidence():
    matcher = AnnotationFilter(ALIASES)
    matched = matcher.evaluate(
        annotation("shop", ocr=["老王面馆"]),
        SearchQuery(raw_text="老王面馆", ocr_terms=["老王面馆"]),
    )
    missing = matcher.evaluate(
        annotation("other", ocr=["便利店"]),
        SearchQuery(raw_text="老王面馆", ocr_terms=["老王面馆"]),
    )
    assert matched.allowed and matched.matched_fields == ["ocr_text"]
    assert not missing.allowed
