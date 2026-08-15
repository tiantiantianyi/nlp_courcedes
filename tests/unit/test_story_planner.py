from __future__ import annotations

from anima_search.m7.story_planner import (
    build_story_gaps,
    order_story_candidates,
    scene_similarity,
    time_bucket,
)
from anima_search.schemas import ImageAnnotation, SearchResult


def annotation(
    image_id: str,
    *,
    time: str,
    scene: str,
    objects: list[str] | None = None,
) -> ImageAnnotation:
    return ImageAnnotation(
        image_id=image_id,
        split="Val",
        relative_path=f"Val/{image_id}.jpg",
        sha256=image_id,
        summary=f"{time}{scene}",
        objects=objects or [],
        scene=scene,
        attributes=[f"time_of_day:{time}"],
        search_queries=["a", "b", "c"],
        generation_prompt=f"{time} {scene}",
        model_version="fake",
        prompt_version="v1",
    )


def result(image_id: str) -> SearchResult:
    return SearchResult(
        image_id=image_id,
        relative_path=f"Val/{image_id}.jpg",
        fused_score=1.0,
    )


def test_story_order_uses_morning_to_night_time_buckets():
    annotations = {
        "night": annotation("night", time="夜晚", scene="城市"),
        "morning": annotation("morning", time="早晨", scene="公园"),
        "dusk": annotation("dusk", time="黄昏", scene="街道"),
    }
    ordered = order_story_candidates(
        [result("night"), result("morning"), result("dusk")],
        annotations,
    )
    assert [item.image_id for item in ordered] == ["morning", "dusk", "night"]
    assert time_bucket(annotations["morning"]) == (1, "早晨")


def test_scene_similarity_keeps_related_images_adjacent_within_same_time():
    annotations = {
        "city-a": annotation(
            "city-a", time="夜晚", scene="城市", objects=["高楼", "汽车"]
        ),
        "nature": annotation(
            "nature", time="夜晚", scene="自然", objects=["树木"]
        ),
        "city-b": annotation(
            "city-b", time="夜晚", scene="城市", objects=["高楼"]
        ),
    }
    ordered = order_story_candidates(
        [result("city-a"), result("nature"), result("city-b")],
        annotations,
    )
    assert [item.image_id for item in ordered] == ["city-a", "city-b", "nature"]
    assert scene_similarity(annotations["city-a"], annotations["city-b"]) > 0


def test_story_gap_records_generation_contract_and_ai_marker():
    annotations = {
        "morning": annotation("morning", time="早晨", scene="公园"),
        "night": annotation("night", time="夜晚", scene="城市"),
    }
    gaps = build_story_gaps(
        [result("morning"), result("night")],
        annotations,
    )
    assert len(gaps) == 1
    assert gaps[0].status == "missing"
    assert gaps[0].source == "generated"
    assert gaps[0].ai_generated
    assert "早晨" in gaps[0].reason and "夜晚" in gaps[0].reason


def test_story_order_preserves_selection_without_annotations():
    selected = [result("b"), result("a"), result("c")]
    assert order_story_candidates(selected, {}) == selected


def test_canonical_dawn_dusk_uses_explicit_twilight_bucket():
    item = annotation("twilight", time="dawn_dusk", scene="海边")

    assert time_bucket(item) == (4, "晨昏")
