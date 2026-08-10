from anima_search.retrieval.filters import AnnotationFilter
from anima_search.retrieval.query_parser import QueryParser
from anima_search.schemas import ImageAnnotation


ALIASES = {
    "positive_filter_mode": "hybrid",
    "aliases": {
        "汽车": ["车", "轿车"], "人物": ["人", "行人"],
        "街道": ["街景"], "夜晚": ["夜间", "晚上"], "雨天": ["下雨", "雨夜"],
    },
    "fields": {
        "objects": ["汽车", "人物"], "scene": ["街道"],
        "time_of_day": ["夜晚"], "weather": ["雨天"],
    },
}


def annotation(*, cars: int, time: str = "夜晚", weather: str = "雨天") -> ImageAnnotation:
    return ImageAnnotation(
        image_id="val-1", split="Val", relative_path="../Val/1.jpg", sha256="1",
        summary="雨夜街道", objects=["汽车"], object_counts={"汽车": cars}, scene="街道",
        attributes=[f"time_of_day:{time}", f"weather:{weather}"],
        search_queries=["a", "b", "c"], generation_prompt="city",
        model_version="qwen", prompt_version="v4",
    )


def test_parser_extracts_count_time_and_weather():
    parsed = QueryParser(aliases=ALIASES).parse("至少三辆车的雨夜街景")
    assert parsed.query_type == "count"
    assert parsed.count_target == "汽车"
    assert parsed.count_value == 3 and parsed.count_operator == "gte"
    assert parsed.time_of_day == ["夜晚"]
    assert parsed.weather == ["雨天"]
    assert parsed.scene == ["街道"]


def test_count_and_time_are_executable_hard_filters():
    query = QueryParser(aliases=ALIASES).parse("至少三辆车的雨夜街景")
    matcher = AnnotationFilter(ALIASES)
    assert matcher.evaluate(annotation(cars=3), query).allowed
    assert not matcher.evaluate(annotation(cars=2), query).allowed
    assert not matcher.evaluate(annotation(cars=3, time="白天"), query).allowed


def test_soft_mode_keeps_unmatched_positive_scene():
    aliases = {**ALIASES, "positive_filter_mode": "soft"}
    query = QueryParser(aliases=aliases).parse("街景")
    other = annotation(cars=1).model_copy(update={"scene": "室内"})
    assert AnnotationFilter(aliases).evaluate(other, query).allowed
