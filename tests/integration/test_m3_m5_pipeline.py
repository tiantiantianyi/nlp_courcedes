from anima_search.retrieval.query_parser import QueryParser
from anima_search.retrieval.search import HybridSearcher
from anima_search.schemas import ImageAnnotation


class FixtureIndex:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, limit):
        return self.ranking[:limit]


def test_annotation_to_query_to_three_branch_search():
    aliases = {
        "aliases": {"人物": ["人", "行人"], "冷色": ["冷色调"], "城市": ["城区"]},
        "fields": {"objects": ["人物"], "colors": ["冷色"], "scene": ["城市"]},
        "negative_prefixes": ["不要", "没有", "无"],
    }
    annotations = {
        "with-person": ImageAnnotation(
            image_id="with-person", split="Val", relative_path="Val/1.jpg", sha256="1",
            summary="冷色城市中的行人", objects=["行人"], scene="城市", colors=["冷色"],
            search_queries=["a", "b", "c"], generation_prompt="city", model_version="qwen",
            prompt_version="v1",
        ),
        "empty-city": ImageAnnotation(
            image_id="empty-city", split="Val", relative_path="Val/2.jpg", sha256="2",
            summary="空旷的冷色城市街景", objects=["汽车"], scene="城市", colors=["冷色"],
            search_queries=["a", "b", "c"], generation_prompt="city", model_version="qwen",
            prompt_version="v1",
        ),
    }
    indexes = {
        "image": FixtureIndex([("with-person", 0.9), ("empty-city", 0.8)]),
        "text": FixtureIndex([("empty-city", 0.9), ("with-person", 0.8)]),
        "bm25": FixtureIndex([("with-person", 3.0), ("empty-city", 2.0)]),
    }
    query = QueryParser(aliases=aliases).parse("不要人物，找冷色调的城市")
    results = HybridSearcher(annotations, indexes=indexes, aliases=aliases).search(query)

    assert query.excluded_terms == ["人物"]
    assert [result.image_id for result in results] == ["empty-city"]
    assert results[0].active_branches == ["image", "text", "bm25"]
    assert set(results[0].matched_fields) == {"scene", "colors"}
