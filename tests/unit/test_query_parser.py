from anima_search.retrieval.query_parser import QueryParser

ALIASES = {
    "aliases": {
        "人物": ["人", "行人"],
        "城市": ["城区"],
        "雨天": ["雨夜", "下雨"],
        "冷色": ["冷色调"],
    },
    "fields": {
        "objects": ["人物"],
        "scene": ["城市", "雨天"],
        "colors": ["冷色"],
    },
    "negative_prefixes": ["不要", "没有", "无"],
    "negative_exceptions": ["无人机"],
}


class BrokenGenerator:
    def generate_text(self, prompt):
        raise RuntimeError("offline")


class ExtraGenerator:
    def generate_text(self, prompt):
        return '{"semantic_text":"雨夜城区","scene":["城市"],"actions":["行走"],"query_type":"compositional"}'


class ScalarGenerator:
    def generate_text(self, prompt):
        return (
            '{"semantic_text":"雨夜城市","scene":"城市","mood":"安静",'
            '"excluded_terms":["不要","没有"],"query_type":"compositional"}'
        )


def test_rules_extract_negative_compositional_query():
    parsed = QueryParser(aliases=ALIASES).parse("不要人物，寻找冷色调的雨夜城市")
    assert parsed.query_type == "negative"
    assert parsed.excluded_terms == ["人物"]
    assert set(parsed.scene) == {"城市", "雨天"}
    assert parsed.colors == ["冷色"]
    assert "不要人物" not in parsed.semantic_text


def test_rules_extract_quoted_ocr_term():
    parsed = QueryParser(aliases=ALIASES).parse("找招牌写着“老王面馆”的照片")
    assert parsed.query_type == "ocr"
    assert parsed.ocr_terms == ["老王面馆"]


def test_negative_exception_does_not_treat_drone_as_no_people():
    parsed = QueryParser(aliases=ALIASES).parse("无人机拍摄的城市")
    assert parsed.excluded_terms == []
    assert parsed.objects == []


def test_broken_generator_keeps_rule_results():
    parser = QueryParser(aliases=ALIASES)
    parsed = parser.parse("没有行人的城市", BrokenGenerator())
    assert parsed.excluded_terms == ["人物"]
    assert parsed.scene == ["城市"]
    assert parser.last_backend == "rules_fallback"
    assert parser.last_error == "RuntimeError: offline"


def test_generator_adds_fields_without_overwriting_explicit_negative():
    parser = QueryParser(aliases=ALIASES)
    parsed = parser.parse("不要人物的雨夜城市", ExtraGenerator())
    assert parsed.query_type == "negative"
    assert parsed.excluded_terms == ["人物"]
    assert parsed.actions == ["行走"]
    assert parser.last_backend == "llm"
    assert parser.last_error is None


def test_generator_scalar_fields_are_normalized_and_cannot_invent_exclusions():
    parser = QueryParser(aliases=ALIASES)
    parsed = parser.parse("安静的雨夜城市", ScalarGenerator())
    assert parsed.scene == ["城市", "雨天"]
    assert parsed.mood == ["安静"]
    assert parsed.excluded_terms == []
    assert parser.last_backend == "llm"
