from __future__ import annotations

import re

from anima_search.annotation.validation import extract_json_object
from anima_search.retrieval.terms import AliasCatalog, unique_strings
from anima_search.schemas import SearchQuery


_COUNT_PATTERN = re.compile(
    r"(?P<prefix>至少|不少于|不低于|最多|至多|不超过)?\s*"
    r"(?P<number>\d+|[零一二两三四五六七八九十]+)\s*"
    r"(?:个|名|辆|只|张)?\s*"
    r"(?P<target>人物|人|行人|游客|汽车|轿车|车辆|车|自行车|单车|摩托车|狗|猫)"
)


def _number(value: str) -> int:
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        return (digits.get(left, 1) * 10) + digits.get(right, 0)
    return digits[value]


class QueryParser:
    def __init__(self, text_generator: object | None = None, prompt: str = "",
                 aliases: dict | None = None) -> None:
        self.text_generator = text_generator
        self.prompt = prompt
        self.catalog = AliasCatalog(aliases)

    @staticmethod
    def _quoted_terms(query: str) -> list[str]:
        return unique_strings(
            match[0] or match[1]
            for match in re.findall(r'[“"]([^”"]+)[”"]|\'([^\']+)\'', query)
        )

    def _count(self, query: str) -> tuple[str | None, int | None, str | None]:
        match = _COUNT_PATTERN.search(query)
        if match is None:
            return None, None, None
        prefix = match.group("prefix") or ""
        operator = "gte" if prefix in {"至少", "不少于", "不低于"} else (
            "lte" if prefix in {"最多", "至多", "不超过"} else "eq"
        )
        return (
            self.catalog.canonicalize(match.group("target")),
            _number(match.group("number")),
            operator,
        )

    def _rule_parse(self, query: str) -> SearchQuery:
        excluded = self.catalog.find_excluded(query)
        ocr_terms = self._quoted_terms(query)
        semantic = self.catalog.remove_negative_phrases(query, excluded)
        objects = [term for term in self.catalog.find(query, "objects") if term not in excluded]
        scene = [term for term in self.catalog.find(query, "scene") if term not in excluded]
        time_of_day = self.catalog.find(query, "time_of_day")
        if not time_of_day and "夜" in query:
            time_of_day = [self.catalog.canonicalize("夜晚")]
        weather = self.catalog.find(query, "weather")
        colors = [term for term in self.catalog.find(query, "colors") if term not in excluded]
        count_target, count_value, count_operator = self._count(query)
        if count_target and count_target not in objects:
            objects.append(count_target)
        field_count = sum(bool(values) for values in (
            objects, scene, time_of_day, weather, colors, ocr_terms,
        ))
        if excluded:
            query_type = "negative"
        elif ocr_terms:
            query_type = "ocr"
        elif count_value is not None:
            query_type = "count"
        elif field_count >= 2:
            query_type = "compositional"
        else:
            query_type = "simple"
        return SearchQuery(
            raw_text=query,
            semantic_text=semantic,
            query_type=query_type,
            objects=objects,
            scene=scene,
            time_of_day=time_of_day,
            weather=weather,
            colors=colors,
            count_target=count_target,
            count_value=count_value,
            count_operator=count_operator,
            excluded_terms=excluded,
            ocr_terms=ocr_terms,
        )

    @staticmethod
    def _merge(base: SearchQuery, generated: SearchQuery) -> SearchQuery:
        payload = base.model_dump()
        list_fields = (
            "objects", "actions", "scene", "time_of_day", "weather", "mood", "colors", "style",
            "required_terms", "excluded_terms", "ocr_terms",
        )
        for field_name in list_fields:
            payload[field_name] = unique_strings([
                *getattr(base, field_name), *getattr(generated, field_name)
            ])
        for field_name in ("count_target", "count_value", "count_operator"):
            if payload.get(field_name) is None and getattr(generated, field_name) is not None:
                payload[field_name] = getattr(generated, field_name)
        if generated.semantic_text.strip():
            payload["semantic_text"] = generated.semantic_text.strip()
        if base.query_type == "simple" and generated.query_type != "simple":
            payload["query_type"] = generated.query_type
        payload["raw_text"] = base.raw_text
        return SearchQuery.model_validate(payload)

    def parse(self, query: str, generator: object | None = None) -> SearchQuery:
        base = self._rule_parse(query)
        source = generator or self.text_generator
        if source is None:
            return base
        try:
            source = source() if callable(source) else source
            payload = extract_json_object(source.generate_text(f"{self.prompt}\n用户查询：{query}"))
            payload["raw_text"] = query
            return self._merge(base, SearchQuery.model_validate(payload))
        except Exception:  # optional model failures must preserve deterministic parsing
            return base
