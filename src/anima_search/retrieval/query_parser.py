import re

from anima_search.annotation.validation import extract_json_object
from anima_search.retrieval.terms import AliasCatalog, unique_strings
from anima_search.schemas import SearchQuery


class QueryParser:
    def __init__(self, text_generator: object | None = None, prompt: str = "",
                 aliases: dict | None = None) -> None:
        self.text_generator = text_generator
        self.prompt = prompt
        self.catalog = AliasCatalog(aliases)

    @staticmethod
    def _quoted_terms(query: str) -> list[str]:
        return unique_strings(match[0] or match[1] for match in re.findall(r'[“"]([^”"]+)[”"]|\'([^\']+)\'', query))

    def _rule_parse(self, query: str) -> SearchQuery:
        excluded = self.catalog.find_excluded(query)
        ocr_terms = self._quoted_terms(query)
        semantic = self.catalog.remove_negative_phrases(query, excluded)
        objects = [term for term in self.catalog.find(query, "objects") if term not in excluded]
        scene = [term for term in self.catalog.find(query, "scene") if term not in excluded]
        colors = [term for term in self.catalog.find(query, "colors") if term not in excluded]
        field_count = sum(bool(values) for values in (objects, scene, colors, ocr_terms))
        if excluded:
            query_type = "negative"
        elif ocr_terms:
            query_type = "ocr"
        elif re.search(r"\d+|[一二三四五六七八九十]+(?:个|名|辆|只|张)", query):
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
            colors=colors,
            excluded_terms=excluded,
            ocr_terms=ocr_terms,
        )

    @staticmethod
    def _merge(base: SearchQuery, generated: SearchQuery) -> SearchQuery:
        payload = base.model_dump()
        list_fields = (
            "objects", "actions", "scene", "mood", "colors", "style",
            "required_terms", "excluded_terms", "ocr_terms",
        )
        for field_name in list_fields:
            payload[field_name] = unique_strings([
                *getattr(base, field_name), *getattr(generated, field_name)
            ])
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
            generated = SearchQuery.model_validate(payload)
            return self._merge(base, generated)
        except Exception:  # noqa: BLE001 - optional model failures must preserve rule parsing
            return base
