from __future__ import annotations

from dataclasses import dataclass, field

from anima_search.indexing.documents import annotation_fields
from anima_search.retrieval.terms import AliasCatalog, normalize_text, unique_strings
from anima_search.schemas import ImageAnnotation, SearchQuery


@dataclass
class FilterDecision:
    allowed: bool = True
    matched_fields: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    mismatch: list[str] = field(default_factory=list)


class AnnotationFilter:
    def __init__(self, aliases: dict | None = None) -> None:
        self.catalog = AliasCatalog(aliases)

    def _term_matches_values(self, term: str, values: list[str]) -> bool:
        aliases = [normalize_text(value) for value in self.catalog.expand(term)]
        return any(
            (normalized_value == alias if len(alias) == 1 else alias in normalized_value)
            for alias in aliases
            for normalized_value in (normalize_text(value) for value in values)
        )

    def _matching_fields(self, annotation: ImageAnnotation, term: str,
                         preferred_field: str | None = None) -> list[str]:
        fields = annotation_fields(annotation)
        names = [preferred_field] if preferred_field in fields else list(fields)
        return [name for name in names if self._term_matches_values(term, fields[name])]

    def evaluate(self, annotation: ImageAnnotation, query: SearchQuery) -> FilterDecision:
        decision = FilterDecision()

        for term in unique_strings(query.excluded_terms):
            preferred = self.catalog.field_for(term)
            matches = self._matching_fields(annotation, term, preferred)
            if matches:
                decision.allowed = False
                decision.mismatch.append(f"排除条件命中:{term}({','.join(matches)})")

        for term in unique_strings(query.required_terms):
            matches = self._matching_fields(annotation, term)
            if matches:
                decision.matched_fields.extend(matches)
                decision.evidence.append(f"必须词命中:{term}({','.join(matches)})")
            else:
                decision.allowed = False
                decision.mismatch.append(f"缺少必须词:{term}")

        for term in unique_strings(query.ocr_terms):
            if self._term_matches_values(term, annotation.ocr_text):
                decision.matched_fields.append("ocr_text")
                decision.evidence.append(f"OCR命中:{term}")
            else:
                decision.allowed = False
                decision.mismatch.append(f"OCR未命中:{term}")

        structured = {
            "objects": query.objects,
            "actions": query.actions,
            "scene": query.scene,
            "mood": query.mood,
            "colors": query.colors,
            "style": query.style,
        }
        for field_name, terms in structured.items():
            values = annotation_fields(annotation).get(field_name, [])
            for term in terms:
                if term not in query.excluded_terms and self._term_matches_values(term, values):
                    decision.matched_fields.append(field_name)
                    decision.evidence.append(f"{field_name}命中:{term}")

        decision.matched_fields = unique_strings(decision.matched_fields)
        decision.evidence = unique_strings(decision.evidence)
        decision.mismatch = unique_strings(decision.mismatch)
        return decision
