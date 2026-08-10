from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


def unique_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        key = normalize_text(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


class AliasCatalog:
    def __init__(self, payload: dict | None = None) -> None:
        payload = payload or {}
        raw_aliases = payload.get("aliases", {})
        self.aliases: dict[str, list[str]] = {
            str(canonical): unique_strings([str(canonical), *map(str, values or [])])
            for canonical, values in raw_aliases.items()
        }
        self.fields: dict[str, list[str]] = {
            str(field): unique_strings(map(str, values or []))
            for field, values in payload.get("fields", {}).items()
        }
        self.negative_prefixes = unique_strings(
            map(str, payload.get("negative_prefixes", ["不要", "没有", "无", "不含", "排除", "避开"]))
        )
        self.negative_exceptions = unique_strings(map(str, payload.get("negative_exceptions", [])))
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_to_field: dict[str, str] = {}
        for canonical, aliases in self.aliases.items():
            for alias in aliases:
                self._alias_to_canonical[normalize_text(alias)] = canonical
        for field, canonicals in self.fields.items():
            for canonical in canonicals:
                self._canonical_to_field[canonical] = field

    def canonicalize(self, term: str) -> str:
        return self._alias_to_canonical.get(normalize_text(term), term.strip())

    def expand(self, term: str) -> list[str]:
        canonical = self.canonicalize(term)
        return self.aliases.get(canonical, [canonical])

    def field_for(self, term: str) -> str | None:
        return self._canonical_to_field.get(self.canonicalize(term))

    def find(self, text: str, field: str | None = None) -> list[str]:
        normalized = normalize_text(text)
        for exception in self.negative_exceptions:
            normalized = normalized.replace(normalize_text(exception), "")
        candidates = self.fields.get(field, []) if field else list(self.aliases)
        found = []
        for canonical in candidates:
            if any(normalize_text(alias) in normalized for alias in self.expand(canonical)):
                found.append(canonical)
        return unique_strings(found)

    def find_excluded(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        for exception in self.negative_exceptions:
            normalized = normalized.replace(normalize_text(exception), "")
        found: list[str] = []
        for canonical, aliases in self.aliases.items():
            for prefix in self.negative_prefixes:
                for alias in aliases:
                    patterns = (
                        f"{normalize_text(prefix)}{normalize_text(alias)}",
                        f"{normalize_text(prefix)}有{normalize_text(alias)}",
                        f"{normalize_text(prefix)}任何{normalize_text(alias)}",
                        f"{normalize_text(prefix)}的{normalize_text(alias)}",
                    )
                    if any(pattern in normalized for pattern in patterns):
                        found.append(canonical)
                        break
                else:
                    continue
                break
        return unique_strings(found)

    def remove_negative_phrases(self, text: str, excluded_terms: list[str]) -> str:
        result = text
        for term in excluded_terms:
            for prefix in self.negative_prefixes:
                for alias in self.expand(term):
                    pattern = re.compile(
                        rf"{re.escape(prefix)}\s*(?:有|任何|的)?\s*{re.escape(alias)}",
                        flags=re.IGNORECASE,
                    )
                    result = pattern.sub("", result)
        return re.sub(r"[，,。；;]+", " ", result).strip() or text.strip()
