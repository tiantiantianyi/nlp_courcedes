from __future__ import annotations

import re

_CITATION = re.compile(r"\[img_([^\]]+)\]")


def citation_token(image_id: str) -> str:
    return f"[img_{image_id}]"


def extract_citations(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for image_id in _CITATION.findall(text):
        if image_id not in seen:
            seen.add(image_id)
            result.append(image_id)
    return result


def validate_citations(text: str, declared: list[str], allowed: set[str], *, refused: bool) -> list[str]:
    inline = extract_citations(text)
    normalized = list(dict.fromkeys([*declared, *inline]))
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise ValueError(f"answer cites images outside the current evidence set: {invalid}")
    if not refused and not normalized:
        raise ValueError("a non-refusal answer must cite at least one evidence image")
    return normalized
