from __future__ import annotations

import json
from typing import Any


_LIST_FIELDS = (
    "objects",
    "actions",
    "attributes",
    "spatial_relations",
    "style",
    "mood",
    "colors",
    "ocr_text",
    "search_queries",
    "uncertainty",
)
_ANNOTATION_FIELDS = set(_LIST_FIELDS) | {"summary", "scene", "generation_prompt"}


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output does not contain a valid JSON object")


def extract_annotation_json(text: str, max_ocr_items: int = 10) -> dict[str, Any]:
    try:
        return extract_json_object(text)
    except ValueError:
        pass

    marker_index = text.find('"ocr_text"')
    array_index = text.find("[", marker_index)
    if marker_index < 0 or array_index < 0:
        raise ValueError("Model output does not contain a valid JSON object")

    decoder = json.JSONDecoder()
    values: list[str] = []
    cursor = array_index + 1
    while len(values) < max_ocr_items:
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
            cursor += 1
        if cursor >= len(text) or text[cursor] != '"':
            break
        try:
            value, consumed = decoder.raw_decode(text[cursor:])
        except json.JSONDecodeError:
            break
        cursor += consumed
        if isinstance(value, str) and value not in values:
            values.append(value)

    repaired = text[:array_index] + json.dumps(values, ensure_ascii=False) + "}"
    return extract_json_object(repaired)


def normalize_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce common VLM JSON variants into the schema's list-shaped fields."""
    normalized = dict(payload)
    for key in list(normalized):
        canonical = key.lstrip("_")
        if canonical in _ANNOTATION_FIELDS and canonical not in normalized:
            normalized[canonical] = normalized.pop(key)
    for field in _LIST_FIELDS:
        value = normalized.get(field, [])
        if value is None or value == "":
            normalized[field] = []
        elif isinstance(value, dict):
            normalized[field] = [f"{key}:{item}" for key, item in value.items()][:10]
        elif isinstance(value, list):
            normalized[field] = list(dict.fromkeys(str(item) for item in value))[:10]
        else:
            normalized[field] = [str(value)]
    if isinstance(normalized.get("scene"), list):
        normalized["scene"] = "、".join(str(item) for item in normalized["scene"])
    return normalized
