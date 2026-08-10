from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from anima_search.schemas import ImageAnnotation, ManifestItem


def _strings(value: object) -> list[str]:
    if value is None or value == "":
        return []
    source = value if isinstance(value, list) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in source:
        text = str(item).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _object_fields(value: object) -> tuple[list[str], dict[str, int], list[str]]:
    names: list[str] = []
    counts: dict[str, int] = {}
    attributes: list[str] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            names.extend(_strings(item))
            continue
        data = _mapping(item)
        name = str(data.get("name", "")).strip()
        if not name:
            continue
        names.extend(_strings(name))
        count = data.get("count")
        if isinstance(count, int) and count >= 0:
            counts[name] = count
            attributes.append(f"count:{name}={count}")
        for key, attr_value in _mapping(data.get("attrs")).items():
            if str(attr_value).strip():
                attributes.append(f"{key}:{attr_value}")
    return _strings(names), counts, _strings(attributes)


def _query_seeds(summary: str, scene: str, keywords: list[str]) -> list[str]:
    result = [item.strip() for item in (summary, " ".join(keywords), scene) if item.strip()]
    while len(result) < 3:
        result.append(summary)
    return result[:3]


def adapt_annotation(payload: Mapping[str, Any], manifest: ManifestItem | None = None,
                     *, default_split: str = "Val") -> ImageAnnotation:
    """Convert the proposal's nested schema or the existing flat schema to ImageAnnotation."""
    if "summary" in payload and isinstance(payload.get("objects", []), list) and not any(
        isinstance(item, Mapping) for item in payload.get("objects", [])
    ):
        flat = dict(payload)
        flat["relative_path"] = str(flat.get("relative_path", "")).replace("\\", "/")
        return ImageAnnotation.model_validate(flat)

    scene_data = _mapping(payload.get("scene"))
    scene = str(scene_data.get("type") or payload.get("scene") or "未知场景").strip()
    capture = _mapping(payload.get("capture"))
    affect = _mapping(payload.get("affect"))
    objects, object_counts, object_attributes = _object_fields(payload.get("objects", []))

    persons = _mapping(payload.get("persons"))
    person_count = persons.get("count")
    if isinstance(person_count, int) and person_count >= 0:
        object_counts["人物"] = person_count
        object_attributes.append(f"count:人物={person_count}")
        if person_count > 0:
            objects = _strings([*objects, "人物"])

    attributes = list(object_attributes)
    for key in ("time_of_day", "weather", "lighting", "viewpoint", "shot_type", "blur"):
        value = capture.get(key)
        if value not in (None, ""):
            attributes.append(f"{key}:{value}")
    for key in ("sub_type", "indoor_outdoor"):
        value = scene_data.get(key)
        if value not in (None, ""):
            attributes.append(f"{key}:{value}")
    for key in ("activity", "attire"):
        value = persons.get(key)
        if value not in (None, ""):
            attributes.append(f"person_{key}:{value}")

    ocr_text = []
    for item in payload.get("ocr", []) if isinstance(payload.get("ocr"), list) else []:
        ocr_text.extend(_strings(item if isinstance(item, str) else _mapping(item).get("text")))

    summary = str(payload.get("caption_dense") or payload.get("caption_short") or "").strip()
    if not summary:
        raise ValueError("nested annotation must provide caption_dense or caption_short")
    keywords = _strings([*(_strings(payload.get("keywords_zh"))), *(_strings(payload.get("keywords_en")))])
    image_id = str(payload.get("image_id") or (manifest.image_id if manifest else "")).strip()
    if not image_id:
        raise ValueError("annotation image_id is required")
    split = str(payload.get("split") or (manifest.split if manifest else default_split)).title()
    relative_path = str(
        payload.get("relative_path") or (manifest.relative_path if manifest else "")
    ).replace("\\", "/")
    sha256 = str(payload.get("sha256") or (manifest.sha256 if manifest else ""))
    if not relative_path or not sha256:
        raise ValueError("relative_path and sha256 are required; provide them or a manifest item")

    return ImageAnnotation(
        image_id=image_id,
        split=split,
        relative_path=relative_path,
        sha256=sha256,
        duplicate_of=manifest.duplicate_of if manifest else None,
        summary=summary,
        objects=objects,
        object_counts=object_counts,
        actions=_strings([payload.get("events"), persons.get("activity")]),
        scene=scene,
        attributes=_strings(attributes),
        spatial_relations=_strings(payload.get("relations")),
        style=_strings(payload.get("style")),
        mood=_strings(affect.get("mood")),
        colors=_strings(affect.get("palette")),
        ocr_text=_strings(ocr_text),
        search_queries=_query_seeds(summary, scene, keywords),
        generation_prompt=str(payload.get("generation_prompt") or summary),
        uncertainty=_strings(payload.get("uncertain_fields")),
        model_version=str(payload.get("model_version") or "unknown"),
        prompt_version=str(payload.get("prompt_version") or "proposal-adapter-v1"),
    )
