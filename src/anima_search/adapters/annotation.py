from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from anima_search.schemas import ImageAnnotation, ManifestItem


CANONICAL_QWEN35_VERSION = "qwen35-canonical-v1.3"
CANONICAL_QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"

_RELATION_ZH = {
    "left_of": "位于左侧",
    "right_of": "位于右侧",
    "above": "位于上方",
    "below": "位于下方",
    "in_front_of": "位于前方",
    "behind": "位于后方",
    "inside": "位于内部",
    "on": "位于上面",
    "under": "位于下面",
    "next_to": "邻近",
    "overlapping": "重叠",
    "holding": "拿着",
    "wearing": "穿戴",
    "riding": "骑乘",
    "looking_at": "看向",
    "eating": "正在吃",
}

_ENUM_ZH = {
    "primary_type": {
        "general": "一般场景",
        "indoor": "室内",
        "street_urban": "城市街道",
        "nature": "自然风景",
        "people_activity": "人物活动",
        "food": "美食",
        "transport": "交通出行",
        "animal_plant": "动植物",
        "object_exhibit": "物品展陈",
        "illustration_meme": "插画表情包",
        "document_screen": "文档屏幕",
    },
    "environment": {"indoor": "室内", "outdoor": "室外", "mixed": "室内外混合"},
    "time_of_day": {"day": "白天", "dawn_dusk": "黄昏", "night": "夜晚"},
    "weather": {
        "clear": "晴天",
        "cloudy": "阴天",
        "rain": "雨天",
        "snow": "雪天",
        "fog": "雾天",
    },
    "media_type": {
        "natural_image": "自然图像",
        "illustration": "插画",
        "screenshot": "屏幕截图",
        "document_scan": "文档扫描",
        "mixed": "混合媒介",
    },
}

_COLOR_ZH = {
    "red": "红色",
    "orange": "橙色",
    "yellow": "黄色",
    "green": "绿色",
    "blue": "蓝色",
    "purple": "紫色",
    "pink": "粉色",
    "brown": "棕色",
    "black": "黑色",
    "white": "白色",
    "gray": "灰色",
    "grey": "灰色",
    "silver": "银色",
    "gold": "金色",
}


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


def _localized_enum(field: str, value: object) -> str:
    text = str(value or "").strip()
    return _ENUM_ZH.get(field, {}).get(text, text)


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


def _canonical_qwen35_annotation(
    payload: Mapping[str, Any], manifest: ManifestItem | None
) -> ImageAnnotation:
    if manifest is None:
        raise ValueError("canonical v1.3 annotation requires a manifest item")
    raw_id = str(payload.get("image_id", "")).strip()
    if raw_id not in {manifest.image_id, manifest.image_id.rsplit("-", 1)[-1]}:
        raise ValueError(
            f"annotation image_id {raw_id!r} does not match manifest {manifest.image_id!r}"
        )
    processed_sha256 = str(payload.get("processed_sha256", "")).strip()
    if not processed_sha256:
        raise ValueError("canonical v1.3 processed_sha256 is required")
    if processed_sha256 != manifest.sha256:
        raise ValueError(
            f"processed_sha256 mismatch for {manifest.image_id}: "
            f"{processed_sha256} != {manifest.sha256}"
        )
    source_model_id = str(payload.get("source_model_id", "")).strip()
    if not source_model_id:
        raise ValueError("canonical v1.3 source_model_id is required")
    if source_model_id != CANONICAL_QWEN35_MODEL_ID:
        raise ValueError(
            f"source_model_id mismatch: {source_model_id!r} != {CANONICAL_QWEN35_MODEL_ID!r}"
        )

    annotation = _mapping(payload.get("annotation"))
    scene_data = _mapping(annotation.get("scene"))
    capture = _mapping(annotation.get("capture_visual"))
    subjective = _mapping(annotation.get("subjective"))
    captions = _mapping(annotation.get("captions"))
    event = _mapping(annotation.get("event"))
    entities = [
        _mapping(item)
        for item in annotation.get("entities", [])
        if isinstance(item, Mapping)
    ]
    entity_by_id = {
        str(item.get("entity_id", "")): str(item.get("name_zh", "")).strip()
        for item in entities
    }

    objects: list[str] = []
    object_counts: dict[str, int] = {}
    actions: list[str] = []
    attributes: list[str] = []
    colors: list[str] = []
    for entity in entities:
        name = str(entity.get("name_zh", "")).strip()
        if not name:
            continue
        objects.append(name)
        count = entity.get("count")
        if entity.get("count_exact") is True and isinstance(count, int) and count >= 0:
            object_counts[name] = object_counts.get(name, 0) + count
        for key in ("entity_type", "position_zone", "salience", "visibility"):
            value = entity.get(key)
            if value not in (None, ""):
                attributes.append(f"{key}:{name}={value}")
        entity_attributes = _mapping(entity.get("attributes"))
        colors.extend(
            _COLOR_ZH.get(value.casefold(), value)
            for value in _strings(entity_attributes.get("colors_zh"))
        )
        for key in ("materials_zh", "states_zh", "attire_zh"):
            attributes.extend(
                f"{key}:{name}={value}"
                for value in _strings(entity_attributes.get(key))
            )
        action = str(entity_attributes.get("action_zh") or "").strip()
        if action:
            actions.append(f"{name}{action}")

    for key, value in capture.items():
        if value not in (None, ""):
            localized = _localized_enum(key, value)
            attributes.append(f"{key}:{localized}")
            if localized != str(value):
                attributes.append(f"{key}_code:{value}")
    for key in ("primary_type", "environment", "media_type"):
        value = scene_data.get(key)
        if value not in (None, ""):
            attributes.append(f"scene_{key}:{value}")
    attributes.extend(
        f"scene_secondary:{value}"
        for value in _strings(scene_data.get("secondary_types"))
    )

    relations: list[str] = []
    for relation in annotation.get("relations", []):
        data = _mapping(relation)
        subject = entity_by_id.get(str(data.get("subject_id", "")), "未知实体")
        target = entity_by_id.get(str(data.get("object_id", "")), "未知实体")
        predicate = str(data.get("predicate", "")).strip()
        relation_text = str(data.get("predicate_other_zh") or "").strip()
        relations.append(f"{subject}{relation_text or _RELATION_ZH.get(predicate, predicate)}{target}")

    event_summary = str(event.get("summary_zh") or "").strip()
    if event_summary:
        actions.append(event_summary)
    summary = str(captions.get("dense_zh") or captions.get("short_zh") or "").strip()
    if not summary:
        raise ValueError("canonical v1.3 annotation must provide captions.dense_zh or short_zh")
    secondary_types = _strings(scene_data.get("secondary_types"))
    scene_terms = _strings(
        [
            scene_data.get("sub_type_zh"),
            _localized_enum("primary_type", scene_data.get("primary_type")),
            *(_localized_enum("primary_type", value) for value in secondary_types),
            _localized_enum("environment", scene_data.get("environment")),
        ]
    )
    scene = " ".join(scene_terms) or "未知场景"
    ocr_text = [
        str(_mapping(item).get("text_raw", "")).strip()
        for item in annotation.get("ocr", [])
        if str(_mapping(item).get("text_raw", "")).strip()
    ]
    uncertainty = []
    for item in annotation.get("uncertainties", []):
        data = _mapping(item)
        uncertainty.append(
            " | ".join(
                value
                for value in (
                    str(data.get("field_path", "")).strip(),
                    str(data.get("reason", "")).strip(),
                    str(data.get("note_zh", "")).strip(),
                )
                if value
            )
        )

    mood = _strings(subjective.get("mood_terms_zh"))
    colors = _strings([*colors, *(_strings(subjective.get("palette_terms_zh")))])
    keywords = _strings([*objects, *scene_terms, *mood, *colors, *ocr_text])
    media_type = _localized_enum("media_type", scene_data.get("media_type"))
    return ImageAnnotation(
        image_id=manifest.image_id,
        split=manifest.split,
        relative_path=manifest.relative_path.replace("\\", "/"),
        sha256=manifest.sha256,
        duplicate_of=manifest.duplicate_of,
        summary=summary,
        objects=_strings(objects),
        object_counts=object_counts,
        actions=_strings(actions),
        scene=scene,
        attributes=_strings(attributes),
        spatial_relations=_strings(relations),
        style=_strings(media_type),
        mood=mood,
        colors=colors,
        ocr_text=_strings(ocr_text),
        search_queries=_query_seeds(summary, scene, keywords),
        generation_prompt=summary,
        uncertainty=_strings(uncertainty),
        model_version=source_model_id,
        prompt_version=CANONICAL_QWEN35_VERSION,
    )


def adapt_annotation(payload: Mapping[str, Any], manifest: ManifestItem | None = None,
                     *, default_split: str = "Val") -> ImageAnnotation:
    """Convert the proposal's nested schema or the existing flat schema to ImageAnnotation."""
    if isinstance(payload.get("annotation"), Mapping):
        return _canonical_qwen35_annotation(payload, manifest)
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
