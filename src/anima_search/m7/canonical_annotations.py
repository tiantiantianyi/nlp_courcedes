from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from anima_search.schemas import ImageAnnotation, ManifestItem


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: object) -> list[str]:
    source = value if isinstance(value, list) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in source:
        if item is None:
            continue
        text = str(item).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _manifest_by_numeric_id(
    manifest_paths: list[Path],
) -> dict[str, ManifestItem]:
    result: dict[str, ManifestItem] = {}
    for path in manifest_paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                item = ManifestItem.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid manifest record {path}:{line_number}: {exc}"
                ) from exc
            numeric_id = item.image_id.partition("-")[2]
            if not numeric_id:
                raise ValueError(
                    f"manifest image_id lacks split prefix: {item.image_id}"
                )
            if numeric_id in result:
                raise ValueError(
                    f"duplicate numeric image_id across manifests: {numeric_id}"
                )
            result[numeric_id] = item
    return result


def _entity_fields(
    entities_value: object,
) -> tuple[
    list[str],
    dict[str, int],
    list[str],
    list[str],
    list[str],
]:
    entities = [
        _mapping(item)
        for item in entities_value
        if isinstance(entities_value, list)
    ]
    names = _strings([entity.get("name_zh") for entity in entities])
    counts: dict[str, list[tuple[object, object]]] = defaultdict(list)
    actions: list[str] = []
    colors: list[str] = []
    attributes: list[str] = []
    for entity in entities:
        name = str(entity.get("name_zh") or "").strip()
        if not name:
            continue
        counts[name].append((entity.get("count"), entity.get("count_exact")))
        details = _mapping(entity.get("attributes"))
        actions.extend(_strings(details.get("action_zh")))
        colors.extend(_strings(details.get("colors_zh")))
        for state in _strings(details.get("states_zh")):
            attributes.append(f"state:{name}={state}")
        for material in _strings(details.get("materials_zh")):
            attributes.append(f"material:{name}={material}")
        for attire in _strings(details.get("attire_zh")):
            attributes.append(f"attire:{name}={attire}")

    reliable_counts: dict[str, int] = {}
    for name, observations in counts.items():
        if all(
            exact is True
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            for count, exact in observations
        ):
            reliable_counts[name] = sum(
                int(count) for count, _ in observations
            )
    return (
        names,
        reliable_counts,
        _strings(actions),
        _strings(colors),
        _strings(attributes),
    )


def _map_record(
    record: Mapping[str, Any],
    manifest: ManifestItem,
) -> ImageAnnotation:
    annotation = _mapping(record.get("annotation"))
    scene_data = _mapping(annotation.get("scene"))
    capture = _mapping(annotation.get("capture_visual"))
    captions = _mapping(annotation.get("captions"))
    event = _mapping(annotation.get("event"))
    subjective = _mapping(annotation.get("subjective"))

    dense = str(captions.get("dense_zh") or "").strip()
    short = str(captions.get("short_zh") or "").strip()
    event_summary = str(event.get("summary_zh") or "").strip()
    summary = dense or short or event_summary
    if not summary:
        raise ValueError(
            f"canonical annotation {record.get('image_id')} lacks caption/event summary"
        )
    scene = str(
        scene_data.get("sub_type_zh")
        or scene_data.get("primary_type")
        or "未知场景"
    ).strip()

    (
        objects,
        object_counts,
        actions,
        entity_colors,
        entity_attributes,
    ) = _entity_fields(annotation.get("entities"))

    attributes: list[str] = []
    for key, value in capture.items():
        if value not in (None, ""):
            attributes.append(f"{key}:{value}")
    primary_type = scene_data.get("primary_type")
    if primary_type not in (None, ""):
        attributes.append(f"scene_primary_type:{primary_type}")
    for secondary_type in _strings(scene_data.get("secondary_types")):
        attributes.append(f"scene_secondary_type:{secondary_type}")
    environment = scene_data.get("environment")
    if environment not in (None, ""):
        attributes.append(f"environment:{environment}")
    attributes.extend(entity_attributes)

    ocr_text = _strings(
        [
            _mapping(item).get("text_raw")
            for item in annotation.get("ocr", [])
            if isinstance(annotation.get("ocr"), list)
        ]
    )
    uncertainty = _strings(
        [
            _mapping(item).get("note_zh")
            or _mapping(item).get("reason")
            for item in annotation.get("uncertainties", [])
            if isinstance(annotation.get("uncertainties"), list)
        ]
    )

    entity_names = {
        str(_mapping(item).get("entity_id")): str(
            _mapping(item).get("name_zh") or ""
        ).strip()
        for item in annotation.get("entities", [])
        if isinstance(annotation.get("entities"), list)
    }
    spatial_relations: list[str] = []
    for item in (
        annotation.get("relations", [])
        if isinstance(annotation.get("relations"), list)
        else []
    ):
        relation = _mapping(item)
        subject = entity_names.get(str(relation.get("subject_id")), "")
        target = entity_names.get(str(relation.get("object_id")), "")
        predicate = str(
            relation.get("predicate_other_zh")
            or relation.get("predicate")
            or ""
        ).strip()
        if subject and predicate and target:
            spatial_relations.append(f"{subject} {predicate} {target}")

    colors = _strings(
        [
            *_strings(subjective.get("palette_terms_zh")),
            *entity_colors,
        ]
    )
    search_short = short or summary
    return ImageAnnotation(
        image_id=manifest.image_id,
        split=manifest.split,
        relative_path=manifest.relative_path,
        sha256=manifest.sha256,
        duplicate_of=manifest.duplicate_of,
        summary=summary,
        objects=objects,
        object_counts=object_counts,
        actions=actions,
        scene=scene,
        attributes=_strings(attributes),
        spatial_relations=_strings(spatial_relations),
        style=[],
        mood=_strings(subjective.get("mood_terms_zh")),
        colors=colors,
        ocr_text=ocr_text,
        search_queries=[summary, search_short, scene or summary],
        generation_prompt=summary,
        uncertainty=uncertainty,
        model_version=str(record.get("source_model_id") or "unknown"),
        prompt_version="canonical-v1.3",
        generation_parameters={
            "normalizer_version": record.get("normalizer_version"),
            "repairs_applied": record.get("repairs_applied"),
            "lossy_repairs": record.get("lossy_repairs"),
        },
    )


def load_canonical_m7_annotations(
    annotation_path: Path,
    manifest_paths: list[Path],
    *,
    split: Literal["Train", "Val"] | None = None,
) -> dict[str, ImageAnnotation]:
    manifests = _manifest_by_numeric_id(manifest_paths)
    result: dict[str, ImageAnnotation] = {}
    seen_numeric_ids: set[str] = set()
    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid canonical JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"canonical line {line_number} must contain a JSON object"
            )
        numeric_id = str(record.get("image_id") or "").strip()
        if not numeric_id:
            raise ValueError(f"canonical line {line_number} lacks image_id")
        if numeric_id in seen_numeric_ids:
            raise ValueError(f"duplicate canonical image_id: {numeric_id}")
        seen_numeric_ids.add(numeric_id)
        manifest = manifests.get(numeric_id)
        if manifest is None:
            raise ValueError(
                f"canonical image_id {numeric_id} is absent from manifests"
            )
        if split is not None and manifest.split != split:
            continue
        processed_sha256 = str(record.get("processed_sha256") or "")
        if processed_sha256 != manifest.sha256:
            raise ValueError(
                f"processed_sha256 mismatch for {manifest.image_id}"
            )
        result[manifest.image_id] = _map_record(record, manifest)
    return result
