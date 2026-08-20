#!/usr/bin/env python3
"""Conservatively normalize completed M1 VLM candidates.

The source run is immutable. This script writes a separate record for every image,
including every mechanical change and any validation errors that remain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from m1_validation import (
    diagnostic_json_candidate,
    semantic_validation_errors,
    validation_error_records,
)


NORMALIZER_VERSION = "m1-normalize-v1.0.1"

ENTITY_TYPES = {
    "person",
    "animal",
    "object",
    "vehicle",
    "plant",
    "food",
    "building",
    "document",
    "screen",
    "artwork",
    "other",
}
ENTITY_TYPE_ALIASES = {
    "tree": "plant",
    "flower": "plant",
    "grass": "plant",
    "motorcycle": "vehicle",
    "bike": "vehicle",
    "bicycle": "vehicle",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "transport": "vehicle",
    "bridge": "building",
    "house": "building",
    "sign": "object",
    "clock": "object",
    "keyboard": "object",
    "machine": "object",
    "chart": "document",
}

POSITION_ZONES = {
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
    "spans_multiple",
    "unknown",
}
POSITION_ALIASES = {
    "middle_center": "center",
    "middle": "center",
    "top": "top_center",
    "bottom": "bottom_center",
    "left": "middle_left",
    "right": "middle_right",
    "middle_bottom": "bottom_center",
    "left_center": "middle_left",
    "right_center": "middle_right",
    "bottom_spans_multiple": "spans_multiple",
    "top_spans_multiple": "spans_multiple",
    "background": "unknown",
    "left_of": "middle_left",
    "right_of": "middle_right",
}

SCENE_TYPES = {
    "general",
    "indoor",
    "street_urban",
    "nature",
    "people_activity",
    "food",
    "transport",
    "animal_plant",
    "object_exhibit",
    "illustration_meme",
    "document_screen",
}
MEDIA_TYPE_ALIASES = {
    "document": "document_scan",
    "document_screen": "natural_image",
    "screen": "natural_image",
    "illustration_meme": "illustration",
}
SECONDARY_SCENE_ALIASES = {
    "building": "street_urban",
    "plant": "animal_plant",
    "animal": "animal_plant",
    "object": "object_exhibit",
    "artwork": "object_exhibit",
    "document": "document_screen",
    "screen": "document_screen",
    "person": "people_activity",
    "people": "people_activity",
    "vehicle": "transport",
}

RELATION_PREDICATES = {
    "left_of",
    "right_of",
    "above",
    "below",
    "in_front_of",
    "behind",
    "inside",
    "on",
    "under",
    "next_to",
    "overlapping",
    "holding",
    "wearing",
    "riding",
    "looking_at",
    "eating",
    "other",
}
RELATION_ALIASES = {
    "in": "inside",
    "beside": "next_to",
    "near": "next_to",
    "over": "above",
    "beneath": "under",
    "sitting_on": "on",
    "standing_on": "on",
    "lying_on": "on",
    "standing_in_front_of": "in_front_of",
    "parked_in_front_of": "in_front_of",
}
REVERSED_RELATIONS = {
    "held_by": "holding",
    "worn_by": "wearing",
    "ridden_by": "riding",
    "looked_at_by": "looking_at",
}

UNCERTAINTY_REASONS = {
    "blur",
    "occlusion",
    "too_small",
    "cropped",
    "ambiguous_text",
    "ambiguous_category",
    "count_unreliable",
    "low_resolution",
    "reflection_or_glare",
    "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compact_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 20:
        return {"type": "array", "length": len(value)}
    if isinstance(value, dict) and len(value) > 20:
        return {"type": "object", "keys": sorted(value)[:20]}
    return deepcopy(value)


def add_change(
    changes: list[dict[str, Any]],
    path: str,
    rule: str,
    before: Any,
    after: Any,
    *,
    lossy: bool = False,
    force: bool = False,
) -> None:
    if before == after and not force:
        return
    changes.append(
        {
            "path": path,
            "rule": rule,
            "before": compact_value(before),
            "after": compact_value(after),
            "lossy": lossy,
        }
    )


def normalize_bbox(
    owner: dict[str, Any], path: str, changes: list[dict[str, Any]]
) -> None:
    value = owner.get("bbox_norm_1000")
    if value is None:
        return
    valid = (
        isinstance(value, list)
        and len(value) == 4
        and all(type(coordinate) is int for coordinate in value)
        and all(0 <= coordinate <= 1000 for coordinate in value)
        and value[0] < value[2]
        and value[1] < value[3]
    )
    if not valid:
        owner["bbox_norm_1000"] = None
        add_change(
            changes,
            f"{path}/bbox_norm_1000",
            "invalid_bbox_to_null",
            value,
            None,
            lossy=True,
        )


def normalize_scene(annotation: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    scene = annotation.get("scene")
    if not isinstance(scene, dict):
        return
    primary = scene.get("primary_type")
    if isinstance(primary, str) and primary not in SCENE_TYPES:
        replacement = "street_urban" if primary == "building" else "general"
        scene["primary_type"] = replacement
        add_change(changes, "/scene/primary_type", "canonical_scene_type", primary, replacement)
        primary = replacement

    secondary = scene.get("secondary_types")
    if isinstance(secondary, list):
        normalized: list[Any] = []
        for value in secondary:
            replacement = value
            if isinstance(value, str) and value not in SCENE_TYPES:
                replacement = SECONDARY_SCENE_ALIASES.get(value)
            if replacement is None or replacement == "general" or replacement == primary:
                continue
            if replacement not in normalized:
                normalized.append(replacement)
        if len(normalized) > 3:
            normalized = normalized[:3]
        add_change(
            changes,
            "/scene/secondary_types",
            "canonicalize_secondary_scene_types",
            secondary,
            normalized,
            lossy=len(normalized) < len(secondary),
        )
        scene["secondary_types"] = normalized

    media_type = scene.get("media_type")
    if media_type in MEDIA_TYPE_ALIASES:
        replacement = MEDIA_TYPE_ALIASES[media_type]
        scene["media_type"] = replacement
        add_change(
            changes,
            "/scene/media_type",
            "canonical_media_type",
            media_type,
            replacement,
        )


def normalize_capture(annotation: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    capture = annotation.get("capture_visual")
    if not isinstance(capture, dict):
        return

    enum_values = {
        "time_of_day": {"day", "dawn_dusk", "night", "unknown", "not_applicable"},
        "weather": {"clear", "cloudy", "rain", "snow", "fog", "unknown", "not_applicable"},
        "lighting": {"natural", "artificial", "mixed", "low_light", "unknown", "not_applicable"},
        "viewpoint": {"eye_level", "high_angle", "low_angle", "aerial", "unknown", "not_applicable"},
        "shot_scale": {"close_up", "medium", "wide", "unknown", "not_applicable"},
        "blur_level": {"none", "mild", "severe", "unknown"},
    }
    aliases = {
        "time_of_day": {"dusk": "dawn_dusk", "dawn": "dawn_dusk", "indoor": "not_applicable"},
        "viewpoint": {"overhead": "high_angle", "top": "high_angle"},
        "blur_level": {"not_applicable": "unknown"},
    }

    viewpoint = capture.get("viewpoint")
    if viewpoint in {"close_up", "wide"}:
        shot_scale = capture.get("shot_scale")
        if shot_scale in {"unknown", "not_applicable", None}:
            capture["shot_scale"] = viewpoint
            add_change(
                changes,
                "/capture_visual/shot_scale",
                "move_misplaced_shot_scale",
                shot_scale,
                viewpoint,
            )
        capture["viewpoint"] = "unknown"
        add_change(
            changes,
            "/capture_visual/viewpoint",
            "move_misplaced_shot_scale",
            viewpoint,
            "unknown",
        )

    for field, allowed in enum_values.items():
        value = capture.get(field)
        if not isinstance(value, str) or value in allowed:
            continue
        replacement = aliases.get(field, {}).get(value, "unknown")
        capture[field] = replacement
        add_change(
            changes,
            f"/capture_visual/{field}",
            "canonical_capture_enum",
            value,
            replacement,
        )


def renumber_items(
    items: Any,
    prefix: str,
    collection_path: str,
    changes: list[dict[str, Any]],
) -> tuple[dict[str, str], bool]:
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return {}, False
    id_key = "entity_id" if prefix == "e" else "text_id"
    old_ids = [item.get(id_key) for item in items]
    string_ids = [value for value in old_ids if isinstance(value, str)]
    if len(string_ids) != len(set(string_ids)):
        return {}, False
    mapping: dict[str, str] = {}
    for index, item in enumerate(items, start=1):
        old_id = item.get(id_key)
        new_id = f"{prefix}{index}"
        if isinstance(old_id, str):
            mapping[old_id] = new_id
        if old_id != new_id:
            item[id_key] = new_id
            add_change(
                changes,
                f"{collection_path}/{index - 1}/{id_key}",
                "renumber_sequential_ids",
                old_id,
                new_id,
            )
    return mapping, True


def normalize_entities(
    annotation: dict[str, Any], changes: list[dict[str, Any]]
) -> tuple[dict[str, str], bool]:
    entities = annotation.get("entities")
    if not isinstance(entities, list):
        return {}, False
    mapping, ids_unambiguous = renumber_items(entities, "e", "/entities", changes)
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        path = f"/entities/{index}"
        entity_type = entity.get("entity_type")
        if isinstance(entity_type, str) and entity_type not in ENTITY_TYPES:
            replacement = ENTITY_TYPE_ALIASES.get(entity_type, "other")
            entity["entity_type"] = replacement
            add_change(
                changes,
                f"{path}/entity_type",
                "canonical_entity_type",
                entity_type,
                replacement,
            )

        zone = entity.get("position_zone")
        if isinstance(zone, str) and zone not in POSITION_ZONES:
            replacement = POSITION_ALIASES.get(zone, "unknown")
            entity["position_zone"] = replacement
            add_change(
                changes,
                f"{path}/position_zone",
                "canonical_position_zone",
                zone,
                replacement,
            )

        count = entity.get("count")
        count_exact = entity.get("count_exact")
        if count_exact is False and count is not None:
            entity["count"] = None
            add_change(
                changes,
                f"{path}/count",
                "clear_inexact_count",
                count,
                None,
            )
        elif count_exact is True and type(count) is not int:
            entity["count_exact"] = False
            entity["count"] = None
            add_change(
                changes,
                f"{path}/count_exact",
                "downgrade_missing_exact_count",
                count_exact,
                False,
            )
            add_change(
                changes,
                f"{path}/count",
                "downgrade_missing_exact_count",
                count,
                None,
            )

        normalize_bbox(entity, path, changes)
        attributes = entity.get("attributes")
        if isinstance(attributes, dict):
            bounded_attributes = (
                ("colors_zh", 3, "keep_first_three_salient_colors"),
                ("materials_zh", 3, "keep_first_three_materials"),
                ("states_zh", 4, "keep_first_four_states"),
                ("attire_zh", 5, "keep_first_five_attire_items"),
            )
            for field, limit, rule in bounded_attributes:
                values = attributes.get(field)
                if not isinstance(values, list) or len(values) <= limit:
                    continue
                replacement = values[:limit]
                attributes[field] = replacement
                add_change(
                    changes,
                    f"{path}/attributes/{field}",
                    rule,
                    values,
                    replacement,
                    lossy=True,
                )
            attire = attributes.get("attire_zh")
            if entity.get("entity_type") != "person" and isinstance(attire, list) and attire:
                attributes["attire_zh"] = []
                add_change(
                    changes,
                    f"{path}/attributes/attire_zh",
                    "clear_non_person_attire",
                    attire,
                    [],
                    lossy=True,
                )
    return mapping, ids_unambiguous


def normalize_ocr(annotation: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    ocr_items = annotation.get("ocr")
    if not isinstance(ocr_items, list):
        return
    renumber_items(ocr_items, "t", "/ocr", changes)
    for index, item in enumerate(ocr_items):
        if not isinstance(item, dict):
            continue
        path = f"/ocr/{index}"
        language = item.get("language")
        if isinstance(language, str) and language not in {"zh", "en", "mixed", "other", "unknown"}:
            replacement = "unknown" if language == "not_applicable" else "other"
            item["language"] = replacement
            add_change(
                changes,
                f"{path}/language",
                "canonical_ocr_language",
                language,
                replacement,
            )
        normalize_bbox(item, path, changes)


def normalize_relations(
    annotation: dict[str, Any],
    entity_mapping: dict[str, str],
    entity_ids_unambiguous: bool,
    changes: list[dict[str, Any]],
) -> None:
    relations = annotation.get("relations")
    entities = annotation.get("entities")
    if not isinstance(relations, list) or not isinstance(entities, list):
        return
    entity_ids = {
        entity.get("entity_id")
        for entity in entities
        if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
    }
    normalized: list[Any] = []
    normalized_keys: set[str] = set()
    for index, relation in enumerate(relations):
        path = f"/relations/{index}"
        if not isinstance(relation, dict):
            normalized.append(relation)
            continue
        relation = deepcopy(relation)
        old_predicate = relation.get("predicate")
        if old_predicate in REVERSED_RELATIONS:
            old_subject = relation.get("subject_id")
            old_object = relation.get("object_id")
            relation["subject_id"], relation["object_id"] = old_object, old_subject
            relation["predicate"] = REVERSED_RELATIONS[old_predicate]
            add_change(
                changes,
                path,
                "reverse_passive_relation",
                {"subject_id": old_subject, "predicate": old_predicate, "object_id": old_object},
                {
                    "subject_id": relation["subject_id"],
                    "predicate": relation["predicate"],
                    "object_id": relation["object_id"],
                },
            )
        elif old_predicate in RELATION_ALIASES:
            replacement = RELATION_ALIASES[old_predicate]
            relation["predicate"] = replacement
            add_change(
                changes,
                f"{path}/predicate",
                "canonical_relation_predicate",
                old_predicate,
                replacement,
            )
        elif isinstance(old_predicate, str) and old_predicate not in RELATION_PREDICATES:
            relation["predicate"] = "other"
            relation["predicate_other_zh"] = old_predicate[:20]
            add_change(
                changes,
                f"{path}/predicate",
                "preserve_open_relation_as_other",
                old_predicate,
                "other",
            )
            add_change(
                changes,
                f"{path}/predicate_other_zh",
                "preserve_open_relation_as_other",
                None,
                old_predicate[:20],
            )

        if entity_ids_unambiguous:
            for field in ("subject_id", "object_id"):
                old_id = relation.get(field)
                if isinstance(old_id, str) and old_id in entity_mapping:
                    new_id = entity_mapping[old_id]
                    relation[field] = new_id
                    add_change(
                        changes,
                        f"{path}/{field}",
                        "update_renumbered_entity_reference",
                        old_id,
                        new_id,
                    )

        predicate = relation.get("predicate")
        predicate_other = relation.get("predicate_other_zh")
        if predicate != "other" and predicate_other is not None:
            relation["predicate_other_zh"] = None
            add_change(
                changes,
                f"{path}/predicate_other_zh",
                "clear_unused_other_predicate",
                predicate_other,
                None,
            )
        elif predicate != "other" and "predicate_other_zh" not in relation:
            relation["predicate_other_zh"] = None
            add_change(
                changes,
                f"{path}/predicate_other_zh",
                "fill_required_null",
                None,
                None,
                force=True,
            )

        subject_id = relation.get("subject_id")
        object_id = relation.get("object_id")
        references_valid = subject_id in entity_ids and object_id in entity_ids
        if not references_valid or subject_id == object_id:
            add_change(
                changes,
                path,
                "drop_unresolvable_relation",
                relation,
                None,
                lossy=True,
            )
            continue
        relation_key = json.dumps(
            relation, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if relation_key in normalized_keys:
            add_change(
                changes,
                path,
                "drop_duplicate_relation",
                relation,
                None,
            )
            continue
        normalized_keys.add(relation_key)
        normalized.append(relation)
    annotation["relations"] = normalized


def normalize_event(
    annotation: dict[str, Any],
    entity_mapping: dict[str, str],
    entity_ids_unambiguous: bool,
    changes: list[dict[str, Any]],
) -> None:
    event = annotation.get("event")
    entities = annotation.get("entities")
    if not isinstance(event, dict) or not isinstance(entities, list):
        return
    entity_ids = {
        entity.get("entity_id")
        for entity in entities
        if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
    }
    evidence = event.get("evidence_entity_ids")
    if not isinstance(evidence, list):
        return
    if event.get("summary_zh") is None:
        if evidence:
            event["evidence_entity_ids"] = []
            add_change(
                changes,
                "/event/evidence_entity_ids",
                "clear_evidence_without_event",
                evidence,
                [],
                lossy=True,
            )
        return
    normalized: list[Any] = []
    for value in evidence:
        replacement = entity_mapping.get(value, value) if entity_ids_unambiguous else value
        if replacement in entity_ids and replacement not in normalized:
            normalized.append(replacement)
    if len(normalized) > 12:
        normalized = normalized[:12]
    add_change(
        changes,
        "/event/evidence_entity_ids",
        "canonicalize_event_evidence",
        evidence,
        normalized,
        lossy=len(normalized) < len(evidence),
    )
    event["evidence_entity_ids"] = normalized


def normalize_uncertainties(annotation: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    uncertainties = annotation.get("uncertainties")
    if not isinstance(uncertainties, list):
        return
    for index, item in enumerate(uncertainties):
        if not isinstance(item, dict):
            continue
        reason = item.get("reason")
        if isinstance(reason, str) and reason not in UNCERTAINTY_REASONS:
            item["reason"] = "other"
            add_change(
                changes,
                f"/uncertainties/{index}/reason",
                "canonical_uncertainty_reason",
                reason,
                "other",
            )


def normalize_subjective(annotation: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    subjective = annotation.get("subjective")
    if not isinstance(subjective, dict):
        return
    bounded_terms = (
        ("mood_terms_zh", 3, "keep_first_three_mood_terms"),
        ("palette_terms_zh", 5, "keep_first_five_palette_terms"),
    )
    for field, limit, rule in bounded_terms:
        values = subjective.get(field)
        if not isinstance(values, list) or len(values) <= limit:
            continue
        replacement = values[:limit]
        subjective[field] = replacement
        add_change(
            changes,
            f"/subjective/{field}",
            rule,
            values,
            replacement,
            lossy=True,
        )


def normalize_annotation(annotation: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = deepcopy(annotation)
    changes: list[dict[str, Any]] = []
    normalize_scene(result, changes)
    normalize_capture(result, changes)
    entity_mapping, ids_unambiguous = normalize_entities(result, changes)
    normalize_ocr(result, changes)
    normalize_relations(result, entity_mapping, ids_unambiguous, changes)
    normalize_event(result, entity_mapping, ids_unambiguous, changes)
    normalize_subjective(result, changes)
    normalize_uncertainties(result, changes)
    return result, changes


def parse_source(raw: str, summary: dict[str, Any]) -> tuple[Any | None, str]:
    if summary.get("json_parse_ok") is True:
        try:
            return json.loads(raw), "strict"
        except json.JSONDecodeError:
            return None, "unrecoverable"
    annotation, _, duplicate_keys, _ = diagnostic_json_candidate(raw)
    if annotation is not None and not duplicate_keys:
        return annotation, "diagnostic"
    return None, "unrecoverable"


def image_sort_key(path: Path) -> tuple[int, str]:
    image_id = path.parent.name
    return (int(image_id), image_id) if image_id.isdigit() else (10**18, image_id)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    summary_paths = sorted(
        (args.run_dir / "items").glob("*/summary.json"), key=image_sort_key
    )
    if not summary_paths:
        raise FileNotFoundError(f"No source summaries under {args.run_dir}")

    records_path = args.output_dir / "normalization_records.jsonl"
    valid_path = args.output_dir / "normalized_annotations.jsonl"
    review_path = args.output_dir / "review_queue.jsonl"
    counters: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    remaining_schema_paths: Counter[str] = Counter()
    remaining_semantic_paths: Counter[str] = Counter()

    with (
        records_path.open("w", encoding="utf-8") as records_file,
        valid_path.open("w", encoding="utf-8") as valid_file,
        review_path.open("w", encoding="utf-8") as review_file,
    ):
        for summary_path in summary_paths:
            source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            image_id = str(source_summary["image_id"])
            identity_status = str(
                source_summary.get("image_identity_status", "exact_hash_match")
            )
            raw_path = args.run_dir / "raw" / f"{image_id}.txt"
            raw = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
            source_annotation, parse_mode = parse_source(raw, source_summary)
            counters["total"] += 1
            counters[f"parse_mode:{parse_mode}"] += 1
            counters[f"image_identity_status:{identity_status}"] += 1

            changes: list[dict[str, Any]] = []
            normalized_annotation = None
            schema_errors: list[dict[str, str]] = []
            semantic_errors: list[dict[str, str]] = []
            if isinstance(source_annotation, dict):
                normalized_annotation, changes = normalize_annotation(source_annotation)
                schema_errors = validation_error_records(list(validator.iter_errors(normalized_annotation)))
                semantic_errors = semantic_validation_errors(normalized_annotation)
            elif source_annotation is not None:
                schema_errors = [{"path": "$", "message": "annotation must be an object"}]

            valid = (
                normalized_annotation is not None
                and not schema_errors
                and not semantic_errors
            )
            lossy = any(change["lossy"] for change in changes)
            if valid:
                counters["normalized_valid"] += 1
            if lossy:
                counters["lossy_repair"] += 1
            for change in changes:
                repair_counts[change["rule"]] += 1
            for error in schema_errors:
                remaining_schema_paths[error["path"]] += 1
            for error in semantic_errors:
                remaining_semantic_paths[error["path"]] += 1

            if normalized_annotation is None:
                status = "unrecoverable"
            elif not valid:
                status = "review_required"
            elif lossy:
                status = "valid_with_lossy_repairs"
            else:
                status = "valid"
            counters[f"status:{status}"] += 1

            record = {
                "normalizer_version": NORMALIZER_VERSION,
                "image_id": image_id,
                "processed_sha256": source_summary.get("processed_sha256"),
                "source_processed_sha256": source_summary.get(
                    "source_processed_sha256"
                ),
                "image_identity_status": identity_status,
                "source_model_id": source_summary.get("model_id"),
                "source_prompt_version": source_summary.get("prompt_version"),
                "source_schema_sha256": source_summary.get("schema_file_sha256"),
                "source_raw_response_path": str(raw_path),
                "source_annotation_valid": source_summary.get("annotation_valid") is True,
                "parse_mode": parse_mode,
                "status": status,
                "repairs": changes,
                "normalized_schema_valid": not schema_errors and normalized_annotation is not None,
                "normalized_semantic_valid": not semantic_errors and normalized_annotation is not None,
                "normalized_annotation_valid": valid,
                "remaining_schema_errors": schema_errors,
                "remaining_semantic_errors": semantic_errors,
                "annotation": normalized_annotation,
            }
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if valid:
                valid_file.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "processed_sha256": source_summary.get("processed_sha256"),
                            "source_processed_sha256": source_summary.get(
                                "source_processed_sha256"
                            ),
                            "image_identity_status": identity_status,
                            "source_model_id": source_summary.get("model_id"),
                            "normalizer_version": NORMALIZER_VERSION,
                            "repairs_applied": len(changes),
                            "lossy_repairs": lossy,
                            "annotation": normalized_annotation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if status != "valid":
                review_file.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "status": status,
                            "record_line": counters["total"],
                            "lossy_repairs": lossy,
                            "remaining_schema_errors": schema_errors,
                            "remaining_semantic_errors": semantic_errors,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    total = counters["total"]
    source_valid = sum(
        1
        for path in summary_paths
        if json.loads(path.read_text(encoding="utf-8")).get("annotation_valid") is True
    )
    summary = {
        "normalizer_version": NORMALIZER_VERSION,
        "source_run_dir": str(args.run_dir),
        "target_schema": str(args.schema),
        "total_images": total,
        "source_annotation_valid": source_valid,
        "source_annotation_valid_rate": round(source_valid / total, 4),
        "parse_modes": {
            key.removeprefix("parse_mode:"): value
            for key, value in sorted(counters.items())
            if key.startswith("parse_mode:")
        },
        "image_identity_status_counts": {
            key.removeprefix("image_identity_status:"): value
            for key, value in sorted(counters.items())
            if key.startswith("image_identity_status:")
        },
        "statuses": {
            key.removeprefix("status:"): value
            for key, value in sorted(counters.items())
            if key.startswith("status:")
        },
        "normalized_annotation_valid": counters["normalized_valid"],
        "normalized_annotation_valid_rate": round(counters["normalized_valid"] / total, 4),
        "images_with_lossy_repairs": counters["lossy_repair"],
        "repair_counts": dict(repair_counts.most_common()),
        "remaining_schema_error_paths": dict(remaining_schema_paths.most_common()),
        "remaining_semantic_error_paths": dict(remaining_semantic_paths.most_common()),
        "artifacts": {
            "records": str(records_path),
            "normalized_annotations": str(valid_path),
            "review_queue": str(review_path),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
