"""Shared strict parsing and validation for M1 annotation payloads."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator


def extract_prompt_block(markdown: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*\n\s*```text\s*\n(.*?)\n```"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find fenced text block under: {heading}")
    return match.group(1).strip()


def extract_prompt_version(markdown: str) -> str:
    match = re.search(r"^版本：`([^`]+)`\s*$", markdown, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find Prompt version in markdown")
    return match.group(1)


def duplicate_recording_hook(duplicate_keys: list[str]) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result and key not in duplicate_keys:
                duplicate_keys.append(key)
            result[key] = value
        return result

    return hook


def diagnostic_json_candidate(
    raw_content: str,
) -> tuple[Any | None, list[str], list[str], str | None]:
    format_issues: list[str] = []
    candidate_text = raw_content.strip()

    fence_match = re.fullmatch(
        r"```(?:json)?\s*\n(.*)\n```",
        candidate_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        format_issues.append("markdown_code_fence")
        candidate_text = fence_match.group(1)

    duplicate_keys: list[str] = []
    try:
        decoder = json.JSONDecoder(
            object_pairs_hook=duplicate_recording_hook(duplicate_keys)
        )
        candidate, end = decoder.raw_decode(candidate_text.lstrip())
    except json.JSONDecodeError as exc:
        return None, format_issues, duplicate_keys, f"{type(exc).__name__}: {exc}"

    trailing = candidate_text.lstrip()[end:].strip()
    if trailing:
        format_issues.append("trailing_content")
    if duplicate_keys:
        format_issues.append("duplicate_keys")
    return candidate, format_issues, duplicate_keys, None


def semantic_validation_errors(annotation: Any) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def add(path: str, message: str) -> None:
        errors.append({"path": path, "message": message})

    def check_bbox(value: Any, path: str) -> None:
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(type(coordinate) is int for coordinate in value)
        ):
            x1, y1, x2, y2 = value
            if x1 >= x2 or y1 >= y2:
                add(path, "bbox must satisfy x1 < x2 and y1 < y2")

    if not isinstance(annotation, dict):
        return errors

    scene = annotation.get("scene")
    if isinstance(scene, dict):
        primary_type = scene.get("primary_type")
        secondary_types = scene.get("secondary_types")
        if isinstance(secondary_types, list) and primary_type in secondary_types:
            add("$.scene.secondary_types", "must not contain scene.primary_type")

    entities = annotation.get("entities")
    entity_ids: list[str] = []
    if isinstance(entities, list):
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            path = f"$.entities.{index}"
            entity_id = entity.get("entity_id")
            expected_id = f"e{index + 1}"
            if isinstance(entity_id, str):
                entity_ids.append(entity_id)
                if entity_id != expected_id:
                    add(f"{path}.entity_id", f"expected {expected_id!r}")

            count = entity.get("count")
            count_exact = entity.get("count_exact")
            if count_exact is True and type(count) is not int:
                add(f"{path}.count", "must be an integer when count_exact is true")
            if count_exact is False and count is not None:
                add(f"{path}.count", "must be null when count_exact is false")

            check_bbox(entity.get("bbox_norm_1000"), f"{path}.bbox_norm_1000")

            attributes = entity.get("attributes")
            if isinstance(attributes, dict):
                attire = attributes.get("attire_zh")
                if entity.get("entity_type") != "person" and attire not in (None, []):
                    add(
                        f"{path}.attributes.attire_zh",
                        "must be empty for non-person entities",
                    )

        if len(entity_ids) != len(set(entity_ids)):
            add("$.entities", "entity_id values must be unique")

    entity_id_set = set(entity_ids)
    ocr_items = annotation.get("ocr")
    text_ids: list[str] = []
    if isinstance(ocr_items, list):
        for index, ocr_item in enumerate(ocr_items):
            if not isinstance(ocr_item, dict):
                continue
            path = f"$.ocr.{index}"
            text_id = ocr_item.get("text_id")
            expected_id = f"t{index + 1}"
            if isinstance(text_id, str):
                text_ids.append(text_id)
                if text_id != expected_id:
                    add(f"{path}.text_id", f"expected {expected_id!r}")
            check_bbox(ocr_item.get("bbox_norm_1000"), f"{path}.bbox_norm_1000")
        if len(text_ids) != len(set(text_ids)):
            add("$.ocr", "text_id values must be unique")

    relations = annotation.get("relations")
    if isinstance(relations, list):
        for index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                continue
            path = f"$.relations.{index}"
            subject_id = relation.get("subject_id")
            object_id = relation.get("object_id")
            if isinstance(subject_id, str) and subject_id not in entity_id_set:
                add(f"{path}.subject_id", "must reference an existing entity_id")
            if isinstance(object_id, str) and object_id not in entity_id_set:
                add(f"{path}.object_id", "must reference an existing entity_id")
            if subject_id == object_id and isinstance(subject_id, str):
                add(path, "subject_id and object_id must be different")

            predicate = relation.get("predicate")
            predicate_other = relation.get("predicate_other_zh")
            if predicate == "other":
                if not isinstance(predicate_other, str) or not predicate_other.strip():
                    add(
                        f"{path}.predicate_other_zh",
                        "must be non-empty when predicate is 'other'",
                    )
            elif predicate_other is not None:
                add(
                    f"{path}.predicate_other_zh",
                    "must be null unless predicate is 'other'",
                )

    event = annotation.get("event")
    if isinstance(event, dict):
        evidence_ids = event.get("evidence_entity_ids")
        if isinstance(evidence_ids, list):
            for index, entity_id in enumerate(evidence_ids):
                if isinstance(entity_id, str) and entity_id not in entity_id_set:
                    add(
                        f"$.event.evidence_entity_ids.{index}",
                        "must reference an existing entity_id",
                    )
            if event.get("summary_zh") is None and evidence_ids:
                add(
                    "$.event.evidence_entity_ids",
                    "must be empty when event.summary_zh is null",
                )

    return errors


def _json_path(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return "$" if not parts else "$." + ".".join(parts)


def validate_raw_annotation(
    raw_content: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    result: dict[str, Any] = {
        "json_parse_ok": False,
        "schema_valid": False,
        "diagnostic_json_parse_ok": False,
        "diagnostic_schema_valid": False,
        "semantic_valid": False,
        "diagnostic_semantic_valid": False,
        "annotation_valid": False,
        "format_issues": [],
        "duplicate_keys": [],
        "schema_errors": [],
        "semantic_errors": [],
        "error": None,
    }

    annotation: Any | None = None
    strict_duplicate_keys: list[str] = []
    try:
        annotation = json.loads(
            raw_content,
            object_pairs_hook=duplicate_recording_hook(strict_duplicate_keys),
        )
        if strict_duplicate_keys:
            result["format_issues"] = ["duplicate_keys"]
            result["duplicate_keys"] = strict_duplicate_keys
            result["diagnostic_json_parse_ok"] = True
            result["error"] = "DuplicateKeyError: " + ", ".join(
                strict_duplicate_keys
            )
        else:
            result["json_parse_ok"] = True
    except json.JSONDecodeError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        (
            annotation,
            format_issues,
            diagnostic_duplicate_keys,
            diagnostic_error,
        ) = diagnostic_json_candidate(raw_content)
        result["format_issues"] = format_issues
        result["duplicate_keys"] = sorted(
            set(strict_duplicate_keys + diagnostic_duplicate_keys)
        )
        if diagnostic_error:
            result["diagnostic_error"] = diagnostic_error
        if annotation is not None:
            result["diagnostic_json_parse_ok"] = True

    if annotation is None:
        return result, None

    validator = Draft202012Validator(schema)
    schema_errors = sorted(
        validator.iter_errors(annotation),
        key=lambda error: list(error.path),
    )
    result["schema_errors"] = [
        {"path": _json_path(error), "message": error.message}
        for error in schema_errors[:100]
    ]
    result["diagnostic_schema_valid"] = not schema_errors
    result["schema_valid"] = result["json_parse_ok"] and not schema_errors

    semantic_errors = semantic_validation_errors(annotation)
    result["semantic_errors"] = semantic_errors[:100]
    result["diagnostic_semantic_valid"] = not semantic_errors
    result["semantic_valid"] = result["json_parse_ok"] and not semantic_errors
    result["annotation_valid"] = result["schema_valid"] and result["semantic_valid"]

    if result["json_parse_ok"] and schema_errors:
        result["error"] = f"SchemaValidationError: {len(schema_errors)} error(s)"
    elif result["json_parse_ok"] and semantic_errors:
        result["error"] = (
            f"SemanticValidationError: {len(semantic_errors)} error(s)"
        )

    return result, annotation


def candidate_record_validator(
    candidate_schema: dict[str, Any],
    annotation_schema: dict[str, Any],
) -> Draft202012Validator:
    """Build a validator with the package's local annotation ref inlined."""

    def inline_local_ref(value: Any) -> Any:
        if isinstance(value, dict):
            if value == {"$ref": "./annotation_payload.schema.json"}:
                return deepcopy(annotation_schema)
            return {key: inline_local_ref(item) for key, item in value.items()}
        if isinstance(value, list):
            return [inline_local_ref(item) for item in value]
        return value

    resolved_schema = inline_local_ref(candidate_schema)
    Draft202012Validator.check_schema(resolved_schema)
    return Draft202012Validator(resolved_schema)


def validation_error_records(errors: list[Any]) -> list[dict[str, str]]:
    ordered = sorted(errors, key=lambda error: list(error.path))
    return [
        {"path": _json_path(error), "message": error.message}
        for error in ordered[:100]
    ]
