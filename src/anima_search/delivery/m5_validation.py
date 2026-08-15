from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image

from anima_search.delivery.m5_candidates import M5CandidateBatch, SCHEMA_VERSION
from anima_search.indexing.index_manifest import sha256_file


TOP_LEVEL_FIELDS = {
    "schema_version",
    "query_id",
    "query",
    "category",
    "split",
    "fusion_method",
    "top_k",
    "annotation_version",
    "index_manifest_sha256",
    "config_sha256",
    "candidates",
}
CANDIDATE_FIELDS = {
    "rank",
    "image_id",
    "relative_path",
    "fused_score",
    "branch_scores",
    "branch_ranks",
    "matched_fields",
}
BRANCHES = {"image", "text", "bm25"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _strict_json(line: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(line, parse_constant=reject_constant)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_m5_m6_file(
    input_path: Path,
    *,
    project_root: Path,
    train_dir: Path,
    val_dir: Path,
    index_manifest: Path,
    config_snapshot: Path | None = None,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    project_root = project_root.resolve()
    split_roots = {
        "train": (project_root / train_dir).resolve(),
        "val": (project_root / val_dir).resolve(),
    }
    index_manifest = index_manifest.resolve()
    config_snapshot = config_snapshot.resolve() if config_snapshot is not None else None
    manifest = json.loads(index_manifest.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(index_manifest)
    config_sha256 = sha256_file(config_snapshot) if config_snapshot is not None else None
    manifest_records = {
        str(item.get("image_id", "")): item
        for item in manifest.get("image_records", [])
        if isinstance(item, dict)
    }

    errors: list[dict[str, Any]] = []
    seen_errors: set[tuple[int, str, str]] = set()
    seen_query_ids: set[str] = set()
    decoded_images: dict[Path, tuple[str | None, str | None]] = {}
    query_count = 0
    candidate_count = 0

    def add(line_number: int, code: str, message: str, candidate: int | None = None) -> None:
        key = (line_number, code, f"{candidate}:{message}")
        if key in seen_errors:
            return
        seen_errors.add(key)
        issue: dict[str, Any] = {"line": line_number, "code": code, "message": message}
        if candidate is not None:
            issue["candidate"] = candidate
        errors.append(issue)

    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            query_count += 1
            try:
                payload = _strict_json(line)
            except (json.JSONDecodeError, ValueError) as exc:
                add(line_number, "E_JSON_PARSE", str(exc))
                continue
            if not isinstance(payload, dict):
                add(line_number, "E_REQUIRED_FIELD", "top-level JSON value must be an object")
                continue

            missing = sorted(TOP_LEVEL_FIELDS - set(payload))
            unknown = sorted(set(payload) - TOP_LEVEL_FIELDS)
            if missing:
                add(line_number, "E_REQUIRED_FIELD", f"missing top-level fields: {missing}")
            if unknown:
                add(line_number, "E_UNKNOWN_FIELD", f"unknown top-level fields: {unknown}")
            if payload.get("schema_version") != SCHEMA_VERSION:
                add(line_number, "E_SCHEMA_VERSION", f"schema_version must be {SCHEMA_VERSION}")
            if payload.get("top_k") != 20 or isinstance(payload.get("top_k"), bool):
                add(line_number, "E_TOP_K", "top_k must be integer 20")

            query_id = payload.get("query_id")
            if isinstance(query_id, str):
                if query_id in seen_query_ids:
                    add(line_number, "E_DUPLICATE_QUERY_ID", f"duplicate query_id {query_id!r}")
                seen_query_ids.add(query_id)

            if payload.get("index_manifest_sha256") != manifest_sha256:
                add(line_number, "E_MANIFEST_MISMATCH", "index manifest SHA-256 does not match")
            if config_sha256 is not None and payload.get("config_sha256") != config_sha256:
                add(line_number, "E_MANIFEST_MISMATCH", "config snapshot SHA-256 does not match")
            if payload.get("annotation_version") != manifest.get("annotation_version"):
                add(line_number, "E_MANIFEST_MISMATCH", "annotation version does not match manifest")
            split = payload.get("split")
            if split in split_roots and split != manifest.get("split"):
                add(line_number, "E_MANIFEST_MISMATCH", "split does not match index manifest")

            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                add(line_number, "E_CANDIDATE_COUNT", "candidates must be an array of length 20")
                continue
            candidate_count += len(candidates)
            if len(candidates) != 20:
                add(line_number, "E_CANDIDATE_COUNT", f"expected 20 candidates, found {len(candidates)}")

            seen_image_ids: set[str] = set()
            for position, candidate_payload in enumerate(candidates, start=1):
                if not isinstance(candidate_payload, dict):
                    add(line_number, "E_REQUIRED_FIELD", "candidate must be an object", position)
                    continue
                candidate_missing = sorted(CANDIDATE_FIELDS - set(candidate_payload))
                candidate_unknown = sorted(set(candidate_payload) - CANDIDATE_FIELDS)
                if candidate_missing:
                    add(
                        line_number,
                        "E_REQUIRED_FIELD",
                        f"missing candidate fields: {candidate_missing}",
                        position,
                    )
                if candidate_unknown:
                    add(
                        line_number,
                        "E_UNKNOWN_FIELD",
                        f"unknown candidate fields: {candidate_unknown}",
                        position,
                    )
                if candidate_payload.get("rank") != position or isinstance(
                    candidate_payload.get("rank"), bool
                ):
                    add(line_number, "E_RANK_SEQUENCE", "rank must match array position", position)

                image_id = candidate_payload.get("image_id")
                if isinstance(image_id, str):
                    if image_id in seen_image_ids:
                        add(
                            line_number,
                            "E_DUPLICATE_IMAGE_ID",
                            f"duplicate image_id {image_id!r}",
                            position,
                        )
                    seen_image_ids.add(image_id)

                fused_score = candidate_payload.get("fused_score")
                if not _is_number(fused_score) or not math.isfinite(fused_score):
                    add(line_number, "E_NONFINITE_SCORE", "fused_score must be finite", position)

                branch_scores = candidate_payload.get("branch_scores")
                branch_ranks = candidate_payload.get("branch_ranks")
                score_keys = set(branch_scores) if isinstance(branch_scores, dict) else set()
                rank_keys = set(branch_ranks) if isinstance(branch_ranks, dict) else set()
                if not score_keys or score_keys != rank_keys:
                    add(
                        line_number,
                        "E_BRANCH_KEYS",
                        "branch_scores and branch_ranks must have identical non-empty keys",
                        position,
                    )
                if (score_keys | rank_keys) - BRANCHES:
                    add(line_number, "E_BRANCH_NAME", "unknown retrieval branch", position)
                if isinstance(branch_scores, dict) and any(
                    not _is_number(value) or not math.isfinite(value)
                    for value in branch_scores.values()
                ):
                    add(line_number, "E_NONFINITE_SCORE", "branch score must be finite", position)
                if isinstance(branch_ranks, dict) and any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 1
                    for value in branch_ranks.values()
                ):
                    add(line_number, "E_BRANCH_KEYS", "branch ranks must be positive integers", position)

                relative_path = candidate_payload.get("relative_path")
                path_valid = isinstance(relative_path, str) and bool(relative_path)
                if path_valid:
                    relative = Path(relative_path)
                    if "\\" in relative_path or relative.is_absolute():
                        path_valid = False
                if not path_valid:
                    add(
                        line_number,
                        "E_PATH_FORMAT",
                        "relative_path must be a non-absolute POSIX path",
                        position,
                    )
                    continue

                resolved_path = (project_root / relative_path).resolve()
                split_root = split_roots.get(split)
                if split_root is None or not _inside(resolved_path, split_root):
                    add(line_number, "E_PATH_SPLIT", "path is outside the declared split", position)
                    continue
                manifest_record = (
                    manifest_records.get(image_id) if isinstance(image_id, str) else None
                )
                expected_path = (
                    str(manifest_record.get("relative_path", ""))
                    if manifest_record is not None
                    else None
                )
                if expected_path != relative_path:
                    add(
                        line_number,
                        "E_MANIFEST_MISMATCH",
                        "image_id or relative_path does not match manifest",
                        position,
                    )
                if not resolved_path.is_file():
                    add(line_number, "E_IMAGE_MISSING", "image file does not exist", position)
                    continue
                if resolved_path not in decoded_images:
                    try:
                        with Image.open(resolved_path) as image:
                            image.load()
                        decoded_images[resolved_path] = (None, sha256_file(resolved_path))
                    except Exception as exc:  # Pillow exposes several decoder-specific errors.
                        decoded_images[resolved_path] = (
                            f"{type(exc).__name__}: {exc}",
                            None,
                        )
                decode_error, image_sha256 = decoded_images[resolved_path]
                if decode_error:
                    add(line_number, "E_IMAGE_DECODE", decode_error, position)
                expected_sha256 = (
                    str(manifest_record.get("sha256", ""))
                    if manifest_record is not None
                    else None
                )
                if image_sha256 is not None and image_sha256 != expected_sha256:
                    add(
                        line_number,
                        "E_MANIFEST_MISMATCH",
                        "image SHA-256 does not match manifest",
                        position,
                    )

            try:
                M5CandidateBatch.model_validate(payload)
            except Exception as exc:
                validation_errors = getattr(exc, "errors", lambda: [])()
                if validation_errors:
                    for validation_error in validation_errors:
                        location = ".".join(str(part) for part in validation_error.get("loc", ()))
                        message = str(validation_error.get("msg", "schema validation failed"))
                        error_type = str(validation_error.get("type", ""))
                        combined = f"{location} {message}".lower()
                        if error_type == "extra_forbidden":
                            code = "E_UNKNOWN_FIELD"
                        elif "schema_version" in combined:
                            code = "E_SCHEMA_VERSION"
                        elif "top_k" in combined:
                            code = "E_TOP_K"
                        elif "candidate" in combined and ("20" in combined or "length" in combined):
                            code = "E_CANDIDATE_COUNT"
                        elif "rank" in combined and "branch" not in combined:
                            code = "E_RANK_SEQUENCE"
                        elif "duplicate" in combined and "image" in combined:
                            code = "E_DUPLICATE_IMAGE_ID"
                        elif "branch" in combined:
                            code = "E_BRANCH_KEYS"
                        elif "finite" in combined or "nan" in combined or "infinity" in combined:
                            code = "E_NONFINITE_SCORE"
                        elif "relative_path" in combined:
                            code = "E_PATH_FORMAT"
                        else:
                            code = "E_REQUIRED_FIELD"
                        add(
                            line_number,
                            code,
                            f"schema validation failed at {location or '<root>'}: {message}",
                        )
                else:
                    add(line_number, "E_REQUIRED_FIELD", f"schema validation failed: {exc}")

    return {
        "schema_version": "m5-to-m6-validation-v1",
        "valid": not errors,
        "input": str(input_path),
        "index_manifest": str(index_manifest),
        "index_manifest_sha256": manifest_sha256,
        "config_snapshot": str(config_snapshot) if config_snapshot is not None else None,
        "config_sha256": config_sha256,
        "query_count": query_count,
        "candidate_count": candidate_count,
        "decoded_unique_images": sum(value[0] is None for value in decoded_images.values()),
        "hashed_unique_images": sum(value[1] is not None for value in decoded_images.values()),
        "error_count": len(errors),
        "errors": errors,
    }
