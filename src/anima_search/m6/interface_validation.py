from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError

from anima_search.indexing.index_manifest import (
    image_ids_digest,
    sha256_file,
)
from anima_search.m6.contract import M5QueryBatch


class InterfaceIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    line: int | None = None
    query_id: str | None = None
    message: str


class InterfaceValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    query_count: int
    candidate_count: int
    issues: list[InterfaceIssue]


class _NonFiniteNumberError(ValueError):
    pass


def _reject_nonfinite(value: str) -> None:
    raise _NonFiniteNumberError(f"non-finite JSON number is forbidden: {value}")


def _validation_code(error: dict[str, Any]) -> str:
    location = tuple(error.get("loc", ()))
    message = str(error.get("msg", "")).lower()
    error_type = str(error.get("type", ""))
    if location == ("schema_version",):
        return "E_SCHEMA_VERSION"
    if location == ("top_k",):
        return "E_TOP_K"
    if location == ("candidates",) and error_type in {
        "too_short",
        "too_long",
        "list_type",
    }:
        return "E_CANDIDATE_COUNT"
    if (
        ("branch_scores" in location or "branch_ranks" in location)
        and error_type == "literal_error"
    ):
        return "E_BRANCH_NAME"
    if "candidates" in location and location[-1:] == ("rank",):
        return "E_RANK_SEQUENCE"
    if "fused_score" in location or "branch_scores" in location:
        return "E_NONFINITE_SCORE"
    if "branch_ranks" in location or "branch ranks must be positive" in message:
        return "E_BRANCH_KEYS"
    if "branch keys" in message:
        return "E_BRANCH_KEYS"
    if "ordered sequence" in message:
        return "E_RANK_SEQUENCE"
    if "must be unique" in message:
        return "E_DUPLICATE_IMAGE_ID"
    if error_type == "finite_number":
        return "E_NONFINITE_SCORE"
    if error_type == "extra_forbidden":
        return "E_UNKNOWN_FIELD"
    return "E_REQUIRED_FIELD"


def _resolve_candidate_path(project_root: Path, relative_path: str) -> Path:
    if "\\" in relative_path or Path(relative_path).is_absolute():
        raise ValueError("candidate path must be a relative POSIX path")
    return (project_root / relative_path).resolve()


def _load_manifest_catalog(
    index_manifest_path: Path,
    project_root: Path,
    issues: list[InterfaceIssue],
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(
            InterfaceIssue(
                code="E_MANIFEST_MISMATCH",
                message=f"cannot read index manifest: {type(exc).__name__}: {exc}",
            )
        )
        return {}, {}

    declared_path = manifest.get("annotation_path")
    declared_sha256 = manifest.get("annotation_sha256")
    if not isinstance(declared_path, str) or not declared_path.strip():
        issues.append(
            InterfaceIssue(
                code="E_MANIFEST_MISMATCH",
                message="index manifest must declare annotation_path",
            )
        )
        return manifest, {}
    if not isinstance(declared_sha256, str) or not declared_sha256.strip():
        issues.append(
            InterfaceIssue(
                code="E_MANIFEST_MISMATCH",
                message="index manifest must declare annotation_sha256",
            )
        )
        return manifest, {}

    annotations_path = Path(declared_path)
    if not annotations_path.is_absolute():
        annotations_path = project_root / annotations_path
    annotations_path = annotations_path.resolve()
    try:
        actual_sha256 = sha256_file(annotations_path)
        if actual_sha256 != declared_sha256:
            issues.append(
                InterfaceIssue(
                    code="E_MANIFEST_MISMATCH",
                    message=(
                        "manifest annotation_sha256 does not match declared "
                        f"artifact: {annotations_path}"
                    ),
                )
            )

        text = annotations_path.read_text(encoding="utf-8")
        if text.lstrip().startswith("["):
            rows = json.loads(text)
            if not isinstance(rows, list):
                raise ValueError("annotation artifact must contain a JSON array")
        else:
            rows = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("annotation artifact rows must be JSON objects")
        if any(
            "image_id" not in row or "relative_path" not in row
            for row in rows
        ):
            raise ValueError(
                "annotation artifact rows require image_id and relative_path"
            )
        catalog = {
            str(row["image_id"]): str(row["relative_path"])
            for row in rows
        }
        image_ids = [
            str(row["image_id"])
            for row in rows
        ]
    except Exception as exc:
        issues.append(
            InterfaceIssue(
                code="E_MANIFEST_MISMATCH",
                message=(
                    "cannot read declared annotation artifact: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )
        return manifest, {}

    if manifest.get("record_count") != len(image_ids):
        issues.append(
            InterfaceIssue(
                code="E_MANIFEST_MISMATCH",
                message=(
                    "manifest record_count does not match annotation artifact"
                ),
            )
        )
    if manifest.get("image_ids_sha256") != image_ids_digest(image_ids):
        issues.append(
            InterfaceIssue(
                code="E_MANIFEST_MISMATCH",
                message=(
                    "manifest image_ids_sha256 does not match annotation artifact"
                ),
            )
        )
    return manifest, catalog


def _check_batch_against_manifest(
    batch: M5QueryBatch,
    *,
    line_number: int,
    manifest_hash: str,
    config_snapshot_hash: str,
    manifest: dict[str, Any],
    catalog: dict[str, str],
    project_root: Path,
    train_dir: Path,
    val_dir: Path,
    issues: list[InterfaceIssue],
) -> None:
    def add(code: str, message: str) -> None:
        issues.append(
            InterfaceIssue(
                code=code,
                line=line_number,
                query_id=batch.query_id,
                message=message,
            )
        )

    if batch.index_manifest_sha256 != manifest_hash:
        add(
            "E_MANIFEST_MISMATCH",
            "index_manifest_sha256 does not match the supplied manifest",
        )
    if manifest.get("schema_version") != 2:
        add("E_MANIFEST_MISMATCH", "index manifest schema_version must be 2")
    if manifest.get("split") != batch.split:
        add("E_MANIFEST_MISMATCH", "query split does not match index manifest")
    if manifest.get("annotation_version") != batch.annotation_version:
        add(
            "E_MANIFEST_MISMATCH",
            "annotation_version does not match index manifest",
        )
    if batch.config_sha256 != config_snapshot_hash:
        add(
            "E_MANIFEST_MISMATCH",
            "config_sha256 does not match the supplied M5 retrieval snapshot",
        )

    split_root = val_dir.resolve() if batch.split == "val" else train_dir.resolve()
    for candidate in batch.candidates:
        catalog_path = catalog.get(candidate.image_id)
        if catalog_path is None:
            add(
                "E_MANIFEST_MISMATCH",
                f"candidate {candidate.image_id} is absent from index annotations",
            )
        elif catalog_path != candidate.relative_path:
            add(
                "E_MANIFEST_MISMATCH",
                f"candidate {candidate.image_id} path differs from index annotations",
            )

        try:
            image_path = _resolve_candidate_path(
                project_root,
                candidate.relative_path,
            )
        except ValueError as exc:
            add("E_PATH_FORMAT", f"{candidate.image_id}: {exc}")
            continue
        if not image_path.is_relative_to(split_root):
            add(
                "E_PATH_SPLIT",
                f"{candidate.image_id} resolves outside the {batch.split} root",
            )
            continue
        if not image_path.is_file():
            add(
                "E_IMAGE_MISSING",
                f"{candidate.image_id} image does not exist: {image_path}",
            )
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            add(
                "E_IMAGE_DECODE",
                f"{candidate.image_id} cannot be decoded: {type(exc).__name__}: {exc}",
            )


def validate_interface_file(
    input_path: Path,
    project_root: Path,
    train_dir: Path,
    val_dir: Path,
    index_manifest_path: Path,
    config_snapshot_path: Path,
) -> tuple[list[M5QueryBatch], InterfaceValidationReport]:
    issues: list[InterfaceIssue] = []
    parsed_batches: list[tuple[int, M5QueryBatch]] = []
    seen_query_ids: dict[str, int] = {}

    manifest, catalog = _load_manifest_catalog(
        index_manifest_path,
        project_root.resolve(),
        issues,
    )
    try:
        manifest_hash = sha256_file(index_manifest_path)
    except OSError:
        manifest_hash = ""
    try:
        config_snapshot_hash = sha256_file(config_snapshot_path)
    except OSError as exc:
        config_snapshot_hash = ""
        issues.append(
            InterfaceIssue(
                code="E_MANIFEST_MISMATCH",
                message=(
                    "cannot read M5 retrieval snapshot: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )

    for line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line, parse_constant=_reject_nonfinite)
        except _NonFiniteNumberError as exc:
            issues.append(
                InterfaceIssue(
                    code="E_NONFINITE_SCORE",
                    line=line_number,
                    message=str(exc),
                )
            )
            continue
        except (json.JSONDecodeError, TypeError) as exc:
            issues.append(
                InterfaceIssue(
                    code="E_JSON_PARSE",
                    line=line_number,
                    message=str(exc),
                )
            )
            continue

        raw_query_id = (
            str(payload.get("query_id"))
            if isinstance(payload, dict) and payload.get("query_id") is not None
            else None
        )
        if raw_query_id:
            if raw_query_id in seen_query_ids:
                issues.append(
                    InterfaceIssue(
                        code="E_DUPLICATE_QUERY_ID",
                        line=line_number,
                        query_id=raw_query_id,
                        message=(
                            "query_id duplicates line "
                            f"{seen_query_ids[raw_query_id]}"
                        ),
                    )
                )
            else:
                seen_query_ids[raw_query_id] = line_number

        try:
            batch = M5QueryBatch.model_validate(payload)
        except ValidationError as exc:
            for error in exc.errors():
                issues.append(
                    InterfaceIssue(
                        code=_validation_code(error),
                        line=line_number,
                        query_id=raw_query_id,
                        message=str(error.get("msg", "invalid interface row")),
                    )
                )
            continue
        parsed_batches.append((line_number, batch))

    for line_number, batch in parsed_batches:
        _check_batch_against_manifest(
            batch,
            line_number=line_number,
            manifest_hash=manifest_hash,
            config_snapshot_hash=config_snapshot_hash,
            manifest=manifest,
            catalog=catalog,
            project_root=project_root.resolve(),
            train_dir=train_dir,
            val_dir=val_dir,
            issues=issues,
        )

    batches = [batch for _, batch in parsed_batches]
    candidate_count = sum(len(batch.candidates) for batch in batches)
    report = InterfaceValidationReport(
        valid=not issues,
        query_count=len(batches),
        candidate_count=candidate_count,
        issues=issues,
    )
    return (batches if report.valid else []), report
