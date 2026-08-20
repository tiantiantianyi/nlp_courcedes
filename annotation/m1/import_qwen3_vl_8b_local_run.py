#!/usr/bin/env python3
"""Import the coworker Qwen3-VL-8B run into the shared M1 layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from m1_validation import (
    diagnostic_json_candidate,
    semantic_validation_errors,
    validation_error_records,
)


IMPORT_VERSION = "m1-coworker-import-v1.0.0"
MODEL_SLUG = "qwen3_vl_8b_instruct"
SOURCE_ROOT = "local_run"
RAW_MEMBER_RE = re.compile(
    r"^local_run/(train|val)/raw/(?:train|val)-(\d+)\.attempt-01\.txt$"
)
SOURCE_ID_RE = re.compile(r"^(train|val)-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--postprocess-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def normalized_error_prefix(error: Any) -> str:
    if not isinstance(error, str) or not error:
        return "none"
    return error.split(":", 1)[0]


def main() -> None:
    args = parse_args()
    ensure_empty(args.full_dir)
    ensure_empty(args.postprocess_dir)

    archive_sha256 = sha256_file(args.archive)
    prompt_sha256 = sha256_file(args.prompt)
    schema_sha256 = sha256_file(args.schema)
    manifest_records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = {str(record["image_id"]): record for record in manifest_records}
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    raw_dir = args.full_dir / "raw"
    metadata_dir = args.full_dir / "source_metadata"
    error_list_dir = metadata_dir / "error_lists"
    raw_dir.mkdir()
    error_list_dir.mkdir(parents=True)

    source_summaries: dict[str, dict[str, Any]] = {}
    source_error_ids: dict[str, set[str]] = {}
    source_candidates: list[dict[str, Any]] = []
    raw_content: dict[str, str] = {}
    raw_split: dict[str, str] = {}

    with zipfile.ZipFile(args.archive) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")

        for split in ("train", "val"):
            summary_member = f"{SOURCE_ROOT}/{split}/run_summary.json"
            candidates_member = f"{SOURCE_ROOT}/{split}/candidates_local.jsonl"
            error_member = f"{SOURCE_ROOT}/{split}error.txt"

            summary_bytes = archive.read(summary_member)
            error_bytes = archive.read(error_member)
            source_summaries[split] = json.loads(summary_bytes)
            (metadata_dir / f"{split}_run_summary.json").write_bytes(summary_bytes)
            (error_list_dir / f"{split}error.txt").write_bytes(error_bytes)

            source_error_ids[split] = {
                match.group(1)
                for line in error_bytes.decode("utf-8").splitlines()
                if (match := re.search(r"-(\d+)\.attempt-01\.txt$", line.strip()))
            }
            source_candidates.extend(read_jsonl_bytes(archive.read(candidates_member)))

        for member in archive.namelist():
            match = RAW_MEMBER_RE.fullmatch(member)
            if not match:
                continue
            split, image_id = match.groups()
            if image_id in raw_content:
                raise ValueError(f"Duplicate raw image ID in archive: {image_id}")
            content = archive.read(member)
            raw_content[image_id] = content.decode("utf-8")
            raw_split[image_id] = split
            (raw_dir / f"{image_id}.txt").write_bytes(content)

    if set(raw_content) != set(manifest):
        missing = sorted(set(manifest) - set(raw_content), key=int)
        extra = sorted(set(raw_content) - set(manifest), key=int)
        raise ValueError(f"Raw/manifest mismatch; missing={missing}, extra={extra}")

    candidate_by_id: dict[str, dict[str, Any]] = {}
    adapted_candidates: list[dict[str, Any]] = []
    accepted_annotations: list[dict[str, Any]] = []
    for source_record in source_candidates:
        source_image_id = str(source_record["image_id"])
        match = SOURCE_ID_RE.fullmatch(source_image_id)
        if not match:
            raise ValueError(f"Unexpected source image ID: {source_image_id}")
        split, image_id = match.groups()
        if image_id in candidate_by_id:
            raise ValueError(f"Duplicate candidate image ID: {image_id}")
        if image_id not in manifest:
            raise ValueError(f"Candidate image ID is not in manifest: {image_id}")
        if manifest[image_id]["split"] != split or raw_split[image_id] != split:
            raise ValueError(f"Split mismatch for image {image_id}")

        adapted = {
            "image_id": image_id,
            "split": split,
            "source_image_id": source_image_id,
            "source_processed_sha256": source_record["processed_sha256"],
            "workspace_processed_sha256": manifest[image_id]["processed_sha256"],
            "image_sha256_matches_workspace": (
                source_record["processed_sha256"]
                == manifest[image_id]["processed_sha256"]
            ),
            "source_kind": source_record["source_kind"],
            "model_id": source_record["model_id"],
            "prompt_version": source_record["prompt_version"],
            "annotation_schema_version": source_record["annotation_schema_version"],
            "source_status": source_record["status"],
            "raw_response_path": str(raw_dir / f"{image_id}.txt"),
            "annotation": source_record["annotation"],
            "error": source_record["error"],
        }
        candidate_by_id[image_id] = source_record
        adapted_candidates.append(adapted)
        if source_record["status"] == "succeeded":
            accepted_annotations.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "source_processed_sha256": source_record["processed_sha256"],
                    "source_model_id": source_record["model_id"],
                    "source_prompt_version": source_record["prompt_version"],
                    "source_annotation_schema_version": source_record[
                        "annotation_schema_version"
                    ],
                    "annotation": source_record["annotation"],
                }
            )

    if set(candidate_by_id) != set(manifest):
        missing = sorted(set(manifest) - set(candidate_by_id), key=int)
        extra = sorted(set(candidate_by_id) - set(manifest), key=int)
        raise ValueError(f"Candidate/manifest mismatch; missing={missing}, extra={extra}")

    adapted_candidates.sort(key=lambda item: int(item["image_id"]))
    accepted_annotations.sort(key=lambda item: int(item["image_id"]))

    raw_audit: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    source_status_counts: Counter[str] = Counter()
    source_error_prefixes: Counter[str] = Counter()
    status_transition_counts: Counter[str] = Counter()
    source_annotation_matches_current_raw = 0
    current_parse_failures: list[dict[str, Any]] = []

    for image_id in sorted(manifest, key=int):
        source_record = candidate_by_id[image_id]
        split = manifest[image_id]["split"]
        raw = raw_content[image_id]
        source_status = str(source_record["status"])
        source_status_counts[source_status] += 1
        source_error_prefixes[normalized_error_prefix(source_record.get("error"))] += 1
        raw_counts["total"] += 1

        annotation: Any | None = None
        parse_mode = "unrecoverable"
        parse_error: str | None = None
        format_issues: list[str] = []
        duplicate_keys: list[str] = []
        try:
            annotation = json.loads(raw)
            parse_mode = "strict"
        except json.JSONDecodeError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            annotation, format_issues, duplicate_keys, diagnostic_error = (
                diagnostic_json_candidate(raw)
            )
            if annotation is not None and not duplicate_keys:
                parse_mode = "diagnostic"
                parse_error = diagnostic_error

        schema_errors: list[dict[str, str]] = []
        semantic_errors: list[dict[str, str]] = []
        if isinstance(annotation, dict):
            schema_errors = validation_error_records(
                list(validator.iter_errors(annotation))
            )
            semantic_errors = semantic_validation_errors(annotation)
        elif annotation is not None:
            schema_errors = [{"path": "$", "message": "annotation must be an object"}]

        strict_json = parse_mode == "strict"
        recoverable_json = annotation is not None and not duplicate_keys
        schema_valid = recoverable_json and not schema_errors
        semantic_valid = recoverable_json and not semantic_errors
        annotation_valid = schema_valid and semantic_valid
        raw_counts[f"parse_mode:{parse_mode}"] += 1
        raw_counts["schema_valid" if schema_valid else "schema_invalid"] += 1
        raw_counts["semantic_valid" if semantic_valid else "semantic_invalid"] += 1
        raw_counts[
            "annotation_valid" if annotation_valid else "annotation_invalid"
        ] += 1
        status_transition_counts[
            f"source_{source_status}->current_raw_"
            f"{'valid' if annotation_valid else 'invalid'}"
        ] += 1

        if source_record.get("annotation") is not None and source_record.get(
            "annotation"
        ) == annotation:
            source_annotation_matches_current_raw += 1

        listed = image_id in source_error_ids[split]
        audit_record = {
            "image_id": image_id,
            "split": split,
            "raw_response_path": str(raw_dir / f"{image_id}.txt"),
            "raw_sha256": sha256_bytes(raw.encode("utf-8")),
            "source_status": source_status,
            "source_error_prefix": normalized_error_prefix(source_record.get("error")),
            "listed_in_source_error_file": listed,
            "parse_mode": parse_mode,
            "strict_json": strict_json,
            "format_issues": format_issues,
            "duplicate_keys": duplicate_keys,
            "parse_error": parse_error,
            "schema_valid_v1_2": schema_valid,
            "semantic_valid": semantic_valid,
            "annotation_valid_v1_2": annotation_valid,
            "schema_error_count": len(schema_errors),
            "semantic_error_count": len(semantic_errors),
        }
        raw_audit.append(audit_record)
        if not recoverable_json:
            current_parse_failures.append(audit_record)

    source_error_count = sum(len(values) for values in source_error_ids.values())
    unlisted_parse_failure_ids = [
        item["image_id"]
        for item in current_parse_failures
        if not item["listed_in_source_error_file"]
    ]
    listed_but_now_parseable_ids = sorted(
        {
            image_id
            for split_ids in source_error_ids.values()
            for image_id in split_ids
        }
        - {item["image_id"] for item in current_parse_failures},
        key=int,
    )

    prompt_copy = args.full_dir / f"input_prompt_{prompt_sha256[:12]}.md"
    schema_copy = args.full_dir / f"input_schema_{schema_sha256[:12]}.json"
    shutil.copy2(args.prompt, prompt_copy)
    shutil.copy2(args.schema, schema_copy)

    write_jsonl(args.full_dir / "raw_audit.jsonl", raw_audit)
    write_jsonl(metadata_dir / "current_raw_parse_failures.jsonl", current_parse_failures)
    write_jsonl(args.postprocess_dir / "source_candidates.jsonl", adapted_candidates)
    write_jsonl(
        args.postprocess_dir / "source_accepted_annotations.jsonl",
        accepted_annotations,
    )

    model_ids = sorted({str(item["model_id"]) for item in source_candidates})
    prompt_versions = sorted(
        {str(item["prompt_version"]) for item in source_candidates}
    )
    schema_versions = sorted(
        {str(item["annotation_schema_version"]) for item in source_candidates}
    )
    source_processed_hash_matches = sum(
        item["image_sha256_matches_workspace"] for item in adapted_candidates
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "import_version": IMPORT_VERSION,
        "status": "imported_complete_raw",
        "model_slug": MODEL_SLUG,
        "model_ids": model_ids,
        "source_archive": str(args.archive),
        "source_archive_sha256": archive_sha256,
        "source_archive_bytes": args.archive.stat().st_size,
        "manifest_path": str(args.manifest),
        "manifest_images": len(manifest),
        "raw_images": len(raw_content),
        "candidate_records": len(source_candidates),
        "split_counts": dict(Counter(raw_split.values())),
        "prompt": {
            "versions": prompt_versions,
            "sha256": prompt_sha256,
            "copied_file": str(prompt_copy),
        },
        "annotation_schema": {
            "versions": schema_versions,
            "sha256": schema_sha256,
            "copied_file": str(schema_copy),
        },
        "source_candidate_snapshot": {
            "status_counts": dict(source_status_counts),
            "error_prefix_counts": dict(source_error_prefixes),
            "accepted_annotations": len(accepted_annotations),
            "note": (
                "This snapshot predates some raw-file updates and is retained as "
                "source provenance, not as the current raw audit."
            ),
        },
        "current_raw_audit_v1_2": {
            "parse_modes": {
                key.removeprefix("parse_mode:"): value
                for key, value in sorted(raw_counts.items())
                if key.startswith("parse_mode:")
            },
            "schema_valid": raw_counts["schema_valid"],
            "semantic_valid": raw_counts["semantic_valid"],
            "annotation_valid": raw_counts["annotation_valid"],
            "status_transition_counts": dict(status_transition_counts),
            "source_annotations_equal_to_current_raw": (
                source_annotation_matches_current_raw
            ),
        },
        "source_error_lists": {
            "listed_images": source_error_count,
            "current_unrecoverable_raw": len(current_parse_failures),
            "current_unrecoverable_not_listed": unlisted_parse_failure_ids,
            "listed_but_currently_parseable": listed_but_now_parseable_ids,
        },
        "image_identity_check": {
            "numeric_id_coverage": len(raw_content),
            "split_matches_manifest": all(
                raw_split[image_id] == manifest[image_id]["split"]
                for image_id in manifest
            ),
            "source_processed_sha256_matches_workspace": (
                source_processed_hash_matches
            ),
            "source_processed_sha256_mismatches_workspace": (
                len(adapted_candidates) - source_processed_hash_matches
            ),
            "warning": (
                "All source processed-image hashes differ from the current workspace "
                "manifest. Numeric IDs and train/val splits align, but exact image "
                "identity or preprocessing equivalence is not yet proven."
            ),
        },
        "source_run_summary_scope": {
            "train": {
                "requested": source_summaries["train"].get("requested"),
                "succeeded": source_summaries["train"].get("succeeded"),
                "failed": source_summaries["train"].get("failed"),
                "skipped_existing": source_summaries["train"].get(
                    "skipped_existing"
                ),
            },
            "val": {
                "requested": source_summaries["val"].get("requested"),
                "succeeded": source_summaries["val"].get("succeeded"),
                "failed": source_summaries["val"].get("failed"),
                "skipped_existing": source_summaries["val"].get(
                    "skipped_existing"
                ),
            },
            "note": (
                "The source summaries describe their final invocation/resume, not the "
                "aggregate state of all 2369 raw files."
            ),
        },
        "artifacts": {
            "raw_dir": str(raw_dir),
            "raw_audit": str(args.full_dir / "raw_audit.jsonl"),
            "source_candidates": str(
                args.postprocess_dir / "source_candidates.jsonl"
            ),
            "source_accepted_annotations": str(
                args.postprocess_dir / "source_accepted_annotations.jsonl"
            ),
            "source_metadata": str(metadata_dir),
        },
    }
    write_json(args.full_dir / "run_summary.json", summary)
    write_json(args.postprocess_dir / "source_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
