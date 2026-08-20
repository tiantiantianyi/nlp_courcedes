#!/usr/bin/env python3
"""Independently verify normalized M1 artifacts and pairwise coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from m1_validation import semantic_validation_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument(
        "--normalized",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def keyed(records: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        if image_id in result:
            raise ValueError(f"Duplicate image_id {image_id!r} in {source}")
        result[image_id] = record
    return result


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"Invalid labeled path: {value!r}")
    return label, Path(path)


def main() -> None:
    args = parse_args()
    manifest_records = read_jsonl(args.manifest)
    manifest = keyed(manifest_records, args.manifest)
    manifest_ids = set(manifest)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    runs: list[dict[str, Any]] = []
    valid_sets: dict[str, set[str]] = {}
    processed_hashes: dict[str, dict[str, str]] = {}
    for labeled_path in args.normalized:
        label, run_dir = parse_labeled_path(labeled_path)
        records_path = run_dir / "normalization_records.jsonl"
        annotations_path = run_dir / "normalized_annotations.jsonl"
        review_path = run_dir / "review_queue.jsonl"
        records = keyed(read_jsonl(records_path), records_path)
        annotations = keyed(read_jsonl(annotations_path), annotations_path)
        review = read_jsonl(review_path)

        record_ids = set(records)
        valid_ids = set(annotations)
        missing = sorted(manifest_ids - record_ids)
        unexpected = sorted(record_ids - manifest_ids)
        if missing or unexpected:
            raise ValueError(
                f"{label}: record coverage mismatch; missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )
        if not valid_ids <= record_ids:
            raise ValueError(f"{label}: normalized annotations contain unknown IDs")

        hash_mismatches = []
        annotation_errors = []
        status_counts: Counter[str] = Counter()
        image_identity_status_counts: Counter[str] = Counter()
        for image_id, record in records.items():
            expected_hash = manifest[image_id].get("processed_sha256")
            actual_hash = record.get("processed_sha256")
            if actual_hash != expected_hash:
                hash_mismatches.append(image_id)
            status_counts[str(record.get("status"))] += 1
            image_identity_status_counts[
                str(record.get("image_identity_status", "exact_hash_match"))
            ] += 1

        for image_id, record in annotations.items():
            annotation = record.get("annotation")
            schema_errors = list(validator.iter_errors(annotation))
            semantic_errors = semantic_validation_errors(annotation)
            if schema_errors or semantic_errors:
                annotation_errors.append(
                    {
                        "image_id": image_id,
                        "schema_errors": len(schema_errors),
                        "semantic_errors": len(semantic_errors),
                    }
                )

        if hash_mismatches or annotation_errors:
            raise ValueError(
                f"{label}: hash_mismatches={hash_mismatches[:10]}, "
                f"annotation_errors={annotation_errors[:10]}"
            )

        valid_sets[label] = valid_ids
        processed_hashes[label] = {
            image_id: str(record["processed_sha256"])
            for image_id, record in records.items()
        }
        runs.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "normalization_records": len(records),
                "normalized_annotations": len(annotations),
                "review_queue_records": len(review),
                "status_counts": dict(status_counts),
                "manifest_coverage_valid": True,
                "workspace_manifest_hashes_valid": True,
                "processed_hashes_valid": True,
                "source_image_identity_status_counts": dict(
                    image_identity_status_counts
                ),
                "source_image_hashes_exact": (
                    set(image_identity_status_counts) == {"exact_hash_match"}
                ),
                "annotation_schema_and_semantic_valid": True,
            }
        )

    labels = list(valid_sets)
    common_valid = set.intersection(*(valid_sets[label] for label in labels))
    hash_agreement = all(
        len({processed_hashes[label][image_id] for label in labels}) == 1
        for image_id in manifest_ids
    )
    source_image_identity_exact = all(
        run["source_image_hashes_exact"] for run in runs
    )
    report = {
        "verification_status": "passed",
        "manifest_images": len(manifest),
        "schema": str(args.schema),
        "runs": runs,
        "pairwise": {
            "labels": labels,
            "processed_hash_agreement_all_images": hash_agreement,
            "workspace_manifest_hash_agreement_all_images": hash_agreement,
            "source_image_identity_exact_all_runs": source_image_identity_exact,
            "source_image_identity_note": (
                "False means at least one imported run retained source image hashes "
                "that differ from the workspace manifest. It can be structurally "
                "aligned by image_id while exact source image bytes remain unproven."
            ),
            "both_normalized_valid": len(common_valid),
            "both_normalized_valid_rate": round(len(common_valid) / len(manifest), 4),
            "not_ready_for_direct_pairwise_comparison": len(manifest_ids - common_valid),
            "not_ready_image_ids": sorted(manifest_ids - common_valid),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
