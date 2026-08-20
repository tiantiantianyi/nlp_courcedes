#!/usr/bin/env python3
"""Merge normalized M1 retry records without mutating the base artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument(
        "--override-dir",
        type=Path,
        action="append",
        required=True,
        help="Normalized retry directory; later directories take precedence.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
        if not image_id or image_id == "None":
            raise ValueError(f"Missing image_id in {source}")
        if image_id in result:
            raise ValueError(f"Duplicate image_id {image_id!r} in {source}")
        result[image_id] = record
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_path = args.base_dir / "normalization_records.jsonl"
    base_records = read_jsonl(base_path)
    base_by_id = keyed(base_records, base_path)
    ordered_ids = [str(record["image_id"]) for record in base_records]

    merged_by_id = dict(base_by_id)
    override_sources: dict[str, str] = {}
    override_counts: list[dict[str, Any]] = []
    for override_dir in args.override_dir:
        override_path = override_dir / "normalization_records.jsonl"
        override_by_id = keyed(read_jsonl(override_path), override_path)
        unexpected = sorted(set(override_by_id) - set(base_by_id))
        if unexpected:
            raise ValueError(
                f"Override contains IDs absent from base: {unexpected[:10]}"
            )
        merged_by_id.update(override_by_id)
        for image_id in override_by_id:
            override_sources[image_id] = str(override_dir)
        override_counts.append(
            {"override_dir": str(override_dir), "records": len(override_by_id)}
        )

    merged = [merged_by_id[image_id] for image_id in ordered_ids]
    annotations: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    parse_modes: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    remaining_schema_paths: Counter[str] = Counter()
    remaining_semantic_paths: Counter[str] = Counter()
    lossy_images = 0

    for record_line, record in enumerate(merged, start=1):
        status = str(record.get("status"))
        statuses[status] += 1
        parse_modes[str(record.get("parse_mode"))] += 1
        repairs = record.get("repairs") or []
        lossy = any(repair.get("lossy") is True for repair in repairs)
        if lossy:
            lossy_images += 1
        for repair in repairs:
            repair_counts[str(repair.get("rule"))] += 1
        for error in record.get("remaining_schema_errors") or []:
            remaining_schema_paths[str(error.get("path"))] += 1
        for error in record.get("remaining_semantic_errors") or []:
            remaining_semantic_paths[str(error.get("path"))] += 1

        if record.get("normalized_annotation_valid") is True:
            annotations.append(
                {
                    "image_id": str(record["image_id"]),
                    "processed_sha256": record.get("processed_sha256"),
                    "source_model_id": record.get("source_model_id"),
                    "normalizer_version": record.get("normalizer_version"),
                    "repairs_applied": len(repairs),
                    "lossy_repairs": lossy,
                    "annotation": record.get("annotation"),
                }
            )
        if status != "valid":
            review.append(
                {
                    "image_id": str(record["image_id"]),
                    "status": status,
                    "record_line": record_line,
                    "lossy_repairs": lossy,
                    "remaining_schema_errors": record.get("remaining_schema_errors")
                    or [],
                    "remaining_semantic_errors": record.get(
                        "remaining_semantic_errors"
                    )
                    or [],
                }
            )

    records_path = args.output_dir / "normalization_records.jsonl"
    annotations_path = args.output_dir / "normalized_annotations.jsonl"
    review_path = args.output_dir / "review_queue.jsonl"
    write_jsonl(records_path, merged)
    write_jsonl(annotations_path, annotations)
    write_jsonl(review_path, review)

    summary = {
        "normalizer_version": merged[0].get("normalizer_version") if merged else None,
        "source_run_dir": "composite_normalized_records",
        "total_images": len(merged),
        "source_annotation_valid": sum(
            record.get("source_annotation_valid") is True for record in merged
        ),
        "parse_modes": dict(sorted(parse_modes.items())),
        "statuses": dict(sorted(statuses.items())),
        "normalized_annotation_valid": len(annotations),
        "normalized_annotation_valid_rate": round(len(annotations) / len(merged), 4),
        "images_with_lossy_repairs": lossy_images,
        "repair_counts": dict(repair_counts.most_common()),
        "remaining_schema_error_paths": dict(remaining_schema_paths.most_common()),
        "remaining_semantic_error_paths": dict(
            remaining_semantic_paths.most_common()
        ),
        "merge": {
            "base_dir": str(args.base_dir),
            "overrides": override_counts,
            "overridden_image_ids": sorted(
                override_sources,
                key=lambda value: (int(value), value)
                if value.isdigit()
                else (10**18, value),
            ),
            "final_override_source_by_image_id": override_sources,
        },
        "artifacts": {
            "records": str(records_path),
            "normalized_annotations": str(annotations_path),
            "review_queue": str(review_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
