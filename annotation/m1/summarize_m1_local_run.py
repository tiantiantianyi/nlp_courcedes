#!/usr/bin/env python3
"""Merge per-image local M1 artifacts and calculate structural smoke metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m1_validation import candidate_record_validator, validation_error_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--candidate-schema", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def main() -> int:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    annotation_schema = json.loads(args.schema.read_text(encoding="utf-8"))
    candidate_schema = json.loads(args.candidate_schema.read_text(encoding="utf-8"))
    validator = candidate_record_validator(candidate_schema, annotation_schema)

    candidates: list[dict[str, Any]] = []
    item_summaries: list[dict[str, Any]] = []
    missing_image_ids: list[str] = []
    candidate_errors: dict[str, list[dict[str, str]]] = {}

    for manifest_record in manifest:
        image_id = manifest_record["image_id"]
        item_dir = args.run_dir / "items" / image_id
        candidate_path = item_dir / "candidate_record.json"
        summary_path = item_dir / "summary.json"
        if not candidate_path.is_file() or not summary_path.is_file():
            missing_image_ids.append(image_id)
            continue
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        errors = validation_error_records(list(validator.iter_errors(candidate)))
        if candidate.get("image_id") != image_id:
            errors.append({"path": "$.image_id", "message": "does not match manifest"})
        if candidate.get("processed_sha256") != manifest_record["processed_sha256"]:
            errors.append(
                {"path": "$.processed_sha256", "message": "does not match manifest"}
            )
        if errors:
            candidate_errors[image_id] = errors
        candidates.append(candidate)
        item_summaries.append(summary)

    write_jsonl(args.run_dir / "candidates_local.jsonl", candidates)

    total = len(manifest)
    completed = len(item_summaries)
    strict_json = sum(item.get("json_parse_ok") is True for item in item_summaries)
    schema_valid = sum(item.get("schema_valid") is True for item in item_summaries)
    semantic_valid = sum(item.get("semantic_valid") is True for item in item_summaries)
    annotation_valid = sum(item.get("annotation_valid") is True for item in item_summaries)
    generation_times = [
        float(item["generation_seconds"])
        for item in item_summaries
        if isinstance(item.get("generation_seconds"), (int, float))
    ]
    output_tokens = [
        int(item["output_tokens"])
        for item in item_summaries
        if isinstance(item.get("output_tokens"), int)
    ]
    peak_allocated = [
        int(item["gpu_peak_allocated_bytes"])
        for item in item_summaries
        if isinstance(item.get("gpu_peak_allocated_bytes"), int)
    ]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete"
            if completed == total and not candidate_errors
            else "partial_or_invalid"
        ),
        "manifest_path": str(args.manifest),
        "total_images": total,
        "completed_images": completed,
        "missing_image_ids": missing_image_ids,
        "candidate_record_schema_valid": len(candidate_errors) == 0,
        "candidate_record_errors": candidate_errors,
        "strict_json_count": strict_json,
        "strict_json_rate": rate(strict_json, completed),
        "annotation_schema_valid_count": schema_valid,
        "annotation_schema_valid_rate": rate(schema_valid, completed),
        "semantic_valid_count": semantic_valid,
        "semantic_valid_rate": rate(semantic_valid, completed),
        "annotation_valid_count": annotation_valid,
        "annotation_valid_rate": rate(annotation_valid, completed),
        "generation_seconds_total": (
            round(sum(generation_times), 3) if generation_times else None
        ),
        "generation_seconds_mean": (
            round(statistics.mean(generation_times), 3) if generation_times else None
        ),
        "generation_seconds_median": (
            round(statistics.median(generation_times), 3) if generation_times else None
        ),
        "output_tokens_total": sum(output_tokens) if output_tokens else None,
        "output_tokens_mean": (
            round(statistics.mean(output_tokens), 1) if output_tokens else None
        ),
        "gpu_peak_allocated_bytes_max": max(peak_allocated) if peak_allocated else None,
        "per_image": [
            {
                "image_id": item.get("image_id"),
                "json_parse_ok": item.get("json_parse_ok"),
                "schema_valid": item.get("schema_valid"),
                "semantic_valid": item.get("semantic_valid"),
                "annotation_valid": item.get("annotation_valid"),
                "generation_seconds": item.get("generation_seconds"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
                "error": item.get("error"),
            }
            for item in item_summaries
        ],
        "manual_quality_review": {
            "status": "pending",
            "note_zh": "结构通过不代表视觉事实正确；仍需逐图检查实体、OCR、bbox 和图中内容边界。",
        },
    }
    write_json(args.run_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if completed == total and not candidate_errors else 1


if __name__ == "__main__":
    sys.exit(main())
