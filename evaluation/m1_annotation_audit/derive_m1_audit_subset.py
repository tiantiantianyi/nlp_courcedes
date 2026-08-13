#!/usr/bin/env python3
"""Derive a smaller M1 audit while preserving compatible reviews."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from m1_audit_common import read_jsonl, sha256_file, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    return parser.parse_args()


def snapshot_reviews(source: Path, output: Path, selected_ids: set[str]) -> int:
    if not source.is_file():
        return 0
    source_connection = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    output_connection = sqlite3.connect(output)
    try:
        source_connection.backup(output_connection)
        placeholders = ",".join("?" for _ in selected_ids)
        if selected_ids:
            output_connection.execute(
                f"DELETE FROM reviews WHERE image_id NOT IN ({placeholders})",
                sorted(selected_ids),
            )
        else:
            output_connection.execute("DELETE FROM reviews")
        output_connection.commit()
        row = output_connection.execute("SELECT COUNT(*) FROM reviews").fetchone()
        return int(row[0])
    finally:
        output_connection.close()
        source_connection.close()


def count_values(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item[field]) for item in items).items()))


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")

    source_manifest_path = args.source_audit_dir / "audit_manifest.json"
    source_samples_path = args.source_audit_dir / "sample_manifest.jsonl"
    source_tasks_path = args.source_audit_dir / "audit_tasks.jsonl"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_samples = sorted(read_jsonl(source_samples_path), key=lambda item: int(item["sample_index"]))
    source_tasks = sorted(read_jsonl(source_tasks_path), key=lambda item: int(item["sample_index"]))
    if args.sample_size >= len(source_tasks):
        raise ValueError(
            f"sample-size must be smaller than the source task count ({len(source_tasks)})"
        )

    selected_tasks = source_tasks[: args.sample_size]
    selected_ids = {str(item["image_id"]) for item in selected_tasks}
    selected_samples = [item for item in source_samples if str(item["image_id"]) in selected_ids]
    selected_samples.sort(key=lambda item: int(item["sample_index"]))
    if len(selected_samples) != args.sample_size or len(selected_ids) != args.sample_size:
        raise ValueError("source sample/task files do not contain the same unique image IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_samples_path = args.output_dir / "sample_manifest.jsonl"
    output_tasks_path = args.output_dir / "audit_tasks.jsonl"
    write_jsonl(output_samples_path, selected_samples)
    write_jsonl(output_tasks_path, selected_tasks)
    migrated_reviews = snapshot_reviews(
        args.source_audit_dir / "reviews.sqlite3",
        args.output_dir / "reviews.sqlite3",
        selected_ids,
    )

    coverage_counts = Counter(
        str(tag) for item in selected_samples for tag in item.get("coverage_tags", [])
    )
    output_manifest = dict(source_manifest)
    output_manifest.update(
        {
            "sample_name": f"m1_audit_{args.sample_size}",
            "sample_size": args.sample_size,
            "sampling_scope": "train_only_risk_stratified_frozen_prefix_subset",
            "selection_policy": (
                f"retain sample_index 1..{args.sample_size} from {source_manifest.get('sample_name')}"
            ),
            "stratum_quotas": count_values(selected_samples, "primary_stratum"),
            "stratum_counts": count_values(selected_samples, "primary_stratum"),
            "coverage_tag_counts": dict(sorted(coverage_counts.items())),
            "derivation": {
                "source_audit_dir": str(args.source_audit_dir),
                "source_audit_manifest_sha256": sha256_file(source_manifest_path),
                "source_sample_manifest_sha256": sha256_file(source_samples_path),
                "source_audit_tasks_sha256": sha256_file(source_tasks_path),
                "source_sample_size": len(source_tasks),
                "migrated_review_rows": migrated_reviews,
                "task_hashes_preserved": True,
            },
            "outputs": {
                "sample_manifest.jsonl": sha256_file(output_samples_path),
                "audit_tasks.jsonl": sha256_file(output_tasks_path),
            },
        }
    )
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
