#!/usr/bin/env python3
"""Export submitted blind ratings and reveal model-level summary metrics."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from blind_rating_common import (
    SOURCE_NAMES,
    compute_metrics,
    metrics_markdown,
    read_jsonl,
    reviewer_name,
    sha256_file,
    slot_source_map,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rating-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    return parser.parse_args()


def load_submitted(db_path: Path, reviewer: str) -> list[dict[str, Any]]:
    if not db_path.is_file():
        return []
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM reviews WHERE reviewer = ? AND phase = 'submitted' "
            "ORDER BY CAST(image_id AS INTEGER)",
            (reviewer,),
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "reviewer": row["reviewer"],
            "image_id": row["image_id"],
            "task_sha256": row["task_sha256"],
            "phase": row["phase"],
            "rating": json.loads(row["rating_json"]),
            "created_at_utc": row["created_at_utc"],
            "updated_at_utc": row["updated_at_utc"],
            "submitted_at_utc": row["submitted_at_utc"],
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    reviewer = reviewer_name(args.reviewer)
    rating_dir = args.rating_dir.resolve()
    tasks_path = rating_dir / "rating_tasks.jsonl"
    manifest_path = rating_dir / "rating_manifest.json"
    db_path = rating_dir / "reviews.sqlite3"
    tasks = read_jsonl(tasks_path)
    tasks_by_id = {str(task["image_id"]): task for task in tasks}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blind_seed = int(manifest["blind_seed"])
    submitted = load_submitted(db_path, reviewer)

    export_rows = []
    for record in submitted:
        image_id = str(record["image_id"])
        task = tasks_by_id.get(image_id)
        if task is None:
            raise ValueError(f"submitted review references unknown image {image_id}")
        if record["task_sha256"] != task["task_sha256"]:
            raise ValueError(f"task changed after review for image {image_id}")
        mapping = slot_source_map(image_id, reviewer, blind_seed)
        revealed_ratings = {
            source_id: {
                "source_name": SOURCE_NAMES[source_id],
                **record["rating"]["ratings"][slot],
            }
            for slot, source_id in mapping.items()
        }
        best_choice = record["rating"]["best_choice"]
        revealed_best = mapping.get(best_choice, best_choice)
        export_rows.append(
            {
                **record,
                "sample_index": task["sample_index"],
                "sample_group": task["sample_group"],
                "split": task["split"],
                "processed_path": task["processed_path"],
                "processed_sha256": task["processed_sha256"],
                "blind_slot_to_source": mapping,
                "revealed_ratings": revealed_ratings,
                "revealed_best_choice": revealed_best,
            }
        )

    metrics = compute_metrics(submitted, tasks_by_id, blind_seed=blind_seed)
    metrics["reviewer"] = reviewer
    metrics["task_count"] = len(tasks)
    metrics["task_sha256"] = sha256_file(tasks_path)
    exports_dir = rating_dir / "exports"
    reports_dir = rating_dir / "reports"
    exports_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    write_jsonl(exports_dir / "reviews.jsonl", export_rows)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "metrics.md").write_text(
        metrics_markdown(metrics, reviewer), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
