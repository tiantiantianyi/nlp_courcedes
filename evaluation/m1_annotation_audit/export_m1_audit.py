#!/usr/bin/env python3
"""Export submitted M1 reviews and compute before/after metrics."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from m1_audit_common import compute_metrics, metrics_markdown, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    return parser.parse_args()


def load_submitted(db_path: Path, reviewer: str) -> list[dict[str, Any]]:
    if not db_path.is_file():
        return []
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM reviews WHERE reviewer = ? AND phase = 'submitted' ORDER BY image_id",
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
            "gold": json.loads(row["gold_json"]),
            "candidate_reviews": json.loads(row["candidate_reviews_json"]),
            "created_at_utc": row["created_at_utc"],
            "updated_at_utc": row["updated_at_utc"],
            "submitted_at_utc": row["submitted_at_utc"],
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.audit_dir / "audit_tasks.jsonl")
    tasks_by_id = {str(item["image_id"]): item for item in tasks}
    audit_manifest = json.loads((args.audit_dir / "audit_manifest.json").read_text(encoding="utf-8"))
    reviews = load_submitted(args.audit_dir / "reviews.sqlite3", args.reviewer)
    for review in reviews:
        task = tasks_by_id.get(review["image_id"])
        if not task or task["task_sha256"] != review["task_sha256"]:
            raise ValueError(f"review task hash mismatch for image {review['image_id']}")
    export_dir = args.audit_dir / "exports"
    report_dir = args.audit_dir / "reports"
    export_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(export_dir / "reviews.jsonl", reviews)
    report = compute_metrics(reviews, tasks_by_id, blind_seed=int(audit_manifest["blind_seed"]))
    report["reviewer"] = args.reviewer
    (report_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_dir / "metrics.md").write_text(metrics_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
