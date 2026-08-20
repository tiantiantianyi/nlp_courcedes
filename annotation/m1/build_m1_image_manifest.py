#!/usr/bin/env python3
"""Decode, hash, and index M1 image splits into a deterministic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--expected-train", type=int, default=2000)
    parser.add_argument("--expected-val", type=int, default=369)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Image is outside project root: {path}") from exc


def image_paths(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    return sorted(paths, key=lambda path: int(path.stem))


def inspect_image(path: Path, split: str, project_root: Path) -> dict[str, Any]:
    if not path.stem.isdigit():
        raise ValueError(f"Image filename stem must be numeric: {path.name}")
    image_bytes = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        exif_orientation = image.getexif().get(274)
        image_format = image.format
    if image_format != "JPEG":
        raise ValueError(f"Expected JPEG content: {path}")

    coverage_tags: list[str] = []
    if min(width, height) <= 128:
        coverage_tags.append("low_resolution")
    if max(width, height) / min(width, height) >= 3:
        coverage_tags.append("extreme_aspect_ratio")
    if width * height >= 20_000_000:
        coverage_tags.append("high_resolution")
    if exif_orientation not in (None, 1):
        coverage_tags.append("exif_orientation")

    return {
        "image_id": path.stem,
        "split": split,
        "processed_path": relative_path(path, project_root),
        "processed_sha256": sha256_bytes(image_bytes),
        "width": width,
        "height": height,
        "mode": mode,
        "image_bytes": len(image_bytes),
        "exif_orientation": exif_orientation,
        "coverage_tags": coverage_tags,
        "duplicate_group": None,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    split_paths = {
        "train": image_paths(args.train_dir),
        "val": image_paths(args.val_dir),
    }
    expected = {"train": args.expected_train, "val": args.expected_val}
    for split, paths in split_paths.items():
        if len(paths) != expected[split]:
            raise ValueError(
                f"{split} image count is {len(paths)}, expected {expected[split]}"
            )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for split in ("train", "val"):
        for path in split_paths[split]:
            record = inspect_image(path, split, project_root)
            image_id = record["image_id"]
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image_id across splits: {image_id}")
            seen_ids.add(image_id)
            records.append(record)

    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        hashes[record["processed_sha256"]].append(record["image_id"])
    duplicate_groups = [ids for ids in hashes.values() if len(ids) > 1]
    duplicate_lookup = {
        image_id: f"dup-{index:04d}"
        for index, ids in enumerate(duplicate_groups, 1)
        for image_id in ids
    }
    for record in records:
        record["duplicate_group"] = duplicate_lookup.get(record["image_id"])

    write_jsonl(args.output, records)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    tag_counts: dict[str, int] = defaultdict(int)
    for record in records:
        for tag in record["coverage_tags"]:
            tag_counts[tag] += 1
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.output),
        "manifest_sha256": sha256_bytes(args.output.read_bytes()),
        "total_images": len(records),
        "split_counts": {split: len(paths) for split, paths in split_paths.items()},
        "duplicate_groups": duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
        "coverage_tag_counts": dict(sorted(tag_counts.items())),
        "exif_orientation_count": tag_counts.get("exif_orientation", 0),
        "decode_failures": 0,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
