#!/usr/bin/env python3
"""Build transient per-image summaries for an imported raw-only M1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-candidates", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--source-schema", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            values.append(value)
    return values


def keyed(values: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        image_id = str(value.get("image_id"))
        if image_id in result:
            raise ValueError(f"Duplicate image_id {image_id!r} in {source}")
        result[image_id] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    items_dir = args.run_dir / "items"
    if items_dir.exists() and any(items_dir.iterdir()):
        raise FileExistsError(f"Items directory is not empty: {items_dir}")
    items_dir.mkdir(parents=True, exist_ok=True)

    manifest = keyed(read_jsonl(args.manifest), args.manifest)
    candidates = keyed(read_jsonl(args.source_candidates), args.source_candidates)
    audit = keyed(read_jsonl(args.raw_audit), args.raw_audit)
    if set(manifest) != set(candidates) or set(manifest) != set(audit):
        raise ValueError("Manifest, source candidates, and raw audit IDs differ")

    schema_sha256 = sha256_file(args.source_schema)
    hash_mismatches = 0
    for image_id in sorted(manifest, key=int):
        manifest_item = manifest[image_id]
        source = candidates[image_id]
        raw_status = audit[image_id]
        raw_path = args.run_dir / "raw" / f"{image_id}.txt"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if source.get("split") != manifest_item.get("split"):
            raise ValueError(f"Split mismatch for image {image_id}")

        source_hash = source.get("source_processed_sha256")
        workspace_hash = manifest_item.get("processed_sha256")
        hash_matches = source_hash == workspace_hash
        if not hash_matches:
            hash_mismatches += 1

        summary = {
            "image_id": image_id,
            "processed_sha256": workspace_hash,
            "source_processed_sha256": source_hash,
            "image_identity_status": (
                "exact_hash_match" if hash_matches else "source_hash_mismatch"
            ),
            "model_id": source.get("model_id"),
            "prompt_version": source.get("prompt_version"),
            "schema_file_sha256": schema_sha256,
            "json_parse_ok": raw_status.get("strict_json") is True,
            "annotation_valid": raw_status.get("annotation_valid_v1_2") is True,
            "source_candidate_status": source.get("source_status"),
            "source_candidate_error": source.get("error"),
            "compatibility_summary": True,
        }
        item_dir = items_dir / image_id
        item_dir.mkdir()
        (item_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "summaries_written": len(manifest),
                "items_dir": str(items_dir),
                "source_hash_mismatches": hash_mismatches,
                "schema_sha256": schema_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
