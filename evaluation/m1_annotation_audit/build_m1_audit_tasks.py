#!/usr/bin/env python3
"""Build a deterministic, risk-stratified 50-image M1 audit set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from m1_audit_common import AUDIT_VERSION, canonical_json, read_jsonl, sha256_file, sha256_text, write_jsonl


DEFAULT_SEED = 20260812
STRATA: tuple[tuple[str, int], ...] = (
    ("high_agreement", 6),
    ("candidate_unavailable_or_lossy", 6),
    ("ocr_challenge", 10),
    ("entity_count_challenge", 10),
    ("relation_challenge", 8),
    ("scene_capture_challenge", 6),
    ("low_quality_or_extreme", 2),
    ("general", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--internvl", type=Path, required=True)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--blind-seed", type=int, default=2026081201)
    return parser.parse_args()


def by_image(path: Path) -> dict[str, dict[str, Any]]:
    values = read_jsonl(path)
    result = {str(value["image_id"]): value for value in values}
    if len(result) != len(values):
        raise ValueError(f"duplicate image_id in {path}")
    return result


def candidate_from_normalization(source_id: str, value: dict[str, Any] | None) -> dict[str, Any]:
    available = bool(
        value
        and value.get("normalized_annotation_valid")
        and isinstance(value.get("annotation"), dict)
    )
    return {
        "source_id": source_id,
        "available": available,
        "normalization_status": value.get("status") if value else "missing",
        "lossy_repairs": sum(1 for repair in (value or {}).get("repairs", []) if repair.get("lossy") is True),
        "annotation": value.get("annotation") if available else None,
    }


def disagreement_tags(items: list[dict[str, Any]], fusion: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    categories = {str(item["category"]) for item in items}
    tags: set[str] = set()
    if any(not ref.get("available") or ref.get("lossy_repairs", 0) for ref in fusion["candidate_refs"]):
        tags.add("candidate_unavailable_or_lossy")
    if categories.intersection({"candidate_unavailable", "lossy_source_repair"}):
        tags.add("candidate_unavailable_or_lossy")
    if categories.intersection({"ocr_unmatched", "ocr_field_conflict"}):
        tags.add("ocr_challenge")
    if categories.intersection({"entity_unmatched", "entity_field_conflict"}):
        tags.add("entity_count_challenge")
    if any(item["field_path"].endswith("/count") for item in items):
        tags.add("entity_count_challenge")
    if "relation_unmatched" in categories:
        tags.add("relation_challenge")
    if any(
        item["category"] in {"scalar_conflict", "set_difference"}
        and item["field_path"].startswith(("/scene", "/capture_visual"))
        for item in items
    ):
        tags.add("scene_capture_challenge")
    if set(manifest.get("coverage_tags") or []).intersection({"low_resolution", "extreme_aspect_ratio"}):
        tags.add("low_quality_or_extreme")
    required = sum(bool(item.get("requires_resolution")) for item in items)
    if fusion.get("review_status") == "auto_accepted" or required <= 2:
        tags.add("high_agreement")
    tags.add("general")
    return sorted(tags)


def choose_sample(
    pool: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    if sample_size != sum(quota for _, quota in STRATA):
        raise ValueError(f"this audit design requires sample_size={sum(quota for _, quota in STRATA)}")
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    chosen_duplicate_groups: set[str] = set()

    def available(item: dict[str, Any]) -> bool:
        if item["image_id"] in chosen_ids:
            return False
        group = item.get("duplicate_group")
        return group is None or str(group) not in chosen_duplicate_groups

    for stratum, quota in STRATA:
        workload_limit = 60 if stratum == "low_quality_or_extreme" else 45
        candidates = [
            item
            for item in pool
            if stratum in item["coverage_tags"]
            and item["review_item_count"] <= workload_limit
            and available(item)
        ]
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise ValueError(f"stratum {stratum} has only {len(candidates)} available images for quota {quota}")
        for item in candidates[:quota]:
            selected = dict(item)
            selected["primary_stratum"] = stratum
            chosen.append(selected)
            chosen_ids.add(item["image_id"])
            if item.get("duplicate_group") is not None:
                chosen_duplicate_groups.add(str(item["duplicate_group"]))

    rng.shuffle(chosen)
    for index, item in enumerate(chosen, 1):
        item["sample_index"] = index
    return chosen


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = read_jsonl(args.manifest)
    manifest_by_id = {str(item["image_id"]): item for item in manifest_records}
    qwen = by_image(args.qwen)
    internvl = by_image(args.internvl)
    fusion = by_image(args.fusion_dir / "annotations.jsonl")
    disagreements_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in read_jsonl(args.fusion_dir / "disagreements.jsonl"):
        disagreements_by_id[str(item["image_id"])].append(item)

    pool: list[dict[str, Any]] = []
    for manifest in manifest_records:
        if manifest["split"] != "train":
            continue
        image_id = str(manifest["image_id"])
        fused = fusion[image_id]
        if fused["processed_sha256"] != manifest["processed_sha256"]:
            raise ValueError(f"fusion/manifest hash mismatch for {image_id}")
        tags = disagreement_tags(disagreements_by_id[image_id], fused, manifest)
        review_item_counts = {"entities": 0, "ocr": 0, "relations": 0}
        for candidate in (
            candidate_from_normalization("qwen35_9b", qwen.get(image_id)),
            candidate_from_normalization("internvl35_14b", internvl.get(image_id)),
            {"annotation": fused.get("annotation")},
        ):
            annotation = candidate.get("annotation") or {}
            review_item_counts["entities"] += len(annotation.get("entities", []))
            review_item_counts["ocr"] += len(annotation.get("ocr", []))
            review_item_counts["relations"] += len(annotation.get("relations", []))
        pool.append(
            {
                "image_id": image_id,
                "split": "train",
                "processed_path": manifest["processed_path"],
                "processed_sha256": manifest["processed_sha256"],
                "width": manifest["width"],
                "height": manifest["height"],
                "duplicate_group": manifest.get("duplicate_group"),
                "coverage_tags": tags,
                "required_disagreements": sum(
                    bool(item.get("requires_resolution")) for item in disagreements_by_id[image_id]
                ),
                "total_disagreements": len(disagreements_by_id[image_id]),
                "review_item_counts": review_item_counts,
                "review_item_count": sum(review_item_counts.values()),
            }
        )

    selected = choose_sample(pool, args.sample_size, args.seed)
    sample_manifest = args.output_dir / "sample_manifest.jsonl"
    write_jsonl(sample_manifest, selected)

    tasks = []
    for sample in selected:
        image_id = sample["image_id"]
        candidates = [
            candidate_from_normalization("qwen35_9b", qwen.get(image_id)),
            candidate_from_normalization("internvl35_14b", internvl.get(image_id)),
            {
                "source_id": "fusion",
                "available": True,
                "normalization_status": fusion[image_id]["review_status"],
                "lossy_repairs": 0,
                "annotation": fusion[image_id]["annotation"],
            },
        ]
        task = {
            "audit_version": AUDIT_VERSION,
            "image_id": image_id,
            "sample_index": sample["sample_index"],
            "processed_path": sample["processed_path"],
            "processed_sha256": sample["processed_sha256"],
            "width": sample["width"],
            "height": sample["height"],
            "primary_stratum": sample["primary_stratum"],
            "coverage_tags": sample["coverage_tags"],
            "candidates": candidates,
        }
        task["task_sha256"] = sha256_text(canonical_json(task))
        tasks.append(task)
    tasks_path = args.output_dir / "audit_tasks.jsonl"
    write_jsonl(tasks_path, tasks)

    stratum_counts = Counter(item["primary_stratum"] for item in selected)
    tag_counts = Counter(tag for item in selected for tag in item["coverage_tags"])
    manifest = {
        "audit_version": AUDIT_VERSION,
        "sample_name": "m1_audit_50",
        "sample_size": len(selected),
        "single_reviewer": True,
        "seed": args.seed,
        "blind_seed": args.blind_seed,
        "sampling_scope": "train_only_risk_stratified",
        "prompt_dev_exclusion_status": "not_checked_no_frozen_prompt_dev_manifest",
        "stratum_quotas": dict(STRATA),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "coverage_tag_counts": dict(sorted(tag_counts.items())),
        "inputs": {
            "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
            "qwen": {"path": str(args.qwen), "sha256": sha256_file(args.qwen)},
            "internvl": {"path": str(args.internvl), "sha256": sha256_file(args.internvl)},
            "fusion_annotations": {
                "path": str(args.fusion_dir / "annotations.jsonl"),
                "sha256": sha256_file(args.fusion_dir / "annotations.jsonl"),
            },
            "fusion_disagreements": {
                "path": str(args.fusion_dir / "disagreements.jsonl"),
                "sha256": sha256_file(args.fusion_dir / "disagreements.jsonl"),
            },
        },
        "outputs": {
            "sample_manifest.jsonl": sha256_file(sample_manifest),
            "audit_tasks.jsonl": sha256_file(tasks_path),
        },
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
