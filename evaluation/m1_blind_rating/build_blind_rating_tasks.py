#!/usr/bin/env python3
"""Freeze 50 blind-rating tasks from three canonical M1 annotation files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from blind_rating_common import (
    RATING_VERSION,
    SOURCE_IDS,
    canonical_json,
    keyed,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_jsonl,
)


DEFAULT_SEED = 20260813
DEFAULT_BLIND_SEED = 2026081301


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--internvl35", type=Path, required=True)
    parser.add_argument("--qwen3-vl-8b", dest="qwen3_vl_8b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--high-disagreement", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--blind-seed", type=int, default=DEFAULT_BLIND_SEED)
    return parser.parse_args()


def caption_text(annotation: dict[str, Any]) -> str:
    captions = annotation.get("captions") or {}
    return f"{captions.get('short_zh') or ''}{captions.get('dense_zh') or ''}"


def character_bigrams(value: str) -> set[str]:
    compact = "".join(str(value).split()).casefold()
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def pairwise_range(values: list[int]) -> float:
    return float(max(values) - min(values)) if values else 0.0


def disagreement_score(annotations: list[dict[str, Any]]) -> float:
    scenes = [annotation.get("scene") or {} for annotation in annotations]
    captures = [annotation.get("capture_visual") or {} for annotation in annotations]
    scene_conflicts = len({str(scene.get("primary_type")) for scene in scenes}) - 1
    environment_conflicts = len({str(scene.get("environment")) for scene in scenes}) - 1
    media_conflicts = len({str(scene.get("media_type")) for scene in scenes}) - 1
    capture_conflicts = sum(
        len({str(capture.get(field)) for capture in captures}) - 1
        for field in ("time_of_day", "weather", "viewpoint", "shot_scale", "blur_level")
    )
    entity_counts = [len(annotation.get("entities") or []) for annotation in annotations]
    ocr_counts = [len(annotation.get("ocr") or []) for annotation in annotations]
    relation_counts = [len(annotation.get("relations") or []) for annotation in annotations]
    caption_sets = [character_bigrams(caption_text(annotation)) for annotation in annotations]
    caption_distance = sum(
        jaccard_distance(caption_sets[left], caption_sets[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ) / 3
    score = (
        scene_conflicts * 4.0
        + environment_conflicts * 2.0
        + media_conflicts * 2.0
        + capture_conflicts * 0.5
        + min(pairwise_range(entity_counts), 12) * 0.7
        + min(pairwise_range(ocr_counts), 12) * 0.45
        + min(pairwise_range(relation_counts), 12) * 0.4
        + caption_distance * 5.0
    )
    return round(score, 6)


def deterministic_rank(image_id: str, seed: int, namespace: str) -> str:
    return sha256_text(f"{seed}\0{namespace}\0{image_id}")


def allocate_stratified(
    records: list[dict[str, Any]], count: int, *, seed: int, namespace: str
) -> list[dict[str, Any]]:
    """Round-robin scene strata so common classes do not consume the sample."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["scene_primary_type"])].append(record)
    for scene, bucket in buckets.items():
        bucket.sort(
            key=lambda item: deterministic_rank(
                str(item["image_id"]), seed, f"{namespace}:{scene}"
            )
        )
    scene_order = sorted(
        buckets,
        key=lambda scene: deterministic_rank(scene, seed, f"{namespace}:scene"),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < count and scene_order:
        next_order = []
        for scene in scene_order:
            if buckets[scene] and len(selected) < count:
                selected.append(buckets[scene].pop(0))
            if buckets[scene]:
                next_order.append(scene)
        scene_order = next_order
    if len(selected) != count:
        raise ValueError(f"only selected {len(selected)} of requested {count} records")
    return selected


def allocate_diverse_high_disagreement(
    records: list[dict[str, Any]], count: int, *, seed: int, namespace: str
) -> list[dict[str, Any]]:
    """Take the highest disagreement while covering scene types before repeats."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["scene_primary_type"])].append(record)
    for scene, bucket in buckets.items():
        bucket.sort(
            key=lambda item: (
                -float(item["disagreement_score"]),
                deterministic_rank(
                    str(item["image_id"]), seed, f"{namespace}:{scene}"
                ),
            )
        )
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [scene for scene, bucket in buckets.items() if bucket]
        if not available:
            break
        available.sort(
            key=lambda scene: (
                -float(buckets[scene][0]["disagreement_score"]),
                deterministic_rank(scene, seed, f"{namespace}:scene"),
            )
        )
        for scene in available:
            if len(selected) >= count:
                break
            selected.append(buckets[scene].pop(0))
    if len(selected) != count:
        raise ValueError(f"only selected {len(selected)} of requested {count} records")
    return selected


def choose_records(
    pool: list[dict[str, Any]],
    *,
    sample_size: int,
    high_disagreement: int,
    val_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not 0 <= high_disagreement <= sample_size:
        raise ValueError("high_disagreement must be between 0 and sample_size")
    if not 0 <= val_size <= sample_size:
        raise ValueError("val_size must be between 0 and sample_size")
    high_val = round(high_disagreement * val_size / sample_size)
    high_train = high_disagreement - high_val
    ordinary_val = val_size - high_val
    ordinary_train = sample_size - val_size - high_train

    by_split = {
        split: [record for record in pool if record["split"] == split]
        for split in ("train", "val")
    }
    high: list[dict[str, Any]] = []
    for split, count in (("train", high_train), ("val", high_val)):
        high.extend(
            allocate_diverse_high_disagreement(
                by_split[split], count, seed=seed, namespace=f"high:{split}"
            )
        )
    high_ids = {str(record["image_id"]) for record in high}

    ordinary: list[dict[str, Any]] = []
    for split, count in (("train", ordinary_train), ("val", ordinary_val)):
        candidates = [
            record for record in by_split[split] if str(record["image_id"]) not in high_ids
        ]
        ordinary.extend(
            allocate_stratified(candidates, count, seed=seed, namespace=f"ordinary:{split}")
        )

    for record in high:
        record["sample_group"] = "high_disagreement"
    for record in ordinary:
        record["sample_group"] = "ordinary"
    selected = ordinary + high
    random.Random(seed).shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = keyed(read_jsonl(args.manifest), args.manifest)
    source_paths = {
        "qwen35_9b": args.qwen35,
        "internvl35_14b": args.internvl35,
        "qwen3_vl_8b_instruct": args.qwen3_vl_8b,
    }
    sources = {
        source_id: keyed(read_jsonl(path), path)
        for source_id, path in source_paths.items()
    }
    common_ids = set(manifest).intersection(*(set(source) for source in sources.values()))
    pool = []
    for image_id in sorted(common_ids, key=lambda value: int(value)):
        manifest_record = manifest[image_id]
        hashes = {
            str(sources[source_id][image_id].get("processed_sha256"))
            for source_id in SOURCE_IDS
        }
        if hashes != {str(manifest_record.get("processed_sha256"))}:
            raise ValueError(f"workspace image hash mismatch for image {image_id}")
        annotations = [sources[source_id][image_id]["annotation"] for source_id in SOURCE_IDS]
        pool.append(
            {
                "image_id": image_id,
                "split": manifest_record["split"],
                "processed_path": manifest_record["processed_path"],
                "processed_sha256": manifest_record["processed_sha256"],
                "width": manifest_record["width"],
                "height": manifest_record["height"],
                "scene_primary_type": annotations[0]["scene"]["primary_type"],
                "disagreement_score": disagreement_score(annotations),
            }
        )

    selected = choose_records(
        pool,
        sample_size=args.sample_size,
        high_disagreement=args.high_disagreement,
        val_size=args.val_size,
        seed=args.seed,
    )
    tasks = []
    sample_manifest = []
    for sample_index, selected_record in enumerate(selected, 1):
        image_id = str(selected_record["image_id"])
        task = {
            "rating_version": RATING_VERSION,
            "sample_index": sample_index,
            **selected_record,
            "candidates": [
                {
                    "source_id": source_id,
                    "annotation": sources[source_id][image_id]["annotation"],
                }
                for source_id in SOURCE_IDS
            ],
        }
        task["task_sha256"] = sha256_text(canonical_json(task))
        tasks.append(task)
        sample_manifest.append(
            {
                key: task[key]
                for key in (
                    "sample_index",
                    "image_id",
                    "split",
                    "processed_path",
                    "processed_sha256",
                    "width",
                    "height",
                    "sample_group",
                    "scene_primary_type",
                    "disagreement_score",
                    "task_sha256",
                )
            }
        )

    tasks_path = output_dir / "rating_tasks.jsonl"
    sample_path = output_dir / "sample_manifest.jsonl"
    write_jsonl(tasks_path, tasks)
    write_jsonl(sample_path, sample_manifest)
    manifest_report = {
        "rating_version": RATING_VERSION,
        "sample_name": "m1_three_model_blind_rating_50",
        "sample_size": len(tasks),
        "seed": args.seed,
        "blind_seed": args.blind_seed,
        "sampling_scope": "three_model_common_valid_only",
        "selection_policy": "40 scene-stratified ordinary + 10 structural high-disagreement",
        "common_valid_pool": len(common_ids),
        "split_counts": dict(Counter(task["split"] for task in tasks)),
        "sample_group_counts": dict(Counter(task["sample_group"] for task in tasks)),
        "scene_counts": dict(Counter(task["scene_primary_type"] for task in tasks)),
        "inputs": {
            "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
            **{
                source_id: {"path": str(path), "sha256": sha256_file(path)}
                for source_id, path in source_paths.items()
            },
        },
        "outputs": {
            "rating_tasks.jsonl": sha256_file(tasks_path),
            "sample_manifest.jsonl": sha256_file(sample_path),
        },
        "source_image_identity_note": (
            "Qwen3-VL-8B-Instruct records use workspace manifest hashes for alignment, "
            "but the coworker source run retained different original image hashes."
        ),
    }
    (output_dir / "rating_manifest.json").write_text(
        json.dumps(manifest_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
