"""Shared helpers for the lightweight three-model M1 blind rating."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RATING_VERSION = "m1-blind-rating-v1.0.0"
SOURCE_IDS = ("qwen35_9b", "internvl35_14b", "qwen3_vl_8b_instruct")
SOURCE_NAMES = {
    "qwen35_9b": "Qwen3.5-9B",
    "internvl35_14b": "InternVL3.5-14B",
    "qwen3_vl_8b_instruct": "Qwen3-VL-8B-Instruct",
}
SCORE_FIELDS = ("accuracy", "completeness", "usability")
SAMPLE_GROUPS = ("ordinary", "high_disagreement")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            values.append(value)
    return values


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def keyed(records: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        if image_id in result:
            raise ValueError(f"duplicate image_id {image_id!r} in {source}")
        result[image_id] = record
    return result


def blind_source_order(image_id: str, reviewer: str, seed: int) -> list[str]:
    """Return a deterministic per-image order without exposing model labels."""

    ranked = []
    for source_id in SOURCE_IDS:
        digest = sha256_text(f"{seed}\0{reviewer}\0{image_id}\0{source_id}")
        ranked.append((digest, source_id))
    return [source_id for _, source_id in sorted(ranked)]


def reviewer_name(value: Any) -> str:
    name = re.sub(r"\s+", " ", str(value or "")).strip()
    if not name or len(name) > 40 or any(ord(character) < 32 for character in name):
        raise ValueError("评审者名称必须为 1 到 40 个可见字符")
    return name


def slot_source_map(image_id: str, reviewer: str, seed: int) -> dict[str, str]:
    return {
        f"candidate_{index + 1}": source_id
        for index, source_id in enumerate(blind_source_order(image_id, reviewer, seed))
    }


def _clean_score(value: Any, field: str, *, required: bool) -> int | None:
    if value is None and not required:
        return None
    if type(value) is not int or not 1 <= value <= 5:
        raise ValueError(f"{field} 必须是 1 到 5 的整数")
    return value


def _clean_bool(value: Any, field: str, *, required: bool) -> bool | None:
    if value is None and not required:
        return None
    if type(value) is not bool:
        raise ValueError(f"{field} 必须选择是或否")
    return value


def validate_rating(payload: Any, *, require_complete: bool) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("评分内容必须是 JSON 对象")
    ratings = payload.get("ratings")
    if not isinstance(ratings, dict):
        raise ValueError("ratings 必须是对象")
    expected_slots = {f"candidate_{index + 1}" for index in range(len(SOURCE_IDS))}
    if not set(ratings).issubset(expected_slots):
        raise ValueError("ratings 包含未知候选")
    if require_complete and set(ratings) != expected_slots:
        raise ValueError("提交前必须完成 A、B、C 三份评分")

    clean_ratings: dict[str, Any] = {}
    for slot in sorted(ratings):
        rating = ratings[slot]
        if not isinstance(rating, dict):
            raise ValueError(f"{slot} 评分必须是对象")
        clean_ratings[slot] = {
            field: _clean_score(
                rating.get(field), f"{slot}.{field}", required=require_complete
            )
            for field in SCORE_FIELDS
        }
        clean_ratings[slot]["severe_error"] = _clean_bool(
            rating.get("severe_error"),
            f"{slot}.severe_error",
            required=require_complete,
        )

    best_choice = payload.get("best_choice")
    allowed_best = expected_slots | {"tie", "all_unacceptable"}
    if best_choice in {None, ""} and not require_complete:
        best_choice = None
    elif best_choice not in allowed_best:
        raise ValueError("最佳结果必须选择 A、B、C、并列或全部不合格")

    notes = re.sub(r"\s+", " ", str(payload.get("notes") or "")).strip()
    if len(notes) > 1000:
        raise ValueError("备注不能超过 1000 个字符")
    return {"ratings": clean_ratings, "best_choice": best_choice, "notes": notes}


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _new_accumulator() -> dict[str, Any]:
    return {
        "reviewed_images": 0,
        "scores": {field: [] for field in SCORE_FIELDS},
        "severe_error_count": 0,
        "unique_best_count": 0,
    }


def _finalize_accumulator(source_id: str, acc: dict[str, Any]) -> dict[str, Any]:
    reviewed = acc["reviewed_images"]
    return {
        "source_name": SOURCE_NAMES[source_id],
        "reviewed_images": reviewed,
        "mean_scores": {
            field: _mean(acc["scores"][field]) for field in SCORE_FIELDS
        },
        "score_distributions": {
            field: {
                str(score): Counter(acc["scores"][field]).get(score, 0)
                for score in range(1, 6)
            }
            for field in SCORE_FIELDS
        },
        "severe_error_count": acc["severe_error_count"],
        "severe_error_rate": (
            round(acc["severe_error_count"] / reviewed, 4) if reviewed else None
        ),
        "unique_best_count": acc["unique_best_count"],
        "unique_best_rate": (
            round(acc["unique_best_count"] / reviewed, 4) if reviewed else None
        ),
    }


def compute_metrics(
    submitted: list[dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
    *,
    blind_seed: int,
) -> dict[str, Any]:
    overall = {source_id: _new_accumulator() for source_id in SOURCE_IDS}
    grouped = {
        group: {source_id: _new_accumulator() for source_id in SOURCE_IDS}
        for group in SAMPLE_GROUPS
    }
    best_choice_counts = Counter()

    for record in submitted:
        image_id = str(record["image_id"])
        task = tasks_by_id[image_id]
        group = str(task["sample_group"])
        mapping = slot_source_map(
            image_id, str(record["reviewer"]), blind_seed
        )
        rating_payload = record["rating"]
        for slot, rating in rating_payload["ratings"].items():
            source_id = mapping[slot]
            for target in (overall[source_id], grouped[group][source_id]):
                target["reviewed_images"] += 1
                for field in SCORE_FIELDS:
                    target["scores"][field].append(int(rating[field]))
                target["severe_error_count"] += bool(rating["severe_error"])

        best_choice = rating_payload["best_choice"]
        if best_choice in mapping:
            source_id = mapping[best_choice]
            best_choice_counts[source_id] += 1
            overall[source_id]["unique_best_count"] += 1
            grouped[group][source_id]["unique_best_count"] += 1
        else:
            best_choice_counts[str(best_choice)] += 1

    return {
        "rating_version": RATING_VERSION,
        "submitted_images": len(submitted),
        "models": {
            source_id: _finalize_accumulator(source_id, overall[source_id])
            for source_id in SOURCE_IDS
        },
        "sample_groups": {
            group: {
                source_id: _finalize_accumulator(source_id, grouped[group][source_id])
                for source_id in SOURCE_IDS
            }
            for group in SAMPLE_GROUPS
        },
        "best_choice_counts": {
            **{source_id: best_choice_counts[source_id] for source_id in SOURCE_IDS},
            "tie": best_choice_counts["tie"],
            "all_unacceptable": best_choice_counts["all_unacceptable"],
        },
    }


def metrics_markdown(metrics: dict[str, Any], reviewer: str) -> str:
    lines = [
        "# M1 三模型简化盲评结果",
        "",
        f"- 评审者：`{reviewer}`",
        f"- 已提交图片：{metrics['submitted_images']}",
        "",
        "## 总体结果",
        "",
        "| 模型 | 准确性均分 | 完整性均分 | 可用性均分 | 严重错误 | 唯一最佳 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source_id in SOURCE_IDS:
        row = metrics["models"][source_id]
        score = row["mean_scores"]
        severe = f"{row['severe_error_count']}/{row['reviewed_images']}"
        best = f"{row['unique_best_count']}/{row['reviewed_images']}"
        lines.append(
            f"| {row['source_name']} | {score['accuracy']} | "
            f"{score['completeness']} | {score['usability']} | {severe} | {best} |"
        )

    best_counts = metrics["best_choice_counts"]
    lines.extend(
        [
            "",
            "## 最佳结果选择",
            "",
            f"- 并列：{best_counts['tie']} 张",
            f"- 全部不合格：{best_counts['all_unacceptable']} 张",
            "",
            "## 解释边界",
            "",
            "这是单人 50 张简化盲评。分数反映带锚点的整体人工判断，不等于逐字段准确率；高差异样本是有意选择的困难样本，不用于估计完整数据集自然错误率。",
            "",
        ]
    )
    return "\n".join(lines)
