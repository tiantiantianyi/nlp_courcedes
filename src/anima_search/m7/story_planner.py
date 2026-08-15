from __future__ import annotations

from collections import defaultdict

from anima_search.m7.schemas import StoryGap
from anima_search.schemas import ImageAnnotation, SearchResult


_TIME_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("黎明", ("黎明", "凌晨", "日出", "dawn", "sunrise")),
    ("早晨", ("早晨", "清晨", "上午", "morning")),
    ("中午", ("中午", "正午", "noon", "midday")),
    ("下午", ("下午", "afternoon")),
    ("黄昏", ("黄昏", "傍晚", "日落", "dusk", "sunset", "evening")),
    ("夜晚", ("夜晚", "夜间", "夜景", "深夜", "night")),
)


def time_bucket(annotation: ImageAnnotation | None) -> tuple[int, str | None]:
    if annotation is None:
        return len(_TIME_BUCKETS), None
    canonical_time = next(
        (
            value.split(":", 1)[1].strip().lower()
            for value in annotation.attributes
            if value.startswith("time_of_day:") and ":" in value
        ),
        None,
    )
    canonical_buckets = {
        "night": (5, "夜晚"),
        "dawn_dusk": (4, "晨昏"),
    }
    if canonical_time in canonical_buckets:
        return canonical_buckets[canonical_time]
    text = " ".join(
        [annotation.scene, annotation.summary, *annotation.attributes]
    ).lower()
    for rank, (label, terms) in enumerate(_TIME_BUCKETS):
        if any(term.lower() in text for term in terms):
            return rank, label
    return len(_TIME_BUCKETS), None


def scene_features(annotation: ImageAnnotation | None) -> set[str]:
    if annotation is None:
        return set()
    values = [
        annotation.scene,
        *annotation.objects,
        *annotation.colors,
        *annotation.mood,
        *annotation.style,
    ]
    return {value.strip().lower() for value in values if value.strip()}


def scene_similarity(
    left: ImageAnnotation | None,
    right: ImageAnnotation | None,
) -> float:
    left_features = scene_features(left)
    right_features = scene_features(right)
    union = left_features | right_features
    if not union:
        return 0.0
    return len(left_features & right_features) / len(union)


def order_story_candidates(
    candidates: list[SearchResult],
    annotations: dict[str, ImageAnnotation],
) -> list[SearchResult]:
    """Order by time bucket, then greedily keep visually related scenes adjacent."""
    if len(candidates) < 2:
        return list(candidates)
    if not any(item.image_id in annotations for item in candidates):
        return list(candidates)

    indexed = list(enumerate(candidates))
    groups: dict[int, list[tuple[int, SearchResult]]] = defaultdict(list)
    for original_index, item in indexed:
        rank, _ = time_bucket(annotations.get(item.image_id))
        groups[rank].append((original_index, item))

    ordered: list[SearchResult] = []
    for rank in sorted(groups):
        remaining = list(groups[rank])
        current = remaining.pop(0)
        ordered.append(current[1])
        while remaining:
            current_annotation = annotations.get(current[1].image_id)
            next_index = max(
                range(len(remaining)),
                key=lambda index: (
                    scene_similarity(
                        current_annotation,
                        annotations.get(remaining[index][1].image_id),
                    ),
                    -remaining[index][0],
                ),
            )
            current = remaining.pop(next_index)
            ordered.append(current[1])
    return ordered


def build_story_gaps(
    ordered: list[SearchResult],
    annotations: dict[str, ImageAnnotation],
    *,
    max_gaps: int = 2,
    scene_similarity_threshold: float = 0.15,
) -> list[StoryGap]:
    if max_gaps <= 0:
        return []
    gaps: list[StoryGap] = []
    for left, right in zip(ordered, ordered[1:]):
        left_annotation = annotations.get(left.image_id)
        right_annotation = annotations.get(right.image_id)
        if left_annotation is None or right_annotation is None:
            continue

        left_time, left_label = time_bucket(left_annotation)
        right_time, right_label = time_bucket(right_annotation)
        reasons: list[str] = []
        if (
            left_label is not None
            and right_label is not None
            and right_time - left_time >= 2
        ):
            reasons.append(f"时间从{left_label}跳到{right_label}")

        similarity = scene_similarity(left_annotation, right_annotation)
        if (
            left_annotation.scene.strip()
            and right_annotation.scene.strip()
            and left_annotation.scene != right_annotation.scene
            and similarity < scene_similarity_threshold
        ):
            reasons.append(
                f"场景从{left_annotation.scene}切换到{right_annotation.scene}"
            )
        if not reasons:
            continue

        gap_number = len(gaps) + 1
        prompt = (
            f"为视觉故事补充一张自然过渡图片：前一幕是{left_annotation.summary}；"
            f"后一幕是{right_annotation.summary}。表现{'；'.join(reasons)}，"
            "保持摄影感、自然光影，不添加文字、水印、商标或无法确认的具体地点。"
        )
        gaps.append(
            StoryGap(
                gap_id=f"gap-{gap_number:02d}",
                after_image_id=left.image_id,
                before_image_id=right.image_id,
                reason="；".join(reasons),
                generation_prompt=prompt,
            )
        )
        if len(gaps) >= max_gaps:
            break
    return gaps
