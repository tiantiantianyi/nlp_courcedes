from __future__ import annotations

from pathlib import Path

from anima_search.schemas import ImageAnnotation, ManifestItem


IMAGE_ONLY_ANNOTATION_VERSION = "image-only-manifest-v1"


def load_manifest_items(path: Path, split: str, limit: int | None = None) -> list[ManifestItem]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    items = [
        ManifestItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usable = [
        item for item in items
        if item.valid and item.duplicate_of is None and item.split.lower() == split.lower()
    ]
    usable.sort(key=lambda item: item.image_id)
    if limit is not None:
        usable = usable[:limit]
    if not usable:
        raise ValueError(f"manifest contains no usable {split} images")
    ids = [item.image_id for item in usable]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate image IDs")
    return usable


def placeholder_annotations(items: list[ManifestItem]) -> list[ImageAnnotation]:
    return [
        ImageAnnotation(
            image_id=item.image_id,
            split=item.split,
            relative_path=item.relative_path.replace("\\", "/"),
            sha256=item.sha256,
            duplicate_of=item.duplicate_of,
            summary="标注尚未提供；当前仅启用图像向量检索。",
            scene="未知场景",
            search_queries=["image-only", "clip", "unannotated"],
            generation_prompt="",
            model_version="none",
            prompt_version=IMAGE_ONLY_ANNOTATION_VERSION,
        )
        for item in items
    ]
