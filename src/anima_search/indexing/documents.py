from __future__ import annotations

from anima_search.schemas import ImageAnnotation


def _attribute_values(annotation: ImageAnnotation, prefix: str) -> list[str]:
    marker = f"{prefix}:"
    return [value[len(marker):] for value in annotation.attributes if value.startswith(marker)]


def annotation_fields(annotation: ImageAnnotation) -> dict[str, list[str]]:
    return {
        "summary": [annotation.summary],
        "objects": annotation.objects,
        "object_counts": [f"{name}:{count}" for name, count in annotation.object_counts.items()],
        "actions": annotation.actions,
        "scene": [annotation.scene],
        "time_of_day": _attribute_values(annotation, "time_of_day"),
        "weather": _attribute_values(annotation, "weather"),
        "attributes": annotation.attributes,
        "spatial_relations": annotation.spatial_relations,
        "style": annotation.style,
        "mood": annotation.mood,
        "colors": annotation.colors,
        "ocr_text": annotation.ocr_text,
    }


def _labeled_document(fields: list[tuple[str, list[str]]]) -> str:
    return " ".join(f"{name}:{' '.join(values)}" for name, values in fields if any(values))


def annotation_to_dense_document(annotation: ImageAnnotation) -> str:
    values = annotation_fields(annotation)
    fields = [
        ("摘要", values["summary"]), ("主体", values["objects"]),
        ("数量", values["object_counts"]), ("动作", values["actions"]),
        ("场景", values["scene"]), ("时间", values["time_of_day"]),
        ("天气", values["weather"]), ("属性", values["attributes"]),
        ("关系", values["spatial_relations"]), ("风格", values["style"]),
        ("情绪", values["mood"]), ("颜色", values["colors"]),
        ("文字", values["ocr_text"]),
    ]
    return _labeled_document(fields)


def annotation_to_sparse_document(annotation: ImageAnnotation) -> str:
    values = annotation_fields(annotation)
    fields = [
        ("主体", values["objects"]), ("数量", values["object_counts"]),
        ("动作", values["actions"]), ("场景", values["scene"]),
        ("时间", values["time_of_day"]), ("天气", values["weather"]),
        ("属性", values["attributes"]), ("关系", values["spatial_relations"]),
        ("风格", values["style"]), ("情绪", values["mood"]),
        ("颜色", values["colors"]), ("文字", values["ocr_text"]),
        ("摘要", values["summary"]),
    ]
    return _labeled_document(fields)


def annotation_to_document(annotation: ImageAnnotation) -> str:
    """Backward-compatible alias for the dense retrieval document."""
    return annotation_to_dense_document(annotation)
