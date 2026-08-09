from anima_search.schemas import ImageAnnotation


def annotation_fields(annotation: ImageAnnotation) -> dict[str, list[str]]:
    return {
        "summary": [annotation.summary],
        "objects": annotation.objects,
        "actions": annotation.actions,
        "scene": [annotation.scene],
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
    fields = [("摘要", values["summary"]), ("主体", values["objects"]),
              ("动作", values["actions"]), ("场景", values["scene"]),
              ("属性", values["attributes"]), ("关系", values["spatial_relations"]),
              ("风格", values["style"]), ("情绪", values["mood"]),
              ("颜色", values["colors"]), ("文字", values["ocr_text"])]
    return _labeled_document(fields)


def annotation_to_sparse_document(annotation: ImageAnnotation) -> str:
    values = annotation_fields(annotation)
    fields = [("主体", values["objects"]), ("动作", values["actions"]),
              ("场景", values["scene"]), ("属性", values["attributes"]),
              ("关系", values["spatial_relations"]), ("风格", values["style"]),
              ("情绪", values["mood"]), ("颜色", values["colors"]),
              ("文字", values["ocr_text"]), ("摘要", values["summary"])]
    return _labeled_document(fields)


def annotation_to_document(annotation: ImageAnnotation) -> str:
    """Backward-compatible alias for the dense retrieval document."""
    return annotation_to_dense_document(annotation)
