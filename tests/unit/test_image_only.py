from pathlib import Path

from anima_search.indexing.image_only import (
    IMAGE_ONLY_ANNOTATION_VERSION,
    load_manifest_items,
    placeholder_annotations,
)
from anima_search.schemas import ManifestItem


def test_image_only_manifest_filters_invalid_and_duplicate_records(tmp_path: Path):
    records = [
        ManifestItem(image_id="val-2", split="Val", relative_path="../Val/2.jpg",
                     sha256="2", size_bytes=2),
        ManifestItem(image_id="val-1", split="Val", relative_path="..\\Val\\1.jpg",
                     sha256="1", size_bytes=1),
        ManifestItem(image_id="bad", split="Val", relative_path="../Val/bad.jpg",
                     sha256="bad", size_bytes=0, valid=False, error="broken"),
        ManifestItem(image_id="dup", split="Val", relative_path="../Val/dup.jpg",
                     sha256="1", size_bytes=1, duplicate_of="val-1"),
    ]
    path = tmp_path / "val.jsonl"
    path.write_text("".join(item.model_dump_json() + "\n" for item in records), encoding="utf-8")
    items = load_manifest_items(path, "Val")
    annotations = placeholder_annotations(items)
    assert [item.image_id for item in annotations] == ["val-1", "val-2"]
    assert annotations[0].relative_path == "../Val/1.jpg"
    assert annotations[0].prompt_version == IMAGE_ONLY_ANNOTATION_VERSION


def test_image_only_manifest_honors_limit(tmp_path: Path):
    records = [
        ManifestItem(image_id=f"train-{index}", split="Train", relative_path=f"../Train/{index}.jpg",
                     sha256=str(index), size_bytes=1)
        for index in range(3)
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(item.model_dump_json() + "\n" for item in records), encoding="utf-8")
    assert len(load_manifest_items(path, "Train", limit=1)) == 1
