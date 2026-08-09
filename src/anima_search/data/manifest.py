from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PIL import Image

from anima_search.schemas import ManifestItem


def _numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**63 - 1, path.name


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_split(root: Path, split: str, project_root: Path | None = None) -> list[ManifestItem]:
    base = project_root or root.parent
    items: list[ManifestItem] = []
    first_by_hash: dict[str, str] = {}
    for path in sorted(root.glob("*.jpg"), key=_numeric_key):
        digest = sha256_file(path)
        image_id = f"{split.lower()}-{path.stem}"
        metadata: dict[str, object] = {}
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                metadata.update(width=image.width, height=image.height, mode=image.mode)
        except Exception as exc:
            metadata.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        duplicate_of = first_by_hash.get(digest)
        first_by_hash.setdefault(digest, image_id)
        items.append(ManifestItem(
            image_id=image_id, split=split,
            relative_path=Path(os.path.relpath(path, base)).as_posix(),
            sha256=digest, size_bytes=path.stat().st_size, duplicate_of=duplicate_of, **metadata,
        ))
    return items


def mark_cross_split_duplicates(groups: list[list[ManifestItem]]) -> None:
    first_by_hash: dict[str, str] = {}
    for items in groups:
        for item in items:
            duplicate = first_by_hash.get(item.sha256)
            if duplicate and not item.duplicate_of:
                item.duplicate_of = duplicate
            first_by_hash.setdefault(item.sha256, item.image_id)
