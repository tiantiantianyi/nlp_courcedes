from __future__ import annotations

import hashlib
from pathlib import Path


def model_directory_fingerprint(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix()
        stat = file_path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if file_path.suffix.lower() in {".json", ".txt", ".yaml", ".yml"} and stat.st_size <= 10_000_000:
            digest.update(file_path.read_bytes())
    return digest.hexdigest()
