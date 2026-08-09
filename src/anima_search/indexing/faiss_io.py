from __future__ import annotations

from pathlib import Path

import numpy as np


def write_faiss_index(index: object, path: Path) -> None:
    """Write through Python so Windows paths containing Unicode remain supported."""
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = faiss.serialize_index(index)
    path.write_bytes(np.asarray(serialized, dtype=np.uint8).tobytes())


def read_faiss_index(path: Path):
    import faiss

    payload = np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()
    return faiss.deserialize_index(payload)
