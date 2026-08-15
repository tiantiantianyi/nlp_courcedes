from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anima_search.indexing.faiss_io import read_faiss_index, write_faiss_index
from anima_search.provenance import model_directory_fingerprint


class VectorIndex:
    def __init__(self, model_path: str | Path, device: str | None = None,
                 annotation_version: str = "", build_parameters: dict | None = None,
                 encoder: object | None = None) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.annotation_version = annotation_version
        self.build_parameters = build_parameters or {}
        self.model_digest = model_directory_fingerprint(self.model_path)
        self.model = encoder
        self.index = None
        self.image_ids: list[str] = []

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path, device=self.device)
        return self.model

    def build(self, image_ids: list[str], documents: list[str], batch_size: int = 32) -> None:
        import faiss
        if len(image_ids) != len(documents):
            raise ValueError("image_ids and documents must have the same length")
        if not image_ids:
            raise ValueError("cannot build an empty vector index")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("image_ids must be unique")
        vectors = self._load_model().encode(documents, batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(image_ids):
            raise ValueError("encoder returned an invalid vector matrix")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.image_ids = image_ids

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("vector index has not been built or loaded")
        if limit <= 0:
            return []
        vector = self._load_model().encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(np.asarray(vector, dtype=np.float32), min(limit, len(self.image_ids)))
        return [(self.image_ids[int(i)], float(score)) for score, i in zip(scores[0], indices[0]) if i >= 0]

    def save(self, directory: Path) -> None:
        if self.index is None:
            raise RuntimeError("cannot save a vector index before build")
        directory.mkdir(parents=True, exist_ok=True)
        write_faiss_index(self.index, directory / "vectors.faiss")
        (directory / "metadata.json").write_text(json.dumps(
            {"image_ids": self.image_ids, "model_path": self.model_path, "device": self.device,
             "annotation_version": self.annotation_version,
             "model_digest": self.model_digest,
             "build_parameters": self.build_parameters}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: Path,
        encoder: object | None = None,
        *,
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> VectorIndex:
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(
            model_path or payload["model_path"],
            device if device is not None else payload.get("device"),
            payload.get("annotation_version", ""),
            payload.get("build_parameters", {}),
            encoder=encoder,
        )
        instance.model_digest = payload.get("model_digest", instance.model_digest)
        instance.image_ids = payload["image_ids"]
        instance.index = read_faiss_index(directory / "vectors.faiss")
        if instance.index.ntotal != len(instance.image_ids):
            raise ValueError("vector index count does not match metadata image_ids")
        return instance
