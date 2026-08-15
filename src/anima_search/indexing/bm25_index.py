from __future__ import annotations

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


class BM25Index:
    def __init__(self, image_ids: list[str], documents: list[str],
                 annotation_version: str = "", build_parameters: dict | None = None) -> None:
        if len(image_ids) != len(documents):
            raise ValueError("image_ids and documents must have the same length")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("image_ids must be unique")
        self.image_ids = image_ids
        self.documents = documents
        self.annotation_version = annotation_version
        self.build_parameters = build_parameters or {}
        self.model = BM25Okapi([tokenize(doc) for doc in documents])

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        if limit <= 0 or not self.image_ids:
            return []
        scores = self.model.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.image_ids[i]))[:limit]
        return [(self.image_ids[i], float(scores[i])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"image_ids": self.image_ids, "documents": self.documents,
                         "annotation_version": self.annotation_version,
                         "build_parameters": self.build_parameters}, handle)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(payload["image_ids"], payload["documents"],
                   payload.get("annotation_version", ""), payload.get("build_parameters", {}))
