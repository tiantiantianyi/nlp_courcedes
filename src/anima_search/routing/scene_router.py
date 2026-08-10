from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SceneDefinition:
    key: str
    label: str
    prompts: tuple[str, ...]
    prompt_suffix: str

    @classmethod
    def from_mapping(cls, key: str, payload: dict[str, object]) -> "SceneDefinition":
        prompts = tuple(str(value).strip() for value in payload.get("prompts", []) if str(value).strip())
        if not prompts:
            raise ValueError(f"scene category {key!r} requires at least one text prompt")
        label = str(payload.get("label", key)).strip()
        suffix = str(payload.get("prompt_suffix", "")).strip()
        if not label or not suffix:
            raise ValueError(f"scene category {key!r} requires label and prompt_suffix")
        return cls(key=key, label=label, prompts=prompts, prompt_suffix=suffix)


@dataclass(frozen=True)
class SceneRoute:
    image_id: str
    category: str
    label: str
    score: float
    top_scores: tuple[tuple[str, float], ...]
    prompt_suffix: str

    def as_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "category": self.category,
            "label": self.label,
            "score": self.score,
            "top_scores": [
                {"category": category, "score": score}
                for category, score in self.top_scores
            ],
            "prompt_suffix": self.prompt_suffix,
        }


def _normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("scene routing vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("scene routing received a zero vector")
    return matrix / norms


class SceneRouter:
    def __init__(self, encoder: object, definitions: Iterable[SceneDefinition]) -> None:
        self.encoder = encoder
        self.definitions = list(definitions)
        if not self.definitions:
            raise ValueError("at least one scene category is required")
        keys = [definition.key for definition in self.definitions]
        if len(set(keys)) != len(keys):
            raise ValueError("scene category keys must be unique")
        self._category_vectors: np.ndarray | None = None

    @classmethod
    def from_config(cls, encoder: object, payload: dict[str, object]) -> "SceneRouter":
        categories = payload.get("categories")
        if not isinstance(categories, dict):
            raise ValueError("scene routing config requires a categories mapping")
        definitions = [
            SceneDefinition.from_mapping(str(key), dict(value))
            for key, value in categories.items()
        ]
        return cls(encoder, definitions)

    def category_vectors(self) -> np.ndarray:
        if self._category_vectors is not None:
            return self._category_vectors
        prompts = [
            prompt
            for definition in self.definitions
            for prompt in definition.prompts
        ]
        encoded = _normalize(self.encoder.encode_texts(prompts))
        vectors = []
        offset = 0
        for definition in self.definitions:
            count = len(definition.prompts)
            vectors.append(encoded[offset:offset + count].mean(axis=0))
            offset += count
        self._category_vectors = _normalize(np.stack(vectors))
        return self._category_vectors

    def route_vectors(
        self,
        image_ids: list[str],
        image_vectors: np.ndarray,
        *,
        top_n: int = 3,
    ) -> list[SceneRoute]:
        vectors = _normalize(image_vectors)
        if len(image_ids) != vectors.shape[0]:
            raise ValueError("image ID count does not match vector count")
        if not 1 <= top_n <= len(self.definitions):
            raise ValueError("top_n must be between 1 and the number of scene categories")
        similarities = vectors @ self.category_vectors().T
        routes: list[SceneRoute] = []
        for image_id, scores in zip(image_ids, similarities):
            order = np.argsort(-scores, kind="stable")[:top_n]
            winner = self.definitions[int(order[0])]
            routes.append(
                SceneRoute(
                    image_id=image_id,
                    category=winner.key,
                    label=winner.label,
                    score=float(scores[int(order[0])]),
                    top_scores=tuple(
                        (self.definitions[int(index)].key, float(scores[int(index)]))
                        for index in order
                    ),
                    prompt_suffix=winner.prompt_suffix,
                )
            )
        return routes

    def annotation_prompt(self, base_prompt: str, route: SceneRoute) -> str:
        return (
            f"{base_prompt.rstrip()}\n\n"
            f"场景路由：{route.label}（{route.category}）。"
            f"{route.prompt_suffix}"
        )
