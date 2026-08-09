from __future__ import annotations

import random
from dataclasses import dataclass, asdict

from anima_search.indexing.documents import annotation_to_document
from anima_search.schemas import ImageAnnotation


@dataclass(frozen=True)
class TrainingPair:
    query: str
    positive: str
    negative: str
    image_id: str
    negative_image_id: str
    negative_kind: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_training_pairs(annotations: list[ImageAnnotation], seed: int = 20260802) -> list[TrainingPair]:
    if any(item.split != "Train" for item in annotations):
        raise ValueError("Training pairs may only be built from Train annotations")
    rng = random.Random(seed)
    by_scene: dict[str, list[ImageAnnotation]] = {}
    for item in annotations:
        by_scene.setdefault(item.scene, []).append(item)
    pairs: list[TrainingPair] = []
    for item in annotations:
        hard = [candidate for candidate in by_scene[item.scene] if candidate.image_id != item.image_id]
        random_pool = [candidate for candidate in annotations if candidate.image_id != item.image_id]
        if not random_pool:
            continue
        for index, query in enumerate(item.search_queries):
            if index == 0 and hard:
                negative, kind = rng.choice(hard), "hard"
            else:
                negative, kind = rng.choice(random_pool), "random"
            pairs.append(TrainingPair(query=query, positive=annotation_to_document(item),
                negative=annotation_to_document(negative), image_id=item.image_id,
                negative_image_id=negative.image_id, negative_kind=kind))
    return pairs
