from __future__ import annotations

from itertools import product


def ablation_matrix() -> list[dict[str, object]]:
    rows = []
    for prompt, retrieval, trained, rerank in product(
        ["caption_basic_v1", "caption_structured_v1", "caption_verified_v1"],
        ["bm25", "dense", "hybrid"], [False, True], [False, True]):
        rows.append({"prompt": prompt, "retrieval": retrieval, "trained": trained, "rerank": rerank})
    return rows
