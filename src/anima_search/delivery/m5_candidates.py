from __future__ import annotations

from typing import Literal

from anima_search.m6.contract import M5QueryBatch
from anima_search.schemas import SearchResult


def build_m5_candidate_batch(
    *,
    query_id: str,
    query: str,
    category: Literal["simple", "compositional", "negative", "count", "ocr"],
    split: Literal["train", "val"],
    fusion_method: Literal["rrf", "weighted"],
    annotation_version: str,
    index_manifest_sha256: str,
    config_sha256: str,
    results: list[SearchResult],
) -> M5QueryBatch:
    if len(results) != 20:
        raise ValueError("M5 delivery requires exactly 20 results")
    return M5QueryBatch.model_validate(
        {
            "schema_version": "m5-to-m6-v1.0",
            "query_id": query_id,
            "query": query,
            "category": category,
            "split": split,
            "fusion_method": fusion_method,
            "top_k": 20,
            "annotation_version": annotation_version,
            "index_manifest_sha256": index_manifest_sha256,
            "config_sha256": config_sha256,
            "candidates": [
                {
                    "rank": rank,
                    "image_id": result.image_id,
                    "relative_path": result.relative_path,
                    "fused_score": result.fused_score,
                    "branch_scores": result.branch_scores,
                    "branch_ranks": result.branch_ranks,
                    "matched_fields": result.matched_fields,
                }
                for rank, result in enumerate(results, start=1)
            ],
        }
    )
