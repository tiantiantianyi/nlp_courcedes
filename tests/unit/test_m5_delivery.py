from __future__ import annotations

import pytest
from pydantic import ValidationError

from anima_search.delivery.m5_candidates import build_m5_candidate_batch
from anima_search.schemas import SearchResult


HASH_A = "a" * 64
HASH_B = "b" * 64


def results() -> list[SearchResult]:
    return [
        SearchResult(
            image_id=f"val-{2001 + rank}",
            relative_path=f"../Val/{2001 + rank}.jpg",
            fused_score=1.0 / rank,
            branch_scores={"image": 1.0 / rank, "text": 0.5 / rank},
            branch_ranks={"image": rank, "text": 21 - rank},
            matched_fields=["scene"] if rank == 1 else [],
        )
        for rank in range(1, 21)
    ]


def build(rows: list[SearchResult] | None = None):
    return build_m5_candidate_batch(
        query_id="m6-q001",
        query="夜晚的城市街道",
        category="simple",
        split="val",
        fusion_method="rrf",
        annotation_version="qwen35-canonical-v1.3",
        index_manifest_sha256=HASH_A,
        config_sha256=HASH_B,
        results=rows or results(),
    )


def test_m5_delivery_has_exact_v1_fields_and_preserves_fusion_order():
    record = build()
    payload = record.model_dump()
    assert set(payload) == {
        "schema_version",
        "query_id",
        "query",
        "category",
        "split",
        "fusion_method",
        "top_k",
        "annotation_version",
        "index_manifest_sha256",
        "config_sha256",
        "candidates",
    }
    assert [item["rank"] for item in payload["candidates"]] == list(range(1, 21))
    assert "rerank_score" not in payload["candidates"][0]


def test_m5_delivery_rejects_non_top20_and_mismatched_branch_keys():
    with pytest.raises(ValueError, match="exactly 20"):
        build(results()[:19])

    rows = results()
    rows[0].branch_ranks = {"image": 1}
    with pytest.raises(ValidationError, match="identical keys"):
        build(rows)


def test_m5_delivery_rejects_nonfinite_scores():
    rows = results()
    rows[0].fused_score = float("nan")
    with pytest.raises(ValidationError):
        build(rows)
