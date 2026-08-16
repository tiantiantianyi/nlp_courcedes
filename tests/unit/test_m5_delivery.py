from __future__ import annotations

import pytest

from anima_search.delivery.m5_candidates import build_m5_candidate_batch
from anima_search.m6.contract import M5QueryBatch
from anima_search.schemas import SearchResult


def _results(count: int = 20) -> list[SearchResult]:
    return [
        SearchResult(
            image_id=f"val-{index}",
            relative_path=f"../Val/{index}.jpg",
            fused_score=1.0 / index,
            rerank_score=99.0,
            branch_scores={"image": 1.0 / index, "bm25": float(21 - index)},
            branch_ranks={"image": index, "bm25": index},
            matched_fields=["scene"],
            active_branches=["image", "bm25"],
        )
        for index in range(1, count + 1)
    ]


def _build(results: list[SearchResult]) -> M5QueryBatch:
    return build_m5_candidate_batch(
        query_id="m6-q01",
        query="夜晚的城市街道",
        category="simple",
        split="val",
        fusion_method="rrf",
        annotation_version="qwen35-canonical-v1.3",
        index_manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        results=results,
    )


def test_builder_returns_canonical_contract_and_preserves_m5_scores() -> None:
    batch = _build(_results())

    assert isinstance(batch, M5QueryBatch)
    assert [candidate.rank for candidate in batch.candidates] == list(range(1, 21))
    assert batch.candidates[0].branch_scores == {"image": 1.0, "bm25": 20.0}
    assert batch.candidates[0].branch_ranks == {"image": 1, "bm25": 1}
    assert batch.candidates[0].matched_fields == ["scene"]
    assert all(
        "rerank_score" not in candidate
        for candidate in batch.model_dump()["candidates"]
    )


def test_builder_rejects_fewer_than_exactly_twenty_results() -> None:
    with pytest.raises(ValueError, match="exactly 20"):
        _build(_results(19))


def test_builder_rejects_duplicate_image_ids() -> None:
    results = _results()
    results[-1] = results[-1].model_copy(update={"image_id": results[0].image_id})

    with pytest.raises(ValueError, match="must be unique"):
        _build(results)
