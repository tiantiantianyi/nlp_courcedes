from __future__ import annotations

import pytest
from pydantic import ValidationError

from anima_search.m6.contract import M5QueryBatch


def candidate(rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "image_id": f"val-{2001 + rank}",
        "relative_path": f"../Val/{2001 + rank}.jpg",
        "fused_score": 1.0 / rank,
        "branch_scores": {"image": 0.9 / rank, "text": 0.8 / rank},
        "branch_ranks": {"image": rank, "text": rank + 1},
        "matched_fields": ["scene"],
    }


def batch_payload() -> dict[str, object]:
    return {
        "schema_version": "m5-to-m6-v1.0",
        "query_id": "m6-q001",
        "query": "夜晚的城市街道",
        "category": "simple",
        "split": "val",
        "fusion_method": "rrf",
        "top_k": 20,
        "annotation_version": "qwen35-canonical-v1.3",
        "index_manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "candidates": [candidate(rank) for rank in range(1, 21)],
    }


def test_valid_batch_maps_m5_evidence_without_reordering() -> None:
    batch = M5QueryBatch.model_validate(batch_payload())

    results = batch.to_search_results()

    assert [item.image_id for item in results] == [
        f"val-{number}" for number in range(2002, 2022)
    ]
    assert results[0].fused_score == 1.0
    assert results[0].branch_scores == {"image": 0.9, "text": 0.8}
    assert results[0].branch_ranks == {"image": 1, "text": 2}
    assert results[0].matched_fields == ["scene"]


@pytest.mark.parametrize("count", [19, 21])
def test_batch_requires_exactly_twenty_candidates(count: int) -> None:
    payload = batch_payload()
    payload["candidates"] = [
        candidate(rank) for rank in range(1, count + 1)
    ]

    with pytest.raises(ValidationError):
        M5QueryBatch.model_validate(payload)


def test_branch_score_and_rank_keys_must_match() -> None:
    payload = batch_payload()
    payload["candidates"][0]["branch_ranks"] = {"image": 1}  # type: ignore[index]

    with pytest.raises(ValidationError, match="branch keys"):
        M5QueryBatch.model_validate(payload)


def test_candidate_rank_sequence_must_be_ordered() -> None:
    payload = batch_payload()
    payload["candidates"][1]["rank"] = 1  # type: ignore[index]

    with pytest.raises(ValidationError, match="ordered sequence"):
        M5QueryBatch.model_validate(payload)


def test_candidate_image_ids_must_be_unique() -> None:
    payload = batch_payload()
    payload["candidates"][1]["image_id"] = payload["candidates"][0]["image_id"]  # type: ignore[index]

    with pytest.raises(ValidationError, match="must be unique"):
        M5QueryBatch.model_validate(payload)
