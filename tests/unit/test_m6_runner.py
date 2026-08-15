from __future__ import annotations

from pathlib import Path

from PIL import Image

from anima_search.m6.contract import M5QueryBatch
from anima_search.m6.runner import rerank_query_batch
from anima_search.retrieval.reranker import VisualReranker
from anima_search.schemas import SearchResult


def _batch() -> M5QueryBatch:
    candidates = []
    for rank in range(1, 21):
        candidates.append(
            {
                "rank": rank,
                "image_id": f"val-{2001 + rank}",
                "relative_path": f"../Val/{2001 + rank}.jpg",
                "fused_score": 1.0 / rank,
                "branch_scores": {"image": 0.9 / rank, "text": 0.8 / rank},
                "branch_ranks": {"image": rank, "text": rank + 1},
                "matched_fields": ["scene"],
            }
        )
    return M5QueryBatch.model_validate(
        {
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
            "candidates": candidates,
        }
    )


class ReverseReranker:
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        assert query == "夜晚的城市街道"
        for index, candidate in enumerate(candidates):
            candidate.rerank_score = float(100 - index)
        return list(reversed(candidates))


class DuplicateAndMissingReranker:
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        for candidate in candidates:
            candidate.rerank_score = 75.0
        return [*candidates[:-1], candidates[0]]


class UnknownIdReranker:
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        return [
            *candidates[:-1],
            SearchResult(
                image_id="val-unknown",
                relative_path="../Val/unknown.jpg",
                fused_score=0.0,
                rerank_score=100.0,
            ),
        ]


class MetadataDegradedReranker(ReverseReranker):
    last_error = None
    last_degraded_reason = "appended missing IDs: ['val-2021']"


def test_reverse_rerank_preserves_every_m5_field() -> None:
    batch = _batch()

    result = rerank_query_batch(batch, ReverseReranker(), method="listwise")

    assert result.schema_version == "m6-rerank-v1.0"
    assert result.source_schema_version == "m5-to-m6-v1.0"
    assert result.annotation_version == "qwen35-canonical-v1.3"
    assert result.index_manifest_sha256 == "a" * 64
    assert result.config_sha256 == "b" * 64
    assert [item.rerank_rank for item in result.candidates] == list(
        range(1, 21)
    )
    assert result.candidates[0].image_id == "val-2021"
    assert result.candidates[0].rank == 20
    assert result.candidates[0].fused_score == 0.05
    assert result.candidates[0].branch_scores == {
        "image": 0.045,
        "text": 0.04,
    }
    assert result.candidates[0].branch_ranks == {"image": 20, "text": 21}
    assert result.candidates[0].matched_fields == ["scene"]
    assert not result.degraded


def test_duplicate_is_dropped_and_missing_id_is_appended() -> None:
    result = rerank_query_batch(
        _batch(),
        DuplicateAndMissingReranker(),
        method="listwise",
    )

    assert len(result.candidates) == 20
    assert len({item.image_id for item in result.candidates}) == 20
    assert result.candidates[-1].image_id == "val-2021"
    assert result.candidates[-1].rerank_score == 0.0
    assert result.degraded
    assert any("dropped duplicate" in item for item in result.mismatch)
    assert any("appended missing" in item for item in result.mismatch)


def test_unknown_id_causes_auditable_hard_fallback() -> None:
    result = rerank_query_batch(
        _batch(),
        UnknownIdReranker(),
        method="listwise",
    )

    assert [item.image_id for item in result.candidates] == [
        f"val-{number}" for number in range(2002, 2022)
    ]
    assert all(item.rerank_score == 0.0 for item in result.candidates)
    assert result.degraded
    assert any("unknown image_id" in item for item in result.mismatch)
    assert all(result.candidates[0].mismatch)


def test_reranker_degradation_metadata_reaches_output() -> None:
    result = rerank_query_batch(
        _batch(),
        MetadataDegradedReranker(),
        method="listwise",
    )

    assert result.degraded
    assert result.mismatch == ["appended missing IDs: ['val-2021']"]


class InvalidJsonClient:
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        return "this is not JSON"


def test_pointwise_all_candidate_failures_restore_exact_m5_order(
    tmp_path: Path,
) -> None:
    payload = _batch().model_dump(mode="json")
    for rank, candidate in enumerate(payload["candidates"], start=1):
        candidate["fused_score"] = float(rank)
        image_path = tmp_path / candidate["relative_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "navy").save(image_path)
    batch = M5QueryBatch.model_validate(payload)
    reranker = VisualReranker(
        InvalidJsonClient(),
        "return JSON",
        tmp_path,
    )

    result = rerank_query_batch(batch, reranker, method="pointwise")

    assert [item.image_id for item in result.candidates] == [
        item.image_id for item in batch.candidates
    ]
    assert [item.rank for item in result.candidates] == list(range(1, 21))
    assert all(item.rerank_score == 0.0 for item in result.candidates)
    assert result.degraded
    assert any(
        message.startswith("视觉重排不可用：")
        for message in result.mismatch
    )
