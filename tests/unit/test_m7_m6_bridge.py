from __future__ import annotations

from pathlib import Path

import pytest

from anima_search.m6.results import M6QueryResult
from anima_search.m7.m6_bridge import load_m6_query, select_story_candidates


def _result(query_id: str = "q001") -> M6QueryResult:
    return M6QueryResult.model_validate(
        {
            "schema_version": "m6-rerank-v1.0",
            "source_schema_version": "m5-to-m6-v1.0",
            "query_id": query_id,
            "query": "夜晚街道",
            "category": "simple",
            "split": "val",
            "fusion_method": "rrf",
            "top_k": 20,
            "annotation_version": "qwen35-canonical-v1.3",
            "index_manifest_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "rerank_method": "listwise",
            "degraded": False,
            "mismatch": [],
            "candidates": [
                {
                    "rank": 20 - index,
                    "image_id": f"val-{2021 - index}",
                    "relative_path": f"../Val/{2021 - index}.jpg",
                    "fused_score": 1.0 / (20 - index),
                    "branch_scores": {"image": 0.5},
                    "branch_ranks": {"image": 20 - index},
                    "matched_fields": ["scene"],
                    "rerank_rank": index + 1,
                    "rerank_score": float(100 - index),
                    "mismatch": [],
                }
                for index in range(20)
            ],
        }
    )


def test_loads_unique_query_and_preserves_m6_story_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m6.jsonl"
    path.write_text(
        _result("q001").model_dump_json() + "\n"
        + _result("q002").model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    loaded = load_m6_query(path, "q001")
    selected = select_story_candidates(loaded, 3)

    assert [item.image_id for item in selected] == [
        "val-2021",
        "val-2020",
        "val-2019",
    ]
    assert [item.relative_path for item in selected] == [
        "../Val/2021.jpg",
        "../Val/2020.jpg",
        "../Val/2019.jpg",
    ]
    assert selected[0].rerank_score == 100.0
    assert selected[0].branch_ranks == {"image": 20}


@pytest.mark.parametrize("count", [2, 9])
def test_story_selection_requires_three_to_eight_images(count: int) -> None:
    with pytest.raises(ValueError, match="3 to 8"):
        select_story_candidates(_result(), count)


def test_load_rejects_missing_query(tmp_path: Path) -> None:
    path = tmp_path / "m6.jsonl"
    path.write_text(_result("q001").model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        load_m6_query(path, "missing")


def test_load_rejects_duplicate_query_id(tmp_path: Path) -> None:
    path = tmp_path / "m6.jsonl"
    payload = _result("q001").model_dump_json()
    path.write_text(payload + "\n" + payload + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_m6_query(path, "q001")
