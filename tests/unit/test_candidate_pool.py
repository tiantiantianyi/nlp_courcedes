from __future__ import annotations

import json
import sys

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from anima_search.evaluation.candidate_pool import (
    build_candidate_pool,
    select_balanced_queries,
)
from anima_search.evaluation.manual_set import write_tasks
from scripts import build_relevance_pool


def _query(query_id: str, category: str, source_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "text": f"查询 {query_id}",
        "category": category,
        "source_image_id": source_id,
        "source_relative_path": f"../Val/{source_id.removeprefix('val-')}.jpg",
        "reviewed": True,
        "annotator": "张添翼",
        "note": "",
    }


def test_build_candidate_pool_pins_source_and_preserves_union_provenance():
    query = _query("q001", "simple", "val-1")

    pool = build_candidate_pool(
        queries=[query],
        rankings_by_variant={
            "clip_only": {
                "q001": [
                    {"image_id": "val-2", "relative_path": "../Val/2.jpg"},
                    {"image_id": "val-1", "relative_path": "../Val/1.jpg"},
                ]
            },
            "text_only": {
                "q001": [
                    {"image_id": "val-3", "relative_path": "../Val/3.jpg"},
                    {"image_id": "val-2", "relative_path": "../Val/2.jpg"},
                ]
            },
        },
        source_ids={"q001": "val-1"},
        per_variant_k=2,
    )

    candidates = pool[0]["candidates"]
    assert [row["image_id"] for row in candidates] == ["val-1", "val-2", "val-3"]
    assert candidates[0] == {
        "image_id": "val-1",
        "relative_path": "../Val/1.jpg",
        "is_source": True,
        "retrieved_by": ["clip_only"],
        "best_rank": 2,
        "grade": None,
        "annotator": "",
        "reviewed": False,
    }
    assert candidates[1]["retrieved_by"] == ["clip_only", "text_only"]
    assert candidates[1]["best_rank"] == 1
    assert pool[0]["schema_version"] == "formal-relevance-pool-v1.0"


def test_build_candidate_pool_deduplicates_within_variant_and_caps_candidates():
    query = _query("q001", "simple", "val-1")

    pool = build_candidate_pool(
        queries=[query],
        rankings_by_variant={
            "clip_only": {"q001": ["val-2", "val-2", "val-3"]},
            "text_only": {"q001": ["val-4", "val-5"]},
        },
        source_ids={"q001": "val-1"},
        per_variant_k=3,
        candidate_cap=3,
    )

    assert [row["image_id"] for row in pool[0]["candidates"]] == [
        "val-1",
        "val-2",
        "val-3",
    ]


def test_build_candidate_pool_rejects_foreign_or_missing_query_rankings():
    query = _query("q001", "simple", "val-1")
    with pytest.raises(ValueError, match="foreign query IDs"):
        build_candidate_pool(
            [query],
            {"clip_only": {"q001": ["val-1"], "q999": ["val-9"]}},
            {"q001": "val-1"},
            1,
        )

    with pytest.raises(ValueError, match="missing query IDs"):
        build_candidate_pool(
            [query],
            {"clip_only": {}},
            {"q001": "val-1"},
            1,
        )


def test_select_balanced_queries_is_deterministic_across_five_categories():
    categories = ["simple", "compositional", "negative", "count", "ocr"]
    queries = [
        _query(f"q{index:03d}", category, f"val-{index}")
        for index, category in enumerate(categories * 3, start=1)
    ]

    selected = select_balanced_queries(queries, count=10)

    assert Counter(str(row["category"]) for row in selected) == Counter(
        {category: 2 for category in categories}
    )
    assert [row["query_id"] for row in selected] == [
        "q001",
        "q002",
        "q003",
        "q004",
        "q005",
        "q006",
        "q007",
        "q008",
        "q009",
        "q010",
    ]


def test_select_balanced_queries_rejects_insufficient_category_coverage():
    queries = [
        _query(f"q{index:03d}", "simple", f"val-{index}")
        for index in range(1, 11)
    ]

    with pytest.raises(ValueError, match="category coverage"):
        select_balanced_queries(queries, count=5)


def test_build_relevance_pool_cli_runs_variants_and_releases_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    categories = ["simple", "compositional", "negative", "count", "ocr"]
    query_path = tmp_path / "queries.jsonl"
    write_tasks(
        query_path,
        [
            _query(f"q{index:03d}", category, f"val-{index}")
            for index, category in enumerate(categories, start=1)
        ],
    )
    released: list[str] = []

    class FakeService:
        def __init__(self, label: str) -> None:
            self.label = label

        def search(self, text: str, use_reranker: bool = False):
            query_id = text.rsplit(" ", 1)[-1]
            return [
                SimpleNamespace(
                    image_id=f"{self.label}-{query_id}",
                    relative_path=f"../Val/{self.label}-{query_id}.jpg",
                )
            ]

        def release_retrieval_encoders(self) -> list[str]:
            released.append(self.label)
            return [self.label]

    monkeypatch.setattr(
        build_relevance_pool,
        "a5_ablation_matrix",
        lambda: [
            {"variant": "clip_only", "branches": ["image"], "fusion_method": "rrf"},
            {"variant": "text_only", "branches": ["text"], "fusion_method": "rrf"},
        ],
    )
    monkeypatch.setattr(
        build_relevance_pool,
        "create_service",
        lambda _config, _split, branches, _fusion: FakeService(str(branches[0])),
    )
    output = tmp_path / "relevance_pool.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_relevance_pool.py",
            "--queries",
            str(query_path),
            "--graded-query-count",
            "5",
            "--per-variant-k",
            "1",
            "--candidate-cap",
            "3",
            "--output",
            str(output),
        ],
    )

    build_relevance_pool.main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    assert summary["query_count"] == 5
    assert summary["category_counts"] == {category: 1 for category in categories}
    assert summary["variants"] == ["clip_only", "text_only"]
    assert released == ["image", "text"]
