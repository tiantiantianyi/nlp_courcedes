from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from anima_search.evaluation.ablation import (
    a5_ablation_matrix,
    run_a5_ablation,
    run_formal_a5_ablation,
    write_ablation_results,
    write_formal_a5_results,
)
from anima_search.evaluation.manual_set import write_relevance, write_tasks
from anima_search.schemas import SearchResult
from scripts import run_ablation


class BranchService:
    def __init__(self, branches: list[str]) -> None:
        self.branches = branches
        self.released = False

    def search(self, query: str, use_reranker: bool):
        return [SearchResult(image_id="a", relative_path="Val/a.jpg", fused_score=1.0)]

    def release_retrieval_encoders(self) -> list[str]:
        self.released = True
        return list(self.branches)


def test_a5_matrix_matches_proposal_baselines():
    rows = a5_ablation_matrix()
    assert [row["variant"] for row in rows] == [
        "clip_only",
        "text_only",
        "bm25_only",
        "rrf_three_way",
        "weighted_three_way",
    ]
    assert rows[-1]["branches"] == ["image", "text", "bm25"]
    assert rows[-1]["fusion_method"] == "weighted"
    assert all(row["reranker"] is False for row in rows)


def test_a5_runner_executes_each_branch_combination(tmp_path: Path):
    calls: list[tuple[list[str], str]] = []

    def factory(branches: list[str], fusion_method: str):
        calls.append((branches, fusion_method))
        return BranchService(branches)

    queries = [{"query_id": "q1", "text": "城市", "category": "simple", "reviewed": True}]
    rows = run_a5_ablation(factory, queries, {"q1": {"a": 2}})
    assert calls == [
        (["image"], "rrf"),
        (["text"], "rrf"),
        (["bm25"], "rrf"),
        (["image", "text", "bm25"], "rrf"),
        (["image", "text", "bm25"], "weighted"),
    ]
    assert len(rows) == 5
    assert all(row["recall@1"] == 1.0 for row in rows)
    paths = write_ablation_results(tmp_path, rows)
    assert all(path.is_file() for path in paths.values())
    assert len(json.loads(paths["json"].read_text(encoding="utf-8"))) == 5


def test_formal_a5_runner_splits_all_and_complete_graded_queries(tmp_path: Path):
    queries = [
        {"query_id": "q1", "text": "城市", "category": "simple", "reviewed": True},
        {"query_id": "q2", "text": "文字", "category": "ocr", "reviewed": True},
    ]
    relevance = {"q1": {"a": 2, "b": 0}, "q2": {"a": 2}}
    services: list[BranchService] = []

    def formal_factory(branches: list[str], _fusion: str) -> BranchService:
        service = BranchService(branches)
        services.append(service)
        return service


    variants = run_formal_a5_ablation(
        formal_factory,
        queries,
        relevance,
        graded_candidate_ids={"q1": {"a", "b"}},
    )

    assert len(variants) == 5
    assert all(row["all_queries"]["overall"]["query_count"] == 2 for row in variants)
    assert all(row["graded_queries"]["overall"]["query_count"] == 1 for row in variants)
    assert all("simple" in row["graded_queries"]["by_category"] for row in variants)
    assert all("failure_rate" in row["all_queries"]["overall"] for row in variants)
    assert all("latency_p95_seconds" in row["graded_queries"]["overall"] for row in variants)
    assert all(service.released for service in services)

    paths = write_formal_a5_results(
        tmp_path,
        variants,
        provenance={"queries_sha256": "a" * 64, "qrels_sha256": "b" * 64},
    )
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["provenance"]["queries_sha256"] == "a" * 64
    assert len(payload["variants"]) == 5
    assert all(path.is_file() for path in paths.values())


def test_formal_a5_runner_rejects_incomplete_graded_candidate_qrels():
    queries = [
        {"query_id": "q1", "text": "城市", "category": "simple", "reviewed": True}
    ]

    with pytest.raises(ValueError, match="complete candidate-pool judgments"):
        run_formal_a5_ablation(
            lambda branches, _fusion: BranchService(branches),
            queries,
            {"q1": {"a": 2}},
            graded_candidate_ids={"q1": {"a", "b"}},
        )


def test_run_ablation_cli_writes_formal_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    query_path = tmp_path / "queries.jsonl"
    relevance_path = tmp_path / "relevance.csv"
    pool_path = tmp_path / "pool.jsonl"
    config_path = tmp_path / "configs" / "formal.yaml"
    index_manifest = tmp_path / "artifacts" / "indexes" / "val" / "manifest.json"
    output_dir = tmp_path / "a5"
    write_tasks(
        query_path,
        [{"query_id": "q1", "text": "城市", "category": "simple", "reviewed": True}],
    )
    write_relevance(
        relevance_path,
        [
            {"query_id": "q1", "image_id": "a", "relevance": 2, "annotator": "甲", "note": ""},
            {"query_id": "q1", "image_id": "b", "relevance": 0, "annotator": "甲", "note": ""},
        ],
    )
    pool_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "candidates": [{"image_id": "a"}, {"image_id": "b"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text("data:\n  artifacts_dir: artifacts\n", encoding="utf-8")
    index_manifest.parent.mkdir(parents=True)
    index_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        run_ablation,
        "create_service",
        lambda _config, _split, branches, _fusion: BranchService(list(branches)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation.py",
            "--config",
            str(config_path),
            "--queries",
            str(query_path),
            "--relevance",
            str(relevance_path),
            "--pool",
            str(pool_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    run_ablation.main()

    payload = json.loads((output_dir / "a5_formal_results.json").read_text(encoding="utf-8"))
    assert len(payload["variants"]) == 5
    assert payload["provenance"]["queries_sha256"]
    assert payload["provenance"]["qrels_sha256"]
    assert payload["provenance"]["config_sha256"]
    assert payload["provenance"]["index_manifest_sha256"]
    assert len(payload["provenance"]["actual_variants"]) == 5
    assert payload["provenance"]["runtime_timestamp"]
