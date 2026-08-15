from __future__ import annotations

import json
from pathlib import Path

from anima_search.evaluation.ablation import (
    a5_ablation_matrix,
    run_a5_ablation,
    write_ablation_results,
)
from anima_search.schemas import SearchResult


class BranchService:
    def __init__(self, branches: list[str]) -> None:
        self.branches = branches

    def search(self, query: str, use_reranker: bool):
        return [SearchResult(image_id="a", relative_path="Val/a.jpg", fused_score=1.0)]


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
