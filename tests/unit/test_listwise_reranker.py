from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from anima_search.retrieval.listwise_reranker import (
    ListwiseVisualReranker,
    build_contact_sheet,
)
from anima_search.schemas import SearchResult


def _candidate(root: Path, index: int) -> SearchResult:
    relative_path = f"Val/{index}.jpg"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80 + index, 60 + index), (index * 30, 50, 100)).save(path)
    return SearchResult(
        image_id=f"val-{index}",
        relative_path=relative_path,
        fused_score=1.0 / index,
    )


class Client:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[Image.Image, str, int]] = []

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        self.calls.append((image.copy(), prompt, max_new_tokens))
        return json.dumps(self.payload)


def test_contact_sheet_contains_all_candidates(tmp_path: Path):
    candidates = [_candidate(tmp_path, index) for index in range(1, 5)]
    sheet = build_contact_sheet(candidates, tmp_path, columns=2, tile_size=100)
    assert sheet.size == (200, 200)


def test_listwise_reranker_uses_one_call_and_validates_ids(tmp_path: Path):
    candidates = [_candidate(tmp_path, index) for index in range(1, 4)]
    client = Client(
        {
            "ranking": [
                {"image_id": "val-3", "score": 93},
                {"image_id": "val-2", "score": 75},
                {"image_id": "val-1", "score": 20},
            ]
        }
    )
    reranker = ListwiseVisualReranker(
        client,
        "只输出 JSON",
        tmp_path,
        max_new_tokens=512,
        columns=3,
        tile_size=96,
    )
    result = reranker.rerank("城市夜景", candidates)
    assert [item.image_id for item in result] == ["val-3", "val-2", "val-1"]
    assert [item.rerank_score for item in result] == [93, 75, 20]
    assert len(client.calls) == 1
    assert client.calls[0][2] == 512
    assert "03 -> val-3" in client.calls[0][1]
    assert reranker.last_error is None


def test_listwise_reranker_maps_valid_sheet_numbers_to_exact_ids(tmp_path: Path):
    candidates = [_candidate(tmp_path, index) for index in range(1, 4)]
    client = Client({"ranking": ["03", "02", "01"]})
    result = ListwiseVisualReranker(client, "prompt", tmp_path).rerank(
        "城市夜景",
        candidates,
    )
    assert [item.image_id for item in result] == ["val-3", "val-2", "val-1"]
    assert [item.rerank_score for item in result] == [100, 50, 0]


def test_duplicate_and_missing_entries_use_auditable_partial_fallback(tmp_path: Path):
    candidates = [_candidate(tmp_path, index) for index in range(1, 3)]
    client = Client(
        {
            "ranking": [
                {"image_id": "val-1", "score": 90},
                {"image_id": "val-1", "score": 80},
            ]
        }
    )
    reranker = ListwiseVisualReranker(client, "prompt", tmp_path)
    result = reranker.rerank("公路", candidates)
    assert [item.image_id for item in result] == ["val-1", "val-2"]
    assert [item.rerank_score for item in result] == [90, 0]
    assert reranker.last_error is None
    assert "dropped duplicates" in str(reranker.last_degraded_reason)
    assert "appended missing IDs" in str(reranker.last_degraded_reason)


def test_unknown_entry_remains_a_hard_failure(tmp_path: Path):
    candidates = [_candidate(tmp_path, index) for index in range(1, 3)]
    client = Client({"ranking": [{"image_id": "99", "score": 90}]})
    reranker = ListwiseVisualReranker(client, "prompt", tmp_path)
    result = reranker.rerank("公路", candidates)
    assert [item.image_id for item in result] == ["val-1", "val-2"]
    assert reranker.last_error is not None
    assert all(
        any(message.startswith("视觉重排不可用：Listwise") for message in item.mismatch)
        for item in result
    )


def test_contact_sheet_rejects_more_than_twenty_candidates(tmp_path: Path):
    candidates = [_candidate(tmp_path, index) for index in range(1, 22)]
    with pytest.raises(ValueError, match="1-20"):
        build_contact_sheet(candidates, tmp_path)
