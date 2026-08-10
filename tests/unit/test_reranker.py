from __future__ import annotations

from pathlib import Path

from PIL import Image

from anima_search.retrieval.reranker import VisualReranker
from anima_search.schemas import SearchResult


class RerankClient:
    def __init__(self, output: str | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error

    def generate(self, image, prompt, max_new_tokens):
        if self.error:
            raise self.error
        return self.output


def _candidate(relative_path: str) -> SearchResult:
    return SearchResult(
        image_id="train-1",
        relative_path=relative_path,
        fused_score=0.5,
        active_branches=["image", "text", "bm25"],
    )


def test_visual_reranker_parses_single_object_contract(tmp_path: Path):
    image_path = tmp_path / "1.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    client = RerankClient(
        '{"image_id":"train-1","score":88,"evidence":["公路"],'
        '"mismatch":[],"confidence":0.9}'
    )
    result = VisualReranker(client, "只输出一个 JSON 对象", tmp_path).rerank(
        "高速公路", [_candidate("1.jpg")]
    )[0]
    assert result.rerank_score == 88
    assert result.evidence == ["公路"]
    assert result.mismatch == []


def test_visual_reranker_normalizes_scalar_evidence(tmp_path: Path):
    image_path = tmp_path / "1.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    client = RerankClient(
        '{"image_id":"train-1","score":95,"evidence":"日落公路",'
        '"mismatch":"","confidence":1.0}'
    )
    result = VisualReranker(client, "prompt", tmp_path).rerank(
        "高速公路", [_candidate("1.jpg")]
    )[0]
    assert result.evidence == ["日落公路"]
    assert result.mismatch == []


def test_visual_reranker_records_explicit_degradation(tmp_path: Path):
    image_path = tmp_path / "1.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    client = RerankClient(error=RuntimeError("offline"))
    result = VisualReranker(client, "prompt", tmp_path).rerank(
        "高速公路", [_candidate("1.jpg")]
    )[0]
    assert result.rerank_score == 0.0
    assert result.mismatch == ["视觉重排不可用：RuntimeError: offline"]


def test_visual_reranker_rejects_mismatched_image_id(tmp_path: Path):
    image_path = tmp_path / "1.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    client = RerankClient(
        '{"image_id":"train-999","score":99,"evidence":["公路"],'
        '"mismatch":[],"confidence":1.0}'
    )
    result = VisualReranker(client, "prompt", tmp_path).rerank(
        "高速公路", [_candidate("1.jpg")]
    )[0]
    assert result.rerank_score == 0.0
    assert result.evidence == []
    assert result.mismatch == ["视觉重排不可用：ValueError: reranker image_id 'train-999' does not match 'train-1'"]
