from __future__ import annotations

from pathlib import Path

from anima_search.app.service import SearchService
from anima_search.m7.schemas import StoryGap, StorySection, VisualStory
from anima_search.schemas import SearchQuery, SearchResult


class Parser:
    def parse(self, query: str, client=None):
        return SearchQuery(raw_text=query)


class ReleasableIndex:
    def __init__(self) -> None:
        self.released = False

    def unload_encoder(self) -> None:
        self.released = True


class Searcher:
    def __init__(self, index: ReleasableIndex) -> None:
        self.indexes = {"image": index}

    def search(self, query, candidate_count, result_count):
        return []


class Manager:
    def unload_all(self) -> None:
        pass


class M7Stub:
    def __init__(self) -> None:
        self.answer_candidates = []
        self.story_candidates = []

    def answer(self, question, candidates, top_k):
        self.answer_candidates = candidates
        return {"question": question, "top_k": top_k}

    def create_story(self, candidates, tone, theme):
        self.story_candidates = candidates
        return VisualStory(
            title=theme,
            sections=[
                StorySection(image_id=item.image_id, subtitle=tone, text="片段")
                for item in candidates
            ],
            ordered_image_ids=[item.image_id for item in candidates],
            gaps=[
                StoryGap(
                    gap_id="gap-01",
                    after_image_id=candidates[0].image_id,
                    before_image_id=candidates[1].image_id,
                    reason="时间跨度",
                    generation_prompt="生成自然过渡画面",
                )
            ],
        )


def service(tmp_path: Path) -> tuple[SearchService, ReleasableIndex, M7Stub]:
    qwen_path = tmp_path / "qwen"
    qwen_path.mkdir()
    index = ReleasableIndex()
    instance = SearchService(
        {
            "project_root": str(tmp_path),
            "models": {"qwen_vl": str(qwen_path)},
            "retrieval": {
                "candidate_count": 5,
                "result_count": 5,
                "rerank_count": 5,
                "query_parser_use_llm": False,
            },
        },
        Parser(),
        Searcher(index),
        Manager(),
        {},
        "",
        "",
        "",
    )
    stub = M7Stub()
    instance.m7 = stub
    return instance, index, stub


def candidates(count: int) -> list[SearchResult]:
    return [
        SearchResult(
            image_id=f"val-{index}",
            relative_path=f"../Val/{index}.jpg",
            fused_score=1.0,
        )
        for index in range(count)
    ]


def test_grounded_answer_releases_retrieval_encoder_and_limits_selection(tmp_path: Path):
    instance, index, stub = service(tmp_path)
    result = instance.answer_with_evidence(
        "有什么？",
        candidates(4),
        ["val-0", "val-1"],
        top_k=3,
    )
    assert index.released
    assert result["top_k"] == 2
    assert [item.image_id for item in stub.answer_candidates] == ["val-0", "val-1"]


def test_visual_story_requires_three_to_eight_current_results(tmp_path: Path):
    instance, index, stub = service(tmp_path)
    result = instance.create_visual_story(
        candidates(3),
        ["val-0", "val-1", "val-2"],
        theme="旅程",
        tone="自然",
    )
    assert index.released
    assert result.title == "旅程"
    assert result.sections[0].subtitle == "自然"
    assert len(stub.story_candidates) == 3


def test_visual_story_can_fill_gap_and_marks_generated_asset(
    tmp_path: Path,
    monkeypatch,
):
    instance, _, _ = service(tmp_path)
    generated = tmp_path / "artifacts" / "generated" / "generated-42.png"
    generated.parent.mkdir(parents=True)
    generated.touch()
    calls = []

    def fake_generate(query: str, image_id: str | None, seed: int):
        calls.append((query, image_id, seed))
        return generated

    monkeypatch.setattr(instance, "generate_image", fake_generate)
    result = instance.create_visual_story(
        candidates(3),
        ["val-0", "val-1", "val-2"],
        fill_gaps=True,
        seed=42,
    )

    gap = result.gaps[0]
    assert calls == [("生成自然过渡画面", "val-0", 42)]
    assert gap.status == "generated"
    assert gap.source == "generated" and gap.ai_generated
    assert gap.generated_image_id == "generated-42"
    assert gap.relative_path == "artifacts/generated/generated-42.png"
