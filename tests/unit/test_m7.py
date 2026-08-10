from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from anima_search.m7.citations import extract_citations, validate_citations
from anima_search.m7.service import M7Service, REFUSAL_TEXT
from anima_search.schemas import SearchResult


class FakeClient:
    def __init__(self, image_outputs: list[str], text_outputs: list[str]) -> None:
        self.image_outputs = iter(image_outputs)
        self.text_outputs = iter(text_outputs)

    def generate(self, image, prompt, max_new_tokens):
        return next(self.image_outputs)

    def generate_text(self, prompt, max_new_tokens):
        return next(self.text_outputs)


class FakeManager:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    @contextmanager
    def qwen_session(self):
        yield self.client


def candidates(root: Path, count: int) -> list[SearchResult]:
    results = []
    for index in range(count):
        path = root / f"{index}.jpg"
        Image.new("RGB", (16, 16), color=(index * 20, 0, 0)).save(path)
        results.append(SearchResult(
            image_id=f"val-{index}", relative_path=path.name,
            fused_score=1.0 / (index + 1), active_branches=["mock"],
        ))
    return results


def test_citation_validation_rejects_unknown_image():
    assert extract_citations("可见汽车[img_val-1]") == ["val-1"]
    try:
        validate_citations("错误[img_missing]", [], {"val-1"}, refused=False)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("unknown citation must be rejected")


def test_annotation_optional_grounded_answer(tmp_path: Path):
    client = FakeClient(
        ['{"relevant":true,"facts":["画面中有一辆自行车"],"uncertainty":[]}'],
        ['{"answer":"画面中有一辆自行车[img_val-0]","citations":["val-0"],'
         '"confidence":0.9,"refused":false}'],
    )
    service = M7Service(FakeManager(client), tmp_path)
    answer = service.answer("有什么交通工具？", candidates(tmp_path, 1))
    assert answer.citations == ["val-0"]
    assert not answer.refused
    assert not answer.evidence[0].used_annotation


def test_grounded_answer_refuses_without_evidence(tmp_path: Path):
    client = FakeClient(
        ['{"relevant":false,"facts":[],"uncertainty":["无法确认"]}'],
        [],
    )
    service = M7Service(FakeManager(client), tmp_path)
    answer = service.answer("这是什么城市？", candidates(tmp_path, 1))
    assert answer.refused and answer.answer == REFUSAL_TEXT


def test_story_preserves_selected_order(tmp_path: Path):
    selected = candidates(tmp_path, 3)
    image_outputs = [
        f'{{"relevant":true,"facts":["场景{index}"],"uncertainty":[]}}'
        for index in range(3)
    ]
    text_output = (
        '{"title":"三段旅程","sections":['
        '{"image_id":"val-0","subtitle":"一","text":"场景0"},'
        '{"image_id":"val-1","subtitle":"二","text":"场景1"},'
        '{"image_id":"val-2","subtitle":"三","text":"场景2"}]}'
    )
    service = M7Service(FakeManager(FakeClient(image_outputs, [text_output])), tmp_path)
    story = service.create_story(selected)
    assert [section.image_id for section in story.sections] == ["val-0", "val-1", "val-2"]


def test_route_distinguishes_story_and_search():
    assert M7Service.route("帮我把这些照片写成游记").intent == "story"
    assert M7Service.route("寻找夜晚街道").intent == "search"
