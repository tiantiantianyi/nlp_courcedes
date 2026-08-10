from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from anima_search.annotation.validation import extract_json_object
from anima_search.m7.citations import citation_token, validate_citations
from anima_search.m7.schemas import GroundedAnswer, ImageEvidence, IntentRoute, StorySection, VisualStory
from anima_search.schemas import ImageAnnotation, SearchResult


REFUSAL_TEXT = "在当前检索到的图片中没有足够依据回答这个问题。你可以扩大检索范围或换一种描述。"


class M7Service:
    """Grounded M7 workflows that operate with or without image annotations."""

    def __init__(self, manager: object, project_root: str | Path,
                 annotations: dict[str, ImageAnnotation] | None = None,
                 max_new_tokens: int = 384) -> None:
        self.manager = manager
        self.project_root = Path(project_root)
        self.annotations = annotations or {}
        self.max_new_tokens = max_new_tokens
        self._evidence_cache: dict[tuple[str, str], ImageEvidence] = {}

    @staticmethod
    def route(text: str, selected_image_ids: list[str] | None = None) -> IntentRoute:
        normalized = text.strip().lower()
        if any(term in normalized for term in ("游记", "故事", "日记", "朋友圈文案")):
            return IntentRoute(intent="story", reason="检测到叙事生成意图")
        if any(term in normalized for term in ("补图", "生成图片", "画一张", "插图")):
            return IntentRoute(intent="generate", reason="检测到图像生成意图")
        if selected_image_ids or any(term in normalized for term in ("什么", "哪些", "是否", "有没有", "为什么")):
            return IntentRoute(intent="qa", reason="检测到针对图片的问答意图")
        return IntentRoute(intent="search", reason="默认执行图片检索")

    def _image_path(self, result: SearchResult) -> Path:
        path = Path(result.relative_path)
        resolved = path if path.is_absolute() else self.project_root / path
        if not resolved.is_file():
            raise FileNotFoundError(f"image for {result.image_id} does not exist: {resolved}")
        return resolved

    def _annotation_context(self, image_id: str) -> str:
        annotation = self.annotations.get(image_id)
        return annotation.model_dump_json() if annotation is not None else ""

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    def _extract_evidence(self, client: object, result: SearchResult, question: str) -> ImageEvidence:
        cache_key = (result.image_id, question.strip())
        if cache_key in self._evidence_cache:
            return self._evidence_cache[cache_key]
        annotation_context = self._annotation_context(result.image_id)
        prompt = (
            "你是图片证据提取器。只记录图片中可以直接确认且与问题有关的事实；"
            "不得猜测人物身份、地点或事件背景。只输出 JSON 对象，字段为 "
            "relevant(bool)、facts(字符串数组)、uncertainty(字符串数组)。\n"
            f"问题：{question}\n"
        )
        if annotation_context:
            prompt += f"可选的已有标注（仍须以原图为准）：{annotation_context}\n"
        with Image.open(self._image_path(result)) as image:
            raw = client.generate(image.convert("RGB").copy(), prompt, max_new_tokens=self.max_new_tokens)
        payload = extract_json_object(raw)
        evidence = ImageEvidence(
            image_id=result.image_id,
            relevant=bool(payload.get("relevant", True)),
            facts=self._string_list(payload.get("facts")),
            uncertainty=self._string_list(payload.get("uncertainty")),
            used_annotation=bool(annotation_context),
        )
        self._evidence_cache[cache_key] = evidence
        return evidence

    @staticmethod
    def _fallback_answer(evidence: list[ImageEvidence]) -> GroundedAnswer:
        usable = [item for item in evidence if item.relevant and item.facts]
        if not usable:
            return GroundedAnswer(answer=REFUSAL_TEXT, refused=True, evidence=evidence)
        statements = [f"{item.facts[0]}{citation_token(item.image_id)}" for item in usable]
        return GroundedAnswer(
            answer="；".join(statements) + "。",
            citations=[item.image_id for item in usable],
            confidence=0.5,
            evidence=evidence,
        )

    def answer(self, question: str, candidates: list[SearchResult], top_k: int = 3) -> GroundedAnswer:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        selected = candidates[:top_k]
        if not selected:
            return GroundedAnswer(answer=REFUSAL_TEXT, refused=True)
        with self.manager.qwen_session() as client:
            evidence = [self._extract_evidence(client, item, question) for item in selected]
            usable = [item for item in evidence if item.relevant and item.facts]
            if not usable:
                return GroundedAnswer(answer=REFUSAL_TEXT, refused=True, evidence=evidence)
            request = (
                "仅根据给定证据回答问题。每个事实后使用 [img_图片ID] 引用；证据不足时拒答。"
                "只输出 JSON：answer、citations、confidence、refused。\n"
                f"问题：{question}\n证据："
                + json.dumps([item.model_dump() for item in usable], ensure_ascii=False)
            )
            try:
                payload = extract_json_object(client.generate_text(request, max_new_tokens=self.max_new_tokens))
                refused = bool(payload.get("refused", False))
                answer = str(payload.get("answer", "")).strip() or REFUSAL_TEXT
                citations = validate_citations(
                    answer,
                    self._string_list(payload.get("citations")),
                    {item.image_id for item in usable},
                    refused=refused,
                )
                return GroundedAnswer(
                    answer=answer,
                    citations=citations,
                    confidence=float(payload.get("confidence", 0.0)),
                    refused=refused,
                    evidence=evidence,
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                return self._fallback_answer(evidence)

    def create_story(self, candidates: list[SearchResult], tone: str = "自然",
                     theme: str = "图文游记") -> VisualStory:
        if not 3 <= len(candidates) <= 8:
            raise ValueError("a visual story requires 3 to 8 selected images")
        question = f"为{theme}提取场景、主体、动作、氛围和可见细节，语气为{tone}"
        with self.manager.qwen_session() as client:
            evidence = [self._extract_evidence(client, item, question) for item in candidates]
            request = (
                "根据按顺序提供的图片证据生成图文故事。不得虚构具体地点、身份和真实经历。"
                "只输出 JSON：title 和 sections；每个 section 包含 image_id、subtitle、text，"
                "section 顺序和图片顺序必须一致。\n"
                f"主题：{theme}\n语气：{tone}\n证据："
                + json.dumps([item.model_dump() for item in evidence], ensure_ascii=False)
            )
            try:
                payload = extract_json_object(client.generate_text(request, max_new_tokens=768))
                story = VisualStory.model_validate(payload)
                expected = [item.image_id for item in candidates]
                actual = [section.image_id for section in story.sections]
                if actual != expected:
                    raise ValueError("story sections do not preserve selected image order")
                return story
            except (AttributeError, KeyError, TypeError, ValueError):
                sections = []
                for index, item in enumerate(evidence, start=1):
                    detail = item.facts[0] if item.facts else "这张图片的可见信息不足。"
                    sections.append(StorySection(
                        image_id=item.image_id,
                        subtitle=f"片段 {index}",
                        text=f"{detail}{citation_token(item.image_id)}",
                    ))
                return VisualStory(title=theme, sections=sections)
