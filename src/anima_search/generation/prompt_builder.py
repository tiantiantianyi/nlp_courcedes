from __future__ import annotations

from anima_search.annotation.validation import extract_json_object
from anima_search.schemas import GenerationPrompts


class SDPromptBuilder:
    def __init__(self, text_generator: object, template: str) -> None:
        self.text_generator = text_generator
        self.template = template

    def build(self, query: str, context: str = "") -> GenerationPrompts:
        request = f"{self.template}\n用户需求：{query}\n参考内容：{context}"
        for attempt in range(2):
            raw = self.text_generator.generate_text(request)
            try:
                return GenerationPrompts.model_validate(extract_json_object(raw))
            except (KeyError, TypeError, ValueError):
                request += (
                    "\n上一次输出无法解析。请只返回单个 JSON 对象，不要解释、"
                    "不要 Markdown 代码块。"
                )

        # Image generation should remain available when an optional prompt LLM
        # violates the JSON contract. SD 1.5 accepts mixed-language prompts, so
        # preserve the user's request while adding stable photographic anchors.
        positive = (
            "documentary photo, coherent transition, natural light, "
            f"balanced composition, {query.strip()[:24]}"
        )
        negative = (
            "low quality, blurry, deformed, duplicate, text, watermark, logo, "
            "signature, oversaturated"
        )
        return GenerationPrompts(
            positive_prompt=positive,
            negative_prompt=negative,
        )
