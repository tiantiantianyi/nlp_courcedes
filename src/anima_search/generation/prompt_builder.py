from __future__ import annotations

from anima_search.annotation.validation import extract_json_object
from anima_search.schemas import GenerationPrompts


class SDPromptBuilder:
    def __init__(self, text_generator: object, template: str) -> None:
        self.text_generator = text_generator
        self.template = template

    def build(self, query: str, context: str = "") -> GenerationPrompts:
        raw = self.text_generator.generate_text(f"{self.template}\n用户需求：{query}\n参考内容：{context}")
        return GenerationPrompts.model_validate(extract_json_object(raw))
