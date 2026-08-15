from anima_search.generation.prompt_builder import SDPromptBuilder


class TextGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls = 0

    def generate_text(self, prompt: str) -> str:
        self.calls += 1
        return next(self.outputs)


def test_prompt_builder_retries_invalid_json_once():
    generator = TextGenerator([
        "not json",
        '{"positive_prompt":"city at dusk","negative_prompt":"watermark"}',
    ])

    prompts = SDPromptBuilder(generator, "template").build("黄昏城市")

    assert generator.calls == 2
    assert prompts.positive_prompt == "city at dusk"


def test_prompt_builder_falls_back_when_llm_breaks_json_contract():
    generator = TextGenerator(["invalid", "still invalid"])

    prompts = SDPromptBuilder(generator, "template").build("自然过渡画面")

    assert generator.calls == 2
    assert "自然过渡画面" in prompts.positive_prompt
    assert "watermark" in prompts.negative_prompt
