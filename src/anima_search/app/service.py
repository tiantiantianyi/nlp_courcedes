from __future__ import annotations

from pathlib import Path

from anima_search.annotation.validation import extract_json_object
from anima_search.generation.prompt_builder import SDPromptBuilder
from anima_search.schemas import ImageAnnotation, SearchResult


class SearchService:
    def __init__(self, config: dict, parser: object, searcher: object, manager: object,
                 annotations: dict[str, ImageAnnotation], reranker_prompt: str,
                 content_prompt: str, sd_prompt: str) -> None:
        self.config, self.parser, self.searcher, self.manager = config, parser, searcher, manager
        self.annotations = annotations
        self.reranker_prompt, self.content_prompt, self.sd_prompt = reranker_prompt, content_prompt, sd_prompt

    def search(self, query: str, use_reranker: bool = True) -> list[SearchResult]:
        settings = self.config["retrieval"]
        use_parser_llm = bool(settings.get("query_parser_use_llm", False))
        target_count = settings["rerank_count"] if use_reranker else settings["result_count"]
        if use_reranker or use_parser_llm:
            with self.manager.qwen_session() as qwen:
                parsed = self.parser.parse(query, qwen if use_parser_llm else None)
                results = self.searcher.search(parsed, settings["candidate_count"], target_count)
                if not use_reranker:
                    return results[:settings["result_count"]]
                from anima_search.retrieval.reranker import VisualReranker
                reranker = VisualReranker(qwen, self.reranker_prompt,
                    Path(self.config["project_root"]), settings["rrf_weight"], settings["vlm_weight"],
                    settings.get("rerank_max_new_tokens", 128))
                results = reranker.rerank(query, results)
            return results[:settings["result_count"]]

        parsed = self.parser.parse(query)
        return self.searcher.search(parsed, settings["candidate_count"], settings["result_count"])

    def answer_about_image(self, image_id: str, question: str) -> str:
        from PIL import Image
        annotation = self.annotations[image_id]
        image = Image.open(Path(self.config["project_root"]) / annotation.relative_path)
        prompt = f"仅根据图片回答问题；无法确认时明确说不确定。\n问题：{question}"
        with self.manager.qwen_session() as qwen:
            return qwen.generate(image, prompt, max_new_tokens=384)

    def write_content(self, image_id: str, content_type: str, tone: str) -> dict[str, str]:
        annotation = self.annotations[image_id]
        request = f"{self.content_prompt}\n类型：{content_type}\n语气：{tone}\n图片标注：{annotation.model_dump_json()}"
        with self.manager.qwen_session() as qwen:
            return extract_json_object(qwen.generate_text(request, max_new_tokens=512))

    def generate_image(self, query: str, image_id: str | None = None, seed: int | None = None) -> Path:
        context = self.annotations[image_id].model_dump_json() if image_id else ""
        with self.manager.qwen_session() as qwen:
            prompts = SDPromptBuilder(qwen, self.sd_prompt).build(query, context)
        settings = self.config["generation"]
        output = Path(self.config["project_root"]) / self.config["data"]["artifacts_dir"] / "generated"
        with self.manager.sd_session() as generator:
            return generator.generate(prompts, output, seed if seed is not None else settings["seed"],
                settings["width"], settings["height"], settings["steps"], settings["guidance_scale"])
