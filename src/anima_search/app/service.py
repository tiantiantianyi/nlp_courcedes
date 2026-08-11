from __future__ import annotations

from pathlib import Path

from anima_search.annotation.validation import extract_json_object
from anima_search.generation.prompt_builder import SDPromptBuilder
from anima_search.m7.service import M7Service
from anima_search.schemas import ImageAnnotation, SearchResult


class SearchService:
    def __init__(self, config: dict, parser: object, searcher: object, manager: object,
                 annotations: dict[str, ImageAnnotation], reranker_prompt: str,
                 content_prompt: str, sd_prompt: str) -> None:
        self.config, self.parser, self.searcher, self.manager = config, parser, searcher, manager
        self.annotations = annotations
        self.reranker_prompt, self.content_prompt, self.sd_prompt = reranker_prompt, content_prompt, sd_prompt
        m7_settings = config.get("m7", {})
        self.m7 = M7Service(
            manager,
            config["project_root"],
            annotations,
            max_new_tokens=int(m7_settings.get("max_new_tokens", 384)),
            max_story_gaps=int(m7_settings.get("max_story_gaps", 2)),
            gap_scene_similarity_threshold=float(
                m7_settings.get("gap_scene_similarity_threshold", 0.15)
            ),
        )

    def _model_path(self, key: str, label: str) -> Path:
        configured = self.config.get("models", {}).get(key)
        if not configured:
            raise FileNotFoundError(
                f"{label} model path is not configured; set models.{key} in configs/default.yaml"
            )
        path = Path(configured)
        if not path.is_absolute():
            path = Path(self.config["project_root"]) / path
        if not path.exists():
            raise FileNotFoundError(
                f"{label} model directory does not exist: {path}; "
                f"download the model locally or update models.{key}"
            )
        return path

    def _annotation(self, image_id: str) -> ImageAnnotation:
        if image_id not in self.annotations:
            raise KeyError(
                f"unknown image_id {image_id!r}; select an ID returned by the current index"
            )
        return self.annotations[image_id]

    def _image_path(self, annotation: ImageAnnotation) -> Path:
        path = Path(self.config["project_root"]) / annotation.relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"image file for {annotation.image_id!r} does not exist: {path}; "
                "check data.train_dir/data.val_dir and relative_path normalization"
            )
        return path

    def release_retrieval_encoders(self) -> list[str]:
        released: list[str] = []
        for name, index in getattr(self.searcher, "indexes", {}).items():
            if hasattr(index, "unload_encoder"):
                index.unload_encoder()
                released.append(str(name))
            elif hasattr(index, "model") and getattr(index, "model") is not None:
                index.model = None
                released.append(str(name))
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return released

    def search(self, query: str, use_reranker: bool = False) -> list[SearchResult]:
        settings = self.config["retrieval"]
        use_parser_llm = bool(settings.get("query_parser_use_llm", False))
        target_count = settings["rerank_count"] if use_reranker else settings["result_count"]
        if use_parser_llm:
            self._model_path("qwen_vl", "Qwen-VL")
            with self.manager.qwen_session() as qwen:
                parsed = self.parser.parse(query, qwen)
            self.manager.unload_all()
        else:
            parsed = self.parser.parse(query)

        results = self.searcher.search(parsed, settings["candidate_count"], target_count)
        if not use_reranker:
            return results[:settings["result_count"]]

        from anima_search.retrieval.listwise_reranker import ListwiseVisualReranker
        from anima_search.retrieval.reranker import VisualReranker

        self.release_retrieval_encoders()
        self._model_path("qwen_vl", "Qwen-VL")
        with self.manager.qwen_session() as qwen:
            method = settings.get("rerank_method", "pointwise")
            project_root = Path(self.config["project_root"])
            if method == "pointwise":
                reranker = VisualReranker(
                    qwen,
                    self.reranker_prompt,
                    project_root,
                    settings["rrf_weight"],
                    settings["vlm_weight"],
                    settings.get("rerank_max_new_tokens", 128),
                )
            elif method == "listwise":
                prompt_path = project_root / settings.get(
                    "rerank_listwise_prompt",
                    "configs/prompts/reranker_listwise.txt",
                )
                reranker = ListwiseVisualReranker(
                    qwen,
                    prompt_path.read_text(encoding="utf-8"),
                    project_root,
                    max_new_tokens=settings.get(
                        "rerank_listwise_max_new_tokens", 768
                    ),
                    columns=settings.get("rerank_listwise_columns", 5),
                    tile_size=settings.get("rerank_listwise_tile_size", 192),
                )
            else:
                raise ValueError(
                    "retrieval.rerank_method must be 'pointwise' or 'listwise'"
                )
            results = reranker.rerank(query, results)
        return results[:settings["result_count"]]

    @staticmethod
    def _selected_candidates(
        candidates: list[SearchResult],
        selected_image_ids: list[str] | None,
        *,
        minimum: int,
        maximum: int,
    ) -> list[SearchResult]:
        if selected_image_ids:
            by_id = {item.image_id: item for item in candidates}
            unknown = [image_id for image_id in selected_image_ids if image_id not in by_id]
            if unknown:
                raise ValueError(f"selected image IDs are not in the current results: {unknown}")
            selected = [by_id[image_id] for image_id in dict.fromkeys(selected_image_ids)]
        else:
            selected = candidates[:maximum]
        if not minimum <= len(selected) <= maximum:
            raise ValueError(
                f"select between {minimum} and {maximum} images; received {len(selected)}"
            )
        return selected

    def answer_with_evidence(
        self,
        question: str,
        candidates: list[SearchResult],
        selected_image_ids: list[str] | None = None,
        *,
        top_k: int = 3,
    ):
        if not question.strip():
            raise ValueError("question must not be empty")
        if not 1 <= top_k <= 3:
            raise ValueError("M7 grounded QA top_k must be between 1 and 3")
        selected = self._selected_candidates(
            candidates,
            selected_image_ids,
            minimum=1,
            maximum=top_k,
        )
        self.release_retrieval_encoders()
        self._model_path("qwen_vl", "Qwen-VL")
        return self.m7.answer(question.strip(), selected, top_k=len(selected))

    def create_visual_story(
        self,
        candidates: list[SearchResult],
        selected_image_ids: list[str],
        *,
        theme: str = "图文游记",
        tone: str = "自然",
        fill_gaps: bool = False,
        seed: int | None = None,
    ):
        selected = self._selected_candidates(
            candidates,
            selected_image_ids,
            minimum=3,
            maximum=8,
        )
        self.release_retrieval_encoders()
        self._model_path("qwen_vl", "Qwen-VL")
        story = self.m7.create_story(
            selected,
            tone=tone,
            theme=theme.strip() or "图文游记",
        )
        if not fill_gaps:
            return story

        base_seed = (
            int(seed)
            if seed is not None
            else int(self.config["generation"]["seed"])
        )
        project_root = Path(self.config["project_root"]).resolve()
        for offset, gap in enumerate(getattr(story, "gaps", [])):
            try:
                output = self.generate_image(
                    gap.generation_prompt,
                    gap.after_image_id,
                    base_seed + offset,
                ).resolve()
                try:
                    relative_path = output.relative_to(project_root).as_posix()
                except ValueError:
                    relative_path = str(output)
                gap.status = "generated"
                gap.generated_image_id = output.stem
                gap.relative_path = relative_path
                gap.error = None
            except Exception as exc:
                gap.status = "failed"
                gap.error = f"{type(exc).__name__}: {exc}"
        return story

    def answer_about_image(self, image_id: str, question: str) -> str:
        from PIL import Image

        annotation = self._annotation(image_id)
        image_path = self._image_path(annotation)
        self.release_retrieval_encoders()
        self._model_path("qwen_vl", "Qwen-VL")
        prompt = f"仅根据图片回答问题；无法确认时明确说不确定。\n问题：{question}"
        with Image.open(image_path) as image:
            prepared = image.copy()
        with self.manager.qwen_session() as qwen:
            return qwen.generate(prepared, prompt, max_new_tokens=384)

    def write_content(self, image_id: str, content_type: str, tone: str) -> dict[str, str]:
        annotation = self._annotation(image_id)
        self.release_retrieval_encoders()
        self._model_path("qwen_vl", "Qwen-VL")
        request = (
            f"{self.content_prompt}\n类型：{content_type}\n语气：{tone}\n"
            f"图片标注：{annotation.model_dump_json()}"
        )
        with self.manager.qwen_session() as qwen:
            return extract_json_object(qwen.generate_text(request, max_new_tokens=512))

    def generate_image(self, query: str, image_id: str | None = None,
                       seed: int | None = None) -> Path:
        annotation = self._annotation(image_id) if image_id else None
        context = annotation.model_dump_json() if annotation else ""
        self.release_retrieval_encoders()
        self._model_path("qwen_vl", "Qwen-VL")
        self._model_path("stable_diffusion", "Stable Diffusion")
        with self.manager.qwen_session() as qwen:
            prompts = SDPromptBuilder(qwen, self.sd_prompt).build(query, context)
        settings = self.config["generation"]
        output = (
            Path(self.config["project_root"])
            / self.config["data"]["artifacts_dir"]
            / "generated"
        )
        with self.manager.sd_session() as generator:
            return generator.generate(
                prompts,
                output,
                seed if seed is not None else settings["seed"],
                settings["width"],
                settings["height"],
                settings["steps"],
                settings["guidance_scale"],
            )
