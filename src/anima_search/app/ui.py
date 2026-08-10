from __future__ import annotations

from pathlib import Path

import gradio as gr

from anima_search.app.service import SearchService


def build_app(service: SearchService) -> gr.Blocks:
    root = Path(service.config["project_root"])

    def run_search(query: str, rerank: bool):
        results = service.search(query, rerank)
        gallery = []
        details = []
        for item in results:
            caption = f"{item.image_id} | RRF {item.fused_score:.4f}"
            if item.rerank_score is not None:
                caption += f" | VLM {item.rerank_score:.0f}"
            gallery.append((str(root / item.relative_path), caption))
            details.append(item.model_dump())
        image_ids = [item.image_id for item in results]
        selected = image_ids[:min(3, len(image_ids))]
        return gallery, details, results, gr.CheckboxGroup(choices=image_ids, value=selected)

    def select_result(results, event: gr.SelectData):
        if not results or event.index >= len(results):
            return "", None, {}
        item = results[event.index]
        annotation = service.annotations[item.image_id]
        return item.image_id, str(root / item.relative_path), annotation.model_dump()

    def answer(image_id: str, question: str):
        return service.answer_about_image(image_id, question) if image_id else "请先选择图片。"

    def write(image_id: str, content_type: str, tone: str):
        if not image_id:
            return {"error": "请先选择图片。"}
        return service.write_content(image_id, content_type, tone)

    def generate(query: str, image_id: str, seed: int):
        return str(service.generate_image(query, image_id or None, int(seed)))

    def answer_with_evidence(results, selected_ids, question: str):
        if not results:
            return "请先搜索图片。", {"error": "没有检索结果"}
        try:
            answer_result = service.answer_with_evidence(
                question,
                results,
                list(selected_ids or []),
                top_k=3,
            )
        except Exception as exc:
            return f"回答失败：{type(exc).__name__}: {exc}", {"error": str(exc)}
        citation_text = "、".join(
            f"[img_{image_id}]" for image_id in answer_result.citations
        ) or "无"
        markdown = (
            f"{answer_result.answer}\n\n"
            f"**引用：** {citation_text}　"
            f"**置信度：** {answer_result.confidence:.2f}　"
            f"**拒答：** {'是' if answer_result.refused else '否'}"
        )
        return markdown, answer_result.model_dump()

    def create_story(results, selected_ids, theme: str, tone: str):
        if not results:
            return "请先搜索并选择图片。", {"error": "没有检索结果"}
        try:
            story = service.create_visual_story(
                results,
                list(selected_ids or []),
                theme=theme,
                tone=tone,
            )
        except Exception as exc:
            return f"故事生成失败：{type(exc).__name__}: {exc}", {"error": str(exc)}
        sections = [
            f"### {section.subtitle}\n\n{section.text}\n\n`[img_{section.image_id}]`"
            for section in story.sections
        ]
        markdown = f"# {story.title}\n\n" + "\n\n".join(sections)
        markdown += f"\n\n> {story.disclaimer}"
        return markdown, story.model_dump()

    with gr.Blocks(title="Anima 智能图文搜索") as app:
        selected_id = gr.State("")
        search_results = gr.State([])
        gr.Markdown("# Anima 智能图文搜索与内容生成")
        with gr.Tab("智能搜索"):
            with gr.Row():
                query = gr.Textbox(label="中文自然语言查询", placeholder="例如：不要人物，寻找冷色调的雨夜城市")
                use_rerank = gr.Checkbox(
                    value=bool(service.config["retrieval"].get("rerank_default", False)),
                    label="Qwen3-VL 视觉重排",
                )
                search_button = gr.Button("搜索", variant="primary")
            gallery = gr.Gallery(label="检索结果", columns=4, rows=2, height=560, object_fit="contain")
            result_json = gr.JSON(label="分数与匹配解释")
        with gr.Tab("图片详情"):
            with gr.Row():
                selected_image = gr.Image(label="当前图片", height=460)
                annotation_json = gr.JSON(label="结构化标注")
            question = gr.Textbox(label="关于图片的问题")
            answer_button = gr.Button("回答")
            answer_text = gr.Textbox(label="回答")
            answer_button.click(answer, [selected_id, question], answer_text)
        with gr.Tab("M7 证据问答与视觉故事"):
            gr.Markdown(
                "从当前检索结果中选择图片。问答支持 1–3 张，视觉故事要求 3–8 张；"
                "所有回答引用必须来自当前选择。"
            )
            candidate_ids = gr.CheckboxGroup(
                choices=[],
                label="候选图片 ID",
                info="每次搜索后自动选择前三张，可按需要调整。",
            )
            grounded_question = gr.Textbox(
                label="跨图片问题",
                placeholder="例如：这些图片中哪些出现了车辆？",
            )
            grounded_button = gr.Button("生成带引用回答", variant="primary")
            grounded_markdown = gr.Markdown()
            grounded_json = gr.JSON(label="回答、引用与逐图证据")
            grounded_button.click(
                answer_with_evidence,
                [search_results, candidate_ids, grounded_question],
                [grounded_markdown, grounded_json],
            )
            with gr.Row():
                story_theme = gr.Textbox(value="图文游记", label="故事主题")
                story_tone = gr.Dropdown(
                    ["自然", "治愈", "纪实", "幽默"],
                    value="自然",
                    label="语气",
                )
                story_button = gr.Button("生成视觉故事")
            story_markdown = gr.Markdown()
            story_json = gr.JSON(label="故事结构")
            story_button.click(
                create_story,
                [search_results, candidate_ids, story_theme, story_tone],
                [story_markdown, story_json],
            )
        with gr.Tab("内容生成"):
            with gr.Row():
                content_type = gr.Dropdown(["title", "moments", "story"], value="moments", label="内容类型")
                tone = gr.Dropdown(["formal", "humorous", "healing"], value="healing", label="语气")
                content_button = gr.Button("生成文案")
            content_json = gr.JSON(label="生成文案")
            content_button.click(write, [selected_id, content_type, tone], content_json)
            generation_query = gr.Textbox(label="补图需求", placeholder="例如：保留雨夜街道，增加电影感霓虹灯")
            seed = gr.Number(value=service.config["generation"]["seed"], precision=0, label="Seed")
            generation_button = gr.Button("生成补充图片")
            generated_image = gr.Image(label="AI 生成图片", height=460)
            generation_button.click(generate, [generation_query, selected_id, seed], generated_image)
        with gr.Tab("实验信息"):
            gr.JSON(value={
                "qwen": service.config["models"]["qwen_vl"],
                "stable_diffusion": service.config["models"]["stable_diffusion"],
                "embedder": service.config["models"]["embedder"],
                "prompt_version": service.config["annotation"]["prompt_version"],
            }, label="当前配置")
        search_button.click(
            run_search,
            [query, use_rerank],
            [gallery, result_json, search_results, candidate_ids],
        )
        gallery.select(select_result, [search_results], [selected_id, selected_image, annotation_json])
    return app
