from __future__ import annotations

from html import escape
from pathlib import Path

import gradio as gr

from anima_search.app.service import SearchService


APP_CSS = """
:root {
  --anima-ink: #172033;
  --anima-muted: #667085;
  --anima-indigo: #5b5bd6;
  --anima-violet: #8b5cf6;
  --anima-card: rgba(255, 255, 255, 0.88);
}
.gradio-container {
  background:
    radial-gradient(circle at 8% 0%, rgba(139, 92, 246, .12), transparent 30rem),
    radial-gradient(circle at 92% 8%, rgba(59, 130, 246, .11), transparent 28rem),
    #f7f8fc !important;
  color: var(--anima-ink);
}
.anima-shell { max-width: 1440px; margin: 0 auto; }
.anima-hero {
  position: relative;
  overflow: hidden;
  padding: 2.4rem 2.5rem;
  margin: .8rem 0 1.2rem;
  border-radius: 28px;
  color: white;
  background: linear-gradient(125deg, #25265e 0%, #5b5bd6 48%, #8b5cf6 100%);
  box-shadow: 0 24px 70px rgba(67, 56, 202, .22);
}
.anima-hero:after {
  content: "";
  position: absolute;
  width: 19rem;
  height: 19rem;
  right: -5rem;
  top: -9rem;
  border: 1px solid rgba(255,255,255,.35);
  border-radius: 50%;
  box-shadow: 0 0 0 42px rgba(255,255,255,.05), 0 0 0 84px rgba(255,255,255,.04);
}
.anima-kicker {
  font-size: .75rem;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
  opacity: .76;
}
.anima-hero h1 { margin: .35rem 0 .55rem; font-size: clamp(2rem, 5vw, 3.7rem); line-height: 1; }
.anima-hero p { max-width: 47rem; margin: 0; font-size: 1rem; opacity: .88; line-height: 1.7; }
.anima-pills { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.25rem; }
.anima-pill {
  padding: .42rem .72rem;
  border: 1px solid rgba(255,255,255,.24);
  border-radius: 999px;
  background: rgba(255,255,255,.11);
  font-size: .76rem;
  font-weight: 650;
}
.anima-panel {
  padding: .45rem !important;
  border: 1px solid rgba(91, 91, 214, .10) !important;
  border-radius: 22px !important;
  background: var(--anima-card) !important;
  box-shadow: 0 10px 35px rgba(34, 46, 80, .06) !important;
}
.anima-gallery {
  border: 0 !important;
  border-radius: 20px !important;
  overflow: hidden;
}
.anima-gallery img { transition: transform .25s ease, filter .25s ease; }
.anima-gallery img:hover { transform: scale(1.025); filter: saturate(1.08); }
.anima-section-note { color: var(--anima-muted); font-size: .9rem; line-height: 1.65; }
.story-empty {
  min-height: 25rem;
  display: grid;
  place-items: center;
  padding: 2rem;
  border: 1px dashed rgba(91, 91, 214, .3);
  border-radius: 22px;
  color: var(--anima-muted);
  background: linear-gradient(145deg, rgba(255,255,255,.85), rgba(245,243,255,.75));
  text-align: center;
}
.story-empty strong { display: block; color: var(--anima-ink); font-size: 1.15rem; margin-bottom: .35rem; }
.story-head { margin-bottom: 1.25rem; }
.story-head h2 { margin: 0 0 .35rem; font-size: 1.7rem; }
.story-head p { color: var(--anima-muted); margin: 0; line-height: 1.6; }
.story-timeline { position: relative; padding: .4rem .25rem .4rem 1.45rem; }
.story-timeline:before {
  content: "";
  position: absolute;
  left: .42rem;
  top: .8rem;
  bottom: .8rem;
  width: 2px;
  background: linear-gradient(#6366f1, #c4b5fd);
}
.story-node, .story-gap-card {
  position: relative;
  margin: 0 0 1rem;
  padding: 1rem 1.05rem;
  border-radius: 17px;
  background: white;
  border: 1px solid #eaecf5;
  box-shadow: 0 7px 20px rgba(30, 41, 59, .055);
}
.story-node:before, .story-gap-card:before {
  content: "";
  position: absolute;
  left: -1.36rem;
  top: 1.2rem;
  width: .66rem;
  height: .66rem;
  border-radius: 50%;
  background: #6366f1;
  box-shadow: 0 0 0 4px #eef2ff;
}
.story-gap-card {
  border: 1px dashed #c4b5fd;
  background: linear-gradient(145deg, #fafaff, #f5f3ff);
}
.story-gap-card:before { background: #a78bfa; }
.story-node h3, .story-gap-card h4 { margin: .35rem 0 .4rem; color: var(--anima-ink); }
.story-node p, .story-gap-card p { margin: 0; color: #475467; line-height: 1.65; }
.story-meta { color: #7c8294; font-size: .74rem; font-weight: 650; letter-spacing: .02em; }
.story-badge {
  display: inline-flex;
  align-items: center;
  padding: .24rem .52rem;
  border-radius: 999px;
  font-size: .7rem;
  font-weight: 800;
  background: #eef2ff;
  color: #4f46e5;
}
.story-badge.ai { background: #f3e8ff; color: #7e22ce; }
.story-badge.failed { background: #fef2f2; color: #b42318; }
.story-disclaimer {
  margin-top: .9rem;
  padding: .8rem 1rem;
  border-radius: 13px;
  background: #f8fafc;
  color: #667085;
  font-size: .78rem;
  line-height: 1.55;
}
.primary-action button { min-height: 46px; font-weight: 750 !important; }
footer { display: none !important; }
@media (max-width: 760px) {
  .anima-hero { padding: 1.7rem 1.35rem; border-radius: 22px; }
  .anima-hero h1 { font-size: 2.25rem; }
}
"""


EMPTY_STORY_HTML = """
<div class="story-empty">
  <div><strong>你的视觉故事会出现在这里</strong>
  搜索并选择 3–8 张图片，系统将自动排序、识别叙事断点，并按需生成过渡图。</div>
</div>
"""


def build_app(service: SearchService) -> gr.Blocks:
    root = Path(service.config["project_root"]).resolve()
    generation_seed = int(service.config.get("generation", {}).get("seed", 42))
    fill_gaps_default = bool(
        service.config.get("m7", {}).get("fill_gaps_default", False)
    )

    def result_path(relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else (root / path).resolve()

    def run_search(query: str, rerank: bool):
        results = service.search(query, rerank)
        gallery = []
        details = []
        for item in results:
            caption = f"{item.image_id} · 融合 {item.fused_score:.4f}"
            if item.rerank_score is not None:
                caption += f" · VLM {item.rerank_score:.0f}"
            gallery.append((str(result_path(item.relative_path)), caption))
            details.append(item.model_dump())
        image_ids = [item.image_id for item in results]
        selected = image_ids[:min(3, len(image_ids))]
        return (
            gallery,
            details,
            results,
            gr.CheckboxGroup(choices=image_ids, value=selected),
        )

    def select_result(results, event: gr.SelectData):
        if not results or not isinstance(event.index, int) or event.index >= len(results):
            return "", None, {}
        item = results[event.index]
        annotation = service.annotations.get(item.image_id)
        annotation_payload = (
            annotation.model_dump()
            if annotation is not None and hasattr(annotation, "model_dump")
            else {"status": "当前图片没有正式结构化标注"}
        )
        return item.image_id, str(result_path(item.relative_path)), annotation_payload

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

    def story_html(story) -> str:
        gap_by_after: dict[str, list[object]] = {}
        for gap in story.gaps:
            gap_by_after.setdefault(gap.after_image_id, []).append(gap)
        nodes = []
        for position, section in enumerate(story.sections, start=1):
            source_badge = (
                '<span class="story-badge ai">✨ AI 生成</span>'
                if section.ai_generated
                else '<span class="story-badge">原始图片</span>'
            )
            nodes.append(
                '<article class="story-node">'
                f'{source_badge}<div class="story-meta">片段 {position:02d} · '
                f'{escape(section.image_id)}</div>'
                f'<h3>{escape(section.subtitle)}</h3>'
                f'<p>{escape(section.text)}</p></article>'
            )
            for gap in gap_by_after.get(section.image_id, []):
                if gap.status == "generated":
                    badge = '<span class="story-badge ai">✨ AI 生成</span>'
                    detail = "过渡图已生成，并与原始图片分开展示。"
                elif gap.status == "failed":
                    badge = '<span class="story-badge failed">生成失败</span>'
                    detail = escape(gap.error or "请检查本地生成模型。")
                else:
                    badge = '<span class="story-badge ai">缺图占位 · 待补全</span>'
                    detail = "勾选“自动检测并补全缺图”后可调用本地生成模型。"
                nodes.append(
                    '<article class="story-gap-card">'
                    f'{badge}<div class="story-meta">{escape(gap.gap_id)}</div>'
                    '<h4>叙事过渡画面</h4>'
                    f'<p>{escape(gap.reason)}<br>{detail}</p></article>'
                )
        return (
            '<section class="story-head">'
            f'<h2>{escape(story.title)}</h2>'
            f'<p>{escape(story.ordering_reason)}</p></section>'
            f'<div class="story-timeline">{"".join(nodes)}</div>'
            f'<div class="story-disclaimer">{escape(story.disclaimer)}</div>'
        )

    def create_story(
        results,
        selected_ids,
        theme: str,
        tone: str,
        fill_gaps: bool,
        seed: int,
    ):
        if not results:
            return EMPTY_STORY_HTML, [], {"error": "没有检索结果"}
        try:
            story = service.create_visual_story(
                results,
                list(selected_ids or []),
                theme=theme,
                tone=tone,
                fill_gaps=bool(fill_gaps),
                seed=int(seed),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return (
                f'<div class="story-empty"><div><strong>故事生成失败</strong>'
                f'{escape(error)}</div></div>',
                [],
                {"error": str(exc)},
            )

        by_id = {item.image_id: item for item in results}
        gap_by_after: dict[str, list[object]] = {}
        for gap in story.gaps:
            gap_by_after.setdefault(gap.after_image_id, []).append(gap)
        gallery = []
        for section in story.sections:
            item = by_id.get(section.image_id)
            if item is not None:
                gallery.append(
                    (
                        str(result_path(item.relative_path)),
                        f"🖼️ 原图 · {section.image_id} · {section.subtitle}",
                    )
                )
            for gap in gap_by_after.get(section.image_id, []):
                if gap.status == "generated" and gap.relative_path:
                    generated_path = result_path(gap.relative_path)
                    if generated_path.is_file():
                        gallery.append(
                            (
                                str(generated_path),
                                f"✨ AI 生成 · {gap.gap_id} · {gap.reason}",
                            )
                        )
        return story_html(story), gallery, story.model_dump()

    with gr.Blocks(
        title="Anima · 视觉语言相册",
        fill_width=True,
    ) as app:
        selected_id = gr.State("")
        search_results = gr.State([])
        with gr.Column(elem_classes=["anima-shell"]):
            gr.HTML(
                """
                <header class="anima-hero">
                  <div class="anima-kicker">Visual Language Album · M3–M7</div>
                  <h1>Anima</h1>
                  <p>让自然语言成为相册的入口：检索、理解、证据问答，
                  再把零散影像自动编排为可追溯的视觉故事。</p>
                  <div class="anima-pills">
                    <span class="anima-pill">多路检索</span>
                    <span class="anima-pill">Qwen3-VL 重排</span>
                    <span class="anima-pill">证据引用</span>
                    <span class="anima-pill">缺图补全</span>
                  </div>
                </header>
                """
            )
            with gr.Tabs():
                with gr.Tab("探索相册"):
                    with gr.Column(elem_classes=["anima-panel"]):
                        with gr.Row():
                            query = gr.Textbox(
                                label="用自然语言寻找图片",
                                placeholder="例如：不要人物，寻找冷色调的雨夜城市",
                                scale=7,
                            )
                            use_rerank = gr.Checkbox(
                                value=bool(
                                    service.config["retrieval"].get(
                                        "rerank_default", False
                                    )
                                ),
                                label="Qwen3-VL 视觉重排",
                                scale=2,
                            )
                            search_button = gr.Button(
                                "开始探索",
                                variant="primary",
                                scale=1,
                                elem_classes=["primary-action"],
                            )
                    gallery = gr.Gallery(
                        label="检索结果",
                        columns=4,
                        rows=2,
                        height=570,
                        object_fit="cover",
                        elem_classes=["anima-gallery"],
                    )
                    with gr.Accordion("查看融合分数与匹配解释", open=False):
                        result_json = gr.JSON()

                with gr.Tab("图片详情"):
                    with gr.Row():
                        selected_image = gr.Image(
                            label="当前图片",
                            height=500,
                            elem_classes=["anima-panel"],
                        )
                        with gr.Column(elem_classes=["anima-panel"]):
                            annotation_json = gr.JSON(label="结构化标注")
                            question = gr.Textbox(
                                label="关于这张图片的问题",
                                placeholder="例如：画面中的主要物体是什么？",
                            )
                            answer_button = gr.Button("基于图片回答", variant="primary")
                            answer_text = gr.Textbox(label="回答", lines=4)
                            answer_button.click(
                                answer,
                                [selected_id, question],
                                answer_text,
                            )

                with gr.Tab("视觉故事 · M7"):
                    gr.Markdown(
                        "从当前结果选择图片：证据问答支持 1–3 张；视觉故事支持 "
                        "3–8 张，并按时间与场景自动排序。",
                        elem_classes=["anima-section-note"],
                    )
                    candidate_ids = gr.CheckboxGroup(
                        choices=[],
                        label="当前故事候选",
                        info="每次搜索后自动选择前三张，也可以手动调整。",
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=4, elem_classes=["anima-panel"]):
                            gr.Markdown("### 证据问答")
                            grounded_question = gr.Textbox(
                                label="跨图片问题",
                                placeholder="例如：这些图片中哪些出现了车辆？",
                                lines=2,
                            )
                            grounded_button = gr.Button(
                                "生成带引用回答",
                                variant="secondary",
                            )
                            grounded_markdown = gr.Markdown()
                            with gr.Accordion("逐图证据与引用 JSON", open=False):
                                grounded_json = gr.JSON()
                            grounded_button.click(
                                answer_with_evidence,
                                [search_results, candidate_ids, grounded_question],
                                [grounded_markdown, grounded_json],
                            )
                            gr.Markdown("### 故事控制")
                            story_theme = gr.Textbox(
                                value="图文游记",
                                label="故事主题",
                            )
                            story_tone = gr.Dropdown(
                                ["自然", "治愈", "纪实", "幽默"],
                                value="自然",
                                label="叙事语气",
                            )
                            fill_story_gaps = gr.Checkbox(
                                value=fill_gaps_default,
                                label="自动检测并补全缺图",
                                info="会串行调用本地 Qwen 与 Stable Diffusion。",
                            )
                            story_seed = gr.Number(
                                value=generation_seed,
                                precision=0,
                                label="补图 Seed",
                            )
                            story_button = gr.Button(
                                "自动编排视觉故事",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )
                        with gr.Column(scale=7, elem_classes=["anima-panel"]):
                            story_timeline = gr.HTML(value=EMPTY_STORY_HTML)
                            story_gallery = gr.Gallery(
                                label="故事影像序列",
                                columns=3,
                                height=360,
                                object_fit="cover",
                                elem_classes=["anima-gallery"],
                            )
                            with gr.Accordion("查看故事结构 JSON", open=False):
                                story_json = gr.JSON()
                    story_button.click(
                        create_story,
                        [
                            search_results,
                            candidate_ids,
                            story_theme,
                            story_tone,
                            fill_story_gaps,
                            story_seed,
                        ],
                        [story_timeline, story_gallery, story_json],
                    )

                with gr.Tab("创作工作台"):
                    with gr.Row():
                        with gr.Column(elem_classes=["anima-panel"]):
                            content_type = gr.Dropdown(
                                ["title", "moments", "story"],
                                value="moments",
                                label="内容类型",
                            )
                            tone = gr.Dropdown(
                                ["formal", "humorous", "healing"],
                                value="healing",
                                label="语气",
                            )
                            content_button = gr.Button("生成文案", variant="secondary")
                            content_json = gr.JSON(label="生成文案")
                            content_button.click(
                                write,
                                [selected_id, content_type, tone],
                                content_json,
                            )
                        with gr.Column(elem_classes=["anima-panel"]):
                            generation_query = gr.Textbox(
                                label="补图需求",
                                placeholder="例如：保留雨夜街道，增加电影感霓虹灯",
                                lines=2,
                            )
                            seed = gr.Number(
                                value=generation_seed,
                                precision=0,
                                label="Seed",
                            )
                            generation_button = gr.Button(
                                "生成补充图片",
                                variant="primary",
                            )
                            generated_image = gr.Image(
                                label="✨ AI 生成图片",
                                height=420,
                            )
                            generation_button.click(
                                generate,
                                [generation_query, selected_id, seed],
                                generated_image,
                            )

                with gr.Tab("实验信息"):
                    gr.JSON(
                        value={
                            "qwen": service.config.get("models", {}).get("qwen_vl"),
                            "stable_diffusion": service.config.get("models", {}).get(
                                "stable_diffusion"
                            ),
                            "embedder": service.config.get("models", {}).get("embedder"),
                            "prompt_version": service.config.get("annotation", {}).get(
                                "prompt_version"
                            ),
                            "m7_fill_gaps_default": fill_gaps_default,
                        },
                        label="当前运行配置",
                    )

            search_button.click(
                run_search,
                [query, use_rerank],
                [gallery, result_json, search_results, candidate_ids],
            )
            gallery.select(
                select_result,
                [search_results],
                [selected_id, selected_image, annotation_json],
            )
    return app
