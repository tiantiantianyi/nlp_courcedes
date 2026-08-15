from __future__ import annotations

from pathlib import Path

import gradio as gr


def build_mock_app(service: object) -> gr.Blocks:
    root = Path(service.config["project_root"])

    def run_search(query: str):
        results = service.search(query, use_reranker=False)
        gallery = [
            (str(root / item.relative_path), f"{item.image_id} | MOCK {item.fused_score:.4f}")
            for item in results
        ]
        return gallery, [item.model_dump() for item in results]

    with gr.Blocks(title="Anima 无标注 Mock 演示") as app:
        gr.Markdown(
            "# Anima 无标注 Mock 演示\n"
            "> 该模式只验证 UI 与接口，结果顺序不代表检索相关性。"
        )
        with gr.Row():
            query = gr.Textbox(label="查询", value="夜晚街道")
            button = gr.Button("模拟搜索", variant="primary")
        gallery = gr.Gallery(label="模拟结果", columns=4, rows=2, height=560)
        details = gr.JSON(label="SearchResult 契约")
        button.click(run_search, query, [gallery, details])
    return app
