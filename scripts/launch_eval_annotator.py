from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr

from anima_search.config import load_config, resolve_path
from anima_search.evaluation.manual_set import (
    format_judgments,
    load_relevance_rows,
    load_tasks,
    save_review,
)


CATEGORY_CHOICES = [
    ("简单查询", "simple"),
    ("组合查询", "compositional"),
    ("否定查询", "negative"),
    ("数量查询", "count"),
    ("文字识别查询", "ocr"),
]
DEFAULT_ANNOTATOR = "张添翼"


def _task_form_values(
    task: dict[str, object],
    rows: list[dict[str, object]],
    project_root: Path,
) -> tuple[object, ...]:
    image_path = project_root / str(task["source_relative_path"])
    query_id = str(task["query_id"])
    judgments = format_judgments(rows, query_id) or f"{task['source_image_id']}:2"
    return (
        str(image_path),
        query_id,
        str(task["source_image_id"]),
        str(task.get("text", "")),
        str(task.get("category", "")) or None,
        str(task.get("annotator", "")).strip() or DEFAULT_ANNOTATOR,
        str(task.get("note", "")),
        judgments,
        bool(task.get("reviewed", False)),
    )


def build_app(
    task_path: Path,
    relevance_path: Path,
    project_root: Path,
) -> gr.Blocks:
    def load_position(position: int):
        tasks = load_tasks(task_path)
        rows = load_relevance_rows(relevance_path)
        index = max(0, min(len(tasks) - 1, int(position) - 1))
        task = tasks[index]
        completed = sum(bool(row.get("reviewed")) for row in tasks)
        progress = f"**进度：{completed}/{len(tasks)} 已审核**"
        return (
            index + 1,
            *_task_form_values(task, rows, project_root),
            progress,
            "",
        )

    def save_position(
        position: int,
        text: str,
        category: str,
        annotator: str,
        note: str,
        judgments: str,
        reviewed: bool,
    ):
        index = int(position) - 1
        try:
            task = save_review(
                task_path,
                relevance_path,
                index=index,
                text=text,
                category=category,
                annotator=annotator,
                note=note,
                judgments=judgments,
                reviewed=reviewed,
            )
        except Exception as exc:
            return f"保存失败：{type(exc).__name__}: {exc}", load_position(position)[-2]
        progress = load_position(position)[-2]
        return f"已保存 {task['query_id']}", progress

    def previous(position: int):
        return max(1, int(position) - 1)

    def following(position: int):
        return min(len(load_tasks(task_path)), int(position) + 1)

    task_count = len(load_tasks(task_path))
    with gr.Blocks(title="人工检索评测集标注") as app:
        gr.Markdown(
            "# 人工检索评测集\n"
            "请直接观察原图撰写查询。不要复制自动 caption。"
            "相关性格式为每行 `image_id:grade`，其中 2=高度相关、1=部分相关、0=不相关。"
        )
        with gr.Row():
            previous_button = gr.Button("上一条")
            position = gr.Slider(1, task_count, value=1, step=1, label="任务编号")
            next_button = gr.Button("下一条")
        progress = gr.Markdown()
        with gr.Row():
            source_image = gr.Image(label="查询来源原图", height=520, type="filepath")
            with gr.Column():
                query_id = gr.Textbox(label="Query ID", interactive=False)
                source_image_id = gr.Textbox(label="来源图 ID", interactive=False)
                query_text = gr.Textbox(
                    label="人工查询",
                    lines=3,
                    placeholder="直接看图写自然查询，不要复制模型描述。",
                )
                category = gr.Dropdown(CATEGORY_CHOICES, label="查询类别")
                annotator = gr.Textbox(label="标注者")
                note = gr.Textbox(label="备注", lines=2)
        judgments = gr.Textbox(
            label="相关性判断",
            lines=8,
            placeholder="val-2002:2\nval-2035:1",
        )
        reviewed = gr.Checkbox(label="已由人工审核，可用于正式评测")
        save_button = gr.Button("保存当前任务", variant="primary")
        status = gr.Markdown()

        outputs = [
            position,
            source_image,
            query_id,
            source_image_id,
            query_text,
            category,
            annotator,
            note,
            judgments,
            reviewed,
            progress,
            status,
        ]
        app.load(load_position, position, outputs)
        position.change(load_position, position, outputs)
        previous_button.click(previous, position, position)
        next_button.click(following, position, position)
        save_button.click(
            save_position,
            [position, query_text, category, annotator, note, judgments, reviewed],
            [status, progress],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the human retrieval evaluation annotator.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--queries", type=Path, default=Path("evaluation/manual_val/queries.jsonl"))
    parser.add_argument("--relevance", type=Path, default=Path("evaluation/manual_val/relevance.csv"))
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    app = build_app(args.queries, args.relevance, Path(config["project_root"]))
    app.launch(
        server_port=args.port,
        share=args.share,
        allowed_paths=[
            str(resolve_path(config, str(config["data"]["train_dir"])).resolve()),
            str(resolve_path(config, str(config["data"]["val_dir"])).resolve()),
        ],
    )


if __name__ == "__main__":
    main()
