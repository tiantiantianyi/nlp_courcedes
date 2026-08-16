from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr

from anima_search.config import load_config, resolve_path
from anima_search.evaluation.candidate_review import (
    load_candidate_pool,
    load_candidate_relevance,
    render_candidate_contact_sheet,
    review_progress,
    save_candidate_review,
)


DEFAULT_ANNOTATOR = "张添翼"
CATEGORY_LABELS = {
    "simple": "简单查询",
    "compositional": "组合查询",
    "negative": "否定查询",
    "count": "数量查询",
    "ocr": "文字识别查询",
}


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _form_values(
    pool_row: dict[str, object],
    rows: list[dict[str, object]],
    project_root: Path,
) -> tuple[object, ...]:
    query_id = str(pool_row["query_id"])
    saved = [row for row in rows if str(row.get("query_id")) == query_id]
    by_image = {str(row.get("image_id")): row for row in saved}
    grade_lines: list[str] = []
    for candidate in list(pool_row["candidates"]):
        image_id = str(candidate["image_id"])
        saved_row = by_image.get(image_id)
        if saved_row is not None:
            grade = str(saved_row.get("relevance", ""))
        elif image_id == str(pool_row["source_image_id"]):
            grade = "2"
        else:
            grade = ""
        grade_lines.append(f"{image_id}:{grade}")

    annotator = next(
        (
            str(row.get("annotator", "")).strip()
            for row in saved
            if str(row.get("annotator", "")).strip()
        ),
        DEFAULT_ANNOTATOR,
    )
    note = next((str(row.get("note", "")) for row in saved), "")
    reviewed = bool(saved) and len(saved) == len(list(pool_row["candidates"])) and all(
        _is_true(row.get("reviewed")) for row in saved
    )
    return (
        render_candidate_contact_sheet(pool_row, project_root),
        query_id,
        str(pool_row.get("text", "")),
        CATEGORY_LABELS.get(
            str(pool_row.get("category", "")), str(pool_row.get("category", ""))
        ),
        str(pool_row.get("source_image_id", "")),
        annotator,
        note,
        "\n".join(grade_lines),
        reviewed,
    )


def build_app(pool_path: Path, output_path: Path, project_root: Path) -> gr.Blocks:
    pool = load_candidate_pool(pool_path)

    def progress_text() -> str:
        complete, total = review_progress(pool, load_candidate_relevance(output_path))
        return f"**进度：{complete}/{total} 个查询已完成候选级审核**"

    def load_position(position: int):
        index = max(0, min(len(pool) - 1, int(position) - 1))
        return (
            index + 1,
            *_form_values(
                pool[index], load_candidate_relevance(output_path), project_root
            ),
            progress_text(),
            "",
        )

    def save_position(
        position: int,
        annotator: str,
        note: str,
        grades_text: str,
        reviewed: bool,
    ):
        index = int(position) - 1
        try:
            save_candidate_review(
                pool,
                output_path,
                index=index,
                grades_text=grades_text,
                annotator=annotator,
                note=note,
                reviewed=reviewed,
            )
        except Exception as exc:
            return f"保存失败：{type(exc).__name__}: {exc}", progress_text()
        return f"已保存 {pool[index]['query_id']}", progress_text()

    def previous(position: int) -> int:
        return max(1, int(position) - 1)

    def following(position: int) -> int:
        return min(len(pool), int(position) + 1)

    with gr.Blocks(title="正式候选相关性标注") as app:
        gr.Markdown(
            "# 正式候选相关性标注（0/1/2）\n"
            "请逐图判断：`2=高度相关`、`1=部分相关`、`0=不相关`。"
            "绿色边框是来源图，必须保持为 2。每个候选都必须显式填写等级；"
            "模型分数和检索名次不能代替人工判断。"
        )
        with gr.Row():
            previous_button = gr.Button("上一条")
            position = gr.Slider(1, len(pool), value=1, step=1, label="任务编号")
            next_button = gr.Button("下一条")
        progress = gr.Markdown()
        contact_sheet = gr.Image(label="候选联系表", type="pil", height=620)
        with gr.Row():
            with gr.Column():
                query_id = gr.Textbox(label="Query ID", interactive=False)
                query_text = gr.Textbox(label="查询文本", interactive=False, lines=2)
                category = gr.Textbox(label="查询类别", interactive=False)
                source_image_id = gr.Textbox(label="来源图 ID", interactive=False)
            with gr.Column():
                annotator = gr.Textbox(label="标注者", value=DEFAULT_ANNOTATOR)
                note = gr.Textbox(label="备注", lines=2)
                reviewed = gr.Checkbox(label="已逐图审核，可用于正式评测")
        grades = gr.Textbox(
            label="候选相关性（每行 image_id:grade）",
            lines=25,
            placeholder="val-2001:2\nval-2002:1\nval-2003:0",
        )
        save_button = gr.Button("保存当前查询", variant="primary")
        status = gr.Markdown()

        outputs = [
            position,
            contact_sheet,
            query_id,
            query_text,
            category,
            source_image_id,
            annotator,
            note,
            grades,
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
            [position, annotator, note, grades, reviewed],
            [status, progress],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the formal candidate-level relevance annotator."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("artifacts/evaluation/formal/relevance_pool.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/formal/candidate_relevance.csv"),
    )
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config["project_root"])
    app = build_app(args.pool, args.output, project_root)
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
