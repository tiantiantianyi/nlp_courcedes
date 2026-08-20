from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr


DEFAULT_ANNOTATOR = "张添翼"
GRADE_CHOICES = [
    ("2 · 高度相关", "2"),
    ("1 · 部分相关/不确定", "1"),
    ("0 · 不相关", "0"),
]
DOMAIN_LABELS = {
    "wikiart": "WikiArt 艺术图片",
    "mimic_cxr": "MIMIC-CXR 医学影像",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"标注文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def progress_text(rows: list[dict[str, str]]) -> str:
    complete = sum(
        row.get("human_relevance", "").strip() in {"0", "1", "2"}
        and row.get("review_status", "") == "已复核"
        for row in rows
    )
    return f"**复核进度：{complete}/{len(rows)} 条**"


def save_row(
    path: Path,
    index: int,
    grade: str,
    annotator: str,
    note: str,
) -> None:
    rows = load_rows(path)
    if not 0 <= index < len(rows):
        raise IndexError(f"任务编号越界：{index + 1}")
    if str(grade).strip() not in {"0", "1", "2"}:
        raise ValueError("请选择 0、1 或 2 后再保存")
    rows[index]["human_relevance"] = str(grade).strip()
    rows[index]["annotator"] = str(annotator).strip() or DEFAULT_ANNOTATOR
    rows[index]["review_status"] = "已复核"
    rows[index]["review_note"] = str(note or "").strip()

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for key in ("human_relevance", "annotator", "review_status", "review_note"):
        if key not in fieldnames:
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def form_values(
    row: dict[str, str],
    index: int,
    dataset_dir: Path,
) -> tuple[object, ...]:
    image_path = dataset_dir / row["relative_path"]
    grade = row.get("human_relevance", "").strip() or None
    domain = DOMAIN_LABELS.get(row.get("domain", ""), row.get("domain", ""))
    instruction = (
        "艺术域：以图片为主，元数据只用于复核边界。"
        if row.get("domain") == "wikiart"
        else "医学域：以随附报告文字核对术语，不进行医学诊断。"
    )
    return (
        index + 1,
        str(image_path.resolve()),
        row.get("query", ""),
        domain,
        row.get("image_id", ""),
        row.get("reference_text", ""),
        row.get("auto_relevance", ""),
        instruction,
        row.get("annotator", "").strip() or DEFAULT_ANNOTATOR,
        grade,
        row.get("review_note", ""),
    )


def build_app(review_path: Path, dataset_dir: Path) -> gr.Blocks:
    rows = load_rows(review_path)
    if not rows:
        raise ValueError("人工复核文件为空")

    def load_position(position: int):
        current_rows = load_rows(review_path)
        index = max(0, min(len(current_rows) - 1, int(position) - 1))
        return (
            *form_values(current_rows[index], index, dataset_dir),
            progress_text(current_rows),
            "",
        )

    def save_position(
        position: int,
        annotator: str,
        grade: str,
        note: str,
    ):
        index = int(position) - 1
        try:
            save_row(review_path, index, grade, annotator, note)
        except Exception as exc:
            current_rows = load_rows(review_path)
            return (
                f"保存失败：{type(exc).__name__}: {exc}",
                progress_text(current_rows),
            )
        current_rows = load_rows(review_path)
        return (
            f"已保存第 {position} 条：{current_rows[index]['image_id']}",
            progress_text(current_rows),
        )

    def next_unreviewed(position: int) -> int:
        current_rows = load_rows(review_path)
        start = int(position) % len(current_rows)
        for offset in range(len(current_rows)):
            index = (start + offset) % len(current_rows)
            if current_rows[index].get("review_status") != "已复核":
                return index + 1
        return int(position)

    def previous(position: int) -> int:
        return max(1, int(position) - 1)

    def following(position: int) -> int:
        return min(len(rows), int(position) + 1)

    with gr.Blocks(title="A9 域外迁移人工复核") as app:
        gr.Markdown(
            "# A9 域外迁移人工复核\n\n"
            "逐条判断当前查询与图片是否相关：**2=高度相关，1=部分相关/不确定，0=不相关**。"
            "人工复核只覆盖 50 条抽样任务，保存后会自动写入 CSV。"
        )
        gr.Markdown(
            "医学图片只核对附带报告中的关键词，不用于任何诊断；"
            "自动标签仅供发现边界案例，最终以你的判断为准。"
        )
        with gr.Row():
            previous_button = gr.Button("上一条")
            position = gr.Slider(
                1, len(rows), value=1, step=1, label="任务编号"
            )
            next_button = gr.Button("下一条")
            unreviewed_button = gr.Button("跳到下一条未复核")
        progress = gr.Markdown(progress_text(rows))
        with gr.Row():
            with gr.Column(scale=5):
                image = gr.Image(
                    label="待复核图片",
                    type="filepath",
                    height=560,
                    interactive=False,
                )
            with gr.Column(scale=4):
                query = gr.Textbox(
                    label="查询文本", lines=2, interactive=False
                )
                domain = gr.Textbox(label="数据域", interactive=False)
                image_id = gr.Textbox(label="图片 ID", interactive=False)
                reference = gr.Textbox(
                    label="参考元数据/医学报告",
                    lines=10,
                    interactive=False,
                )
                auto_grade = gr.Textbox(
                    label="自动弱标注（仅供核对）",
                    interactive=False,
                )
                instruction = gr.Markdown()
        with gr.Row():
            annotator = gr.Textbox(label="标注者", value=DEFAULT_ANNOTATOR)
            grade = gr.Radio(
                GRADE_CHOICES,
                label="人工相关性判断",
                value=None,
            )
        note = gr.Textbox(
            label="备注（可选）",
            lines=2,
            placeholder="例如：报告是否定句、风格标签存在边界争议等",
        )
        save_button = gr.Button("保存当前条目", variant="primary")
        status = gr.Markdown()

        load_outputs = [
            position,
            image,
            query,
            domain,
            image_id,
            reference,
            auto_grade,
            instruction,
            annotator,
            grade,
            note,
            progress,
            status,
        ]
        app.load(load_position, position, load_outputs)
        position.change(load_position, position, load_outputs)
        previous_button.click(previous, position, position)
        next_button.click(following, position, position)
        unreviewed_button.click(next_unreviewed, position, position)
        save_button.click(
            save_position,
            [position, annotator, grade, note],
            [status, progress],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the A9 domain-transfer review annotator."
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("../domain_transfer_data/a9_subset/manual_review_50.csv"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../domain_transfer_data/a9_subset"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    review_path = args.review.resolve()
    dataset_dir = args.dataset_dir.resolve()
    app = build_app(review_path, dataset_dir)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        show_error=True,
        allowed_paths=[str((dataset_dir / "images").resolve())],
    )


if __name__ == "__main__":
    main()
