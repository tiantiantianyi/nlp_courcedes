from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


MODULE_ROWS = [
    ("M0", "场景路由", "待实现", "待实现", "待评测", "未满足"),
    ("M1", "结构化标注", "队友基线", "标注进行中", "待评测", "部分满足"),
    ("M2", "可验证性校验", "仅基础校验", "待实现", "待评测", "未满足"),
    ("M3", "多路索引", "三路框架", "image-only 实跑", "待正式数据", "部分满足"),
    ("M4", "查询理解", "规则与结构化解析", "测试通过", "待分类评测", "部分满足"),
    ("M5", "混合召回", "RRF 与降级", "工程验证", "待 relevance", "部分满足"),
    ("M6", "VLM 重排", "pointwise", "8GB 实测", "待 A6", "部分满足"),
    ("M7", "输出 Agent", "问答与故事核心", "单元测试", "待人工评测", "部分满足"),
]

STATUS_LEVEL = {
    "待实现": 0,
    "待评测": 0,
    "未满足": 0,
    "标注进行中": 1,
    "仅基础校验": 1,
    "三路框架": 1,
    "队友基线": 1,
    "待正式数据": 1,
    "待分类评测": 1,
    "待 relevance": 1,
    "待 A6": 1,
    "待人工评测": 1,
    "部分满足": 1,
    "规则与结构化解析": 2,
    "RRF 与降级": 2,
    "pointwise": 2,
    "问答与故事核心": 2,
    "image-only 实跑": 2,
    "测试通过": 2,
    "工程验证": 2,
    "8GB 实测": 2,
    "单元测试": 2,
}


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 180,
    })


def plot_alignment(output_path: Path) -> None:
    columns = ["代码/接口", "本地验证", "正式指标", "完整符合方案"]
    values = np.array([
        [STATUS_LEVEL[value] for value in row[2:]]
        for row in MODULE_ROWS
    ])
    labels = np.array([row[2:] for row in MODULE_ROWS])
    ylabels = [f"{row[0]}  {row[1]}" for row in MODULE_ROWS]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    cmap = ListedColormap(["#FDE2E2", "#FFF0C2", "#D9F2E6"])
    ax.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(columns)), labels=columns)
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    ax.tick_params(axis="x", labelsize=11, pad=8)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title("技术方案 M0–M7 阶段工程就绪度（截至 2026-08-10）", fontsize=15, pad=18)
    ax.set_xlabel("绿色：已完成工程验证；黄色：部分完成或依赖外部数据；红色：尚未完成",
                  labelpad=14, color="#555555")

    for row_index in range(labels.shape[0]):
        for col_index in range(labels.shape[1]):
            ax.text(
                col_index,
                row_index,
                labels[row_index, col_index],
                ha="center",
                va="center",
                fontsize=9,
                color="#202020",
            )

    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.01,
        0.01,
        "注：该图表示工程就绪度，不是论文效果分数；mock/image-only 结果不替代正式标注与 relevance judgments。",
        fontsize=8.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"benchmark file is empty: {path}")
    return rows


def plot_benchmark(summary_path: Path, rows_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = load_rows(rows_path)
    labels = [str(row["image_id"]) for row in rows]
    latency_seconds = [float(row["latency_ms"]) / 1000 for row in rows]
    scores = [float(row["rerank_score"]) for row in rows]
    peak_gib = float(summary["peak_cuda_memory_bytes"]) / (1024 ** 3)
    budget_gib = 8.0

    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.35], hspace=0.52, wspace=0.30)
    latency_ax = fig.add_subplot(grid[0, 0])
    score_ax = fig.add_subplot(grid[0, 1])
    budget_ax = fig.add_subplot(grid[1, :])

    x = np.arange(len(labels))
    latency_colors = ["#E07A5F", "#4C78A8", "#4C78A8"]
    bars = latency_ax.bar(x, latency_seconds, color=latency_colors[:len(labels)], width=0.62)
    latency_ax.bar_label(bars, fmt="%.2f s", padding=4, fontsize=9)
    latency_ax.set_xticks(x, labels=labels)
    latency_ax.set_ylabel("耗时（秒）")
    latency_ax.set_title("Top-3 pointwise 重排耗时")
    latency_ax.spines[["top", "right"]].set_visible(False)
    latency_ax.text(
        0.02,
        0.96,
        "首张包含冷启动/模型加载开销",
        transform=latency_ax.transAxes,
        va="top",
        fontsize=9,
        color="#8A3F2D",
    )

    score_bars = score_ax.bar(x, scores, color="#59A14F", width=0.62)
    score_ax.bar_label(score_bars, fmt="%.0f", padding=4, fontsize=9)
    score_ax.set_xticks(x, labels=labels)
    score_ax.set_ylim(0, 105)
    score_ax.set_ylabel("VLM 相关性分数（0–100）")
    score_ax.set_title("单次查询的候选打分")
    score_ax.spines[["top", "right"]].set_visible(False)
    score_ax.text(
        0.02,
        0.96,
        "仅为模型打分，不等同于人工相关性标注",
        transform=score_ax.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )

    budget_ax.barh([0], [peak_gib], color="#4C78A8", height=0.38, label="实测峰值")
    budget_ax.barh(
        [0],
        [max(0.0, budget_gib - peak_gib)],
        left=[peak_gib],
        color="#D9E2EC",
        height=0.38,
        label="名义余量",
    )
    budget_ax.set_xlim(0, budget_gib)
    budget_ax.set_yticks([])
    budget_ax.set_xlabel("RTX 4060 Laptop 名义显存预算（GiB）")
    budget_ax.set_title(
        f"峰值显存 {peak_gib:.2f} GiB / 8 GiB　|　"
        f"总耗时 {float(summary['total_candidate_latency_ms']) / 1000:.2f} s　|　"
        f"失败率 {float(summary['failure_rate']):.0%}"
    )
    budget_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2, frameon=False)
    budget_ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        f"M6 本地实测：{summary['query']}（{summary['split']}，Top-{summary['top_k']}）",
        fontsize=16,
        y=0.99,
    )
    fig.text(
        0.01,
        0.01,
        "运行模式：image-only 候选召回 + Qwen3-VL-2B-Instruct pointwise 重排；"
        "无 relevance judgments，因此不宣称检索质量提升。",
        fontsize=8.5,
        color="#666666",
    )
    fig.subplots_adjust(top=0.90, bottom=0.13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible stage-report figures.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/evaluation/reranker_top3_8gb.summary.json"),
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("artifacts/evaluation/reranker_top3_8gb.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/stage_report"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    plot_alignment(args.output_dir / "proposal_alignment.png")
    plot_benchmark(args.summary, args.rows, args.output_dir / "m6_benchmark.png")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
