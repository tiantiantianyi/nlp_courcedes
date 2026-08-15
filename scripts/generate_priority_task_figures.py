from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def configure_plotting() -> None:
    plt.rcParams.update(
        {
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
        }
    )


def plot_m6(summary: dict[str, object], output: Path) -> None:
    categories = list(summary["by_category"])
    labels = {
        "simple": "简单",
        "compositional": "组合",
        "negative": "否定",
        "count": "数量",
        "ocr": "OCR",
    }
    p50 = [float(summary["by_category"][key]["latency_p50_ms"]) / 1000 for key in categories]
    p95 = [float(summary["by_category"][key]["latency_p95_ms"]) / 1000 for key in categories]
    x = list(range(len(categories)))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    width = 0.36
    ax.bar([value - width / 2 for value in x], p50, width, label="P50", color="#4C78A8")
    ax.bar([value + width / 2 for value in x], p95, width, label="P95", color="#F28E2B")
    ax.set_xticks(x, [labels.get(key, key) for key in categories])
    ax.set_ylabel("单候选延迟（秒）")
    ax.set_title(
        "M6 修复后多查询回归：12 queries，Top-3/Top-5，96 candidate runs\n"
        f"失败率 {float(summary['failure_rate']):.0%}｜"
        f"冷启动 {float(summary['cold_start_latency_ms']) / 1000:.2f}s｜"
        f"峰值显存 {float(summary['peak_cuda_memory_bytes']) / (1024 ** 3):.2f}GiB"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.01,
        0.01,
        "仅评价延迟、显存与稳定性；无人工 relevance 时不宣称检索质量提升。",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_m0(summary: dict[str, object], output: Path) -> None:
    distribution = dict(summary["distribution"])
    ordered = sorted(distribution.items(), key=lambda item: item[1])
    labels = {
        "indoor": "室内",
        "street": "街景",
        "nature": "自然风光",
        "portrait": "人像",
        "food": "美食",
        "night": "夜景",
        "text_scene": "文字招牌",
        "transport": "交通工具",
        "animal": "动植物特写",
    }
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    bars = ax.barh(
        [labels.get(key, key) for key, _ in ordered],
        [count for _, count in ordered],
        color="#59A14F",
    )
    ax.bar_label(bars, padding=4)
    ax.set_xlabel("Val 图片数")
    ax.set_title(
        f"M0 Chinese-CLIP zero-shot 场景路由分布（n={summary['record_count']}）"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.text(
        0.01,
        0.01,
        "类别为 zero-shot 路由结果，不是人工真值；作用是选择 M1 场景专用 prompt。",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for the four priority tasks.")
    parser.add_argument(
        "--m6-summary",
        type=Path,
        default=Path("artifacts/evaluation/m6_multiquery_8gb_fixed.summary.json"),
    )
    parser.add_argument(
        "--m0-summary",
        type=Path,
        default=Path("artifacts/routing/val_scene_routes.summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/priority_tasks"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    plot_m6(
        json.loads(args.m6_summary.read_text(encoding="utf-8")),
        args.output_dir / "m6_multiquery_benchmark.png",
    )
    plot_m0(
        json.loads(args.m0_summary.read_text(encoding="utf-8")),
        args.output_dir / "m0_scene_distribution.png",
    )
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
