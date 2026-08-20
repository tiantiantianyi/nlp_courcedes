from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.evaluation.domain_transfer import (
    build_relevance,
    ranking_metrics,
    summarize_details,
)
from anima_search.indexing.image_vector_index import ImageVectorIndex


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_details(path: Path, rows: list[dict[str, object]]) -> None:
    serializable = [
        {
            **row,
            "ranked_ids": json.dumps(
                row["ranked_ids"],
                ensure_ascii=False,
            ),
        }
        for row in rows
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(serializable[0]),
        )
        writer.writeheader()
        writer.writerows(serializable)


def cuda_peak() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return None


def apply_manual_review(
    relevance: dict[str, dict[str, int]],
    queries: list[dict[str, object]],
    review_rows: list[dict[str, str]],
) -> bool:
    complete = bool(review_rows) and all(
        row.get("human_relevance", "").strip() in {"0", "1", "2"}
        for row in review_rows
    )
    if not complete:
        return False
    by_query = {
        str(query["query_id"]): query
        for query in queries
    }
    for row in review_rows:
        source_query = by_query[row["query_id"]]
        related_ids = [
            str(query["query_id"])
            for query in queries
            if query["domain"] == source_query["domain"]
            and query["target_label"] == source_query["target_label"]
        ]
        grade = int(row["human_relevance"])
        image_id = row["image_id"]
        for query_id in related_ids:
            if grade > 0:
                relevance[query_id][image_id] = grade
            else:
                relevance[query_id].pop(image_id, None)
    return True


def render_report(
    summary: dict[str, object],
    manual_review_complete: bool,
) -> str:
    baseline = summary["baseline"]["overall"]
    lines = [
        "# A9 域外迁移能力实验结果",
        "",
        "> 医学影像实验仅用于观察模型跨领域行为，不用于任何临床诊断或医疗决策。",
        "",
        "## 实验设置",
        "",
        f"- 编码器：{summary['encoder']}",
        f"- 图片数：{summary['record_count']}",
        f"- 查询数：{baseline['query_count']}",
        (
            "- 标注方案："
            + (
                "元数据弱标注 + 50 条人工审计"
                if manual_review_complete
                else "元数据/报告弱标注（未进行人工标注）"
            )
        ),
        "",
        "## 零样本检索结果",
        "",
        "| 数据域 | 查询数 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, metrics in summary["baseline"]["by_domain"].items():
        lines.append(
            f"| {domain} | {metrics['query_count']} | "
            f"{metrics['recall@1']:.4f} | "
            f"{metrics['recall@5']:.4f} | "
            f"{metrics['recall@10']:.4f} | "
            f"{metrics['mrr']:.4f} | "
            f"{metrics['ndcg@10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 结果解释要求",
            "",
            "- WikiArt 重点讨论风格词、颜色与构图语义是否仍能被图文编码器识别。",
            "- MIMIC-CXR 标签来自英文放射报告关键词，只能作为弱标注，不能视为模型诊断能力。",
            "- 本结果属于弱监督探索性实验，不应写成专家质量结论。",
            "- 建议从每个领域选择至少 3 个成功案例和 3 个失败案例放入报告。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A9 zero-shot domain-transfer retrieval."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_8gb.yaml",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../domain_transfer_data/a9_subset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/a9"),
    )
    parser.add_argument(
        "--encoder",
        choices=["chinese_clip", "jina_clip_v2"],
        default="chinese_clip",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.top_k <= 0:
        parser.error("--batch-size and --top-k must be positive")

    config = load_config(args.config)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(dataset_dir / "metadata.jsonl")
    queries = load_jsonl(dataset_dir / "queries.jsonl")
    relevance = build_relevance(queries, records)
    review_path = dataset_dir / "manual_review_50.csv"
    review_rows: list[dict[str, str]] = []
    if review_path.is_file():
        with review_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            review_rows = list(csv.DictReader(handle))
    review_complete = apply_manual_review(
        relevance,
        queries,
        review_rows,
    )
    image_ids = [str(record["image_id"]) for record in records]
    image_paths = [
        dataset_dir / str(record["relative_path"])
        for record in records
    ]

    if args.encoder == "jina_clip_v2":
        model_path = resolve_path(
            config,
            config["models"]["jina_clip_v2"],
        )
        encoder_options = {
            "truncate_dim": int(
                config["retrieval"].get(
                    "jina_clip_truncate_dim",
                    512,
                )
            ),
            "local_files_only": bool(
                config["retrieval"].get(
                    "jina_clip_local_files_only",
                    True,
                )
            ),
        }
    else:
        model_path = resolve_path(
            config,
            config["models"]["image_embedder"],
        )
        encoder_options = {}
    index = ImageVectorIndex(
        model_path,
        config["runtime"]["device"],
        config["runtime"]["dtype"],
        annotation_version="a9-weak-label-v1",
        build_parameters={
            "batch_size": args.batch_size,
            "dataset": "a9-domain-transfer",
        },
        encoder_type=args.encoder,
        encoder_options=encoder_options,
    )
    started = time.perf_counter()
    index.build(
        image_ids,
        image_paths,
        batch_size=args.batch_size,
    )
    build_seconds = time.perf_counter() - started
    index.save(output_dir / f"index_{args.encoder}")

    details: list[dict[str, object]] = []
    for query in queries:
        started = time.perf_counter()
        ranking = index.search(
            str(query["text"]),
            limit=args.top_k,
        )
        latency = time.perf_counter() - started
        ranked_ids = [image_id for image_id, _ in ranking]
        details.append(
            {
                "query_id": query["query_id"],
                "query": query["text"],
                "domain": query["domain"],
                "target_label": query["target_label"],
                "ranked_ids": ranked_ids,
                "latency_seconds": latency,
                **ranking_metrics(
                    ranked_ids,
                    relevance[str(query["query_id"])],
                ),
            }
        )
    baseline = summarize_details(details)
    summary = {
        "schema_version": "a9-domain-transfer-results-v1.0",
        "encoder": args.encoder,
        "record_count": len(records),
        "build_seconds": build_seconds,
        "images_per_second": len(records) / build_seconds,
        "peak_cuda_memory_bytes": cuda_peak(),
        "weak_labels_only": not review_complete,
        "label_basis": (
            "metadata weak labels with 50-pair manual audit"
            if review_complete
            else "metadata weak labels; manual audit pending"
        ),
        "manual_review_file": str(review_path),
        "baseline": baseline,
        "medical_notice": (
            "仅用于模型跨领域行为研究，不用于临床诊断或医疗决策。"
        ),
    }
    (output_dir / f"{args.encoder}_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_details(
        output_dir / f"{args.encoder}_details.csv",
        details,
    )
    (output_dir / f"{args.encoder}_report.md").write_text(
        render_report(summary, review_complete),
        encoding="utf-8",
    )
    index.unload_encoder()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
