from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.evaluation.ablation import a5_ablation_matrix
from anima_search.evaluation.candidate_pool import (
    build_candidate_pool,
    select_balanced_queries,
)
from anima_search.evaluation.manual_set import load_tasks
from anima_search.evaluation.runner import validate_formal_queries


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a balanced candidate pool from all formal A5 variants."
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("evaluation/formal_val_100/queries.jsonl")
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--graded-query-count", type=int, default=50)
    parser.add_argument("--per-variant-k", type=int, default=5)
    parser.add_argument("--candidate-cap", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/formal/relevance_pool.jsonl"),
    )
    args = parser.parse_args()

    queries = load_tasks(args.queries)
    validate_formal_queries(queries)
    selected = select_balanced_queries(queries, count=args.graded_query_count)
    source_ids = {
        str(row["query_id"]): str(row["source_image_id"]) for row in selected
    }

    matrix = a5_ablation_matrix()
    rankings_by_variant: dict[str, dict[str, list[dict[str, str]]]] = {}
    for variant in matrix:
        variant_name = str(variant["variant"])
        service = create_service(
            args.config,
            args.split,
            list(variant["branches"]),
            str(variant["fusion_method"]),
        )
        try:
            rankings_by_variant[variant_name] = {
                str(query["query_id"]): [
                    {
                        "image_id": str(result.image_id),
                        "relative_path": str(result.relative_path),
                    }
                    for result in service.search(
                        str(query["text"]), use_reranker=False
                    )
                ]
                for query in selected
            }
        finally:
            release = getattr(service, "release_retrieval_encoders", None)
            if callable(release):
                release()

    pool = build_candidate_pool(
        selected,
        rankings_by_variant,
        source_ids,
        args.per_variant_k,
        args.candidate_cap,
    )
    category_counts = Counter(str(row["category"]) for row in selected)
    candidate_counts = [len(row["candidates"]) for row in pool]
    summary: dict[str, object] = {
        "schema_version": "formal-relevance-pool-summary-v1.0",
        "query_count": len(pool),
        "category_counts": {
            category: category_counts.get(category, 0)
            for category in ("simple", "compositional", "negative", "count", "ocr")
        },
        "variants": [str(row["variant"]) for row in matrix],
        "per_variant_k": args.per_variant_k,
        "candidate_cap": args.candidate_cap,
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_mean": sum(candidate_counts) / len(candidate_counts),
        "queries": str(args.queries),
        "config": str(args.config),
        "split": args.split,
        "output": str(args.output),
    }
    _write_jsonl(args.output, pool)
    _write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
