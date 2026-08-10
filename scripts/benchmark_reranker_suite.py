from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.evaluation.rerank_benchmark import benchmark_candidates, write_benchmark
from anima_search.evaluation.rerank_suite import (
    annotate_suite_records,
    load_operational_queries,
    summarize_suite,
)
from anima_search.retrieval.reranker import VisualReranker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-query M6 pointwise benchmark with cold/warm statistics."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("configs/m6_benchmark_queries.jsonl"),
    )
    parser.add_argument("--config", default="configs/benchmark_8gb.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=["image", "text", "bm25"],
        default=["image"],
    )
    parser.add_argument("--top-k", nargs="+", type=int, choices=[3, 5], default=[3, 5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/m6_multiquery_8gb.jsonl"),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    queries = load_operational_queries(args.queries)
    top_k_values = sorted(set(args.top_k))
    maximum_top_k = max(top_k_values)
    service = create_service(args.config, args.split, args.branches)

    candidate_sets: dict[str, list] = {}
    collection_failures: list[dict[str, object]] = []
    for query in queries:
        try:
            results = service.search(query["text"], use_reranker=False)
            candidate_sets[query["query_id"]] = list(results[:maximum_top_k])
        except Exception as exc:
            candidate_sets[query["query_id"]] = []
            collection_failures.append(
                {
                    "query_id": query["query_id"],
                    "category": query["category"],
                    "stage": "retrieval",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    released = service.release_retrieval_encoders()
    settings = service.config["retrieval"]
    all_records: list[dict[str, object]] = []
    global_run = 1
    with service.manager.qwen_session() as qwen:
        reranker = VisualReranker(
            qwen,
            service.reranker_prompt,
            Path(service.config["project_root"]),
            settings["rrf_weight"],
            settings["vlm_weight"],
            settings.get("rerank_max_new_tokens", 128),
        )
        for top_k in top_k_values:
            for query in queries:
                available = candidate_sets[query["query_id"]]
                if len(available) < top_k:
                    collection_failures.append(
                        {
                            "query_id": query["query_id"],
                            "category": query["category"],
                            "top_k": top_k,
                            "stage": "candidate_count",
                            "error": (
                                f"retrieval returned {len(available)} candidates; "
                                f"top_k={top_k} requested"
                            ),
                        }
                    )
                    continue
                records, _ = benchmark_candidates(
                    query["text"],
                    args.split,
                    available[:top_k],
                    reranker,
                    args.repeats,
                )
                enriched = annotate_suite_records(
                    records,
                    query_id=query["query_id"],
                    category=query["category"],
                    top_k=top_k,
                    starting_run=global_run,
                )
                all_records.extend(enriched)
                global_run += len(enriched)
                print(f"completed {query['query_id']} top_k={top_k}: {len(enriched)} runs", flush=True)

    summary = summarize_suite(
        all_records,
        collection_failures,
        query_count=len(queries),
        top_k_values=top_k_values,
        repeats=args.repeats,
    )
    summary["retrieval_branches"] = args.branches
    summary["released_retrieval_encoders"] = released
    summary_path = write_benchmark(args.output, all_records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSONL: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
