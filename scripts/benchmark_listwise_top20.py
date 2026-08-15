from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.evaluation.listwise_benchmark import benchmark_listwise_candidates
from anima_search.evaluation.rerank_benchmark import benchmark_candidates
from anima_search.evaluation.rerank_suite import load_operational_queries
from anima_search.retrieval.listwise_reranker import ListwiseVisualReranker
from anima_search.retrieval.reranker import VisualReranker


def _aggregate(
    pointwise_records: list[dict[str, object]],
    listwise_records: list[dict[str, object]],
    *,
    query_count: int,
    top_k: int,
    repeats: int,
) -> dict[str, object]:
    pointwise_latency = sum(float(row["latency_ms"]) for row in pointwise_records)
    listwise_latency = sum(float(row["latency_ms"]) for row in listwise_records)

    def peak(rows: list[dict[str, object]]) -> int | None:
        values = [
            int(row["peak_cuda_memory_bytes"])
            for row in rows
            if row["peak_cuda_memory_bytes"] is not None
        ]
        return max(values) if values else None

    pointwise_failures = sum(not bool(row["success"]) for row in pointwise_records)
    listwise_failures = sum(not bool(row["success"]) for row in listwise_records)
    listwise_degradations = sum(
        bool(row.get("degraded")) for row in listwise_records
    )
    return {
        "query_count": query_count,
        "top_k": top_k,
        "repeats": repeats,
        "pointwise": {
            "model_calls": len(pointwise_records),
            "failure_count": pointwise_failures,
            "failure_rate": (
                pointwise_failures / len(pointwise_records)
                if pointwise_records else None
            ),
            "total_latency_ms": pointwise_latency,
            "mean_query_latency_ms": (
                pointwise_latency / (query_count * repeats)
                if query_count else None
            ),
            "peak_cuda_memory_bytes": peak(pointwise_records),
        },
        "listwise": {
            "model_calls": len(listwise_records),
            "failure_count": listwise_failures,
            "failure_rate": (
                listwise_failures / len(listwise_records)
                if listwise_records else None
            ),
            "degraded_count": listwise_degradations,
            "degraded_rate": (
                listwise_degradations / len(listwise_records)
                if listwise_records else None
            ),
            "total_latency_ms": listwise_latency,
            "mean_query_latency_ms": (
                listwise_latency / len(listwise_records)
                if listwise_records else None
            ),
            "peak_cuda_memory_bytes": peak(listwise_records),
        },
        "listwise_speedup_over_pointwise": (
            pointwise_latency / listwise_latency if listwise_latency > 0 else None
        ),
        "quality_claim": "not_evaluated_without_relevance_judgments",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare M6 pointwise and single-contact-sheet listwise Top-20 cost."
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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--query-limit", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/m6_listwise_top20_8gb.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    if args.query_limit <= 0 or args.repeats <= 0:
        parser.error("--query-limit and --repeats must be positive")

    queries = load_operational_queries(args.queries)[: args.query_limit]
    service = create_service(args.config, args.split, args.branches)
    settings = service.config["retrieval"]
    settings["candidate_count"] = max(int(settings["candidate_count"]), args.top_k)
    settings["result_count"] = args.top_k

    candidate_sets: list[tuple[dict[str, object], list]] = []
    for query in queries:
        query_text = str(query["text"])
        candidates = service.search(query_text, use_reranker=False)
        if len(candidates) < args.top_k:
            raise RuntimeError(
                f"{query['query_id']} returned {len(candidates)} candidates; "
                f"Top-{args.top_k} is required"
            )
        candidate_sets.append((query, list(candidates[: args.top_k])))

    released = service.release_retrieval_encoders()
    project_root = Path(service.config["project_root"])
    listwise_prompt_path = project_root / settings.get(
        "rerank_listwise_prompt",
        "configs/prompts/reranker_listwise.txt",
    )
    listwise_prompt = listwise_prompt_path.read_text(encoding="utf-8")

    pointwise_records: list[dict[str, object]] = []
    listwise_records: list[dict[str, object]] = []
    per_query: list[dict[str, object]] = []
    with service.manager.qwen_session() as qwen:
        pointwise = VisualReranker(
            qwen,
            service.reranker_prompt,
            project_root,
            settings["rrf_weight"],
            settings["vlm_weight"],
            settings.get("rerank_max_new_tokens", 256),
        )
        listwise = ListwiseVisualReranker(
            qwen,
            listwise_prompt,
            project_root,
            max_new_tokens=settings.get("rerank_listwise_max_new_tokens", 768),
            columns=settings.get("rerank_listwise_columns", 5),
            tile_size=settings.get("rerank_listwise_tile_size", 192),
        )
        warm_query, warm_candidates = candidate_sets[0]
        warm_result = pointwise.rerank(
            str(warm_query["text"]),
            [warm_candidates[0].model_copy(deep=True)],
        )
        warm_failure = next(
            (
                message
                for message in warm_result[0].mismatch
                if message.startswith("视觉重排不可用：")
            ),
            None,
        )
        if warm_failure:
            raise RuntimeError(f"Qwen warm-up failed: {warm_failure}")
        for query, candidates in candidate_sets:
            query_text = str(query["text"])
            point_rows, point_summary = benchmark_candidates(
                query_text,
                args.split,
                candidates,
                pointwise,
                args.repeats,
            )
            for row in point_rows:
                row["method"] = "pointwise"
                row["query_id"] = query["query_id"]
                row["category"] = query["category"]
                row["top_k"] = args.top_k
            pointwise_records.extend(point_rows)

            list_rows, list_summary = benchmark_listwise_candidates(
                query_text,
                args.split,
                candidates,
                listwise,
                args.repeats,
            )
            for row in list_rows:
                row["query_id"] = query["query_id"]
                row["category"] = query["category"]
                row["contact_sheet_size"] = listwise.last_contact_sheet_size
            listwise_records.extend(list_rows)
            per_query.append(
                {
                    "query_id": query["query_id"],
                    "category": query["category"],
                    "query": query_text,
                    "pointwise": point_summary,
                    "listwise": list_summary,
                }
            )
            print(f"completed {query['query_id']} Top-{args.top_k}", flush=True)

    summary = _aggregate(
        pointwise_records,
        listwise_records,
        query_count=len(candidate_sets),
        top_k=args.top_k,
        repeats=args.repeats,
    )
    summary["retrieval_branches"] = args.branches
    summary["released_retrieval_encoders"] = released
    summary["warmup"] = "one unmeasured pointwise candidate before both methods"
    payload = {
        "summary": summary,
        "per_query": per_query,
        "pointwise_records": pointwise_records,
        "listwise_records": listwise_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
