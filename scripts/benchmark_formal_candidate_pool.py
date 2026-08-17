from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.evaluation.candidate_review import load_candidate_pool
from anima_search.evaluation.listwise_benchmark import benchmark_listwise_candidates
from anima_search.evaluation.ground_truth import load_relevance
from anima_search.evaluation.rerank_benchmark import benchmark_candidates
from anima_search.evaluation.rerank_quality import build_rerank_quality
from anima_search.retrieval.listwise_reranker import ListwiseVisualReranker
from anima_search.retrieval.reranker import VisualReranker
from anima_search.schemas import SearchResult


def _load_queries(path: Path) -> dict[str, dict[str, str]]:
    return {
        str(row["query_id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _candidate_results(row: dict[str, object]) -> list[SearchResult]:
    results = []
    for rank, candidate in enumerate(list(row["candidates"]), start=1):
        results.append(
            SearchResult(
                image_id=str(candidate["image_id"]),
                relative_path=str(candidate["relative_path"]),
                fused_score=float(-rank),
                branch_ranks={"candidate_pool": rank},
                active_branches=["candidate_pool"],
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A6 on the frozen, manually reviewed candidate pool (1--20 candidates/query)."
    )
    parser.add_argument("--config", default="configs/benchmark_8gb.yaml")
    parser.add_argument("--queries", type=Path, default=Path("artifacts/evaluation/formal/val_queries.jsonl"))
    parser.add_argument("--pool", type=Path, default=Path("artifacts/evaluation/formal/relevance_pool.jsonl"))
    parser.add_argument("--relevance", type=Path, default=Path("artifacts/evaluation/formal/val_relevance.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/formal/a6/formal_candidate_pool.json"))
    parser.add_argument("--quality-output", type=Path, default=Path("artifacts/evaluation/formal/a6/formal_candidate_pool_quality.json"))
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    query_by_id = _load_queries(args.queries)
    pool = load_candidate_pool(args.pool)
    if args.query_limit > 0:
        pool = pool[: args.query_limit]
    relevance = load_relevance(args.relevance)

    service = create_service(args.config, "val", ["image"])
    settings = service.config["retrieval"]
    project_root = Path(service.config["project_root"])
    prompt_path = project_root / settings.get("rerank_listwise_prompt", "configs/prompts/reranker_listwise.txt")
    listwise_prompt = prompt_path.read_text(encoding="utf-8")
    service.release_retrieval_encoders()

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
        warm = _candidate_results(pool[0])[:1]
        pointwise.rerank(str(query_by_id[str(pool[0]["query_id"])]["text"]), warm)
        for row in pool:
            query_id = str(row["query_id"])
            query = query_by_id[query_id]
            candidates = _candidate_results(row)
            point_rows, point_summary = benchmark_candidates(
                str(query["text"]), "val", candidates, pointwise, args.repeats
            )
            for record in point_rows:
                record.update({"query_id": query_id, "category": query["category"], "top_k": len(candidates)})
            list_rows, list_summary = benchmark_listwise_candidates(
                str(query["text"]), "val", candidates, listwise, args.repeats
            )
            for record in list_rows:
                record.update({"query_id": query_id, "category": query["category"], "candidate_count": len(candidates)})
            pointwise_records.extend(point_rows)
            listwise_records.extend(list_rows)
            per_query.append({
                "query_id": query_id,
                "category": query["category"],
                "query": query["text"],
                "candidate_count": len(candidates),
                "baseline_image_ids": [item.image_id for item in candidates],
                "pointwise": point_summary,
                "listwise": list_summary,
            })
            print(f"completed {query_id} candidates={len(candidates)}", flush=True)

    baseline_by_query = {
        item["query_id"]: item["baseline_image_ids"] for item in per_query
    }
    quality = build_rerank_quality(
        baseline_by_query=baseline_by_query,
        pointwise_records=pointwise_records,
        listwise_records=listwise_records,
        relevance=relevance,
    )
    payload = {
        "schema_version": "formal-a6-candidate-pool-v1.0",
        "query_count": len(per_query),
        "candidate_count_total": sum(int(item["candidate_count"]) for item in per_query),
        "candidate_count_min": min(int(item["candidate_count"]) for item in per_query),
        "candidate_count_max": max(int(item["candidate_count"]) for item in per_query),
        "per_query": per_query,
        "pointwise_records": pointwise_records,
        "listwise_records": listwise_records,
        "quality": quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.quality_output.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"query_count": payload["query_count"], "candidate_count_total": payload["candidate_count_total"], "quality_summary": quality["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
