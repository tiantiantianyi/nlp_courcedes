from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.app.factory import create_service
from anima_search.config import load_config, resolve_path
from anima_search.evaluation.ground_truth import load_queries, load_relevance
from anima_search.evaluation.metrics import aggregate_query_metrics, average_precision, ndcg_at_k, recall_at_k, reciprocal_rank


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--queries", default="artifacts/evaluation/val_queries.jsonl")
    parser.add_argument("--relevance", default="artifacts/evaluation/val_relevance.csv")
    parser.add_argument("--rerank", action="store_true"); args = parser.parse_args()
    config = load_config(args.config); service = create_service(args.config, "val")
    queries = load_queries(Path(args.queries)); relevance = load_relevance(Path(args.relevance))
    unreviewed = [row["query_id"] for row in queries if not row.get("reviewed", False) or row.get("category") == "auto_seed"]
    if unreviewed:
        raise ValueError(
            "Evaluation refused: rewrite and categorize every auto-generated query, then set reviewed=true. "
            f"Unreviewed query IDs: {', '.join(unreviewed[:10])}"
        )
    metric_rows = []; details = []
    for query in queries:
        started = time.perf_counter(); results = service.search(query["text"], args.rerank); elapsed = time.perf_counter() - started
        ranked = [item.image_id for item in results]; rel = relevance.get(query["query_id"], {})
        metrics = {"recall@1": recall_at_k(ranked, rel, 1), "recall@5": recall_at_k(ranked, rel, 5),
            "recall@10": recall_at_k(ranked, rel, 10), "mrr": reciprocal_rank(ranked, rel),
            "map": average_precision(ranked, rel), "ndcg@10": ndcg_at_k(ranked, rel, 10), "latency": elapsed}
        metric_rows.append(metrics); details.append({"query_id": query["query_id"], "query": query["text"], "ranked_ids": ranked, **metrics})
    output = resolve_path(config, config["data"]["artifacts_dir"]) / "evaluation"; output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_metrics.json").write_text(json.dumps(aggregate_query_metrics(metric_rows), indent=2), encoding="utf-8")
    with (output / "retrieval_details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=details[0].keys()); writer.writeheader(); writer.writerows(details)
    print(json.dumps(aggregate_query_metrics(metric_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
