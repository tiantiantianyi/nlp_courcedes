from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

from anima_search.evaluation.metrics import (
    aggregate_query_metrics,
    average_precision,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


METRIC_FIELDS = (
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "map",
    "ndcg@10",
)


def validate_formal_queries(queries: list[dict[str, object]]) -> None:
    if not queries:
        raise ValueError("Evaluation refused: the query set is empty.")
    query_ids: list[str] = []
    invalid: list[str] = []
    for index, row in enumerate(queries, start=1):
        query_id = str(row.get("query_id", f"row-{index}"))
        query_ids.append(query_id)
        if (
            not str(row.get("text", "")).strip()
            or not str(row.get("category", "")).strip()
            or not bool(row.get("reviewed", False))
            or row.get("category") == "auto_seed"
        ):
            invalid.append(query_id)
    duplicates = sorted({query_id for query_id in query_ids if query_ids.count(query_id) > 1})
    if invalid:
        raise ValueError(
            "Evaluation refused: rewrite and categorize every auto-generated query, then "
            "set reviewed=true. Invalid query IDs: " + ", ".join(invalid[:10])
        )
    if duplicates:
        raise ValueError(
            "Evaluation refused: duplicate query IDs: " + ", ".join(duplicates[:10])
        )


def _query_metrics(ranked_ids: list[str], relevance: dict[str, int]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked_ids, relevance, 1),
        "recall@5": recall_at_k(ranked_ids, relevance, 5),
        "recall@10": recall_at_k(ranked_ids, relevance, 10),
        "mrr": reciprocal_rank(ranked_ids, relevance),
        "map": average_precision(ranked_ids, relevance),
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
    }


def _aggregate_details(rows: list[dict[str, object]]) -> dict[str, float | int]:
    metric_rows = [
        {field: float(row[field]) for field in METRIC_FIELDS}
        for row in rows
    ]
    aggregate = aggregate_query_metrics(metric_rows)
    latencies = [float(row["latency_seconds"]) for row in rows]
    failures = sum(not bool(row["success"]) for row in rows)
    return {
        "query_count": len(rows),
        **aggregate,
        "latency_mean_seconds": sum(latencies) / len(latencies),
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "failure_count": failures,
        "failure_rate": failures / len(rows),
    }


def evaluate_queries(
    service: object,
    queries: list[dict[str, object]],
    relevance: dict[str, dict[str, int]],
    *,
    use_reranker: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    validate_formal_queries(queries)
    missing_relevance = [
        str(row["query_id"]) for row in queries if str(row["query_id"]) not in relevance
    ]
    if missing_relevance:
        raise ValueError(
            "Evaluation refused: no relevance judgments for query IDs: "
            + ", ".join(missing_relevance[:10])
        )

    details: list[dict[str, object]] = []
    for query in queries:
        query_id = str(query["query_id"])
        started = clock()
        error: str | None = None
        try:
            results = service.search(str(query["text"]), use_reranker=use_reranker)
            ranked_ids = [item.image_id for item in results]
        except Exception as exc:
            ranked_ids = []
            error = f"{type(exc).__name__}: {exc}"
        elapsed = clock() - started
        metrics = _query_metrics(ranked_ids, relevance[query_id])
        details.append(
            {
                "query_id": query_id,
                "query": str(query["text"]),
                "category": str(query["category"]),
                "ranked_ids": ranked_ids,
                "latency_seconds": elapsed,
                "success": error is None,
                "error": error,
                **metrics,
            }
        )

    categories = sorted({str(row["category"]) for row in details})
    summary: dict[str, object] = {
        "reranker_enabled": use_reranker,
        "overall": _aggregate_details(details),
        "by_category": {
            category: _aggregate_details(
                [row for row in details if row["category"] == category]
            )
            for category in categories
        },
    }
    return details, summary


def summary_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    overall = dict(summary["overall"])
    rows = [{"scope": "overall", "category": "all", **overall}]
    rows.extend(
        {"scope": "category", "category": category, **metrics}
        for category, metrics in dict(summary["by_category"]).items()
    )
    return rows


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def render_latex_summary(summary: dict[str, object]) -> str:
    rows = summary_rows(summary)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Scope & Category & R@5 & MRR & nDCG@10 & P95 (s) & Failure \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['scope'])} & {_latex_escape(row['category'])} & "
            f"{float(row['recall@5']):.4f} & {float(row['mrr']):.4f} & "
            f"{float(row['ndcg@10']):.4f} & "
            f"{float(row['latency_p95_seconds']):.4f} & "
            f"{float(row['failure_rate']):.2%} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_evaluation(
    output_dir: Path,
    details: list[dict[str, object]],
    summary: dict[str, object],
    *,
    stem: str = "retrieval",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{stem}_metrics.json",
        "details_csv": output_dir / f"{stem}_details.csv",
        "summary_csv": output_dir / f"{stem}_summary.csv",
        "latex": output_dir / f"{stem}_summary.tex",
        "failures": output_dir / f"{stem}_failures.jsonl",
    }
    paths["json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    detail_rows = [
        {**row, "ranked_ids": json.dumps(row["ranked_ids"], ensure_ascii=False)}
        for row in details
    ]
    with paths["details_csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    rows = summary_rows(summary)
    with paths["summary_csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths["latex"].write_text(render_latex_summary(summary), encoding="utf-8")
    failures = [row for row in details if not row["success"]]
    paths["failures"].write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    return paths
