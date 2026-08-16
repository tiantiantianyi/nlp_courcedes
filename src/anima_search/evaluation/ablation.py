from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

from anima_search.evaluation.runner import _aggregate_details, evaluate_queries


_A5_VARIANTS = (
    ("clip_only", ["image"], "rrf"),
    ("text_only", ["text"], "rrf"),
    ("bm25_only", ["bm25"], "rrf"),
    ("rrf_three_way", ["image", "text", "bm25"], "rrf"),
    ("weighted_three_way", ["image", "text", "bm25"], "weighted"),
)


def a5_ablation_matrix() -> list[dict[str, object]]:
    return [
        {
            "variant": variant,
            "branches": list(branches),
            "fusion_method": fusion_method,
            "reranker": False,
        }
        for variant, branches, fusion_method in _A5_VARIANTS
    ]


def ablation_matrix() -> list[dict[str, object]]:
    """Backward-compatible name for the proposal's core A5 matrix."""
    return a5_ablation_matrix()


def run_a5_ablation(
    service_factory: Callable[[list[str], str], object],
    queries: list[dict[str, object]],
    relevance: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in a5_ablation_matrix():
        branches = list(variant["branches"])
        fusion_method = str(variant["fusion_method"])
        service = service_factory(branches, fusion_method)
        _, summary = evaluate_queries(
            service,
            queries,
            relevance,
            use_reranker=False,
        )
        rows.append(
            {
                "variant": variant["variant"],
                "branches": branches,
                **dict(summary["overall"]),
                "fusion_method": fusion_method,
            }
        )
    return rows


def _summarize_details(details: list[dict[str, object]]) -> dict[str, object]:
    if not details:
        raise ValueError("cannot summarize an empty A5 query subset")
    categories = sorted({str(row["category"]) for row in details})
    return {
        "overall": _aggregate_details(details),
        "by_category": {
            category: _aggregate_details(
                [row for row in details if str(row["category"]) == category]
            )
            for category in categories
        },
    }


def run_formal_a5_ablation(
    service_factory: Callable[[list[str], str], object],
    queries: list[dict[str, object]],
    relevance: dict[str, dict[str, int]],
    *,
    graded_candidate_ids: dict[str, set[str]],
) -> list[dict[str, object]]:
    query_ids = {str(row["query_id"]) for row in queries}
    unknown = sorted(set(graded_candidate_ids) - query_ids)
    if unknown:
        raise ValueError(f"graded subset contains unknown query IDs: {unknown}")
    if not graded_candidate_ids:
        raise ValueError("graded subset must not be empty")
    for query_id, expected_ids in graded_candidate_ids.items():
        actual_ids = set(relevance.get(query_id, {}))
        if actual_ids != set(expected_ids):
            raise ValueError(
                "formal A5 requires complete candidate-pool judgments for "
                f"{query_id}; missing={sorted(set(expected_ids) - actual_ids)}, "
                f"extra={sorted(actual_ids - set(expected_ids))}"
            )

    variants: list[dict[str, object]] = []
    for variant in a5_ablation_matrix():
        branches = list(variant["branches"])
        fusion_method = str(variant["fusion_method"])
        service = service_factory(branches, fusion_method)
        try:
            details, all_summary = evaluate_queries(
                service, queries, relevance, use_reranker=False
            )
        finally:
            release = getattr(service, "release_retrieval_encoders", None)
            if callable(release):
                release()
        graded_details = [
            row for row in details if str(row["query_id"]) in graded_candidate_ids
        ]
        variants.append(
            {
                "variant": str(variant["variant"]),
                "branches": branches,
                "fusion_method": fusion_method,
                "reranker": False,
                "all_queries": all_summary,
                "graded_queries": _summarize_details(graded_details),
                "details": details,
            }
        )
    return variants


def _latex_rows(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & R@1 & R@5 & MRR & mAP & nDCG@10 & P95 (s) \\",
        r"\midrule",
    ]
    for row in rows:
        variant = str(row["variant"]).replace("_", r"\_")
        lines.append(
            f"{variant} & {float(row['recall@1']):.4f} & "
            f"{float(row['recall@5']):.4f} & {float(row['mrr']):.4f} & "
            f"{float(row['map']):.4f} & {float(row['ndcg@10']):.4f} & "
            f"{float(row['latency_p95_seconds']):.4f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_ablation_results(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    if not rows:
        raise ValueError("no ablation results to write")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "ablation_results.json",
        "csv": output_dir / "ablation_results.csv",
        "latex": output_dir / "ablation_results.tex",
    }
    paths["json"].write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_rows = [
        {**row, "branches": ",".join(str(value) for value in row["branches"])}
        for row in rows
    ]
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    paths["latex"].write_text(_latex_rows(rows), encoding="utf-8")
    return paths


def write_formal_a5_results(
    output_dir: Path,
    variants: list[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> dict[str, Path]:
    if not variants:
        raise ValueError("no formal A5 results to write")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "a5_formal_results.json",
        "csv": output_dir / "a5_formal_results.csv",
        "latex": output_dir / "a5_formal_results.tex",
    }
    payload = {
        "schema_version": "formal-a5-results-v1.0",
        "provenance": provenance,
        "variants": variants,
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows: list[dict[str, object]] = []
    latex_by_scope: dict[str, list[dict[str, object]]] = {
        "all_queries": [],
        "graded_queries": [],
    }
    for variant in variants:
        for scope in ("all_queries", "graded_queries"):
            overall = dict(dict(variant[scope])["overall"])
            csv_rows.append(
                {
                    "scope": scope,
                    "variant": variant["variant"],
                    "branches": ",".join(str(value) for value in variant["branches"]),
                    "fusion_method": variant["fusion_method"],
                    **overall,
                }
            )
            latex_by_scope[scope].append({"variant": variant["variant"], **overall})
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    paths["latex"].write_text(
        "% All 100 source-positive queries\n"
        + _latex_rows(latex_by_scope["all_queries"])
        + "\n% Balanced 50-query graded candidate pool\n"
        + _latex_rows(latex_by_scope["graded_queries"]),
        encoding="utf-8",
    )
    return paths
