from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

from anima_search.evaluation.runner import evaluate_queries


_A5_VARIANTS = (
    ("clip_only", ["image"]),
    ("text_only", ["text"]),
    ("bm25_only", ["bm25"]),
    ("rrf_three_way", ["image", "text", "bm25"]),
)


def a5_ablation_matrix() -> list[dict[str, object]]:
    return [
        {"variant": variant, "branches": list(branches), "reranker": False}
        for variant, branches in _A5_VARIANTS
    ]


def ablation_matrix() -> list[dict[str, object]]:
    """Backward-compatible name for the proposal's core A5 matrix."""
    return a5_ablation_matrix()


def run_a5_ablation(
    service_factory: Callable[[list[str]], object],
    queries: list[dict[str, object]],
    relevance: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in a5_ablation_matrix():
        branches = list(variant["branches"])
        service = service_factory(branches)
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
            }
        )
    return rows


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
