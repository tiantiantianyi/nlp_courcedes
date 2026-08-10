from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.config import load_config, resolve_path
from anima_search.evaluation.ablation import (
    a5_ablation_matrix,
    run_a5_ablation,
    write_ablation_results,
)
from anima_search.evaluation.ground_truth import load_queries, load_relevance
from anima_search.evaluation.runner import validate_formal_queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run proposal experiment A5: CLIP/text/BM25 single branches vs three-way RRF."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--queries", default="artifacts/evaluation/val_queries.jsonl")
    parser.add_argument("--relevance", default="artifacts/evaluation/val_relevance.csv")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    matrix = a5_ablation_matrix()
    if args.dry_run:
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
        return

    queries = load_queries(Path(args.queries))
    relevance = load_relevance(Path(args.relevance))
    validate_formal_queries(queries)
    missing = [str(row["query_id"]) for row in queries if str(row["query_id"]) not in relevance]
    if missing:
        raise ValueError(
            "Evaluation refused: no relevance judgments for query IDs: "
            + ", ".join(missing[:10])
        )

    config = load_config(args.config)
    output_dir = args.output_dir or (
        resolve_path(config, config["data"]["artifacts_dir"]) / "evaluation"
    )
    rows = run_a5_ablation(
        lambda branches: create_service(args.config, args.split, branches),
        queries,
        relevance,
    )
    paths = write_ablation_results(output_dir, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("Outputs:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
