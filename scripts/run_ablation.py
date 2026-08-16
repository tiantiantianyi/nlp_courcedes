from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.config import load_config, resolve_path
from anima_search.evaluation.ablation import (
    a5_ablation_matrix,
    run_formal_a5_ablation,
    write_formal_a5_results,
)
from anima_search.evaluation.candidate_review import load_candidate_pool
from anima_search.evaluation.ground_truth import load_queries, load_relevance
from anima_search.evaluation.runner import validate_formal_queries
from anima_search.indexing.index_manifest import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run proposal experiment A5: CLIP/text/BM25 single branches vs three-way RRF."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--queries", default="artifacts/evaluation/formal/val_queries.jsonl")
    parser.add_argument("--relevance", default="artifacts/evaluation/formal/val_relevance.csv")
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("artifacts/evaluation/formal/relevance_pool.jsonl"),
    )
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
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    output_dir = args.output_dir or artifacts / "evaluation" / "formal" / "a5"
    pool = load_candidate_pool(args.pool)
    graded_candidate_ids = {
        str(row["query_id"]): {
            str(candidate["image_id"]) for candidate in list(row["candidates"])
        }
        for row in pool
    }
    variants = run_formal_a5_ablation(
        lambda branches, fusion_method: create_service(
            args.config, args.split, branches, fusion_method
        ),
        queries,
        relevance,
        graded_candidate_ids=graded_candidate_ids,
    )
    query_path = Path(args.queries)
    relevance_path = Path(args.relevance)
    config_path = Path(args.config)
    index_manifest_path = artifacts / "indexes" / args.split / "manifest.json"
    provenance = {
        "queries_sha256": sha256_file(query_path),
        "qrels_sha256": sha256_file(relevance_path),
        "candidate_pool_sha256": sha256_file(args.pool),
        "config_sha256": sha256_file(config_path),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "actual_variants": matrix,
        "runtime_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    paths = write_formal_a5_results(output_dir, variants, provenance=provenance)
    print(
        json.dumps(
            {"provenance": provenance, "variants": variants}, ensure_ascii=False, indent=2
        )
    )
    print("Outputs:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
