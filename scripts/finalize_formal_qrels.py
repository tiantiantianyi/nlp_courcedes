from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.evaluation.candidate_review import (
    load_candidate_pool,
    load_candidate_relevance,
)
from anima_search.evaluation.manual_set import load_relevance_rows, load_tasks
from anima_search.evaluation.qrels_validation import (
    finalize_qrels,
    write_finalized_qrels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate candidate judgments and assemble formal retrieval qrels."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/formal_val_100/queries.jsonl"),
    )
    parser.add_argument(
        "--source-relevance",
        type=Path,
        default=Path("evaluation/formal_val_100/relevance.csv"),
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("artifacts/evaluation/formal/relevance_pool.jsonl"),
    )
    parser.add_argument(
        "--candidate-relevance",
        type=Path,
        default=Path("artifacts/evaluation/formal/candidate_relevance.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation/formal")
    )
    args = parser.parse_args()

    final_queries, final_rows, summary = finalize_qrels(
        load_tasks(args.queries),
        load_relevance_rows(args.source_relevance),
        load_candidate_pool(args.pool),
        load_candidate_relevance(args.candidate_relevance),
    )
    paths = write_finalized_qrels(
        args.output_dir,
        final_queries,
        final_rows,
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Outputs:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
