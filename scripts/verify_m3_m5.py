from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end M3-M5 retrieval verification against built indexes."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--query", default="日落高速公路，没有人物")
    parser.add_argument("--expected-top-id")
    parser.add_argument(
        "--require-branches",
        default="image,text,bm25",
        help="Comma-separated branches that must be active; use an empty value to disable.",
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--rerank-count",
        type=int,
        help="Override the number of candidates sent to Qwen for this verification run.",
    )
    args = parser.parse_args()
    if args.rerank_count is not None and args.rerank_count <= 0:
        parser.error("--rerank-count must be positive")

    service = create_service(args.config, args.split)
    if args.rerank_count is not None:
        service.config["retrieval"]["rerank_count"] = args.rerank_count
    results = service.search(args.query, use_reranker=args.rerank)
    if not results:
        raise RuntimeError("M3-M5 verification returned no search results")

    required = {
        branch.strip()
        for branch in args.require_branches.split(",")
        if branch.strip()
    }
    active = set(results[0].active_branches)
    missing = sorted(required - active)
    if missing:
        raise RuntimeError(f"required retrieval branches are inactive: {missing}")
    if args.expected_top_id and results[0].image_id != args.expected_top_id:
        raise RuntimeError(
            f"unexpected top result: expected {args.expected_top_id}, "
            f"got {results[0].image_id}"
        )

    payload = {
        "status": "passed",
        "split": args.split,
        "query": args.query,
        "rerank": args.rerank,
        "active_branches": results[0].active_branches,
        "top_results": [
            {
                "rank": rank,
                "image_id": result.image_id,
                "fused_score": result.fused_score,
                "rerank_score": result.rerank_score,
                "branch_ranks": result.branch_ranks,
            }
            for rank, result in enumerate(results[:5], start=1)
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
