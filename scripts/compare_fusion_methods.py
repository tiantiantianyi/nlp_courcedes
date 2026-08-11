from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.config import load_config, resolve_path
from anima_search.evaluation.fusion_comparison import compare_fusion_methods


def load_queries(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare M5 RRF and normalized weighted fusion without claiming relevance quality."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--queries", type=Path, default=Path("configs/m6_benchmark_queries.jsonl"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    service = create_service(
        args.config,
        args.split,
        ["image", "text", "bm25"],
        "rrf",
    )
    queries = load_queries(args.queries)
    if queries:
        service.search(str(queries[0]["text"]), use_reranker=False)
    rows, summary = compare_fusion_methods(service, queries, top_k=args.top_k)
    config = load_config(args.config)
    output = args.output or (
        resolve_path(config, config["data"]["artifacts_dir"])
        / "evaluation/m5_fusion_comparison.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "queries": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
