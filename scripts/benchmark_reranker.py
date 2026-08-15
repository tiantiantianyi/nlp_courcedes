from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.evaluation.rerank_benchmark import (
    benchmark_candidates,
    collect_candidates,
    write_benchmark,
)
from anima_search.retrieval.reranker import VisualReranker


def _release_retrieval_encoders(service: object) -> list[str]:
    released: list[str] = []
    indexes = getattr(getattr(service, "searcher", None), "indexes", {})
    for name, index in indexes.items():
        if hasattr(index, "unload_encoder"):
            index.unload_encoder()
            released.append(str(name))
        elif hasattr(index, "model") and getattr(index, "model") is not None:
            index.model = None
            released.append(str(name))
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return released


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure pointwise Qwen-VL reranking latency, failures, and peak CUDA memory."
    )
    parser.add_argument("query", help="query used to retrieve a fixed candidate set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=["image", "text", "bm25"],
        help="override candidate retrieval branches, for example: --branches image",
    )
    parser.add_argument("--top-k", type=int, choices=[3, 5], default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/reranker_benchmark.jsonl"),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    service = create_service(args.config, args.split, args.branches)
    candidates = collect_candidates(service, args.query, args.top_k)
    released = _release_retrieval_encoders(service)
    settings = service.config["retrieval"]
    with service.manager.qwen_session() as qwen:
        reranker = VisualReranker(
            qwen,
            service.reranker_prompt,
            Path(service.config["project_root"]),
            settings["rrf_weight"],
            settings["vlm_weight"],
            settings.get("rerank_max_new_tokens", 128),
        )
        records, summary = benchmark_candidates(
            args.query,
            args.split,
            candidates,
            reranker,
            args.repeats,
        )
    summary["retrieval_branches"] = args.branches or settings.get("enabled_branches", [])
    summary["released_retrieval_encoders"] = released
    summary_path = write_benchmark(args.output, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSONL: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
