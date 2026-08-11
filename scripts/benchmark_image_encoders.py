from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.indexing.image_only import load_manifest_items
from anima_search.indexing.image_vector_index import ImageVectorIndex


DEFAULT_QUERIES = [
    "雨夜城市",
    "户外自然风景",
    "室内有人",
    "暖色调建筑",
    "道路上的汽车",
]


def directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.parts
    )


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_cuda_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return None


def run_encoder(
    *,
    name: str,
    model_path: Path,
    encoder_type: str,
    encoder_options: dict,
    image_ids: list[str],
    image_paths: list[Path],
    batch_size: int,
    queries: list[str],
    output_root: Path,
    device: str,
    dtype: str,
) -> dict:
    gc.collect()
    reset_cuda_peak()
    index = ImageVectorIndex(
        model_path,
        device,
        dtype,
        annotation_version="a7-resource-comparison",
        build_parameters={
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "similarity": "inner_product",
            "paired_sample": True,
        },
        encoder_type=encoder_type,
        encoder_options=encoder_options,
    )
    started = time.perf_counter()
    index.build(image_ids, image_paths, batch_size=batch_size)
    cuda_sync()
    build_seconds = time.perf_counter() - started

    index_dir = output_root / name
    index.save(index_dir)
    index_bytes = directory_size(index_dir)

    index.search("测试查询", limit=min(5, len(image_ids)))
    cuda_sync()
    query_rows = []
    for query in queries:
        started = time.perf_counter()
        results = index.search(query, limit=min(5, len(image_ids)))
        cuda_sync()
        query_rows.append({
            "query": query,
            "elapsed_seconds": time.perf_counter() - started,
            "top5": [
                {"image_id": image_id, "score": score}
                for image_id, score in results
            ],
        })

    result = {
        "name": name,
        "encoder_type": encoder_type,
        "encoder_options": encoder_options,
        "model_path": str(model_path),
        "model_bytes": directory_size(model_path),
        "records": len(image_ids),
        "batch_size": batch_size,
        "dimension": int(index.index.d),
        "build_seconds": build_seconds,
        "images_per_second": len(image_ids) / build_seconds,
        "peak_cuda_bytes": peak_cuda_bytes(),
        "index_bytes": index_bytes,
        "mean_warm_query_seconds": statistics.fmean(
            row["elapsed_seconds"] for row in query_rows
        ),
        "queries": query_rows,
    }
    index.unload_encoder()
    del index
    gc.collect()
    reset_cuda_peak()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Chinese-CLIP and jina-clip-v2 resource usage."
    )
    parser.add_argument("--config", default="configs/benchmark_8gb.yaml")
    parser.add_argument("--split", choices=["Train", "Val"], default="Val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--chinese-batch-size", type=int, default=4)
    parser.add_argument("--jina-batch-size", type=int, default=1)
    parser.add_argument(
        "--jina-dim",
        type=int,
        choices=[32, 64, 128, 256, 512, 768, 1024],
        default=512,
    )
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument(
        "--output",
        default="artifacts/a7_encoder_comparison.json",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    config = load_config(args.config)
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    manifest_path = artifacts / "manifests" / f"{args.split.lower()}.jsonl"
    items = load_manifest_items(manifest_path, args.split, args.limit)
    image_ids = [item.image_id for item in items]
    image_paths = [
        Path(config["project_root"]) / item.relative_path for item in items
    ]
    output_path = resolve_path(config, args.output)
    output_root = output_path.parent / "a7_indexes"
    output_root.mkdir(parents=True, exist_ok=True)
    queries = args.queries or DEFAULT_QUERIES

    rows = [
        run_encoder(
            name="chinese_clip_512",
            model_path=resolve_path(config, config["models"]["image_embedder"]),
            encoder_type="chinese_clip",
            encoder_options={},
            image_ids=image_ids,
            image_paths=image_paths,
            batch_size=args.chinese_batch_size,
            queries=queries,
            output_root=output_root,
            device=config["runtime"]["device"],
            dtype=config["runtime"]["dtype"],
        ),
        run_encoder(
            name=f"jina_clip_v2_{args.jina_dim}",
            model_path=resolve_path(config, config["models"]["jina_clip_v2"]),
            encoder_type="jina_clip_v2",
            encoder_options={
                "truncate_dim": args.jina_dim,
                "local_files_only": bool(
                    config["retrieval"].get(
                        "jina_clip_local_files_only",
                        True,
                    )
                ),
            },
            image_ids=image_ids,
            image_paths=image_paths,
            batch_size=args.jina_batch_size,
            queries=queries,
            output_root=output_root,
            device=config["runtime"]["device"],
            dtype=config["runtime"]["dtype"],
        ),
    ]
    payload = {
        "experiment": "A7 image encoder resource comparison",
        "paired_image_ids": image_ids,
        "queries": queries,
        "has_relevance_judgments": False,
        "quality_claim_allowed": False,
        "note": (
            "Resource comparison only. Add human relevance judgments before "
            "reporting retrieval quality."
        ),
        "encoders": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
