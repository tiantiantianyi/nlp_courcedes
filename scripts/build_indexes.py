from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.indexing.bm25_index import BM25Index
from anima_search.indexing.documents import (
    annotation_to_dense_document,
    annotation_to_sparse_document,
)
from anima_search.indexing.image_vector_index import ImageVectorIndex
from anima_search.indexing.index_manifest import sha256_file, write_index_manifest
from anima_search.indexing.vector_index import VectorIndex
from anima_search.schemas import ImageAnnotation

VALID_BRANCHES = ("image", "text", "bm25")


def _parse_branches(value: str | None, config: dict) -> list[str]:
    raw = value.split(",") if value else config["retrieval"].get("enabled_branches", VALID_BRANCHES)
    branches = list(dict.fromkeys(str(item).strip().lower() for item in raw if str(item).strip()))
    invalid = sorted(set(branches) - set(VALID_BRANCHES))
    if invalid:
        raise ValueError(f"unsupported retrieval branches: {', '.join(invalid)}")
    if not branches:
        raise ValueError("at least one retrieval branch must be enabled")
    return branches


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_annotations(path: Path, split: str, limit: int | None) -> list[ImageAnnotation]:
    if not path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {path}")
    annotations = [
        ImageAnnotation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotations = sorted(annotations, key=lambda item: item.image_id)
    if limit is not None:
        annotations = annotations[:limit]
    if not annotations:
        raise ValueError("annotation file contains no usable records")
    ids = [item.image_id for item in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("annotation image_id values must be unique")
    wrong_split = [item.image_id for item in annotations if item.split.lower() != split]
    if wrong_split:
        raise ValueError(f"annotation split mismatch for image IDs: {wrong_split[:5]}")
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M3 sparse, text, and image retrieval indexes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["Train", "Val"], required=True)
    parser.add_argument("--branches", help="Comma-separated subset of image,text,bm25")
    parser.add_argument("--limit", type=int, help="Build only the first N records for a smoke test")
    parser.add_argument("--allow-missing-branches", action="store_true",
                        help="Continue building other branches when one model is unavailable")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    config = load_config(args.config)
    split = args.split.lower()
    branches = _parse_branches(args.branches, config)
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    annotation_path = artifacts / "annotations" / f"{split}.{config['annotation']['prompt_version']}.jsonl"
    annotations = _load_annotations(annotation_path, split, args.limit)
    image_ids = [item.image_id for item in annotations]
    dense_documents = [annotation_to_dense_document(item) for item in annotations]
    sparse_documents = [annotation_to_sparse_document(item) for item in annotations]
    output = artifacts / "indexes" / split
    output.mkdir(parents=True, exist_ok=True)

    built: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for branch in branches:
        try:
            if branch == "bm25":
                index = BM25Index(
                    image_ids, sparse_documents, config["annotation"]["prompt_version"],
                    {"tokenizer": "jieba", "candidate_count": config["retrieval"]["candidate_count"]},
                )
                index.save(output / "bm25.pkl")
                built[branch] = {"kind": "sparse", "record_count": len(image_ids)}
            elif branch == "text":
                model_path = resolve_path(config, config["models"]["embedder"])
                batch_size = int(config["retrieval"].get("text_batch_size", 32))
                index = VectorIndex(
                    model_path, config["runtime"]["device"], config["annotation"]["prompt_version"],
                    {"batch_size": batch_size, "normalize_embeddings": True,
                     "similarity": "inner_product"},
                )
                index.build(image_ids, dense_documents, batch_size=batch_size)
                index.save(output / "text")
                built[branch] = {
                    "kind": "text_vector", "record_count": len(image_ids),
                    "model_digest": index.model_digest, "dimension": index.index.d,
                }
                index.model = None
                _release_cuda()
            elif branch == "image":
                encoder_type = str(
                    config["retrieval"].get("image_encoder_type", "chinese_clip")
                )
                if encoder_type == "jina_clip_v2":
                    model_path = resolve_path(
                        config,
                        config["models"]["jina_clip_v2"],
                    )
                    encoder_options = {
                        "truncate_dim": int(
                            config["retrieval"].get(
                                "jina_clip_truncate_dim",
                                512,
                            )
                        ),
                        "local_files_only": bool(
                            config["retrieval"].get(
                                "jina_clip_local_files_only",
                                True,
                            )
                        ),
                    }
                    batch_size = int(
                        config["retrieval"].get(
                            "jina_clip_image_batch_size",
                            1,
                        )
                    )
                else:
                    model_path = resolve_path(
                        config,
                        config["models"]["image_embedder"],
                    )
                    encoder_options = {}
                    batch_size = int(
                        config["retrieval"].get("image_batch_size", 8)
                    )
                index = ImageVectorIndex(
                    model_path, config["runtime"]["device"], config["runtime"]["dtype"],
                    config["annotation"]["prompt_version"],
                    {"batch_size": batch_size, "normalize_embeddings": True,
                     "similarity": "inner_product"},
                    encoder_type=encoder_type,
                    encoder_options=encoder_options,
                )
                image_paths = [Path(config["project_root"]) / item.relative_path for item in annotations]
                index.build(image_ids, image_paths, batch_size=batch_size)
                index.save(output / "image")
                built[branch] = {
                    "kind": "image_vector", "record_count": len(image_ids),
                    "model_digest": index.model_digest, "dimension": index.index.d,
                }
                index.unload_encoder()
                _release_cuda()
        except Exception as exc:
            failures[branch] = f"{type(exc).__name__}: {exc}"
            if not args.allow_missing_branches:
                raise RuntimeError(f"failed to build {branch} retrieval branch: {failures[branch]}") from exc

    if not built:
        raise RuntimeError(f"no retrieval indexes were built: {failures}")

    (output / "annotations.json").write_text(
        json.dumps([item.model_dump() for item in annotations], ensure_ascii=False), encoding="utf-8"
    )
    manifest = write_index_manifest(
        output / "manifest.json",
        split=split,
        image_ids=image_ids,
        annotation_path=annotation_path,
        annotation_version=config["annotation"]["prompt_version"],
        branches=built,
        config_digest=sha256_file(Path(args.config).resolve()),
    )
    if failures:
        manifest["branch_failures"] = failures
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"output": str(output), "records": len(image_ids),
                      "active_branches": sorted(built), "failures": failures},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
