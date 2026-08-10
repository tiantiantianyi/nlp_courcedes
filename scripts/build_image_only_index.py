from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.indexing.image_only import (
    IMAGE_ONLY_ANNOTATION_VERSION,
    load_manifest_items,
    placeholder_annotations,
)
from anima_search.indexing.image_vector_index import ImageVectorIndex
from anima_search.indexing.index_manifest import sha256_file, write_index_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Chinese-CLIP image branch from a manifest without M1/M2 annotations."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["Train", "Val"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    config = load_config(args.config)
    split = args.split.lower()
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    manifest_path = artifacts / "manifests" / f"{split}.jsonl"
    items = load_manifest_items(manifest_path, args.split, args.limit)
    annotations = placeholder_annotations(items)
    image_ids = [item.image_id for item in annotations]
    image_paths = [Path(config["project_root"]) / item.relative_path for item in annotations]
    output = artifacts / "indexes" / split

    index = ImageVectorIndex(
        resolve_path(config, config["models"]["image_embedder"]),
        config["runtime"]["device"],
        config["runtime"]["dtype"],
        IMAGE_ONLY_ANNOTATION_VERSION,
        {
            "batch_size": int(config["retrieval"].get("image_batch_size", 8)),
            "normalize_embeddings": True,
            "similarity": "inner_product",
            "source": "manifest",
        },
    )
    index.build(
        image_ids,
        image_paths,
        batch_size=int(config["retrieval"].get("image_batch_size", 8)),
    )
    index.save(output / "image")
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotations.json").write_text(
        json.dumps([item.model_dump() for item in annotations], ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = write_index_manifest(
        output / "manifest.json",
        split=split,
        image_ids=image_ids,
        annotation_path=manifest_path,
        annotation_version=IMAGE_ONLY_ANNOTATION_VERSION,
        branches={
            "image": {
                "kind": "image_vector",
                "record_count": len(image_ids),
                "model_digest": index.model_digest,
                "dimension": index.index.d,
            }
        },
        config_digest=sha256_file(Path(args.config).resolve()),
    )
    manifest["source_kind"] = "image_manifest"
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    index.unload_encoder()
    gc.collect()
    print(json.dumps({
        "output": str(output),
        "records": len(image_ids),
        "active_branches": ["image"],
        "annotation_version": IMAGE_ONLY_ANNOTATION_VERSION,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
