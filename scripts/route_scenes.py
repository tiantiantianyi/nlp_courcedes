from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.routing.scene_router import SceneRouter


def reconstruct_vectors(index: object) -> np.ndarray:
    if getattr(index, "index", None) is None:
        raise RuntimeError("image FAISS index is not loaded")
    return np.stack(
        [index.index.reconstruct(position) for position in range(index.index.ntotal)]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route indexed images into M0 scene categories with Chinese-CLIP."
    )
    parser.add_argument("--config", default="configs/benchmark_8gb.yaml")
    parser.add_argument("--routing-config", type=Path, default=Path("configs/scene_routing.yaml"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/routing/val_scene_routes.jsonl"),
    )
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    service = create_service(args.config, args.split, ["image"])
    image_index = service.searcher.indexes["image"]
    encoder = image_index._load_encoder()
    routing_config = yaml.safe_load(args.routing_config.read_text(encoding="utf-8"))
    router = SceneRouter.from_config(encoder, routing_config)
    routes = router.route_vectors(
        list(image_index.image_ids),
        reconstruct_vectors(image_index),
        top_n=args.top_n,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(route.as_dict(), ensure_ascii=False) + "\n" for route in routes),
        encoding="utf-8",
    )
    distribution = Counter(route.category for route in routes)
    summary = {
        "split": args.split,
        "record_count": len(routes),
        "routing_version": routing_config.get("version", ""),
        "category_count": len(router.definitions),
        "distribution": dict(sorted(distribution.items())),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    image_index.unload_encoder()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
