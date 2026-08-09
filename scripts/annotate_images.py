from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.annotation.pipeline import annotate_items
from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.config import load_config, resolve_path
from anima_search.schemas import ManifestItem


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["Train", "Val"], required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument(
        "--image-id",
        action="append",
        dest="image_ids",
        metavar="IMAGE_ID",
        help="Annotate a specific manifest ID; repeat this option for multiple images.",
    )
    args = parser.parse_args(); config = load_config(args.config); split = args.split.lower()
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    items = [ManifestItem.model_validate_json(line) for line in (artifacts / "manifests" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if args.image_ids:
        known_ids = {item.image_id for item in items}
        missing_ids = sorted(set(args.image_ids) - known_ids)
        if missing_ids:
            parser.error(f"Unknown image ID(s) for {args.split}: {', '.join(missing_ids)}")
        requested_ids = set(args.image_ids)
        items = [item for item in items if item.image_id in requested_ids]
    elif args.limit:
        items = items[:args.limit]
    prompt = resolve_path(config, config["annotation"]["prompt"]).read_text(encoding="utf-8")
    client = QwenVLClient(resolve_path(config, config["models"]["qwen_vl"]),
        config["runtime"]["dtype"], config["runtime"]["device"],
        config["runtime"]["max_image_pixels"])
    output = artifacts / "annotations" / f"{split}.{config['annotation']['prompt_version']}.jsonl"
    success, failed = annotate_items(items, client, prompt, output, Path(config["project_root"]),
        config["annotation"]["prompt_version"], config["annotation"]["retries"], config["annotation"]["max_new_tokens"])
    print(json.dumps({"success": success, "failed": failed, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__": main()
