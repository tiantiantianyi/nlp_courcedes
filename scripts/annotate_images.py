from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.annotation.pipeline import annotate_items
from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.config import load_config, resolve_path
from anima_search.schemas import ManifestItem


def load_scene_prompts(path: Path, base_prompt: str) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        image_id = str(payload.get("image_id", "")).strip()
        label = str(payload.get("label", "")).strip()
        category = str(payload.get("category", "")).strip()
        suffix = str(payload.get("prompt_suffix", "")).strip()
        if not image_id or not label or not category or not suffix:
            raise ValueError(f"invalid scene route at {path}:{line_number}")
        if image_id in prompts:
            raise ValueError(f"duplicate scene route for image_id: {image_id}")
        prompts[image_id] = (
            f"{base_prompt.rstrip()}\n\n"
            f"场景路由：{label}（{category}）。{suffix}"
        )
    if not prompts:
        raise ValueError(f"scene route file contains no records: {path}")
    return prompts


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
    parser.add_argument(
        "--scene-routes",
        type=Path,
        help="M0 JSONL output; append each route's scene-specific suffix to the M1 prompt.",
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
    prompts_by_image_id = load_scene_prompts(args.scene_routes, prompt) if args.scene_routes else None
    if prompts_by_image_id is not None:
        expected_ids = {item.image_id for item in items if item.valid}
        missing_routes = sorted(expected_ids - set(prompts_by_image_id))
        if missing_routes:
            raise ValueError(
                f"scene route coverage mismatch; missing={missing_routes[:10]}"
            )
    client = QwenVLClient(resolve_path(config, config["models"]["qwen_vl"]),
        config["runtime"]["dtype"], config["runtime"]["device"],
        config["runtime"]["max_image_pixels"])
    output = artifacts / "annotations" / f"{split}.{config['annotation']['prompt_version']}.jsonl"
    success = failed = 0
    if prompts_by_image_id is None:
        batches = [(prompt, items)]
    else:
        grouped: dict[str, list[ManifestItem]] = {}
        for item in items:
            if item.valid:
                grouped.setdefault(prompts_by_image_id[item.image_id], []).append(item)
        batches = list(grouped.items())
    for selected_prompt, selected_items in batches:
        batch_success, batch_failed = annotate_items(
            selected_items, client, selected_prompt, output, Path(config["project_root"]),
            config["annotation"]["prompt_version"], config["annotation"]["retries"],
            config["annotation"]["max_new_tokens"],
        )
        success += batch_success
        failed += batch_failed
    print(json.dumps({"success": success, "failed": failed, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__": main()
