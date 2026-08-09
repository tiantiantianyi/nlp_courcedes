from __future__ import annotations

import argparse, json, time, sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.config import load_config, resolve_path
from anima_search.schemas import ManifestItem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the same images through three annotation prompts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="Train", choices=["Train", "Val"])
    parser.add_argument("--sample-size", type=int, default=60)
    args = parser.parse_args(); config = load_config(args.config); split = args.split.lower()
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    manifest = artifacts / "manifests" / f"{split}.jsonl"
    items = [ManifestItem.model_validate_json(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line][:args.sample_size]
    client = QwenVLClient(resolve_path(config, config["models"]["qwen_vl"]),
        config["runtime"]["dtype"], config["runtime"]["device"],
        config["runtime"]["max_image_pixels"])
    output = artifacts / "evaluation" / "prompt_outputs.jsonl"; output.parent.mkdir(parents=True, exist_ok=True)
    prompt_dir = Path(config["project_root"]) / "configs" / "prompts"
    with output.open("w", encoding="utf-8") as handle:
        for prompt_name in ("caption_basic", "caption_structured", "caption_verified"):
            prompt = (prompt_dir / f"{prompt_name}.txt").read_text(encoding="utf-8")
            for item in items:
                started = time.perf_counter()
                try:
                    with Image.open(Path(config["project_root"]) / item.relative_path) as image:
                        raw = client.generate(image.copy(), prompt, config["annotation"]["max_new_tokens"])
                    row = {"image_id": item.image_id, "prompt": prompt_name, "output": raw,
                           "elapsed_seconds": time.perf_counter() - started, "error": None}
                except Exception as exc:
                    row = {"image_id": item.image_id, "prompt": prompt_name, "output": None,
                           "elapsed_seconds": time.perf_counter() - started,
                           "error": f"{type(exc).__name__}: {exc}"}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
    print(f"Prompt outputs written to {output}. Add human scores in a spreadsheet for the report.")


if __name__ == "__main__": main()
