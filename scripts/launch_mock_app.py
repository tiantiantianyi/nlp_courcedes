from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.mock_service import MockSearchService
from anima_search.app.mock_ui import build_mock_app
from anima_search.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an annotation/index/model-free UI smoke demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--image-dir")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    image_dir = args.image_dir or resolve_path(config, config["data"][f"{args.split}_dir"])
    service = MockSearchService(
        config["project_root"], image_dir,
        result_count=int(config["retrieval"].get("result_count", 8)),
    )
    build_mock_app(service).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
