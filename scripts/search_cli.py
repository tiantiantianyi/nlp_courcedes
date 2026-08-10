from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the image collection from a terminal."
    )
    parser.add_argument("query")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=["image", "text", "bm25"],
        help="override enabled branches, for example: --branches image",
    )
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    service = create_service(args.config, args.split, args.branches)
    results = service.search(args.query, args.rerank)
    print(
        json.dumps(
            [result.model_dump() for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
