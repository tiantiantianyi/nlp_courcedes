from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.config import load_config, resolve_path
from anima_search.training.train_embedder import train_embedder


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch-size", type=int, default=16); args = parser.parse_args()
    config = load_config(args.config); artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    train_embedder(resolve_path(config, config["models"]["embedder"]), artifacts / "training" / "pairs.jsonl",
                   artifacts / "checkpoints" / "retriever", args.epochs, args.batch_size, config["seed"])


if __name__ == "__main__": main()
