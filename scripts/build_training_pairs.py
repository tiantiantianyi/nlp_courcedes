from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.config import load_config, resolve_path
from anima_search.schemas import ImageAnnotation
from anima_search.training.pairs import build_training_pairs


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml"); args = parser.parse_args()
    config = load_config(args.config); artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    path = artifacts / "annotations" / f"train.{config['annotation']['prompt_version']}.jsonl"
    annotations = [ImageAnnotation.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    pairs = build_training_pairs(annotations, config["seed"]); output = artifacts / "training" / "pairs.jsonl"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(pair.to_dict(), ensure_ascii=False) for pair in pairs) + "\n", encoding="utf-8")
    print(f"Wrote {len(pairs)} training pairs to {output}")


if __name__ == "__main__": main()
