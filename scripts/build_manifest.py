from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.config import load_config, resolve_path
from anima_search.data.manifest import mark_cross_split_duplicates, scan_split


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(); config = load_config(args.config); root = Path(config["project_root"])
    train = scan_split(resolve_path(config, config["data"]["train_dir"]), "Train", root)
    val = scan_split(resolve_path(config, config["data"]["val_dir"]), "Val", root)
    mark_cross_split_duplicates([train, val])
    output = resolve_path(config, config["data"]["artifacts_dir"]) / "manifests"; output.mkdir(parents=True, exist_ok=True)
    for name, items in (("train", train), ("val", val)):
        (output / f"{name}.jsonl").write_text("\n".join(item.model_dump_json() for item in items) + "\n", encoding="utf-8")
    quality = {"train_count": len(train), "val_count": len(val), "invalid": [x.image_id for x in train + val if not x.valid],
               "duplicates": {x.image_id: x.duplicate_of for x in train + val if x.duplicate_of}}
    (output / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
