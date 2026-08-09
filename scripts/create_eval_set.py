from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.config import load_config, resolve_path
from anima_search.schemas import ImageAnnotation


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an editable Val retrieval benchmark seed.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args(); config = load_config(args.config)
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    source = artifacts / "annotations" / f"val.{config['annotation']['prompt_version']}.jsonl"
    annotations = [ImageAnnotation.model_validate_json(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    annotations = [item for item in annotations if item.duplicate_of is None][:args.count]
    output = artifacts / "evaluation"; output.mkdir(parents=True, exist_ok=True)
    query_rows = [{"query_id": f"q{index:03d}", "text": item.search_queries[0],
                   "category": "auto_seed", "source_image_id": item.image_id, "reviewed": False}
                  for index, item in enumerate(annotations, start=1)]
    (output / "val_queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in query_rows) + "\n", encoding="utf-8")
    with (output / "val_relevance.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "image_id", "relevance", "annotator", "note"])
        writer.writeheader()
        for row in query_rows:
            writer.writerow({"query_id": row["query_id"], "image_id": row["source_image_id"],
                "relevance": 2, "annotator": "auto_seed", "note": "人工改写查询、补充相关图片后，将 JSONL reviewed 改为 true"})
    print(f"Created {len(query_rows)} query seeds in {output}")


if __name__ == "__main__": main()
