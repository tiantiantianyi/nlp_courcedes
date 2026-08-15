from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.evaluation.manual_set import (
    load_manifest,
    load_relevance_rows,
    load_tasks,
    validate_manual_set,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a human retrieval evaluation set.")
    parser.add_argument("--queries", type=Path, default=Path("evaluation/manual_val/queries.jsonl"))
    parser.add_argument("--relevance", type=Path, default=Path("evaluation/manual_val/relevance.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/val.jsonl"))
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()

    valid_ids = {item.image_id for item in load_manifest(args.manifest)}
    summary = validate_manual_set(
        load_tasks(args.queries),
        load_relevance_rows(args.relevance),
        expected_count=args.expected_count,
        valid_image_ids=valid_ids,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
