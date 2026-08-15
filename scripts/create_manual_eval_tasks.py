from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.evaluation.manual_set import (
    load_manifest,
    sample_manual_tasks,
    write_relevance,
    write_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create human-written retrieval query tasks directly from Val images."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/manual_val"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    manifest_path = args.manifest or artifacts / "manifests" / "val.jsonl"
    task_path = args.output_dir / "queries.jsonl"
    relevance_path = args.output_dir / "relevance.csv"
    if not args.force and (task_path.exists() or relevance_path.exists()):
        parser.error(
            f"{args.output_dir} already contains evaluation work; use --force only to replace it"
        )

    manifest = load_manifest(manifest_path)
    tasks = sample_manual_tasks(manifest, count=args.count, seed=args.seed)
    write_tasks(task_path, tasks)
    write_relevance(relevance_path, [])
    print(f"Created {len(tasks)} human-review tasks")
    print(f"Queries: {task_path}")
    print(f"Relevance: {relevance_path}")
    print("No query or relevance label was generated automatically.")


if __name__ == "__main__":
    main()
