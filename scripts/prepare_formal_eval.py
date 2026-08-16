from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.evaluation.formal_set import merge_reviewed_sets
from anima_search.evaluation.manual_set import write_relevance, write_tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independently reviewed retrieval query sets."
    )
    parser.add_argument("--queries", type=Path, nargs="+", required=True)
    parser.add_argument("--relevance", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/formal_val_100")
    )
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()

    tasks, rows, summary = merge_reviewed_sets(
        args.queries,
        args.relevance,
        expected_count=args.expected_count,
    )
    input_paths = [*args.queries, *args.relevance]
    report = {
        **summary,
        "schema_version": "formal-evaluation-merge-v1.0",
        "input_sha256": {str(path): _sha256(path) for path in input_paths},
    }
    write_tasks(args.output_dir / "queries.jsonl", tasks)
    write_relevance(args.output_dir / "relevance.csv", rows)
    _write_json(args.output_dir / "merge_report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Queries: {args.output_dir / 'queries.jsonl'}")
    print(f"Relevance: {args.output_dir / 'relevance.csv'}")


if __name__ == "__main__":
    main()
