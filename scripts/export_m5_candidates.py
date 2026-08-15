from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.config import load_config, resolve_path
from anima_search.delivery.m5_candidates import build_m5_candidate_batch
from anima_search.indexing.index_manifest import sha256_file


VALID_CATEGORIES = {"simple", "compositional", "negative", "count", "ocr"}


def _load_queries(path: Path) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid query JSON at line {line_number}: {exc}") from exc
        query_id = str(payload.get("query_id", "")).strip()
        query = str(payload.get("query") or payload.get("text") or "").strip()
        category = str(payload.get("category", "")).strip().lower()
        if not query_id or not query:
            raise ValueError(f"query line {line_number} requires query_id and query/text")
        if query_id in seen:
            raise ValueError(f"duplicate query_id {query_id!r} at line {line_number}")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"query {query_id!r} category must be one of {sorted(VALID_CATEGORIES)}"
            )
        seen.add(query_id)
        queries.append({"query_id": query_id, "query": query, "category": category})
    if not queries:
        raise ValueError("query file contains no usable records")
    return queries


def _write_snapshot(path: Path, config: dict, *, config_path: Path, split: str) -> str:
    snapshot = {
        "schema_version": "m5-retrieval-config-v1",
        "source_config": str(config_path),
        "split": split,
        "annotation_version": config["annotation"]["prompt_version"],
        "retrieval": config["retrieval"],
        "delivery": {
            "top_k": 20,
            "branch_candidate_depth": "full_split",
            "adaptive_positive_filter_fallback": "soft",
            "always_hard": ["excluded_terms", "required_terms", "ocr_terms", "count"],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export M5 fused Top-20 candidates using m5-to-m6-v1.0."
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=["image", "text", "bm25"],
        help="Optional retrieval branch subset for smoke/degraded delivery checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/m5_to_m6_candidates.jsonl"),
    )
    parser.add_argument("--config-snapshot", type=Path)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.branches:
        config["retrieval"]["enabled_branches"] = list(dict.fromkeys(args.branches))
    fusion_method = str(config["retrieval"].get("fusion_method", "rrf"))
    if fusion_method not in {"rrf", "weighted"}:
        raise ValueError("M5 interface supports fusion_method rrf or weighted")
    config["retrieval"]["result_count"] = 20
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    manifest_path = artifacts / "indexes" / args.split / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"index manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_count = int(manifest.get("record_count", 0))
    if record_count < 20:
        raise ValueError(f"index must contain at least 20 records; found {record_count}")
    # M5 delivery filters before fusion. Full-depth branch recall prevents a
    # selective structured filter from accidentally shrinking a valid Top-20.
    config["retrieval"]["candidate_count"] = max(
        record_count, int(config["retrieval"]["candidate_count"])
    )
    annotation_version = str(manifest.get("annotation_version", "")).strip()
    if annotation_version != config["annotation"]["prompt_version"]:
        raise ValueError(
            "index annotation_version does not match the configured Qwen3.5 version"
        )

    snapshot_path = args.config_snapshot or args.output.with_name(
        "m5_retrieval_config.snapshot.json"
    )
    config_sha256 = _write_snapshot(
        snapshot_path.resolve(), config, config_path=config_path, split=args.split
    )

    service = create_service(
        str(config_path),
        args.split,
        enabled_branches=config["retrieval"]["enabled_branches"],
    )
    service.config["retrieval"]["candidate_count"] = config["retrieval"]["candidate_count"]
    service.config["retrieval"]["result_count"] = 20
    queries = _load_queries(args.queries.resolve())
    manifest_sha256 = sha256_file(manifest_path)
    records = []
    softened_queries: list[str] = []
    for query in queries:
        results = service.search(query["query"], use_reranker=False)
        if len(results) < 20:
            filter_instance = service.searcher.annotation_filter
            previous_mode = filter_instance.positive_filter_mode
            filter_instance.positive_filter_mode = "soft"
            try:
                results = service.search(query["query"], use_reranker=False)
            finally:
                filter_instance.positive_filter_mode = previous_mode
            softened_queries.append(query["query_id"])
        if len(results) < 20:
            raise ValueError(
                f"query {query['query_id']!r} has only {len(results)} candidates after "
                "full-depth recall and soft-positive fallback; revise hard constraints"
            )
        records.append(
            build_m5_candidate_batch(
                query_id=query["query_id"],
                query=query["query"],
                category=query["category"],
                split=args.split,
                fusion_method=fusion_method,
                annotation_version=annotation_version,
                index_manifest_sha256=manifest_sha256,
                config_sha256=config_sha256,
                results=results,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "query_count": len(records),
                "candidate_count": len(records) * 20,
                "schema_version": "m5-to-m6-v1.0",
                "index_manifest": str(manifest_path),
                "config_snapshot": str(snapshot_path),
                "softened_query_ids": softened_queries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
