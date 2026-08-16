from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.config import load_config, resolve_path
from anima_search.retrieval.openai_compatible import OpenAICompatibleTextClient
from anima_search.retrieval.query_parser import QueryParser


def build_parser(config: dict, backend: str) -> tuple[QueryParser, object | None]:
    retrieval = config["retrieval"]
    aliases_path = resolve_path(
        config,
        retrieval.get("aliases", "configs/retrieval_aliases.yaml"),
    )
    aliases = (
        yaml.safe_load(aliases_path.read_text(encoding="utf-8"))
        if aliases_path.is_file()
        else {}
    )
    prompt_path = resolve_path(config, "configs/prompts/query_parser.txt")
    prompt = prompt_path.read_text(encoding="utf-8")
    generator = None
    if backend == "local_qwen":
        generator = QwenVLClient(
            resolve_path(config, config["models"]["qwen_vl"]),
            config["runtime"]["dtype"],
            config["runtime"]["device"],
            config["runtime"]["max_image_pixels"],
        )
    elif backend == "openai_compatible":
        api = retrieval.get("query_parser_api", {})
        generator = OpenAICompatibleTextClient(
            str(api.get("base_url", "")),
            str(api.get("model", "")),
            api_key_env=str(api.get("api_key_env", "SILICONFLOW_API_KEY")),
            timeout_seconds=float(api.get("timeout_seconds", 30)),
            max_retries=int(api.get("max_retries", 2)),
        )
    return QueryParser(
        generator if backend == "openai_compatible" else None,
        prompt,
        aliases,
    ), generator


def load_query_records(
    queries: list[str] | None,
    queries_file: Path | None,
) -> list[tuple[str, str]]:
    records = [
        (f"cli-{index:03d}", query.strip())
        for index, query in enumerate(queries or [], start=1)
        if query.strip()
    ]
    if queries_file is not None:
        if not queries_file.is_file():
            raise FileNotFoundError(f"queries file does not exist: {queries_file}")
        for line_number, line in enumerate(
            queries_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {queries_file}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"query record at {queries_file}:{line_number} must be an object"
                )
            query = str(payload.get("query") or payload.get("text") or "").strip()
            if not query:
                raise ValueError(
                    f"query record at {queries_file}:{line_number} needs query or text"
                )
            query_id = str(
                payload.get("query_id") or f"file-{line_number:03d}"
            ).strip()
            records.append((query_id, query))
    query_ids = [query_id for query_id, _ in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify M4 query parser backends.")
    parser.add_argument("--config", default="configs/benchmark_8gb.yaml")
    parser.add_argument(
        "--backend",
        choices=["rules", "local_qwen", "openai_compatible"],
        default="rules",
    )
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.queries and args.queries_file is None:
        parser.error("provide --query or --queries-file")

    config = load_config(args.config)
    query_parser, generator = build_parser(config, args.backend)
    query_records = load_query_records(args.queries, args.queries_file)
    rows = []
    try:
        for query_id, query in query_records:
            started = time.perf_counter()
            if args.backend == "local_qwen":
                parsed = query_parser.parse(query, generator)
            else:
                parsed = query_parser.parse(query)
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "requested_backend": args.backend,
                    "effective_backend": query_parser.last_backend,
                    "fallback_error": query_parser.last_error,
                    "elapsed_seconds": time.perf_counter() - started,
                    "parsed": parsed.model_dump(),
                }
            )
    finally:
        if generator is not None and hasattr(generator, "unload"):
            generator.unload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "query_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
