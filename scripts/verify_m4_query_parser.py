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


DEFAULT_QUERIES = [
    "不要人物，寻找冷色调的雨夜城市",
    "至少三辆车的城市街景",
    "找招牌写着“老王面馆”的照片",
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify M4 query parser backends.")
    parser.add_argument("--config", default="configs/benchmark_8gb.yaml")
    parser.add_argument(
        "--backend",
        choices=["rules", "local_qwen", "openai_compatible"],
        default="rules",
    )
    parser.add_argument("--query", action="append", dest="queries")
    args = parser.parse_args()

    config = load_config(args.config)
    query_parser, generator = build_parser(config, args.backend)
    rows = []
    try:
        for query in args.queries or DEFAULT_QUERIES:
            started = time.perf_counter()
            if args.backend == "local_qwen":
                parsed = query_parser.parse(query, generator)
            else:
                parsed = query_parser.parse(query)
            rows.append(
                {
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
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
