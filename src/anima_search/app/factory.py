from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.app.service import SearchService
from anima_search.config import load_config, resolve_path
from anima_search.generation.sd_generator import StableDiffusionGenerator
from anima_search.indexing.bm25_index import BM25Index
from anima_search.indexing.image_vector_index import ImageVectorIndex
from anima_search.indexing.index_manifest import (
    load_index_manifest,
    validate_index_manifest,
)
from anima_search.indexing.vector_index import VectorIndex
from anima_search.retrieval.openai_compatible import OpenAICompatibleTextClient
from anima_search.retrieval.query_parser import QueryParser
from anima_search.retrieval.search import HybridSearcher
from anima_search.runtime.model_manager import ModelManager
from anima_search.schemas import ImageAnnotation


def create_service(
    config_path: str = "configs/default.yaml",
    split: str = "val",
    enabled_branches: Sequence[str] | None = None,
    fusion_method: str | None = None,
) -> SearchService:
    config = load_config(config_path)
    if enabled_branches is not None:
        config["retrieval"]["enabled_branches"] = list(enabled_branches)
    if fusion_method is not None:
        config["retrieval"]["fusion_method"] = fusion_method
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    index_dir = artifacts / "indexes" / split
    annotation_path = index_dir / "annotations.json"
    annotations_list = [
        ImageAnnotation.model_validate(item)
        for item in json.loads(annotation_path.read_text(encoding="utf-8"))
    ]
    annotations = {item.image_id: item for item in annotations_list}
    indexes: dict[str, object] = {}
    enabled = set(config["retrieval"].get("enabled_branches", ["image", "text", "bm25"]))
    if "bm25" in enabled and (index_dir / "bm25.pkl").is_file():
        indexes["bm25"] = BM25Index.load(index_dir / "bm25.pkl")
    text_dir = index_dir / "text"
    if not text_dir.is_dir() and (index_dir / "vector").is_dir():
        text_dir = index_dir / "vector"
    if "text" in enabled and text_dir.is_dir():
        indexes["text"] = VectorIndex.load(
            text_dir,
            model_path=resolve_path(config, config["models"]["embedder"]),
            device=config["runtime"]["device"],
        )
    if "image" in enabled and (index_dir / "image").is_dir():
        indexes["image"] = ImageVectorIndex.load(
            index_dir / "image",
            model_path=resolve_path(config, config["models"]["image_embedder"]),
            device=config["runtime"]["device"],
            dtype=config["runtime"]["dtype"],
        )
    missing_enabled = enabled - set(indexes)
    if missing_enabled:
        raise FileNotFoundError(
            f"enabled retrieval indexes are missing under {index_dir}: "
            f"{sorted(missing_enabled)}; build these branches before running search"
        )
    if not indexes:
        raise FileNotFoundError(f"no enabled retrieval indexes found under {index_dir}")

    manifest_path = index_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_index_manifest(manifest_path)
        required = enabled.intersection(manifest.get("active_branches", []))
        missing = required - set(indexes)
        if missing:
            raise FileNotFoundError(
                f"index manifest branches are missing on disk: {sorted(missing)}"
            )
        branch_ids = {name: list(index.image_ids) for name, index in indexes.items()}
        validate_index_manifest(
            manifest,
            [item.image_id for item in annotations_list],
            branch_ids,
        )

    aliases_path = resolve_path(
        config,
        config["retrieval"].get("aliases", "configs/retrieval_aliases.yaml"),
    )
    aliases = (
        yaml.safe_load(aliases_path.read_text(encoding="utf-8"))
        if aliases_path.is_file()
        else {}
    )
    manager = ModelManager(
        lambda: QwenVLClient(
            resolve_path(config, config["models"]["qwen_vl"]),
            config["runtime"]["dtype"],
            config["runtime"]["device"],
            config["runtime"]["max_image_pixels"],
        ),
        lambda: StableDiffusionGenerator(
            resolve_path(config, config["models"]["stable_diffusion"]),
            config["runtime"]["dtype"],
            config["runtime"]["device"],
        ),
    )
    prompt_dir = Path(config["project_root"]) / "configs" / "prompts"
    retrieval_settings = config["retrieval"]
    parser_backend = retrieval_settings.get("query_parser_backend")
    if not parser_backend:
        parser_backend = (
            "local_qwen"
            if bool(retrieval_settings.get("query_parser_use_llm", False))
            else "rules"
        )
    parser_backend = str(parser_backend).strip().lower()
    parser_generator = None
    if parser_backend in {"openai_compatible", "api"}:
        api_settings = retrieval_settings.get("query_parser_api", {})
        parser_generator = OpenAICompatibleTextClient(
            str(api_settings.get("base_url", "")),
            str(api_settings.get("model", "")),
            api_key_env=str(
                api_settings.get("api_key_env", "SILICONFLOW_API_KEY")
            ),
            timeout_seconds=float(api_settings.get("timeout_seconds", 30)),
            max_retries=int(api_settings.get("max_retries", 2)),
        )
    parser = QueryParser(
        parser_generator,
        (prompt_dir / "query_parser.txt").read_text(encoding="utf-8"),
        aliases,
    )
    searcher = HybridSearcher(
        annotations,
        rrf_k=config["retrieval"]["rrf_k"],
        indexes=indexes,
        aliases=aliases,
        fusion_method=config["retrieval"].get("fusion_method", "rrf"),
        fusion_weights=config["retrieval"].get("fusion_weights", {}),
    )
    return SearchService(
        config,
        parser,
        searcher,
        manager,
        annotations,
        (prompt_dir / "reranker.txt").read_text(encoding="utf-8"),
        (prompt_dir / "content_writer.txt").read_text(encoding="utf-8"),
        (prompt_dir / "sd_prompt.txt").read_text(encoding="utf-8"),
    )
