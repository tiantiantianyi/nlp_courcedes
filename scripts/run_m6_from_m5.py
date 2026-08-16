#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.config import load_config, resolve_path
from anima_search.m6.interface_validation import validate_interface_file
from anima_search.m6.path_safety import reject_output_path_aliases
from anima_search.m6.runner import rerank_query_batch
from anima_search.retrieval.listwise_reranker import ListwiseVisualReranker
from anima_search.retrieval.reranker import VisualReranker
from anima_search.schemas import SearchResult


class _DryRunReranker:
    last_error = None
    last_degraded_reason = "dry-run: Qwen3-VL was not invoked"

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        return candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank a validated m5-to-m6-v1.0 delivery offline.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--m5-config-snapshot", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("pointwise", "listwise"),
        default="listwise",
    )
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _data_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _real_reranker(
    config: dict[str, object],
    *,
    method: str,
) -> tuple[object, QwenVLClient]:
    project_root = Path(str(config["project_root"]))
    models = config["models"]
    runtime = config["runtime"]
    retrieval = config["retrieval"]
    if not isinstance(models, dict):
        raise ValueError("config.models must be a mapping")
    if not isinstance(runtime, dict):
        raise ValueError("config.runtime must be a mapping")
    if not isinstance(retrieval, dict):
        raise ValueError("config.retrieval must be a mapping")

    client = QwenVLClient(
        resolve_path(config, str(models["qwen_vl"])),
        dtype=str(runtime.get("dtype", "float16")),
        device=str(runtime.get("device", "cuda")),
        max_image_pixels=int(runtime.get("max_image_pixels", 1024 * 1024)),
    )
    if method == "listwise":
        prompt_path = resolve_path(
            config,
            str(
                retrieval.get(
                    "rerank_listwise_prompt",
                    "configs/prompts/reranker_listwise.txt",
                )
            ),
        )
        reranker = ListwiseVisualReranker(
            client,
            prompt_path.read_text(encoding="utf-8"),
            project_root,
            max_new_tokens=int(
                retrieval.get("rerank_listwise_max_new_tokens", 768)
            ),
            columns=int(retrieval.get("rerank_listwise_columns", 5)),
            tile_size=int(retrieval.get("rerank_listwise_tile_size", 192)),
        )
    else:
        prompt_path = resolve_path(
            config,
            str(retrieval.get("rerank_prompt", "configs/prompts/reranker.txt")),
        )
        reranker = VisualReranker(
            client,
            prompt_path.read_text(encoding="utf-8"),
            project_root,
            rrf_weight=float(retrieval.get("rrf_weight", 0.35)),
            vlm_weight=float(retrieval.get("vlm_weight", 0.65)),
            max_new_tokens=int(retrieval.get("rerank_max_new_tokens", 128)),
        )
    return reranker, client


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reject_output_path_aliases(
        read_only={
            "input": args.input,
            "config": args.config,
            "index_manifest": args.index_manifest,
            "m5_config_snapshot": args.m5_config_snapshot,
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
        },
        outputs={
            "output": args.output,
            "validation_report": args.validation_report,
        },
    )
    if args.query_limit is not None and args.query_limit <= 0:
        raise ValueError("--query-limit must be positive")

    config = load_config(args.config)
    project_root = Path(config["project_root"]).resolve()
    batches, report = validate_interface_file(
        input_path=args.input,
        project_root=project_root,
        train_dir=_data_path(project_root, args.train_dir),
        val_dir=_data_path(project_root, args.val_dir),
        index_manifest_path=args.index_manifest,
        config_snapshot_path=args.m5_config_snapshot,
    )
    args.validation_report.parent.mkdir(parents=True, exist_ok=True)
    args.validation_report.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if not report.valid:
        print(f"M5 delivery blocked with {len(report.issues)} interface errors")
        return 1

    selected = batches[: args.query_limit] if args.query_limit else batches
    client: QwenVLClient | None = None
    if args.dry_run:
        reranker: object = _DryRunReranker()
    else:
        reranker, client = _real_reranker(config, method=args.method)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as output:
            for batch in selected:
                result = rerank_query_batch(
                    batch,
                    reranker,
                    method=args.method,
                )
                output.write(
                    json.dumps(
                        result.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output.flush()
    finally:
        if client is not None:
            client.unload()

    print(
        f"queries={len(selected)} method={args.method} "
        f"dry_run={args.dry_run} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
