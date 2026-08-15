#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.app.service import SearchService
from anima_search.config import load_config, resolve_path
from anima_search.generation.sd_generator import StableDiffusionGenerator
from anima_search.m7.canonical_annotations import (
    load_canonical_m7_annotations,
)
from anima_search.m7.m6_bridge import load_m6_query, select_story_candidates
from anima_search.runtime.model_manager import ModelManager
from anima_search.schemas import ImageAnnotation


class _NoRetrieval:
    indexes: dict[str, object] = {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an M7 visual story from one M6 query result.",
    )
    parser.add_argument("--m6-results", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--select-count", type=int, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", default="图文游记")
    parser.add_argument("--tone", default="自然")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fill-gaps", action="store_true")
    return parser


def _build_service(
    config: dict,
    annotations: dict[str, ImageAnnotation],
) -> SearchService:
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
    return SearchService(
        config,
        parser=object(),
        searcher=_NoRetrieval(),
        manager=manager,
        annotations=annotations,
        reranker_prompt=(prompt_dir / "reranker.txt").read_text(
            encoding="utf-8"
        ),
        content_prompt=(prompt_dir / "content_writer.txt").read_text(
            encoding="utf-8"
        ),
        sd_prompt=(prompt_dir / "sd_prompt.txt").read_text(encoding="utf-8"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.resolve() == args.m6_results.resolve():
        raise ValueError("M7 output must differ from M6 results")

    m6_result = load_m6_query(args.m6_results, args.query_id)
    candidates = select_story_candidates(m6_result, args.select_count)
    split = "Val" if m6_result.split == "val" else "Train"
    annotations = load_canonical_m7_annotations(
        args.annotations,
        [args.train_manifest, args.val_manifest],
        split=split,
    )
    selected_ids = [candidate.image_id for candidate in candidates]
    missing_annotations = [
        image_id for image_id in selected_ids if image_id not in annotations
    ]
    if missing_annotations:
        raise ValueError(
            f"selected M6 candidates lack canonical annotations: {missing_annotations}"
        )

    config = load_config(args.config)
    service = _build_service(config, annotations)
    try:
        story = service.create_visual_story(
            candidates,
            selected_ids,
            theme=args.theme,
            tone=args.tone,
            fill_gaps=args.fill_gaps,
            seed=args.seed,
        )
    finally:
        service.manager.unload_all()

    for gap in story.gaps:
        if gap.status == "generated" and (
            gap.source != "generated" or not gap.ai_generated
        ):
            raise ValueError(
                f"generated story gap lacks AI marker: {gap.gap_id}"
            )
    payload = {
        "schema_version": "m7-story-v1.0",
        "source_query_id": m6_result.query_id,
        "source_m6_degraded": m6_result.degraded,
        "selected_image_ids": selected_ids,
        "fill_gaps_requested": args.fill_gaps,
        **story.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"query_id={m6_result.query_id} selected={len(selected_ids)} "
        f"gaps={len(story.gaps)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
