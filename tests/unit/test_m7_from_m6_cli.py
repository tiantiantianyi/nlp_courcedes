from __future__ import annotations

import json
from pathlib import Path

import pytest

from anima_search.app.service import SearchService
from anima_search.m6.results import M6QueryResult
from anima_search.m7.schemas import StoryGap, StorySection, VisualStory
from scripts.run_m7_from_m6 import main


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _m6_result() -> M6QueryResult:
    return M6QueryResult.model_validate(
        {
            "schema_version": "m6-rerank-v1.0",
            "source_schema_version": "m5-to-m6-v1.0",
            "query_id": "q-story",
            "query": "城市故事",
            "category": "simple",
            "split": "val",
            "fusion_method": "rrf",
            "top_k": 20,
            "annotation_version": "qwen35-canonical-v1.3",
            "index_manifest_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "rerank_method": "listwise",
            "degraded": False,
            "mismatch": [],
            "candidates": [
                {
                    "rank": rank,
                    "image_id": f"val-{2001 + rank}",
                    "relative_path": f"../Val/{2001 + rank}.jpg",
                    "fused_score": 1.0 / rank,
                    "branch_scores": {"image": 0.5},
                    "branch_ranks": {"image": rank},
                    "matched_fields": [],
                    "rerank_rank": rank,
                    "rerank_score": float(101 - rank),
                    "mismatch": [],
                }
                for rank in range(1, 21)
            ],
        }
    )


def _files(tmp_path: Path) -> dict[str, Path]:
    m6_path = tmp_path / "m6.jsonl"
    m6_path.write_text(_m6_result().model_dump_json() + "\n", encoding="utf-8")
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text("", encoding="utf-8")
    val_manifest = tmp_path / "val.jsonl"
    annotations = tmp_path / "canonical.jsonl"
    manifest_lines: list[str] = []
    annotation_lines: list[str] = []
    for number in (2002, 2003, 2004):
        digest = f"{number:064x}"[-64:]
        manifest_lines.append(
            json.dumps(
                {
                    "image_id": f"val-{number}",
                    "split": "Val",
                    "relative_path": f"../Val/{number}.jpg",
                    "sha256": digest,
                    "size_bytes": 100,
                    "valid": True,
                    "error": None,
                    "duplicate_of": None,
                }
            )
        )
        annotation_lines.append(
            json.dumps(
                {
                    "image_id": str(number),
                    "processed_sha256": digest,
                    "source_model_id": "Qwen/Qwen3.5-9B",
                    "normalizer_version": "m1-normalize-v1.0.0",
                    "repairs_applied": 0,
                    "lossy_repairs": False,
                    "annotation": {
                        "scene": {
                            "primary_type": "street_urban",
                            "sub_type_zh": "城市街道",
                            "environment": "outdoor",
                        },
                        "capture_visual": {"time_of_day": "night"},
                        "entities": [],
                        "ocr": [],
                        "relations": [],
                        "event": {"summary_zh": None},
                        "subjective": {
                            "mood_terms_zh": [],
                            "palette_terms_zh": [],
                        },
                        "captions": {
                            "short_zh": f"城市图片{number}",
                            "dense_zh": f"夜晚的城市图片{number}",
                        },
                        "uncertainties": [],
                    },
                },
                ensure_ascii=False,
            )
        )
    val_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    annotations.write_text(
        "\n".join(annotation_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "m6": m6_path,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "annotations": annotations,
        "output": tmp_path / "story.json",
    }


def _argv(paths: dict[str, Path]) -> list[str]:
    return [
        "--m6-results",
        str(paths["m6"]),
        "--query-id",
        "q-story",
        "--select-count",
        "3",
        "--annotations",
        str(paths["annotations"]),
        "--train-manifest",
        str(paths["train_manifest"]),
        "--val-manifest",
        str(paths["val_manifest"]),
        "--config",
        str(PROJECT_ROOT / "configs" / "benchmark_8gb.yaml"),
        "--output",
        str(paths["output"]),
        "--fill-gaps",
    ]


def test_cli_outputs_ordered_story_and_generated_asset_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _files(tmp_path)

    def fake_story(
        self: SearchService,
        candidates: list[object],
        selected_image_ids: list[str],
        **_: object,
    ) -> VisualStory:
        return VisualStory(
            title="城市故事",
            sections=[
                StorySection(
                    image_id=image_id,
                    subtitle=f"片段{index}",
                    text="可见城市街道",
                )
                for index, image_id in enumerate(selected_image_ids, start=1)
            ],
            ordered_image_ids=selected_image_ids,
            gaps=[
                StoryGap(
                    gap_id="gap-01",
                    after_image_id=selected_image_ids[0],
                    before_image_id=selected_image_ids[1],
                    reason="场景过渡",
                    generation_prompt="生成过渡图片",
                    status="generated",
                    generated_image_id="generated-1",
                    relative_path="artifacts/generated/generated-1.png",
                )
            ],
        )

    monkeypatch.setattr(SearchService, "create_visual_story", fake_story)

    assert main(_argv(paths)) == 0

    payload = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "m7-story-v1.0"
    assert payload["source_query_id"] == "q-story"
    assert payload["ordered_image_ids"] == [
        "val-2002",
        "val-2003",
        "val-2004",
    ]
    assert [section["image_id"] for section in payload["sections"]] == [
        "val-2002",
        "val-2003",
        "val-2004",
    ]
    assert payload["gaps"][0]["source"] == "generated"
    assert payload["gaps"][0]["ai_generated"] is True


def test_cli_refuses_to_overwrite_m6_results(tmp_path: Path) -> None:
    paths = _files(tmp_path)
    paths["output"] = paths["m6"]

    with pytest.raises(ValueError, match="output path alias"):
        main(_argv(paths))


@pytest.mark.parametrize(
    "protected_key",
    ["m6", "annotations", "train_manifest", "val_manifest"],
)
def test_cli_refuses_output_aliasing_any_read_only_input(
    tmp_path: Path,
    protected_key: str,
) -> None:
    paths = _files(tmp_path)
    protected = paths[protected_key]
    original = protected.read_bytes()
    paths["output"] = protected

    with pytest.raises(ValueError, match="output path alias"):
        main(_argv(paths))

    assert protected.read_bytes() == original


def test_cli_refuses_output_aliasing_config(tmp_path: Path) -> None:
    paths = _files(tmp_path)
    paths["output"] = PROJECT_ROOT / "configs" / "benchmark_8gb.yaml"
    original = paths["output"].read_bytes()

    with pytest.raises(ValueError, match="output path alias"):
        main(_argv(paths))

    assert paths["output"].read_bytes() == original


def test_cli_refuses_output_symlink_aliasing_annotations(
    tmp_path: Path,
) -> None:
    paths = _files(tmp_path)
    original = paths["annotations"].read_bytes()
    paths["output"].symlink_to(paths["annotations"])

    with pytest.raises(ValueError, match="output path alias"):
        main(_argv(paths))

    assert paths["annotations"].read_bytes() == original
