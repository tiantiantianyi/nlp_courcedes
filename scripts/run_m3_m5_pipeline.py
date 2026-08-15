from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.schemas import ImageAnnotation, ManifestItem


def _read_jsonl(path: Path, model_type: type) -> list:
    if not path.is_file():
        raise FileNotFoundError(f"required file does not exist: {path}")
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_annotations(config: dict, split: str) -> int:
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    manifest_path = artifacts / "manifests" / f"{split.lower()}.jsonl"
    annotation_path = (
        artifacts
        / "annotations"
        / f"{split.lower()}.{config['annotation']['prompt_version']}.jsonl"
    )
    manifest = {
        item.image_id: item
        for item in _read_jsonl(manifest_path, ManifestItem)
        if item.valid
    }
    annotations = _read_jsonl(annotation_path, ImageAnnotation)
    by_id = {item.image_id: item for item in annotations}
    if len(by_id) != len(annotations):
        raise ValueError(f"{split} annotations contain duplicate image IDs")

    missing = sorted(set(manifest) - set(by_id))
    extra = sorted(set(by_id) - set(manifest))
    mismatched = sorted(
        image_id
        for image_id in set(manifest).intersection(by_id)
        if (
            manifest[image_id].sha256 != by_id[image_id].sha256
            or manifest[image_id].relative_path != by_id[image_id].relative_path
            or by_id[image_id].prompt_version != config["annotation"]["prompt_version"]
        )
    )
    allow_missing = bool(config.get("annotation", {}).get("allow_missing", False))
    if (missing and not allow_missing) or extra or mismatched:
        raise ValueError(
            f"{split} annotation validation failed: "
            f"missing={missing[:10]}, extra={extra[:10]}, mismatched={mismatched[:10]}"
        )
    return len(annotations)


class PipelineState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, object] = {"status": "running", "stages": {}}
        self.write()

    def update(self, stage: str, status: str, **details: object) -> None:
        stages = self.data.setdefault("stages", {})
        assert isinstance(stages, dict)
        stages[stage] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self.data["current_stage"] = stage
        self.write()

    def finish(self, status: str, **details: object) -> None:
        self.data.update(
            status=status,
            finished_at=datetime.now(timezone.utc).isoformat(),
            **details,
        )
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)


def _run_stage(state: PipelineState, name: str, command: list[str], root: Path) -> None:
    state.update(name, "running", command=command)
    subprocess.run(command, cwd=root, check=True)
    state.update(name, "completed", command=command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the ordered full-data M3-M5 preparation pipeline."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--eval-count", type=int, default=100)
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Use the existing manifests instead of scanning the image directories again.",
    )
    args = parser.parse_args()
    if args.eval_count <= 0:
        parser.error("--eval-count must be positive")

    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    state = PipelineState(artifacts / "cache" / "m3_m5_pipeline_state.json")
    python = sys.executable
    config_arg = str(config_path)

    try:
        if not args.skip_manifest:
            _run_stage(
                state,
                "manifest",
                [python, "scripts/build_manifest.py", "--config", config_arg],
                root,
            )

        counts: dict[str, int] = {}
        annotation_source = config.get("annotation", {}).get("source")
        if annotation_source:
            _run_stage(
                state,
                "import_qwen35_annotations",
                [
                    python,
                    "scripts/import_m1_qwen35.py",
                    "--config",
                    config_arg,
                ],
                root,
            )
        else:
            for split in ("Train", "Val"):
                _run_stage(
                    state,
                    f"annotate_{split.lower()}",
                    [
                        python,
                        "scripts/annotate_images.py",
                        "--config",
                        config_arg,
                        "--split",
                        split,
                    ],
                    root,
                )

        for split in ("Train", "Val"):
            count = validate_annotations(config, split)
            counts[split.lower()] = count
            state.update(f"validate_{split.lower()}", "completed", record_count=count)

        for split in ("Train", "Val"):
            _run_stage(
                state,
                f"index_{split.lower()}",
                [
                    python,
                    "scripts/build_indexes.py",
                    "--config",
                    config_arg,
                    "--split",
                    split,
                    "--branches",
                    "image,text,bm25",
                ],
                root,
            )

        _run_stage(
            state,
            "evaluation_seed",
            [
                python,
                "scripts/create_eval_set.py",
                "--config",
                config_arg,
                "--count",
                str(args.eval_count),
            ],
            root,
        )
        query_path = artifacts / "evaluation" / "val_queries.jsonl"
        query_count = sum(
            1 for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        expected_queries = min(args.eval_count, counts["val"])
        if query_count != expected_queries:
            raise ValueError(
                f"evaluation seed count mismatch: expected {expected_queries}, got {query_count}"
            )
        state.finish("completed", annotation_counts=counts, evaluation_queries=query_count)
        print(json.dumps(state.data, ensure_ascii=False, indent=2))
    except Exception as exc:
        state.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
