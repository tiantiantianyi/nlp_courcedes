from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from anima_search.pipeline.directory import (  # noqa: E402
    PipelineState,
    materialize_runtime_config,
    scan_input_directory,
    validate_annotation_snapshot,
    write_manifest_snapshot,
)


def default_workspace(input_dir: Path) -> Path:
    resolved = input_dir.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in resolved.name
    ).strip("-") or "images"
    return REPOSITORY_ROOT / "artifacts" / "directory_runs" / f"{safe_name}-{digest}"


def build_commands(
    runtime_config: Path,
    workspace: Path,
    routing_config: Path,
    *,
    mode: str,
    launch: bool,
    host: str,
    port: int,
    share: bool,
) -> list[tuple[str, list[str]]]:
    python = sys.executable
    commands: list[tuple[str, list[str]]] = [
        (
            "image_index",
            [
                python,
                str(REPOSITORY_ROOT / "scripts/build_image_only_index.py"),
                "--config",
                str(runtime_config),
                "--split",
                "Val",
            ],
        ),
        (
            "scene_routing",
            [
                python,
                str(REPOSITORY_ROOT / "scripts/route_scenes.py"),
                "--config",
                str(runtime_config),
                "--routing-config",
                str(routing_config),
                "--split",
                "val",
                "--output",
                str(workspace / "artifacts/routing/val_scene_routes.jsonl"),
            ],
        ),
    ]
    if mode == "full":
        route_path = workspace / "artifacts/routing/val_scene_routes.jsonl"
        commands.extend(
            [
                (
                    "annotation",
                    [
                        python,
                        str(REPOSITORY_ROOT / "scripts/annotate_images.py"),
                        "--config",
                        str(runtime_config),
                        "--split",
                        "Val",
                        "--scene-routes",
                        str(route_path),
                    ],
                ),
                (
                    "full_indexes",
                    [
                        python,
                        str(REPOSITORY_ROOT / "scripts/build_indexes.py"),
                        "--config",
                        str(runtime_config),
                        "--split",
                        "Val",
                        "--branches",
                        "image,text,bm25",
                    ],
                ),
            ]
        )
    if launch:
        command = [
            python,
            str(REPOSITORY_ROOT / "scripts/launch_app.py"),
            "--config",
            str(runtime_config),
            "--split",
            "val",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if share:
            command.append("--share")
        commands.append(("launch", command))
    return commands


def _run_stage(state: PipelineState, stage: str, command: list[str], *, resume: bool) -> None:
    if resume and state.completed(stage):
        print(f"[resume] skip completed stage: {stage}")
        return
    state.update(stage, "running", command=command)
    try:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    except Exception as exc:
        state.update(stage, "failed", command=command, error=f"{type(exc).__name__}: {exc}")
        raise
    state.update(stage, "completed", command=command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AskAlbum on an arbitrary image directory: manifest, M0, M1, M3-M7 readiness."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--mode",
        choices=["full", "image-only"],
        default="full",
        help="full performs M0 routing, M1 annotation, and three-branch indexing; image-only skips M1.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--routing-config", type=Path, default=Path("configs/scene_routing.yaml")
    )
    parser.add_argument("--limit", type=int, help="Use only the first N images for a smoke test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true", help="Launch Gradio after indexes are ready")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"--input_dir is not a directory: {input_dir}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    base_config = (
        (REPOSITORY_ROOT / args.config).resolve()
        if not args.config.is_absolute()
        else args.config.resolve()
    )
    routing_config = (
        (REPOSITORY_ROOT / args.routing_config).resolve()
        if not args.routing_config.is_absolute()
        else args.routing_config.resolve()
    )
    if not base_config.is_file():
        parser.error(f"config does not exist: {base_config}")
    if not routing_config.is_file():
        parser.error(f"routing config does not exist: {routing_config}")
    workspace = (args.workspace or default_workspace(input_dir)).resolve()
    runtime_config = workspace / "configs/runtime.yaml"
    commands = build_commands(
        runtime_config,
        workspace,
        routing_config,
        mode=args.mode,
        launch=args.launch,
        host=args.host,
        port=args.port,
        share=args.share,
    )
    plan = {
        "input_dir": str(input_dir),
        "workspace": str(workspace),
        "mode": args.mode,
        "limit": args.limit,
        "stages": ["manifest", *[name for name, _ in commands]],
        "commands": {name: command for name, command in commands},
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    identity = {
        "input_dir": str(input_dir),
        "mode": args.mode,
        "limit": args.limit,
        "base_config": str(base_config),
        "routing_config": str(routing_config),
    }
    state = PipelineState(workspace / "pipeline_state.json", identity, resume=args.resume)
    runtime_config = materialize_runtime_config(
        base_config, workspace, input_dir, mode=args.mode
    )
    artifacts_dir = workspace / "artifacts"

    try:
        if not (args.resume and state.completed("manifest")):
            state.update("manifest", "running")
            items = scan_input_directory(input_dir, workspace, limit=args.limit)
            quality = write_manifest_snapshot(items, artifacts_dir)
            if quality["valid_count"] == 0:
                raise ValueError("all discovered images are invalid")
            state.update("manifest", "completed", **quality)
        else:
            current_items = scan_input_directory(input_dir, workspace, limit=args.limit)
            current_payload = "".join(
                item.model_dump_json() + "\n" for item in current_items
            ).encode("utf-8")
            current_digest = hashlib.sha256(current_payload).hexdigest()
            expected_digest = state.data["stages"]["manifest"].get("manifest_sha256")
            if current_digest != expected_digest:
                raise ValueError(
                    "input directory changed after the saved manifest; "
                    "start a new --workspace instead of resuming stale indexes"
                )
            print("[resume] skip completed stage: manifest")

        non_launch_commands = [(name, command) for name, command in commands if name != "launch"]
        for stage, command in non_launch_commands:
            _run_stage(state, stage, command, resume=args.resume)
            if stage == "annotation":
                runtime = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
                prompt_version = runtime["annotation"]["prompt_version"]
                count = validate_annotation_snapshot(
                    artifacts_dir / "manifests/val.jsonl",
                    artifacts_dir / "annotations" / f"val.{prompt_version}.jsonl",
                    prompt_version,
                )
                state.update(stage, "completed", command=command, validated_records=count)

        index_dir = artifacts_dir / "indexes/val"
        state.finish(
            index_dir=str(index_dir),
            runtime_config=str(runtime_config),
            launch_command=next(
                (command for name, command in commands if name == "launch"),
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/launch_app.py"),
                    "--config",
                    str(runtime_config),
                    "--split",
                    "val",
                ],
            ),
        )
        print(json.dumps(state.data, ensure_ascii=False, indent=2))

        launch_entry = next(
            ((name, command) for name, command in commands if name == "launch"), None
        )
        if launch_entry:
            subprocess.run(launch_entry[1], cwd=REPOSITORY_ROOT, check=True)
    except Exception as exc:
        if state.data.get("status") != "failed":
            state.update(
                str(state.data.get("current_stage", "pipeline")),
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        raise


if __name__ == "__main__":
    main()
