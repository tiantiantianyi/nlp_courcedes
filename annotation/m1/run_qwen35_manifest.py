#!/usr/bin/env python3
"""Run one persistent Hugging Face VLM over a deterministic manifest shard.

The filename is retained for compatibility with the completed Qwen baseline. New
launchers should call run_local_vlm_manifest.py and pass the exact model ID.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from m1_validation import (
    candidate_record_validator,
    extract_prompt_block,
    extract_prompt_version,
    validate_raw_annotation,
    validation_error_records,
)


MODEL_ID = "Qwen/Qwen3.5-9B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--processor-family",
        choices=("qwen35", "internvl35"),
        default="qwen35",
    )
    parser.add_argument(
        "--constrained-decoding",
        choices=("none", "lmfe-json-schema", "xgrammar-json-schema"),
        default="none",
    )
    parser.add_argument("--constraint-max-whitespace", type=int, default=16)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--candidate-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        for value in values
    )
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def schema_version(schema: dict[str, Any]) -> str:
    schema_id = schema.get("$id", "")
    match = re.search(r"annotation-payload-v(\d+)\.(\d+)", schema_id)
    if not match:
        raise ValueError("Could not determine annotation schema version from $id")
    return f"{match.group(1)}.{match.group(2)}.0"


def validate_manifest(
    records: list[dict[str, Any]],
    project_root: Path,
) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    checked: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        required = {
            "image_id",
            "split",
            "processed_path",
            "processed_sha256",
            "width",
            "height",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"manifest row {index} missing: {', '.join(missing)}")
        image_id = record["image_id"]
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"manifest row {index} has invalid image_id")
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id in manifest: {image_id}")
        seen_ids.add(image_id)
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"manifest image {image_id} has invalid split: {split!r}")
        expected_hash = record["processed_sha256"]
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
            raise ValueError(f"manifest image {image_id} has invalid SHA-256")

        image_path = (project_root / record["processed_path"]).resolve()
        image_bytes = image_path.read_bytes()
        actual_hash = sha256_bytes(image_bytes)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {image_id}: {actual_hash} != {expected_hash}"
            )
        with Image.open(image_path) as image:
            dimensions = list(image.size)
        expected_dimensions = [record["width"], record["height"]]
        if dimensions != expected_dimensions:
            raise ValueError(
                f"dimension mismatch for {image_id}: {dimensions} != {expected_dimensions}"
            )
        checked_record = dict(record)
        checked_record["resolved_path"] = str(image_path)
        checked_record["manifest_index"] = index
        checked.append(checked_record)
    return checked


def write_shared_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"existing shared artifact differs: {path}")
        return
    path.write_bytes(data)


def resumable(
    item_dir: Path,
    image: dict[str, Any],
    expected: dict[str, str],
    retry_failed: bool,
    validator: Draft202012Validator,
) -> bool:
    candidate_path = item_dir / "candidate_record.json"
    summary_path = item_dir / "summary.json"
    if not candidate_path.is_file() or not summary_path.is_file():
        return False
    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if list(validator.iter_errors(candidate)):
        return False
    if retry_failed and candidate.get("status") == "failed":
        return False
    checks = {
        "image_id": image["image_id"],
        "processed_sha256": image["processed_sha256"],
        "model_id": expected["model_id"],
        "prompt_version": expected["prompt_version"],
        "annotation_schema_version": expected["annotation_schema_version"],
    }
    if any(candidate.get(key) != value for key, value in checks.items()):
        return False
    summary_checks = {
        "prompt_file_sha256": expected["prompt_file_sha256"],
        "schema_file_sha256": expected["schema_file_sha256"],
        "model_config_sha256": expected["model_config_sha256"],
        "model_index_sha256": expected["model_index_sha256"],
        "constraint_backend": expected["constraint_backend"],
    }
    if expected["constraint_backend"] == "xgrammar-json-schema":
        summary_checks["constraint_max_whitespace"] = expected[
            "constraint_max_whitespace"
        ]
    return all(summary.get(key) == value for key, value in summary_checks.items())


def failure_record(
    image: dict[str, Any],
    expected: dict[str, str],
    error: str,
    raw_response_path: str | None,
) -> dict[str, Any]:
    return {
        "image_id": image["image_id"],
        "processed_sha256": image["processed_sha256"],
        "source_kind": "local",
        "model_id": expected["model_id"],
        "prompt_version": expected["prompt_version"],
        "annotation_schema_version": expected["annotation_schema_version"],
        "status": "failed",
        "raw_response_path": raw_response_path,
        "annotation": None,
        "error": error[:1000],
    }


def initial_item_summary(
    image: dict[str, Any],
    expected: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "started_at_utc": utc_now(),
        "manifest_index": image["manifest_index"],
        "image_id": image["image_id"],
        "split": image["split"],
        "processed_path": image["processed_path"],
        "processed_sha256": image["processed_sha256"],
        "image_dimensions": [image["width"], image["height"]],
        "coverage_tags": image.get("coverage_tags", []),
        "model_id": expected["model_id"],
        "model_config_sha256": expected["model_config_sha256"],
        "model_index_sha256": expected["model_index_sha256"],
        "model_weight_bytes": expected["model_weight_bytes"],
        "prompt_version": expected["prompt_version"],
        "prompt_file_sha256": expected["prompt_file_sha256"],
        "schema_file_sha256": expected["schema_file_sha256"],
        "prompt_composition": f"{expected['prompt_version']}+schema-inline-v1",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "processor_family": args.processor_family,
        "dtype": "bfloat16",
        "do_sample": False,
        "constrained_decoding": args.constrained_decoding != "none",
        "constraint_backend": args.constrained_decoding,
        "constraint_setup_seconds": None,
        "constraint_max_whitespace": (
            args.constraint_max_whitespace
            if args.constrained_decoding == "xgrammar-json-schema"
            else None
        ),
        "thinking": "disabled",
        "max_new_tokens": args.max_new_tokens,
        "preprocess_seconds": None,
        "generation_seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "effective_eos_token_ids": [],
        "last_output_token_id": None,
        "ended_with_eos": None,
        "hit_max_new_tokens": None,
        "out_of_tokenizer_vocab_tokens": None,
        "out_of_tokenizer_vocab_token_ids": [],
        "pixel_values_shape": None,
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "json_parse_ok": False,
        "schema_valid": False,
        "semantic_valid": False,
        "annotation_valid": False,
        "candidate_record_valid": False,
        "candidate_schema_errors": [],
        "error": None,
    }


def persist_failure(
    image: dict[str, Any],
    expected: dict[str, str],
    args: argparse.Namespace,
    candidate_validator: Draft202012Validator,
    error: str,
) -> None:
    item_dir = args.output_dir / "items" / image["image_id"]
    raw_relative = f"raw/{image['image_id']}.txt"
    raw_path = args.output_dir / raw_relative
    candidate = failure_record(
        image,
        expected,
        error,
        raw_relative if raw_path.is_file() else None,
    )
    candidate_errors = validation_error_records(
        list(candidate_validator.iter_errors(candidate))
    )
    summary = initial_item_summary(image, expected, args)
    summary["candidate_record_valid"] = not candidate_errors
    summary["candidate_schema_errors"] = candidate_errors
    summary["error"] = error
    summary["finished_at_utc"] = utc_now()
    write_json(item_dir / "candidate_record.json", candidate)
    write_json(item_dir / "summary.json", summary)


def process_one(
    image_record: dict[str, Any],
    expected: dict[str, str],
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    system_with_schema: str,
    user_prompt: str,
    annotation_schema: dict[str, Any],
    candidate_validator: Draft202012Validator,
    prefix_allowed_tokens_fn: Any | None,
    compiled_xgrammar: Any | None,
    constraint_setup_seconds: float | None,
) -> None:
    image_id = image_record["image_id"]
    item_dir = args.output_dir / "items" / image_id
    raw_relative = f"raw/{image_id}.txt"
    raw_path = args.output_dir / raw_relative
    summary = initial_item_summary(image_record, expected, args)
    summary["constraint_setup_seconds"] = constraint_setup_seconds
    candidate: dict[str, Any] | None = None
    image: Image.Image | None = None
    inputs: Any = None
    generated_ids: Any = None

    try:
        torch.cuda.reset_peak_memory_stats()
        with Image.open(image_record["resolved_path"]) as opened:
            image = opened.convert("RGB")

        messages = [
            {"role": "system", "content": system_with_schema},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        preprocess_started = time.monotonic()
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if args.processor_family == "qwen35":
            template_kwargs["enable_thinking"] = False
        inputs = processor.apply_chat_template(messages, **template_kwargs)
        summary["preprocess_seconds"] = round(time.monotonic() - preprocess_started, 3)
        summary["input_tokens"] = int(inputs["input_ids"].shape[-1])
        if "pixel_values" in inputs:
            summary["pixel_values_shape"] = list(inputs["pixel_values"].shape)
        inputs = inputs.to("cuda:0")

        generation_started = time.monotonic()
        with torch.inference_mode():
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            }
            if prefix_allowed_tokens_fn is not None:
                generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            if compiled_xgrammar is not None:
                from xgrammar_transformers import new_logits_processor

                generate_kwargs["logits_processor"] = [
                    new_logits_processor(compiled_xgrammar)
                ]
                # XGrammar terminates with the tokenizer's stop token. Some VLM
                # checkpoints configure a different EOS token for generate().
                # Align them so generation stops instead of repeating a hidden
                # stop token until max_new_tokens.
                generate_kwargs["eos_token_id"] = list(
                    compiled_xgrammar.stop_token_ids
                )
            effective_eos_token_ids = generate_kwargs.get(
                "eos_token_id", model.generation_config.eos_token_id
            )
            if isinstance(effective_eos_token_ids, int):
                effective_eos_token_ids = [effective_eos_token_ids]
            summary["effective_eos_token_ids"] = list(
                effective_eos_token_ids or []
            )
            generated_ids = model.generate(
                **inputs,
                **generate_kwargs,
            )
        torch.cuda.synchronize()
        summary["generation_seconds"] = round(
            time.monotonic() - generation_started,
            3,
        )

        prompt_length = int(inputs["input_ids"].shape[-1])
        output_ids = generated_ids[:, prompt_length:]
        summary["output_tokens"] = int(output_ids.shape[-1])
        if output_ids.shape[-1] > 0:
            last_output_token_id = int(output_ids[0, -1].item())
            summary["last_output_token_id"] = last_output_token_id
            summary["ended_with_eos"] = last_output_token_id in summary[
                "effective_eos_token_ids"
            ]
        summary["hit_max_new_tokens"] = (
            summary["output_tokens"] >= args.max_new_tokens
        )
        if compiled_xgrammar is not None:
            invalid_token_mask = (
                output_ids >= compiled_xgrammar.tokenizer_vocab_size
            )
            invalid_token_ids = output_ids[invalid_token_mask]
            summary["out_of_tokenizer_vocab_tokens"] = int(
                invalid_token_ids.numel()
            )
            summary["out_of_tokenizer_vocab_token_ids"] = sorted(
                {int(token_id) for token_id in invalid_token_ids.tolist()}
            )[:20]
        raw_content = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_content, encoding="utf-8")

        validation, annotation = validate_raw_annotation(raw_content, annotation_schema)
        summary.update(validation)
        if validation["annotation_valid"]:
            candidate = {
                "image_id": image_id,
                "processed_sha256": image_record["processed_sha256"],
                "source_kind": "local",
                "model_id": expected["model_id"],
                "prompt_version": expected["prompt_version"],
                "annotation_schema_version": expected["annotation_schema_version"],
                "status": "succeeded",
                "raw_response_path": raw_relative,
                "annotation": annotation,
                "error": None,
            }
            write_json(item_dir / "annotation.json", annotation)
        else:
            candidate = failure_record(
                image_record,
                expected,
                validation.get("error") or "AnnotationValidationError",
                raw_relative,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        summary["error"] = error
        candidate = failure_record(
            image_record,
            expected,
            error,
            raw_relative if raw_path.is_file() else None,
        )
    finally:
        if torch.cuda.is_available():
            summary["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            summary["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if image is not None:
            image.close()
        del inputs, generated_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert candidate is not None
    candidate_errors = validation_error_records(
        list(candidate_validator.iter_errors(candidate))
    )
    summary["candidate_schema_errors"] = candidate_errors
    summary["candidate_record_valid"] = not candidate_errors
    if candidate_errors:
        summary["annotation_valid"] = False
        summary["error"] = f"CandidateRecordValidationError: {len(candidate_errors)} error(s)"
        candidate = failure_record(
            image_record,
            expected,
            summary["error"],
            raw_relative if raw_path.is_file() else None,
        )
    summary["finished_at_utc"] = utc_now()
    write_json(item_dir / "candidate_record.json", candidate)
    write_json(item_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "image_id": image_id,
                "annotation_valid": summary["annotation_valid"],
                "generation_seconds": summary["generation_seconds"],
                "error": summary["error"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def aggregate_shard(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    load_seconds: float | None,
    constraint_setup_seconds: float | None,
    started_at: str,
    run_error: str | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for record in records:
        item_dir = args.output_dir / "items" / record["image_id"]
        candidate_path = item_dir / "candidate_record.json"
        summary_path = item_dir / "summary.json"
        if candidate_path.is_file():
            candidates.append(json.loads(candidate_path.read_text(encoding="utf-8")))
        if summary_path.is_file():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    shard_name = f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    write_jsonl(args.output_dir / f"candidates_local.{shard_name}.jsonl", candidates)
    succeeded = sum(item.get("annotation_valid") is True for item in summaries)
    generation_times = [
        item["generation_seconds"]
        for item in summaries
        if isinstance(item.get("generation_seconds"), (int, float))
    ]
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "model_id": args.model_id,
        "constraint_backend": args.constrained_decoding,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_images": len(records),
        "candidate_records": len(candidates),
        "succeeded": succeeded,
        "failed": len(summaries) - succeeded,
        "load_seconds": load_seconds,
        "constraint_setup_seconds": constraint_setup_seconds,
        "generation_seconds_total": round(sum(generation_times), 3),
        "generation_seconds_mean": (
            round(sum(generation_times) / len(generation_times), 3)
            if generation_times
            else None
        ),
        "run_error": run_error,
    }
    write_json(args.output_dir / f"run_summary.{shard_name}.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must satisfy 0 <= index < num_shards")
    if args.constraint_max_whitespace < 1:
        raise ValueError("constraint max whitespace must be positive")
    args.project_root = args.project_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    prompt_bytes = args.prompt.read_bytes()
    prompt_markdown = prompt_bytes.decode("utf-8")
    prompt_version = extract_prompt_version(prompt_markdown)
    system_prompt = extract_prompt_block(prompt_markdown, "System Prompt")
    user_prompt = extract_prompt_block(prompt_markdown, "User Prompt")

    schema_bytes = args.schema.read_bytes()
    annotation_schema = json.loads(schema_bytes)
    Draft202012Validator.check_schema(annotation_schema)
    candidate_schema_bytes = args.candidate_schema.read_bytes()
    candidate_schema = json.loads(candidate_schema_bytes)
    candidate_validator = candidate_record_validator(candidate_schema, annotation_schema)

    config_path = args.model_path / "config.json"
    index_path = args.model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Model directory is incomplete")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_weight_names = sorted(set(model_index.get("weight_map", {}).values()))
    if not expected_weight_names:
        raise ValueError("Model index does not contain a weight_map")
    missing_weight_names = [
        name for name in expected_weight_names if not (args.model_path / name).is_file()
    ]
    if missing_weight_names:
        raise FileNotFoundError(
            "Model directory is missing weight shard(s): "
            + ", ".join(missing_weight_names)
        )
    weight_files = [args.model_path / name for name in expected_weight_names]

    manifest = validate_manifest(read_jsonl(args.manifest), args.project_root)
    assigned = [
        record
        for record in manifest
        if record["manifest_index"] % args.num_shards == args.shard_index
    ]
    expected = {
        "model_id": args.model_id,
        "prompt_version": prompt_version,
        "annotation_schema_version": schema_version(annotation_schema),
        "prompt_file_sha256": sha256_bytes(prompt_bytes),
        "schema_file_sha256": sha256_bytes(schema_bytes),
        "model_config_sha256": sha256_bytes(config_path.read_bytes()),
        "model_index_sha256": sha256_bytes(index_path.read_bytes()),
        "model_weight_bytes": sum(path.stat().st_size for path in weight_files),
        "constraint_backend": args.constrained_decoding,
        "constraint_max_whitespace": (
            args.constraint_max_whitespace
            if args.constrained_decoding == "xgrammar-json-schema"
            else None
        ),
    }

    write_shared_bytes(
        args.output_dir / f"input_prompt_{expected['prompt_file_sha256'][:12]}.md",
        prompt_bytes,
    )
    write_shared_bytes(
        args.output_dir / f"input_schema_{expected['schema_file_sha256'][:12]}.json",
        schema_bytes,
    )
    compact_schema = json.dumps(annotation_schema, ensure_ascii=False, separators=(",", ":"))
    system_with_schema = (
        system_prompt
        + "\n\n以下是本次输出必须遵守的完整 JSON Schema。"
        + "只返回符合该 Schema 的 JSON 对象：\n"
        + compact_schema
    )

    pending: list[dict[str, Any]] = []
    for record in assigned:
        item_dir = args.output_dir / "items" / record["image_id"]
        if args.resume and resumable(
            item_dir,
            record,
            expected,
            args.retry_failed,
            candidate_validator,
        ):
            print(f"resume skip image_id={record['image_id']}", flush=True)
        else:
            pending.append(record)

    if args.dry_run:
        report = {
            "dry_run": True,
            "manifest_images": len(manifest),
            "assigned_images": len(assigned),
            "pending_images": len(pending),
            "assigned_image_ids": [record["image_id"] for record in assigned],
            **expected,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    load_seconds: float | None = None
    constraint_setup_seconds: float | None = None
    compiled_xgrammar: Any | None = None
    run_error: str | None = None
    if pending:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            load_started = time.monotonic()
            processor = AutoProcessor.from_pretrained(
                args.model_path,
                local_files_only=True,
            )
            prefix_allowed_tokens_fn = None
            if args.constrained_decoding == "lmfe-json-schema":
                from lmformatenforcer import JsonSchemaParser

                from lmfe_transformers_compat import build_prefix_allowed_tokens_fn

                constraint_started = time.monotonic()
                prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(
                    processor.tokenizer,
                    JsonSchemaParser(annotation_schema),
                )
                constraint_setup_seconds = round(
                    time.monotonic() - constraint_started,
                    3,
                )
            elif args.constrained_decoding == "xgrammar-json-schema":
                from xgrammar_transformers import (
                    compile_json_schema,
                    model_vocab_size,
                )

                constraint_started = time.monotonic()
                compiled_xgrammar = compile_json_schema(
                    processor.tokenizer,
                    annotation_schema,
                    model_vocab_size(model_config),
                    args.constraint_max_whitespace,
                )
                constraint_setup_seconds = round(
                    time.monotonic() - constraint_started,
                    3,
                )
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_path,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            model.eval()
            load_seconds = round(time.monotonic() - load_started, 3)
            print(
                json.dumps(
                    {
                        "model_loaded": True,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "load_seconds": load_seconds,
                        "constraint_backend": args.constrained_decoding,
                        "constraint_setup_seconds": constraint_setup_seconds,
                        "pending_images": len(pending),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            for record in pending:
                process_one(
                    record,
                    expected,
                    args,
                    model,
                    processor,
                    system_with_schema,
                    user_prompt,
                    annotation_schema,
                    candidate_validator,
                    prefix_allowed_tokens_fn,
                    compiled_xgrammar,
                    constraint_setup_seconds,
                )
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            for record in pending:
                item_dir = args.output_dir / "items" / record["image_id"]
                if not (item_dir / "candidate_record.json").exists():
                    persist_failure(
                        record,
                        expected,
                        args,
                        candidate_validator,
                        run_error,
                    )

    summary = aggregate_shard(
        assigned,
        args,
        load_seconds,
        constraint_setup_seconds,
        started_at,
        run_error,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["succeeded"] == summary["assigned_images"] else 1


if __name__ == "__main__":
    sys.exit(main())
