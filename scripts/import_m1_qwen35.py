from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.adapters.annotation import CANONICAL_QWEN35_VERSION, adapt_annotation
from anima_search.config import load_config, resolve_path
from anima_search.schemas import ManifestItem


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL file does not exist: {path}")
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"record at {path}:{line_number} must be an object")
        records.append(payload)
    return records


def _load_manifests(artifacts: Path) -> dict[str, ManifestItem]:
    result: dict[str, ManifestItem] = {}
    for split in ("train", "val"):
        path = artifacts / "manifests" / f"{split}.jsonl"
        for payload in _read_jsonl(path):
            item = ManifestItem.model_validate(payload)
            numeric_id = item.image_id.rsplit("-", 1)[-1]
            if numeric_id in result:
                raise ValueError(f"numeric image ID is ambiguous across manifests: {numeric_id}")
            result[numeric_id] = item
    return result


def import_annotations(source: Path, artifacts: Path, *, require_complete: bool = False) -> dict:
    manifests = _load_manifests(artifacts)
    by_split: dict[str, list] = {"train": [], "val": []}
    failures: list[dict[str, object]] = []
    seen: set[str] = set()

    for line_number, payload in enumerate(_read_jsonl(source), start=1):
        raw_id = str(payload.get("image_id", "")).strip()
        manifest = manifests.get(raw_id)
        if manifest is None:
            failures.append({"line": line_number, "image_id": raw_id, "error": "not in manifest"})
            continue
        if raw_id in seen:
            failures.append({"line": line_number, "image_id": raw_id, "error": "duplicate ID"})
            continue
        seen.add(raw_id)
        try:
            annotation = adapt_annotation(payload, manifest)
        except Exception as exc:
            failures.append(
                {"line": line_number, "image_id": raw_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        by_split[manifest.split.lower()].append(annotation)

    missing = sorted(set(manifests) - seen, key=lambda value: int(value))
    if require_complete and (missing or failures):
        raise ValueError(
            f"Qwen3.5 coverage is incomplete: missing={missing[:20]}, failures={failures[:5]}"
        )

    output_dir = artifacts / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for split, annotations in by_split.items():
        annotations.sort(key=lambda item: int(item.image_id.rsplit("-", 1)[-1]))
        path = output_dir / f"{split}.{CANONICAL_QWEN35_VERSION}.jsonl"
        path.write_text(
            "".join(item.model_dump_json() + "\n" for item in annotations),
            encoding="utf-8",
        )
        outputs[split] = str(path)

    report = {
        "annotation_version": CANONICAL_QWEN35_VERSION,
        "source": str(source),
        "source_record_count": len(seen),
        "manifest_record_count": len(manifests),
        "imported": {split: len(items) for split, items in by_split.items()},
        "missing_image_ids": missing,
        "failures": failures,
        "outputs": outputs,
    }
    report_path = output_dir / "qwen35_canonical_v1.3_import_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["report"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import M1 Qwen3.5 canonical v1.3 annotations for M3-M5."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    configured_source = config["annotation"].get("source")
    if args.source is None and not configured_source:
        parser.error("provide --source or annotation.source in the config")
    source = args.source or resolve_path(config, str(configured_source))
    artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    report = import_annotations(source.resolve(), artifacts, require_complete=args.require_complete)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
