#!/usr/bin/env python3
"""Audit completed M1 candidate runs without modifying their artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from m1_validation import diagnostic_json_candidate


INDEX_RE = re.compile(r"(?<=\.)\d+(?=\.|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Completed run directory. May be specified more than once.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=12)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"Invalid --run value: {value!r}; expected LABEL=PATH")
    return label, Path(path)


def image_sort_key(path: Path) -> tuple[int, str]:
    image_id = path.parent.name
    return (int(image_id), image_id) if image_id.isdigit() else (10**18, image_id)


def normalized_path(path: str) -> str:
    return INDEX_RE.sub("[]", path).replace(".[]", "[]")


def summarize_value(value: Any) -> str:
    if isinstance(value, list):
        if len(value) <= 8 and all(
            item is None or isinstance(item, (str, int, float, bool))
            for item in value
        ):
            return json.dumps(value, ensure_ascii=False)
        return f"array(len={len(value)})"
    if isinstance(value, dict):
        keys = ",".join(sorted(str(key) for key in value)[:8])
        suffix = ",..." if len(value) > 8 else ""
        return f"object(keys={keys}{suffix})"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def value_at_path(value: Any, path: str) -> Any:
    if path == "$":
        return value
    current = value
    for part in path.removeprefix("$.").split("."):
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        elif isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def classify_schema_message(message: str) -> str:
    if "is not one of" in message:
        return "enum"
    if message.endswith("is too long"):
        return "max_items_or_length"
    if "is greater than the maximum" in message:
        return "maximum"
    if "is less than the minimum" in message:
        return "minimum"
    if "is a required property" in message:
        return "required"
    if message.startswith("Additional properties are not allowed"):
        return "additional_properties"
    if "is not of type" in message:
        return "type"
    if "does not match" in message:
        return "pattern"
    if "has non-unique elements" in message:
        return "unique_items"
    return "other"


def add_sample(samples: dict[str, list[str]], key: str, image_id: str, limit: int) -> None:
    values = samples[key]
    if image_id not in values and len(values) < limit:
        values.append(image_id)


def load_annotation(run_dir: Path, image_id: str) -> Any | None:
    raw_path = run_dir / "raw" / f"{image_id}.txt"
    if not raw_path.exists():
        return None
    raw = raw_path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        annotation, _, duplicate_keys, _ = diagnostic_json_candidate(raw)
        return None if duplicate_keys else annotation


def audit_run(label: str, run_dir: Path, sample_limit: int) -> dict[str, Any]:
    summary_paths = sorted(
        (run_dir / "items").glob("*/summary.json"), key=image_sort_key
    )
    if not summary_paths:
        raise FileNotFoundError(f"No per-image summaries found under {run_dir}")

    metric_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    schema_category_occurrences: Counter[str] = Counter()
    schema_category_images: dict[str, set[str]] = defaultdict(set)
    schema_path_occurrences: Counter[str] = Counter()
    schema_path_images: dict[str, set[str]] = defaultdict(set)
    schema_path_values: dict[str, Counter[str]] = defaultdict(Counter)
    semantic_path_occurrences: Counter[str] = Counter()
    semantic_path_images: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, list[str]] = defaultdict(list)
    image_status: dict[str, dict[str, bool]] = {}

    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        image_id = str(summary["image_id"])
        annotation = None
        image_status[image_id] = {
            key: summary.get(key) is True
            for key in (
                "json_parse_ok",
                "diagnostic_json_parse_ok",
                "schema_valid",
                "semantic_valid",
                "annotation_valid",
            )
        }

        for key in ("json_parse_ok", "schema_valid", "semantic_valid", "annotation_valid"):
            if summary.get(key) is True:
                metric_counts[key] += 1
            else:
                add_sample(samples, f"not_{key}", image_id, sample_limit)

        if summary.get("json_parse_ok") is not True:
            if summary.get("diagnostic_json_parse_ok") is True:
                metric_counts["diagnostic_json_recoverable"] += 1
            else:
                metric_counts["json_unrecoverable"] += 1

        for issue in summary.get("format_issues", []):
            format_counts[str(issue)] += 1
            add_sample(samples, f"format:{issue}", image_id, sample_limit)

        schema_errors = summary.get("schema_errors", [])
        if schema_errors:
            annotation = load_annotation(run_dir, image_id)
        for error in schema_errors:
            path = normalized_path(str(error.get("path", "$")))
            message = str(error.get("message", ""))
            category = classify_schema_message(message)
            schema_category_occurrences[category] += 1
            schema_category_images[category].add(image_id)
            schema_path_occurrences[path] += 1
            schema_path_images[path].add(image_id)
            add_sample(samples, f"schema:{path}", image_id, sample_limit)
            if annotation is not None:
                actual = value_at_path(annotation, str(error.get("path", "$")))
                serialized = summarize_value(actual)
                schema_path_values[path][serialized] += 1

        for error in summary.get("semantic_errors", []):
            path = normalized_path(str(error.get("path", "$")))
            semantic_path_occurrences[path] += 1
            semantic_path_images[path].add(image_id)
            add_sample(samples, f"semantic:{path}", image_id, sample_limit)

    total = len(summary_paths)

    def rate(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    schema_paths = []
    for path, occurrences in schema_path_occurrences.most_common():
        schema_paths.append(
            {
                "path": path,
                "occurrences": occurrences,
                "affected_images": len(schema_path_images[path]),
                "top_values": [
                    {"value": value, "occurrences": count}
                    for value, count in schema_path_values[path].most_common(10)
                ],
                "sample_image_ids": samples[f"schema:{path}"],
            }
        )

    semantic_paths = []
    for path, occurrences in semantic_path_occurrences.most_common():
        semantic_paths.append(
            {
                "path": path,
                "occurrences": occurrences,
                "affected_images": len(semantic_path_images[path]),
                "sample_image_ids": samples[f"semantic:{path}"],
            }
        )

    return {
        "label": label,
        "run_dir": str(run_dir),
        "total_images": total,
        "metrics": {
            key: {"count": metric_counts[key], "rate": rate(metric_counts[key])}
            for key in (
                "json_parse_ok",
                "diagnostic_json_recoverable",
                "json_unrecoverable",
                "schema_valid",
                "semantic_valid",
                "annotation_valid",
            )
        },
        "format_issues": dict(format_counts.most_common()),
        "schema_categories": [
            {
                "category": category,
                "occurrences": occurrences,
                "affected_images": len(schema_category_images[category]),
            }
            for category, occurrences in schema_category_occurrences.most_common()
        ],
        "schema_paths": schema_paths,
        "semantic_paths": semantic_paths,
        "sample_image_ids": dict(samples),
        "image_status": image_status,
    }


def intersection_report(audits: list[dict[str, Any]], sample_limit: int) -> dict[str, Any]:
    if len(audits) < 2:
        return {}
    common_ids = set(audits[0]["image_status"])
    for audit in audits[1:]:
        common_ids &= set(audit["image_status"])

    result: dict[str, Any] = {"common_images": len(common_ids)}
    for metric in ("json_parse_ok", "schema_valid", "semantic_valid", "annotation_valid"):
        all_valid = sorted(
            image_id
            for image_id in common_ids
            if all(audit["image_status"][image_id][metric] for audit in audits)
        )
        all_invalid = sorted(
            image_id
            for image_id in common_ids
            if all(not audit["image_status"][image_id][metric] for audit in audits)
        )
        result[metric] = {
            "all_valid": len(all_valid),
            "all_invalid": len(all_invalid),
            "all_invalid_sample_image_ids": all_invalid[:sample_limit],
        }
    return result


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M1 全量候选结构审计",
        "",
        "> 本报告只统计结构与跨字段规则，不评价视觉事实是否正确。原始候选未被修改。",
        "",
        "## 总览",
        "",
        "| 模型 | 图片 | 严格 JSON | Schema | 语义 | 全部通过 | 可恢复 JSON | 不可恢复 JSON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in report["runs"]:
        metrics = audit["metrics"]
        cells = []
        for key in (
            "json_parse_ok",
            "schema_valid",
            "semantic_valid",
            "annotation_valid",
            "diagnostic_json_recoverable",
            "json_unrecoverable",
        ):
            item = metrics[key]
            cells.append(f"{item['count']} ({item['rate']:.2%})")
        lines.append(
            f"| {audit['label']} | {audit['total_images']} | " + " | ".join(cells) + " |"
        )

    for audit in report["runs"]:
        lines.extend(["", f"## {audit['label']} 错误分布", "", "### Schema 错误类别", ""])
        lines.extend(
            [
                "| 类别 | 错误次数 | 受影响图片 |",
                "|---|---:|---:|",
            ]
        )
        for item in audit["schema_categories"]:
            lines.append(
                f"| `{item['category']}` | {item['occurrences']} | {item['affected_images']} |"
            )
        lines.extend(["", "### 高频 Schema 字段", ""])
        lines.extend(
            [
                "| 字段路径 | 错误次数 | 受影响图片 | 高频实际值 | 示例图片 |",
                "|---|---:|---:|---|---|",
            ]
        )
        for item in audit["schema_paths"][:25]:
            values = ", ".join(
                f"`{entry['value']}` x{entry['occurrences']}"
                for entry in item["top_values"][:4]
            )
            samples = ", ".join(item["sample_image_ids"])
            lines.append(
                f"| `{item['path']}` | {item['occurrences']} | "
                f"{item['affected_images']} | {values or '-'} | {samples} |"
            )
        lines.extend(["", "### 高频语义规则", ""])
        lines.extend(
            [
                "| 字段路径 | 错误次数 | 受影响图片 | 示例图片 |",
                "|---|---:|---:|---|",
            ]
        )
        for item in audit["semantic_paths"][:20]:
            lines.append(
                f"| `{item['path']}` | {item['occurrences']} | "
                f"{item['affected_images']} | {', '.join(item['sample_image_ids'])} |"
            )

    if report["cross_run"]:
        lines.extend(["", "## 两模型交集", ""])
        lines.extend(
            [
                "| 检查项 | 两边都通过 | 两边都未通过 | 双方未通过示例 |",
                "|---|---:|---:|---|",
            ]
        )
        for metric, item in report["cross_run"].items():
            if metric == "common_images":
                continue
            lines.append(
                f"| `{metric}` | {item['all_valid']} | {item['all_invalid']} | "
                f"{', '.join(item['all_invalid_sample_image_ids'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = [parse_run(value) for value in args.run]
    audits = [audit_run(label, path, args.sample_limit) for label, path in runs]
    report = {
        "report_version": "1.0.0",
        "runs": audits,
        "cross_run": intersection_report(audits, args.sample_limit),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "structure_audit.json"
    markdown_path = args.output_dir / "structure_audit.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
