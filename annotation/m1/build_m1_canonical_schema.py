#!/usr/bin/env python3
"""Build the post-normalization M1 schema from the immutable v1.2 source schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_max_length(node: dict[str, Any], value: int) -> None:
    node["maxLength"] = value


def build(source: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(json.dumps(source, ensure_ascii=False))
    schema["$id"] = "https://askalbum.local/schemas/annotation-payload-v1.3.json"
    schema["title"] = "AskAlbum M1 Canonical Annotation Payload v1.3"
    schema["description"] = (
        "M1 候选经可审计机械归一化后的入库载荷。v1.3 只放宽开放文本长度和 OCR 密集图容量；"
        "枚举、引用完整性、bbox、字段类型和颜色上限等硬约束保持不变。"
    )

    properties = schema["properties"]
    scene = properties["scene"]["properties"]
    set_max_length(scene["sub_type_zh"], 60)

    entity = properties["entities"]["items"]["properties"]
    set_max_length(entity["name_zh"], 60)
    attributes = entity["attributes"]["properties"]
    set_max_length(attributes["colors_zh"]["items"], 30)
    set_max_length(attributes["materials_zh"]["items"], 60)
    set_max_length(attributes["states_zh"]["items"], 60)
    set_max_length(attributes["action_zh"], 80)
    set_max_length(attributes["attire_zh"]["items"], 80)

    ocr = properties["ocr"]
    ocr["maxItems"] = 100
    set_max_length(ocr["items"]["properties"]["text_raw"], 2000)

    relation = properties["relations"]["items"]["properties"]
    set_max_length(relation["predicate_other_zh"], 60)

    event = properties["event"]["properties"]
    set_max_length(event["summary_zh"], 160)
    event["evidence_entity_ids"]["maxItems"] = 24

    subjective = properties["subjective"]["properties"]
    set_max_length(subjective["mood_terms_zh"]["items"], 30)
    set_max_length(subjective["palette_terms_zh"]["items"], 30)
    set_max_length(subjective["aesthetic_reason_zh"], 160)

    captions = properties["captions"]["properties"]
    captions["short_zh"]["minLength"] = 1
    captions["short_zh"]["maxLength"] = 80
    captions["dense_zh"]["minLength"] = 1
    captions["dense_zh"]["maxLength"] = 500

    uncertainty = properties["uncertainties"]["items"]["properties"]
    set_max_length(uncertainty["note_zh"], 120)
    return schema


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    schema = build(source)
    Draft202012Validator.check_schema(schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
