#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from build_m1_canonical_schema import build


M1_ROOT = Path(__file__).resolve().parent
SOURCE_SCHEMA = M1_ROOT / "specification/schemas/annotation_payload.schema.json"


class BuildM1CanonicalSchemaTest(unittest.TestCase):
    def test_only_expected_contract_areas_are_relaxed(self) -> None:
        source = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
        canonical = build(source)
        Draft202012Validator.check_schema(canonical)

        source_properties = source["properties"]
        canonical_properties = canonical["properties"]
        self.assertEqual(canonical_properties["ocr"]["maxItems"], 100)
        self.assertEqual(
            canonical_properties["captions"]["properties"]["dense_zh"]["minLength"],
            1,
        )
        self.assertEqual(
            canonical_properties["entities"]["items"]["properties"]["entity_type"],
            source_properties["entities"]["items"]["properties"]["entity_type"],
        )
        self.assertEqual(
            canonical_properties["entities"]["items"]["properties"]["bbox_norm_1000"],
            source_properties["entities"]["items"]["properties"]["bbox_norm_1000"],
        )
        self.assertEqual(
            canonical_properties["relations"]["items"]["properties"]["predicate"]["enum"],
            source_properties["relations"]["items"]["properties"]["predicate"]["enum"],
        )
        self.assertEqual(
            canonical_properties["entities"]["items"]["properties"]["attributes"]["properties"]["colors_zh"]["maxItems"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
