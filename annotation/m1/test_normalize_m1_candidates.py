#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from m1_validation import semantic_validation_errors
from normalize_m1_candidates import normalize_annotation


M1_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = M1_ROOT / "specification/schemas/annotation_payload.schema.json"
SOURCE_ANNOTATION_PATH = M1_ROOT / "specification/examples/m1_annotation_payload.example.json"


class NormalizeM1CandidatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(cls.schema)

    def source_annotation(self) -> dict:
        return json.loads(SOURCE_ANNOTATION_PATH.read_text(encoding="utf-8"))

    def assert_valid(self, annotation: dict) -> None:
        self.assertEqual(list(self.validator.iter_errors(annotation)), [])
        self.assertEqual(semantic_validation_errors(annotation), [])

    def test_repairs_common_structural_aliases(self) -> None:
        annotation = self.source_annotation()
        entity = annotation["entities"][0]
        entity["position_zone"] = "middle_center"
        entity["count"] = 3
        entity["count_exact"] = False
        entity["attributes"]["colors_zh"] = ["红", "黄", "蓝", "绿"]
        relation = annotation["relations"][0]
        relation["predicate"] = "in"
        relation.pop("predicate_other_zh", None)

        normalized, changes = normalize_annotation(annotation)

        self.assertEqual(normalized["entities"][0]["position_zone"], "center")
        self.assertIsNone(normalized["entities"][0]["count"])
        self.assertEqual(normalized["entities"][0]["attributes"]["colors_zh"], ["红", "黄", "蓝"])
        self.assertEqual(normalized["relations"][0]["predicate"], "inside")
        self.assertIsNone(normalized["relations"][0]["predicate_other_zh"])
        self.assertTrue(any(change["rule"] == "fill_required_null" for change in changes))
        self.assert_valid(normalized)

    def test_preserves_unknown_relation_and_coarsens_entity_type(self) -> None:
        annotation = self.source_annotation()
        annotation["entities"][0]["entity_type"] = "mountain"
        relation = annotation["relations"][0]
        relation["predicate"] = "hanging_from"
        relation.pop("predicate_other_zh", None)

        normalized, _ = normalize_annotation(annotation)

        self.assertEqual(normalized["entities"][0]["entity_type"], "other")
        self.assertEqual(normalized["relations"][0]["predicate"], "other")
        self.assertEqual(normalized["relations"][0]["predicate_other_zh"], "hanging_from")
        self.assert_valid(normalized)

    def test_reverses_passive_relation_and_nulls_invalid_bbox(self) -> None:
        annotation = self.source_annotation()
        relation = annotation["relations"][0]
        old_subject = relation["subject_id"]
        old_object = relation["object_id"]
        relation["predicate"] = "held_by"
        relation.pop("predicate_other_zh", None)
        annotation["entities"][0]["bbox_norm_1000"] = [0, 0, 1200, 500]

        normalized, changes = normalize_annotation(annotation)

        relation = normalized["relations"][0]
        self.assertEqual(relation["subject_id"], old_object)
        self.assertEqual(relation["object_id"], old_subject)
        self.assertEqual(relation["predicate"], "holding")
        self.assertIsNone(normalized["entities"][0]["bbox_norm_1000"])
        self.assertTrue(any(change["lossy"] for change in changes))
        self.assert_valid(normalized)

    def test_applies_bounded_lists_media_alias_and_relation_deduplication(self) -> None:
        annotation = self.source_annotation()
        annotation["scene"]["media_type"] = "document"
        entity = annotation["entities"][0]
        entity["attributes"]["materials_zh"] = ["a", "b", "c", "d"]
        entity["attributes"]["states_zh"] = ["a", "b", "c", "d", "e"]
        annotation["subjective"]["palette_terms_zh"] = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ]
        annotation["relations"].append(dict(annotation["relations"][0]))

        normalized, changes = normalize_annotation(annotation)

        self.assertEqual(normalized["scene"]["media_type"], "document_scan")
        self.assertEqual(entity["attributes"]["materials_zh"][:3], ["a", "b", "c"])
        self.assertEqual(normalized["entities"][0]["attributes"]["materials_zh"], ["a", "b", "c"])
        self.assertEqual(normalized["entities"][0]["attributes"]["states_zh"], ["a", "b", "c", "d"])
        self.assertEqual(normalized["subjective"]["palette_terms_zh"], ["a", "b", "c", "d", "e"])
        self.assertEqual(len(normalized["relations"]), len(annotation["relations"]) - 1)
        rules = {change["rule"] for change in changes}
        self.assertIn("canonical_media_type", rules)
        self.assertIn("keep_first_three_materials", rules)
        self.assertIn("keep_first_four_states", rules)
        self.assertIn("keep_first_five_palette_terms", rules)
        self.assertIn("drop_duplicate_relation", rules)
        self.assert_valid(normalized)

    def test_maps_scene_labels_misused_as_media_types(self) -> None:
        expected = {
            "document_screen": "natural_image",
            "screen": "natural_image",
            "illustration_meme": "illustration",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                annotation = self.source_annotation()
                annotation["scene"]["media_type"] = source
                normalized, changes = normalize_annotation(annotation)
                self.assertEqual(normalized["scene"]["media_type"], target)
                self.assertTrue(
                    any(change["rule"] == "canonical_media_type" for change in changes)
                )
                self.assert_valid(normalized)


if __name__ == "__main__":
    unittest.main()
