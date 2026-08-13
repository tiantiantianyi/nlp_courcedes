from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m1_audit_common import (
    SOURCE_IDS,
    blind_source_order,
    compute_metrics,
    validate_candidate_reviews,
    validate_gold,
)
from m1_audit_server import AuditStore


def annotation(entity_name: str = "汽车") -> dict:
    return {
        "scene": {"primary_type": "transport", "environment": "outdoor", "media_type": "natural_image"},
        "capture_visual": {},
        "entities": [
            {
                "entity_id": "e1",
                "entity_type": "vehicle",
                "name_zh": entity_name,
                "count": 1,
                "count_exact": True,
                "position_zone": "center",
                "attributes": {},
            }
        ],
        "ocr": [{"text_id": "t1", "text_raw": "停车", "legibility": "high", "language": "zh"}],
        "relations": [],
        "captions": {"short_zh": "一辆汽车。", "dense_zh": "室外有一辆汽车。"},
    }


def gold() -> dict:
    return validate_gold(
        {
            "assessability": "assessable",
            "scene_primary": "transport",
            "environment": "outdoor",
            "salient_entities": [
                {"gold_id": "g1", "name": "汽车", "count_evaluable": True, "count": 1}
            ],
            "clear_ocr": [{"gold_id": "o1", "text": "停车"}],
            "notes": "",
        }
    )


def task(image_path: str) -> dict:
    return {
        "audit_version": "m1-audit-ui-v0.1.0",
        "image_id": "1",
        "sample_index": 1,
        "processed_path": image_path,
        "processed_sha256": "a" * 64,
        "width": 10,
        "height": 10,
        "primary_stratum": "general",
        "coverage_tags": ["general"],
        "candidates": [
            {"source_id": source_id, "available": True, "normalization_status": "valid", "lossy_repairs": 0, "annotation": annotation()}
            for source_id in SOURCE_IDS
        ],
        "task_sha256": "b" * 64,
    }


def candidate_review() -> dict:
    return {
        "entity_judgments": {"e1": {"support": "supported", "count": "correct"}},
        "salient_coverage": {"g1": True},
        "ocr_judgments": {"t1": {"status": "correct", "corrected_text": ""}},
        "ocr_coverage": {"o1": True},
        "relation_judgments": {},
        "caption_new_fact_count": 0,
        "caption_new_fact_notes": "",
        "caption_correctness": 5,
        "caption_completeness": 5,
        "privacy_violation": False,
        "privacy_notes": "",
        "notes": "",
    }


class M1AuditTest(unittest.TestCase):
    def test_blind_order_is_deterministic_and_complete(self) -> None:
        left = blind_source_order("9", "reviewer", 123)
        right = blind_source_order("9", "reviewer", 123)
        self.assertEqual(left, right)
        self.assertEqual(set(left), set(SOURCE_IDS))

    def test_gold_rejects_invalid_exact_count(self) -> None:
        value = gold()
        value["salient_entities"][0]["count"] = None
        with self.assertRaises(ValueError):
            validate_gold(value)

    def test_candidate_review_requires_every_item_on_submit(self) -> None:
        public_candidates = [
            {"slot": f"candidate_{index + 1}", "available": True, "annotation": annotation()}
            for index in range(3)
        ]
        partial = {"candidate_1": candidate_review()}
        with self.assertRaises(ValueError):
            validate_candidate_reviews(partial, public_candidates, gold(), require_complete=True)

    def test_metrics_resolve_blind_slots_to_real_sources(self) -> None:
        audit_task = task("image.jpg")
        order = blind_source_order("1", "reviewer", 123)
        reviews = {
            f"candidate_{index + 1}": candidate_review() for index in range(3)
        }
        fusion_slot = f"candidate_{order.index('fusion') + 1}"
        reviews[fusion_slot]["entity_judgments"]["e1"]["support"] = "unsupported"
        result = compute_metrics(
            [{"image_id": "1", "reviewer": "reviewer", "gold": gold(), "candidate_reviews": reviews}],
            {"1": audit_task},
            blind_seed=123,
        )
        self.assertEqual(result["metrics"]["fusion"]["entity_mention_precision"], 0.0)
        self.assertEqual(result["metrics"]["qwen35_9b"]["entity_mention_precision"], 1.0)

    def test_sqlite_store_saves_gold_and_submits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            image.write_bytes(b"not-a-real-jpeg")
            audit_dir = root / "audit"
            audit_dir.mkdir()
            audit_task = task("image.jpg")
            (audit_dir / "audit_tasks.jsonl").write_text(json.dumps(audit_task, ensure_ascii=False) + "\n", encoding="utf-8")
            (audit_dir / "audit_manifest.json").write_text(json.dumps({"blind_seed": 123}), encoding="utf-8")
            store = AuditStore(audit_dir, root)
            saved = store.save_gold("1", "reviewer", gold())
            self.assertEqual(saved["phase"], "gold_saved")
            public = store.public_task("1", "reviewer")
            payload = {candidate["slot"]: candidate_review() for candidate in public["candidates"]}
            submitted = store.save_candidates("1", "reviewer", payload, submit=True)
            self.assertEqual(submitted["phase"], "submitted")
            self.assertEqual(store.summary("reviewer")["counts"]["submitted"], 1)
            reopened = store.reopen("1", "reviewer")
            self.assertEqual(reopened["phase"], "gold_saved")
            self.assertEqual(store.summary("reviewer")["counts"]["submitted"], 0)


if __name__ == "__main__":
    unittest.main()
