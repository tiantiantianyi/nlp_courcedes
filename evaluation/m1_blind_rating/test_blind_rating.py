from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from blind_rating_common import (  # noqa: E402
    SOURCE_IDS,
    blind_source_order,
    compute_metrics,
    sha256_text,
    slot_source_map,
    validate_rating,
)
from blind_rating_server import RatingStore  # noqa: E402
from build_blind_rating_tasks import allocate_diverse_high_disagreement  # noqa: E402
from compare_reviewers import DIMENSIONS, cohen_kappa, compare  # noqa: E402


def full_rating(best_choice: str = "candidate_1") -> dict:
    return {
        "ratings": {
            "candidate_1": {
                "accuracy": 5,
                "completeness": 4,
                "usability": 5,
                "severe_error": False,
            },
            "candidate_2": {
                "accuracy": 3,
                "completeness": 3,
                "usability": 3,
                "severe_error": False,
            },
            "candidate_3": {
                "accuracy": 1,
                "completeness": 2,
                "usability": 1,
                "severe_error": True,
            },
        },
        "best_choice": best_choice,
        "notes": "",
    }


class BlindRatingCommonTest(unittest.TestCase):
    def test_blind_order_is_deterministic_and_complete(self) -> None:
        order = blind_source_order("42", "reviewer", 123)
        self.assertEqual(order, blind_source_order("42", "reviewer", 123))
        self.assertEqual(set(order), set(SOURCE_IDS))
        orders = {
            tuple(blind_source_order(str(image_id), "reviewer", 123))
            for image_id in range(30)
        }
        self.assertGreater(len(orders), 1)

    def test_draft_accepts_partial_but_submit_requires_all_fields(self) -> None:
        partial = {
            "ratings": {"candidate_1": {"accuracy": 4}},
            "best_choice": None,
            "notes": "later",
        }
        clean = validate_rating(partial, require_complete=False)
        self.assertEqual(clean["ratings"]["candidate_1"]["accuracy"], 4)
        with self.assertRaises(ValueError):
            validate_rating(partial, require_complete=True)
        self.assertEqual(
            validate_rating(full_rating(), require_complete=True)["best_choice"],
            "candidate_1",
        )

    def test_metrics_reveal_blind_slots(self) -> None:
        image_id = "7"
        reviewer = "r"
        seed = 456
        mapping = slot_source_map(image_id, reviewer, seed)
        submitted = [
            {
                "image_id": image_id,
                "reviewer": reviewer,
                "rating": full_rating("candidate_1"),
            }
        ]
        tasks = {image_id: {"image_id": image_id, "sample_group": "ordinary"}}
        metrics = compute_metrics(submitted, tasks, blind_seed=seed)
        best_source = mapping["candidate_1"]
        severe_source = mapping["candidate_3"]
        self.assertEqual(metrics["models"][best_source]["unique_best_count"], 1)
        self.assertEqual(metrics["models"][severe_source]["severe_error_count"], 1)

    def test_high_disagreement_sampling_covers_scenes_before_repeats(self) -> None:
        records = [
            {"image_id": "1", "scene_primary_type": "document", "disagreement_score": 10},
            {"image_id": "2", "scene_primary_type": "document", "disagreement_score": 9},
            {"image_id": "3", "scene_primary_type": "street", "disagreement_score": 8},
            {"image_id": "4", "scene_primary_type": "food", "disagreement_score": 7},
        ]
        selected = allocate_diverse_high_disagreement(
            records, 3, seed=123, namespace="test"
        )
        self.assertEqual(
            {record["scene_primary_type"] for record in selected},
            {"document", "street", "food"},
        )


class RatingStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.rating_dir = root / "rating"
        image_dir = self.workspace / "data/train"
        image_dir.mkdir(parents=True)
        self.rating_dir.mkdir(parents=True)
        (image_dir / "1.jpg").write_bytes(b"not-a-real-image-but-servable")
        annotation = {
            "scene": {"primary_type": "general"},
            "entities": [],
            "ocr": [],
            "relations": [],
            "captions": {"short_zh": "test", "dense_zh": "test"},
        }
        task = {
            "rating_version": "m1-blind-rating-v1.0.0",
            "sample_index": 1,
            "image_id": "1",
            "processed_path": "data/train/1.jpg",
            "processed_sha256": "hash",
            "width": 10,
            "height": 10,
            "sample_group": "ordinary",
            "task_sha256": "task-hash",
            "candidates": [
                {"source_id": source_id, "annotation": annotation}
                for source_id in SOURCE_IDS
            ],
        }
        (self.rating_dir / "rating_tasks.jsonl").write_text(
            json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (self.rating_dir / "rating_manifest.json").write_text(
            json.dumps({"blind_seed": 99}), encoding="utf-8"
        )
        self.store = RatingStore(self.rating_dir, self.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_public_task_hides_source_identity(self) -> None:
        task = self.store.public_task("1", "reviewer")
        serialized = json.dumps(task, ensure_ascii=False)
        self.assertNotIn("source_id", serialized)
        self.assertNotIn("Qwen", serialized)
        self.assertNotIn("InternVL", serialized)
        self.assertEqual(
            [candidate["slot"] for candidate in task["candidates"]],
            ["candidate_1", "candidate_2", "candidate_3"],
        )

    def test_save_resume_submit_and_reopen(self) -> None:
        draft = self.store.save(
            "1",
            "reviewer",
            {"ratings": {"candidate_1": {"accuracy": 4}}, "notes": ""},
            submit=False,
        )
        self.assertEqual(draft["phase"], "draft")
        submitted = self.store.save("1", "reviewer", full_rating(), submit=True)
        self.assertEqual(submitted["phase"], "submitted")
        with self.assertRaises(ValueError):
            self.store.save("1", "reviewer", full_rating(), submit=True)
        reopened = self.store.reopen("1", "reviewer")
        self.assertEqual(reopened["phase"], "draft")
        self.assertEqual(self.store.summary("reviewer")["counts"]["draft"], 1)


class CompareReviewersTest(unittest.TestCase):
    def test_cohen_kappa_known_values(self) -> None:
        self.assertEqual(
            cohen_kappa([(1, 1), (2, 2), (3, 3)], [1, 2, 3], quadratic=True),
            1.0,
        )
        self.assertAlmostEqual(
            cohen_kappa(
                [(False, False), (False, True), (True, True), (True, True)],
                [False, True],
            ),
            0.5,
        )

    def test_identical_revealed_exports_have_full_agreement(self) -> None:
        rows = []
        for image_id, offset in (("1", 0), ("2", 1)):
            revealed = {}
            for position, source_id in enumerate(SOURCE_IDS):
                score = max(1, 5 - position - offset)
                revealed[source_id] = {
                    "source_name": source_id,
                    "accuracy": score,
                    "completeness": score,
                    "usability": score,
                    "severe_error": score == 1,
                }
            rows.append(
                {
                    "reviewer": "same",
                    "image_id": image_id,
                    "revealed_ratings": revealed,
                    "revealed_best_choice": SOURCE_IDS[0],
                }
            )
        result = compare(rows, rows)
        self.assertTrue(result["composite_ranking"]["same_order"])
        self.assertEqual(result["best_choice_agreement"]["exact_agreement_rate"], 1.0)
        self.assertEqual(
            result["severe_error_agreement"]["overall"]["agreement_rate"], 1.0
        )
        for dimension in DIMENSIONS:
            self.assertEqual(
                result["score_agreement"][dimension]["exact_agreement_rate"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
