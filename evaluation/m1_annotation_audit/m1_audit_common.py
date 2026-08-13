"""Shared helpers for the lightweight M1 human audit."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


AUDIT_VERSION = "m1-audit-ui-v0.1.0"
SOURCE_IDS = ("qwen35_9b", "internvl35_14b", "fusion")
SOURCE_NAMES = {
    "qwen35_9b": "Qwen3.5-9B",
    "internvl35_14b": "InternVL3.5-14B",
    "fusion": "Fusion v0.1.1",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            values.append(value)
    return values


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def blind_source_order(image_id: str, reviewer: str, seed: int) -> list[str]:
    """Return a deterministic source order without persisting model labels in reviews."""

    ranked = []
    for source_id in SOURCE_IDS:
        digest = sha256_text(f"{seed}\0{reviewer}\0{image_id}\0{source_id}")
        ranked.append((digest, source_id))
    return [source_id for _, source_id in sorted(ranked)]


def normalize_characters(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in text if not character.isspace())


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _valid_choice(value: Any, allowed: set[str], field: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}")
    return str(value)


def validate_gold(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("gold must be an object")
    assessability = _valid_choice(
        payload.get("assessability"), {"assessable", "partly_assessable", "unassessable"}, "assessability"
    )
    primary = _valid_choice(
        payload.get("scene_primary"),
        {
            "general", "indoor", "street_urban", "nature", "people_activity", "food",
            "transport", "animal_plant", "object_exhibit", "illustration_meme",
            "document_screen", "uncertain",
        },
        "scene_primary",
    )
    environment = _valid_choice(
        payload.get("environment"), {"indoor", "outdoor", "mixed", "not_applicable", "uncertain"}, "environment"
    )
    entities = payload.get("salient_entities")
    if not isinstance(entities, list) or len(entities) > 30:
        raise ValueError("salient_entities must be an array with at most 30 items")
    clean_entities = []
    seen_entity_ids: set[str] = set()
    for index, item in enumerate(entities):
        if not isinstance(item, dict):
            raise ValueError(f"salient_entities[{index}] must be an object")
        gold_id = str(item.get("gold_id") or f"g{index + 1}")
        if not re.fullmatch(r"g\d+", gold_id) or gold_id in seen_entity_ids:
            raise ValueError("gold entity IDs must be unique g<number> values")
        seen_entity_ids.add(gold_id)
        name = re.sub(r"\s+", " ", str(item.get("name") or "")).strip()
        if not name or len(name) > 80:
            raise ValueError(f"salient_entities[{index}].name is required and must be <= 80 chars")
        count_evaluable = bool(item.get("count_evaluable"))
        count = item.get("count")
        if count_evaluable:
            if type(count) is not int or not 1 <= count <= 999:
                raise ValueError(f"salient_entities[{index}].count must be 1..999")
        else:
            count = None
        clean_entities.append(
            {"gold_id": gold_id, "name": name, "count_evaluable": count_evaluable, "count": count}
        )

    ocr_items = payload.get("clear_ocr")
    if not isinstance(ocr_items, list) or len(ocr_items) > 50:
        raise ValueError("clear_ocr must be an array with at most 50 items")
    clean_ocr = []
    seen_ocr_ids: set[str] = set()
    for index, item in enumerate(ocr_items):
        if not isinstance(item, dict):
            raise ValueError(f"clear_ocr[{index}] must be an object")
        gold_id = str(item.get("gold_id") or f"o{index + 1}")
        if not re.fullmatch(r"o\d+", gold_id) or gold_id in seen_ocr_ids:
            raise ValueError("gold OCR IDs must be unique o<number> values")
        seen_ocr_ids.add(gold_id)
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if not text or len(text) > 300:
            raise ValueError(f"clear_ocr[{index}].text is required and must be <= 300 chars")
        clean_ocr.append({"gold_id": gold_id, "text": text})
    notes = str(payload.get("notes") or "").strip()
    if len(notes) > 2000:
        raise ValueError("gold notes must be <= 2000 chars")
    return {
        "assessability": assessability,
        "scene_primary": primary,
        "environment": environment,
        "salient_entities": clean_entities,
        "clear_ocr": clean_ocr,
        "notes": notes,
    }


def validate_candidate_reviews(
    payload: Any,
    public_candidates: list[dict[str, Any]],
    gold: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("candidate_reviews must be an object")
    expected_slots = {candidate["slot"] for candidate in public_candidates}
    if require_complete and set(payload) != expected_slots:
        raise ValueError("all blind candidate slots must be reviewed before submission")
    if not set(payload).issubset(expected_slots):
        raise ValueError("candidate_reviews contains an unknown blind slot")
    clean: dict[str, Any] = {}
    entity_allowed = {"supported", "unsupported", "uncertain"}
    count_allowed = {"correct", "incorrect", "not_evaluable"}
    ocr_allowed = {"correct", "partial", "invented", "unreadable"}
    relation_allowed = {"correct", "incorrect", "uncertain"}
    gold_entity_ids = {item["gold_id"] for item in gold["salient_entities"]}
    gold_ocr_ids = {item["gold_id"] for item in gold["clear_ocr"]}

    for candidate in public_candidates:
        slot = candidate["slot"]
        if slot not in payload:
            continue
        review = payload[slot]
        if not isinstance(review, dict):
            raise ValueError(f"review for {slot} must be an object")
        annotation = candidate.get("annotation") or {}
        entity_ids = {str(item.get("entity_id")) for item in annotation.get("entities", [])}
        ocr_ids = {str(item.get("text_id")) for item in annotation.get("ocr", [])}
        relation_ids = {f"r{index + 1}" for index, _ in enumerate(annotation.get("relations", []))}

        entity_judgments = review.get("entity_judgments") or {}
        if not isinstance(entity_judgments, dict) or not set(entity_judgments).issubset(entity_ids):
            raise ValueError(f"invalid entity judgments for {slot}")
        clean_entity = {}
        for entity_id, item in entity_judgments.items():
            if not isinstance(item, dict):
                raise ValueError(f"entity judgment {entity_id} must be an object")
            clean_entity[entity_id] = {
                "support": _valid_choice(item.get("support"), entity_allowed, f"{slot}.{entity_id}.support"),
                "count": _valid_choice(item.get("count"), count_allowed, f"{slot}.{entity_id}.count"),
            }
        if require_complete and set(clean_entity) != entity_ids:
            raise ValueError(f"all entities in {slot} must be reviewed")

        coverage = review.get("salient_coverage") or {}
        if not isinstance(coverage, dict) or set(coverage) != gold_entity_ids:
            if require_complete or coverage:
                raise ValueError(f"salient coverage in {slot} must contain every gold entity")
        clean_coverage = {key: bool(value) for key, value in coverage.items()}

        ocr_judgments = review.get("ocr_judgments") or {}
        if not isinstance(ocr_judgments, dict) or not set(ocr_judgments).issubset(ocr_ids):
            raise ValueError(f"invalid OCR judgments for {slot}")
        clean_ocr_judgments = {}
        for text_id, item in ocr_judgments.items():
            if not isinstance(item, dict):
                raise ValueError(f"OCR judgment {text_id} must be an object")
            status = _valid_choice(item.get("status"), ocr_allowed, f"{slot}.{text_id}.status")
            corrected = re.sub(r"\s+", " ", str(item.get("corrected_text") or "")).strip()
            if len(corrected) > 300:
                raise ValueError("corrected OCR text must be <= 300 chars")
            if require_complete and status == "partial" and not corrected:
                raise ValueError("partial OCR judgments require corrected_text")
            clean_ocr_judgments[text_id] = {"status": status, "corrected_text": corrected}
        if require_complete and set(clean_ocr_judgments) != ocr_ids:
            raise ValueError(f"all OCR items in {slot} must be reviewed")

        ocr_coverage = review.get("ocr_coverage") or {}
        if not isinstance(ocr_coverage, dict) or set(ocr_coverage) != gold_ocr_ids:
            if require_complete or ocr_coverage:
                raise ValueError(f"OCR coverage in {slot} must contain every gold OCR item")
        clean_ocr_coverage = {key: bool(value) for key, value in ocr_coverage.items()}

        relation_judgments = review.get("relation_judgments") or {}
        if not isinstance(relation_judgments, dict) or not set(relation_judgments).issubset(relation_ids):
            raise ValueError(f"invalid relation judgments for {slot}")
        clean_relations = {
            key: _valid_choice(value, relation_allowed, f"{slot}.{key}.relation")
            for key, value in relation_judgments.items()
        }
        if require_complete and set(clean_relations) != relation_ids:
            raise ValueError(f"all relations in {slot} must be reviewed")

        new_fact_count = review.get("caption_new_fact_count")
        if type(new_fact_count) is not int or not 0 <= new_fact_count <= 50:
            raise ValueError("caption_new_fact_count must be an integer from 0 to 50")
        new_fact_notes = str(review.get("caption_new_fact_notes") or "").strip()
        if len(new_fact_notes) > 2000:
            raise ValueError("caption_new_fact_notes must be <= 2000 chars")
        if require_complete and new_fact_count > 0 and not new_fact_notes:
            raise ValueError("caption new facts require a short explanation")
        correctness = review.get("caption_correctness")
        completeness = review.get("caption_completeness")
        if type(correctness) is not int or not 1 <= correctness <= 5:
            raise ValueError("caption_correctness must be 1..5")
        if type(completeness) is not int or not 1 <= completeness <= 5:
            raise ValueError("caption_completeness must be 1..5")
        privacy = bool(review.get("privacy_violation"))
        privacy_notes = str(review.get("privacy_notes") or "").strip()
        if len(privacy_notes) > 1000:
            raise ValueError("privacy_notes must be <= 1000 chars")
        if require_complete and privacy and not privacy_notes:
            raise ValueError("privacy violations require a short explanation")
        notes = str(review.get("notes") or "").strip()
        if len(notes) > 2000:
            raise ValueError("candidate notes must be <= 2000 chars")
        clean[slot] = {
            "entity_judgments": clean_entity,
            "salient_coverage": clean_coverage,
            "ocr_judgments": clean_ocr_judgments,
            "ocr_coverage": clean_ocr_coverage,
            "relation_judgments": clean_relations,
            "caption_new_fact_count": new_fact_count,
            "caption_new_fact_notes": new_fact_notes,
            "caption_correctness": correctness,
            "caption_completeness": completeness,
            "privacy_violation": privacy,
            "privacy_notes": privacy_notes,
            "notes": notes,
        }
    return clean


def compute_metrics(
    submitted_reviews: list[dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
    *,
    blind_seed: int,
) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = {}
    for source_id in SOURCE_IDS:
        accumulators[source_id] = {
            "images": 0,
            "scene_correct": 0,
            "scene_total": 0,
            "environment_correct": 0,
            "environment_total": 0,
            "entity_supported": 0,
            "entity_evaluable": 0,
            "entity_uncertain": 0,
            "salient_covered": 0,
            "salient_total": 0,
            "count_correct": 0,
            "count_evaluable": 0,
            "ocr_supported": 0,
            "ocr_exact": 0,
            "ocr_evaluable": 0,
            "ocr_unreadable": 0,
            "ocr_cer_edits": 0,
            "ocr_cer_chars": 0,
            "ocr_gold_covered": 0,
            "ocr_gold_total": 0,
            "relation_correct": 0,
            "relation_evaluable": 0,
            "relation_uncertain": 0,
            "caption_new_fact_images": 0,
            "caption_new_facts": 0,
            "privacy_images": 0,
            "caption_correctness": [],
            "caption_completeness": [],
        }

    for review_record in submitted_reviews:
        image_id = review_record["image_id"]
        reviewer = review_record["reviewer"]
        task = tasks_by_id[image_id]
        source_map = {
            f"candidate_{index + 1}": source_id
            for index, source_id in enumerate(blind_source_order(image_id, reviewer, blind_seed))
        }
        task_sources = {item["source_id"]: item for item in task["candidates"]}
        gold = review_record["gold"]
        for slot, candidate_review in review_record["candidate_reviews"].items():
            source_id = source_map[slot]
            source = task_sources[source_id]
            annotation = source.get("annotation") or {}
            acc = accumulators[source_id]
            acc["images"] += 1
            if gold["scene_primary"] != "uncertain":
                acc["scene_total"] += 1
                acc["scene_correct"] += annotation.get("scene", {}).get("primary_type") == gold["scene_primary"]
            if gold["environment"] != "uncertain":
                acc["environment_total"] += 1
                acc["environment_correct"] += annotation.get("scene", {}).get("environment") == gold["environment"]

            entities = {str(item.get("entity_id")): item for item in annotation.get("entities", [])}
            for entity_id, judgment in candidate_review["entity_judgments"].items():
                if judgment["support"] == "uncertain":
                    acc["entity_uncertain"] += 1
                else:
                    acc["entity_evaluable"] += 1
                    acc["entity_supported"] += judgment["support"] == "supported"
                entity = entities.get(entity_id) or {}
                if entity.get("count_exact") and entity.get("count") is not None:
                    if judgment["count"] != "not_evaluable":
                        acc["count_evaluable"] += 1
                        acc["count_correct"] += judgment["count"] == "correct"
            acc["salient_total"] += len(gold["salient_entities"])
            acc["salient_covered"] += sum(candidate_review["salient_coverage"].values())

            ocr = {str(item.get("text_id")): item for item in annotation.get("ocr", [])}
            for text_id, judgment in candidate_review["ocr_judgments"].items():
                status = judgment["status"]
                if status == "unreadable":
                    acc["ocr_unreadable"] += 1
                    continue
                acc["ocr_evaluable"] += 1
                acc["ocr_supported"] += status in {"correct", "partial"}
                acc["ocr_exact"] += status == "correct"
                if status in {"correct", "partial"}:
                    candidate_text = normalize_characters((ocr.get(text_id) or {}).get("text_raw"))
                    reference_text = normalize_characters(
                        judgment["corrected_text"] if status == "partial" else (ocr.get(text_id) or {}).get("text_raw")
                    )
                    if reference_text:
                        acc["ocr_cer_edits"] += levenshtein(candidate_text, reference_text)
                        acc["ocr_cer_chars"] += len(reference_text)
            acc["ocr_gold_total"] += len(gold["clear_ocr"])
            acc["ocr_gold_covered"] += sum(candidate_review["ocr_coverage"].values())

            for judgment in candidate_review["relation_judgments"].values():
                if judgment == "uncertain":
                    acc["relation_uncertain"] += 1
                else:
                    acc["relation_evaluable"] += 1
                    acc["relation_correct"] += judgment == "correct"
            new_facts = candidate_review["caption_new_fact_count"]
            acc["caption_new_facts"] += new_facts
            acc["caption_new_fact_images"] += new_facts > 0
            acc["privacy_images"] += candidate_review["privacy_violation"]
            acc["caption_correctness"].append(candidate_review["caption_correctness"])
            acc["caption_completeness"].append(candidate_review["caption_completeness"])

    metrics: dict[str, Any] = {}
    for source_id, acc in accumulators.items():
        metrics[source_id] = {
            "source_name": SOURCE_NAMES[source_id],
            "reviewed_images": acc["images"],
            "scene_primary_accuracy": safe_ratio(acc["scene_correct"], acc["scene_total"]),
            "scene_primary_correct": acc["scene_correct"],
            "scene_primary_total": acc["scene_total"],
            "environment_accuracy": safe_ratio(acc["environment_correct"], acc["environment_total"]),
            "environment_correct": acc["environment_correct"],
            "environment_total": acc["environment_total"],
            "entity_mention_precision": safe_ratio(acc["entity_supported"], acc["entity_evaluable"]),
            "entity_supported": acc["entity_supported"],
            "entity_evaluable": acc["entity_evaluable"],
            "entity_uncertain": acc["entity_uncertain"],
            "salient_entity_recall": safe_ratio(acc["salient_covered"], acc["salient_total"]),
            "salient_entity_covered": acc["salient_covered"],
            "salient_entity_total": acc["salient_total"],
            "exact_count_accuracy": safe_ratio(acc["count_correct"], acc["count_evaluable"]),
            "exact_count_correct": acc["count_correct"],
            "exact_count_evaluable": acc["count_evaluable"],
            "ocr_item_precision": safe_ratio(acc["ocr_supported"], acc["ocr_evaluable"]),
            "ocr_exact_accuracy": safe_ratio(acc["ocr_exact"], acc["ocr_evaluable"]),
            "ocr_evaluable": acc["ocr_evaluable"],
            "ocr_unreadable": acc["ocr_unreadable"],
            "ocr_cer": safe_ratio(acc["ocr_cer_edits"], acc["ocr_cer_chars"]),
            "ocr_cer_edits": acc["ocr_cer_edits"],
            "ocr_cer_reference_chars": acc["ocr_cer_chars"],
            "clear_ocr_recall": safe_ratio(acc["ocr_gold_covered"], acc["ocr_gold_total"]),
            "clear_ocr_covered": acc["ocr_gold_covered"],
            "clear_ocr_total": acc["ocr_gold_total"],
            "relation_precision": safe_ratio(acc["relation_correct"], acc["relation_evaluable"]),
            "relation_correct": acc["relation_correct"],
            "relation_evaluable": acc["relation_evaluable"],
            "relation_uncertain": acc["relation_uncertain"],
            "caption_new_fact_image_rate": safe_ratio(acc["caption_new_fact_images"], acc["images"]),
            "caption_new_fact_images": acc["caption_new_fact_images"],
            "caption_new_fact_mean": safe_ratio(acc["caption_new_facts"], acc["images"]),
            "caption_new_fact_total": acc["caption_new_facts"],
            "privacy_violation_image_rate": safe_ratio(acc["privacy_images"], acc["images"]),
            "privacy_violation_images": acc["privacy_images"],
            "caption_correctness_mean": mean(acc["caption_correctness"]),
            "caption_completeness_mean": mean(acc["caption_completeness"]),
        }

    comparison: dict[str, Any] = {}
    comparable = (
        "scene_primary_accuracy", "environment_accuracy", "entity_mention_precision",
        "salient_entity_recall", "exact_count_accuracy", "ocr_item_precision",
        "ocr_exact_accuracy", "clear_ocr_recall", "relation_precision",
        "caption_new_fact_image_rate", "caption_new_fact_mean", "privacy_violation_image_rate",
        "caption_correctness_mean", "caption_completeness_mean",
    )
    for baseline in ("qwen35_9b", "internvl35_14b"):
        deltas = {}
        for name in comparable:
            fused = metrics["fusion"].get(name)
            base = metrics[baseline].get(name)
            deltas[name] = fused - base if fused is not None and base is not None else None
        comparison[f"fusion_minus_{baseline}"] = deltas
    return {
        "audit_version": AUDIT_VERSION,
        "sample_size": len(tasks_by_id),
        "submitted_reviews": len(submitted_reviews),
        "single_reviewer": True,
        "metrics": metrics,
        "comparisons": comparison,
        "limitations": [
            "Single-reviewer stratified audit; no inter-rater agreement is available.",
            f"The {len(tasks_by_id)} images are risk-stratified and are not an unbiased estimate of all 2369 images.",
            "All ratios include their effective denominators; small-denominator differences require caution.",
        ],
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def metrics_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M1 融合前后单人人工评审结果",
        "",
        f"已提交图片：{report['submitted_reviews']} / {report.get('sample_size', report['submitted_reviews'])}",
        "",
        "本报告来自单人、风险分层抽样，不代表 2369 张全量数据的无偏总体估计。每项结果必须结合有效分母解释。",
        "",
        "## 核心指标",
        "",
        "| 指标 | Qwen3.5-9B | InternVL3.5-14B | Fusion |",
        "|---|---:|---:|---:|",
    ]
    rows = [
        ("场景主类准确率", "scene_primary_accuracy"),
        ("实体 mention precision", "entity_mention_precision"),
        ("显著实体 recall", "salient_entity_recall"),
        ("精确计数准确率", "exact_count_accuracy"),
        ("OCR item precision", "ocr_item_precision"),
        ("OCR exact accuracy", "ocr_exact_accuracy"),
        ("清晰 OCR recall", "clear_ocr_recall"),
        ("关系 precision", "relation_precision"),
        ("Caption 新增事实图片率", "caption_new_fact_image_rate"),
        ("Caption correctness", "caption_correctness_mean"),
        ("Caption completeness", "caption_completeness_mean"),
        ("隐私违规图片率", "privacy_violation_image_rate"),
    ]
    for label, key in rows:
        values = [format_metric(report["metrics"][source_id].get(key)) for source_id in SOURCE_IDS]
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines.extend(["", "## 有效分母", ""])
    for source_id in SOURCE_IDS:
        item = report["metrics"][source_id]
        lines.append(
            f"- {item['source_name']}：实体 {item['entity_evaluable']}，显著实体 {item['salient_entity_total']}，"
            f"计数 {item['exact_count_evaluable']}，OCR {item['ocr_evaluable']}，关系 {item['relation_evaluable']}。"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 本轮由一人完成，无法报告评审者间一致性。",
            "- 样本按高风险类别分层，不应用于估计全量数据的自然错误率。",
            "- 对有效分母很小的指标不做稳定优劣结论。",
            "",
        ]
    )
    return "\n".join(lines)
