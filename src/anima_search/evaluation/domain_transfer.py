from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from anima_search.evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


WIKIART_STYLES = (
    "Impressionism",
    "Post_Impressionism",
    "Baroque",
    "Realism",
    "Romanticism",
)

MEDICAL_CATEGORIES = (
    "pleural_effusion",
    "pulmonary_edema",
    "pneumothorax",
    "atelectasis",
    "no_acute_abnormality",
)

MEDICAL_DISPLAY_NAMES = {
    "pleural_effusion": "胸腔积液",
    "pulmonary_edema": "肺水肿",
    "pneumothorax": "气胸",
    "atelectasis": "肺不张",
    "no_acute_abnormality": "无急性心肺异常",
}

_NEGATED_FINDING = re.compile(
    r"\b(?:no|without|negative for|absence of|free of)\b[^.;]{{0,45}}\b{term}\b",
    re.IGNORECASE,
)


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.lstrip("\ufeff").splitlines()))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _positive_finding(report: str, terms: tuple[str, ...]) -> bool:
    lowered = report.lower()
    for term in terms:
        if term not in lowered:
            continue
        negated = re.compile(
            _NEGATED_FINDING.pattern.format(term=re.escape(term)),
            re.IGNORECASE,
        )
        if not negated.search(report):
            return True
    return False


def medical_labels(report: str) -> set[str]:
    labels: set[str] = set()
    if _positive_finding(report, ("pleural effusion", "pleural effusions", "effusion")):
        labels.add("pleural_effusion")
    if _positive_finding(report, ("pulmonary edema", "edema")):
        labels.add("pulmonary_edema")
    if _positive_finding(report, ("pneumothorax",)):
        labels.add("pneumothorax")
    if _positive_finding(report, ("atelectasis", "atelectatic")):
        labels.add("atelectasis")
    if re.search(
        r"\bno acute (?:cardiopulmonary|pulmonary)(?: process| disease| abnormality| findings?)?\b"
        r"|\bno acute findings?\b",
        report,
        re.IGNORECASE,
    ):
        labels.add("no_acute_abnormality")
    return labels


def balanced_sample(
    rows: list[dict[str, str]],
    categories: tuple[str, ...],
    category_getter,
    per_category: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    if per_category <= 0:
        raise ValueError("per_category must be positive")
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, str]]] = {category: [] for category in categories}
    for row in rows:
        for category in category_getter(row):
            if category in pools:
                pools[category].append(row)
    for pool in pools.values():
        rng.shuffle(pool)

    selected_by_category: dict[str, list[dict[str, str]]] = {}
    used: set[str] = set()
    for category in categories:
        chosen = []
        for row in pools[category]:
            identifier = row.get("file") or row.get("image") or row.get("index", "")
            if identifier in used:
                continue
            chosen.append(row)
            used.add(identifier)
            if len(chosen) == per_category:
                break
        if len(chosen) != per_category:
            raise ValueError(
                f"category {category!r} has only {len(chosen)} unique rows; "
                f"{per_category} required"
            )
        selected_by_category[category] = chosen
    selected = [row for category in categories for row in selected_by_category[category]]
    return selected, selected_by_category


def build_queries() -> list[dict[str, object]]:
    art_templates = {
        "Impressionism": ("印象派绘画", "色彩明亮且笔触明显的印象派画作"),
        "Post_Impressionism": ("后印象派绘画", "具有强烈个人风格的后印象派画作"),
        "Baroque": ("巴洛克绘画", "具有强烈明暗对比的巴洛克画作"),
        "Realism": ("写实主义绘画", "表现真实人物或生活场景的写实画作"),
        "Romanticism": ("浪漫主义绘画", "强调情感与戏剧氛围的浪漫主义画作"),
    }
    medical_templates = {
        "pleural_effusion": ("报告提到胸腔积液的胸部X光片", "存在 pleural effusion 的胸片"),
        "pulmonary_edema": ("报告提到肺水肿的胸部X光片", "存在 pulmonary edema 的胸片"),
        "pneumothorax": ("报告提到气胸的胸部X光片", "存在 pneumothorax 的胸片"),
        "atelectasis": ("报告提到肺不张的胸部X光片", "存在 atelectasis 的胸片"),
        "no_acute_abnormality": (
            "报告显示无急性心肺异常的胸部X光片",
            "no acute cardiopulmonary abnormality 的胸片",
        ),
    }
    queries: list[dict[str, object]] = []
    for style, texts in art_templates.items():
        for variant, text_value in enumerate(texts, start=1):
            queries.append(
                {
                    "query_id": f"a9-art-{slug(style)}-{variant}",
                    "text": text_value,
                    "category": "domain_art_style",
                    "domain": "wikiart",
                    "target_label": style,
                    "reviewed": True,
                }
            )
    for label, texts in medical_templates.items():
        for variant, text_value in enumerate(texts, start=1):
            queries.append(
                {
                    "query_id": f"a9-medical-{slug(label)}-{variant}",
                    "text": text_value,
                    "category": "domain_medical_report",
                    "domain": "mimic_cxr",
                    "target_label": label,
                    "reviewed": True,
                }
            )
    return queries


def build_relevance(
    queries: list[dict[str, object]],
    records: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    relevance: dict[str, dict[str, int]] = {}
    for query in queries:
        target = str(query["target_label"])
        relevance[str(query["query_id"])] = {
            str(record["image_id"]): 2
            for record in records
            if target in set(record["labels"])
        }
    return relevance


def build_manual_review_rows(
    queries: list[dict[str, object]],
    records: list[dict[str, object]],
    relevance: dict[str, dict[str, int]],
    *,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    canonical = [query for query in queries if str(query["query_id"]).endswith("-1")]
    rows: list[dict[str, object]] = []
    for query in canonical:
        query_id = str(query["query_id"])
        domain_records = [
            record for record in records if record["domain"] == query["domain"]
        ]
        positives = [
            record for record in domain_records if record["image_id"] in relevance[query_id]
        ]
        negatives = [
            record for record in domain_records if record["image_id"] not in relevance[query_id]
        ]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        selected = [(record, 2) for record in positives[:3]]
        selected.extend((record, 0) for record in negatives[:2])
        for record, auto_relevance in selected:
            if record["domain"] == "mimic_cxr":
                reference_text = str(record.get("report", ""))
            else:
                reference_text = (
                    f"style={record.get('style_name', '')}; "
                    f"genre={record.get('genre_name', '')}; "
                    f"artist={record.get('artist_name', '')}"
                )
            rows.append(
                {
                    "query_id": query_id,
                    "query": query["text"],
                    "domain": query["domain"],
                    "image_id": record["image_id"],
                    "relative_path": record["relative_path"],
                    "reference_text": reference_text,
                    "auto_relevance": auto_relevance,
                    "human_relevance": "",
                    "annotator": "张添翼",
                    "review_status": "待复核",
                    "review_note": "",
                }
            )
    return rows


def ranking_metrics(ranked_ids: list[str], relevance: dict[str, int]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked_ids, relevance, 1),
        "recall@5": recall_at_k(ranked_ids, relevance, 5),
        "recall@10": recall_at_k(ranked_ids, relevance, 10),
        "mrr": reciprocal_rank(ranked_ids, relevance),
        "map": average_precision(ranked_ids, relevance),
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
    }


def summarize_details(details: list[dict[str, object]]) -> dict[str, object]:
    metric_names = ("recall@1", "recall@5", "recall@10", "mrr", "map", "ndcg@10")

    def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "query_count": len(rows),
            **{name: mean(float(row[name]) for row in rows) for name in metric_names},
            "latency_mean_seconds": mean(float(row["latency_seconds"]) for row in rows),
        }

    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in details:
        by_domain[str(row["domain"])].append(row)
    return {
        "overall": aggregate(details),
        "by_domain": {domain: aggregate(rows) for domain, rows in sorted(by_domain.items())},
    }


def dataset_summary(records: list[dict[str, object]]) -> dict[str, object]:
    domain_counts = Counter(str(record["domain"]) for record in records)
    label_counts = Counter(
        label for record in records for label in record.get("labels", [])
    )
    return {
        "record_count": len(records),
        "domain_counts": dict(sorted(domain_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
