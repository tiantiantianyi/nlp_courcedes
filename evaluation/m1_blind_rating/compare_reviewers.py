#!/usr/bin/env python3
"""Compare two revealed M1 blind-rating exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


DIMENSIONS = ("accuracy", "completeness", "usability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cohen_kappa(
    pairs: Iterable[tuple[Any, Any]],
    categories: list[Any],
    *,
    quadratic: bool = False,
) -> float | None:
    values = list(pairs)
    if not values:
        return None
    index = {value: position for position, value in enumerate(categories)}
    size = len(categories)
    observed = [[0.0] * size for _ in categories]
    row_totals = [0.0] * size
    column_totals = [0.0] * size
    for left, right in values:
        row = index[left]
        column = index[right]
        observed[row][column] += 1.0
        row_totals[row] += 1.0
        column_totals[column] += 1.0

    count = float(len(values))
    if quadratic:
        denominator = max(size - 1, 1) ** 2

        def weight(row: int, column: int) -> float:
            return ((row - column) ** 2) / denominator

        observed_disagreement = sum(
            weight(row, column) * observed[row][column] / count
            for row in range(size)
            for column in range(size)
        )
        expected_disagreement = sum(
            weight(row, column)
            * (row_totals[row] / count)
            * (column_totals[column] / count)
            for row in range(size)
            for column in range(size)
        )
        if expected_disagreement == 0:
            return 1.0 if observed_disagreement == 0 else None
        return 1.0 - observed_disagreement / expected_disagreement

    observed_agreement = sum(observed[position][position] for position in range(size)) / count
    expected_agreement = sum(
        (row_totals[position] / count) * (column_totals[position] / count)
        for position in range(size)
    )
    if expected_agreement == 1.0:
        return 1.0 if observed_agreement == 1.0 else None
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def mean_model_scores(
    rows: dict[str, dict[str, Any]], source_ids: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_id in source_ids:
        source_name = next(
            row["revealed_ratings"][source_id]["source_name"] for row in rows.values()
        )
        scores = {
            dimension: fmean(
                row["revealed_ratings"][source_id][dimension] for row in rows.values()
            )
            for dimension in DIMENSIONS
        }
        result[source_id] = {
            "source_name": source_name,
            "mean_scores": {key: round(value, 4) for key, value in scores.items()},
            "composite_mean": round(fmean(scores.values()), 4),
            "severe_error_count": sum(
                bool(row["revealed_ratings"][source_id]["severe_error"])
                for row in rows.values()
            ),
            "best_choice_count": sum(
                row["revealed_best_choice"] == source_id for row in rows.values()
            ),
        }
    return result


def compare(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> dict[str, Any]:
    indexed_a = {str(row["image_id"]): row for row in rows_a}
    indexed_b = {str(row["image_id"]): row for row in rows_b}
    if len(indexed_a) != len(rows_a) or len(indexed_b) != len(rows_b):
        raise ValueError("review export contains duplicate image IDs")
    if indexed_a.keys() != indexed_b.keys():
        missing_a = sorted(indexed_b.keys() - indexed_a.keys(), key=int)
        missing_b = sorted(indexed_a.keys() - indexed_b.keys(), key=int)
        raise ValueError(f"review image sets differ: missing_a={missing_a}, missing_b={missing_b}")
    image_ids = sorted(indexed_a, key=int)
    source_ids = sorted(indexed_a[image_ids[0]]["revealed_ratings"])
    if any(sorted(row["revealed_ratings"]) != source_ids for row in rows_a + rows_b):
        raise ValueError("review exports contain different model sets")

    dimensions: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        pairs = [
            (
                int(indexed_a[image_id]["revealed_ratings"][source_id][dimension]),
                int(indexed_b[image_id]["revealed_ratings"][source_id][dimension]),
            )
            for image_id in image_ids
            for source_id in source_ids
        ]
        absolute_differences = [abs(left - right) for left, right in pairs]
        dimensions[dimension] = {
            "rating_pairs": len(pairs),
            "exact_agreement_count": sum(left == right for left, right in pairs),
            "exact_agreement_rate": rounded(fmean(left == right for left, right in pairs)),
            "within_one_count": sum(abs(left - right) <= 1 for left, right in pairs),
            "within_one_rate": rounded(fmean(abs(left - right) <= 1 for left, right in pairs)),
            "mean_absolute_difference": rounded(fmean(absolute_differences)),
            "quadratic_weighted_kappa": rounded(
                cohen_kappa(pairs, [1, 2, 3, 4, 5], quadratic=True)
            ),
        }

    severe: dict[str, Any] = {}
    all_severe_pairs: list[tuple[bool, bool]] = []
    severe_union: list[dict[str, Any]] = []
    for source_id in source_ids:
        records = [
            (
                image_id,
                bool(indexed_a[image_id]["revealed_ratings"][source_id]["severe_error"]),
                bool(indexed_b[image_id]["revealed_ratings"][source_id]["severe_error"]),
            )
            for image_id in image_ids
        ]
        pairs = [(left, right) for _, left, right in records]
        all_severe_pairs.extend(pairs)
        source_name = indexed_a[image_ids[0]]["revealed_ratings"][source_id]["source_name"]
        for image_id, left, right in records:
            if left or right:
                severe_union.append(
                    {
                        "image_id": image_id,
                        "source_id": source_id,
                        "source_name": source_name,
                        "review_a_severe_error": left,
                        "review_b_severe_error": right,
                        "agreement": left == right,
                    }
                )
        severe[source_id] = {
            "source_name": source_name,
            "both_positive": sum(left and right for left, right in pairs),
            "a_only": sum(left and not right for left, right in pairs),
            "b_only": sum(right and not left for left, right in pairs),
            "both_negative": sum(not left and not right for left, right in pairs),
            "both_positive_image_ids": [
                image_id for image_id, left, right in records if left and right
            ],
            "a_only_image_ids": [
                image_id for image_id, left, right in records if left and not right
            ],
            "b_only_image_ids": [
                image_id for image_id, left, right in records if right and not left
            ],
            "agreement_rate": rounded(fmean(left == right for left, right in pairs)),
            "cohen_kappa": rounded(cohen_kappa(pairs, [False, True])),
        }
    severe["overall"] = {
        "rating_pairs": len(all_severe_pairs),
        "agreement_rate": rounded(
            fmean(left == right for left, right in all_severe_pairs)
        ),
        "cohen_kappa": rounded(cohen_kappa(all_severe_pairs, [False, True])),
    }

    best_pairs = [
        (
            indexed_a[image_id]["revealed_best_choice"],
            indexed_b[image_id]["revealed_best_choice"],
        )
        for image_id in image_ids
    ]
    best_categories = sorted({value for pair in best_pairs for value in pair})
    best_choice = {
        "image_count": len(best_pairs),
        "exact_agreement_count": sum(left == right for left, right in best_pairs),
        "exact_agreement_rate": rounded(fmean(left == right for left, right in best_pairs)),
        "cohen_kappa": rounded(cohen_kappa(best_pairs, best_categories)),
        "review_a_counts": dict(Counter(left for left, _ in best_pairs)),
        "review_b_counts": dict(Counter(right for _, right in best_pairs)),
        "agreement_image_ids": [
            image_id
            for image_id, (left, right) in zip(image_ids, best_pairs)
            if left == right
        ],
        "disagreement_image_ids": [
            image_id
            for image_id, (left, right) in zip(image_ids, best_pairs)
            if left != right
        ],
    }

    reviewer_a = rows_a[0]["reviewer"]
    reviewer_b = rows_b[0]["reviewer"]
    models_a = mean_model_scores(indexed_a, source_ids)
    models_b = mean_model_scores(indexed_b, source_ids)
    ranking_a = sorted(source_ids, key=lambda source: models_a[source]["composite_mean"], reverse=True)
    ranking_b = sorted(source_ids, key=lambda source: models_b[source]["composite_mean"], reverse=True)
    return {
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b,
        "image_count": len(image_ids),
        "candidate_rating_pairs": len(image_ids) * len(source_ids),
        "source_ids": source_ids,
        "model_metrics": {
            source_id: {
                "source_name": models_a[source_id]["source_name"],
                "review_a": models_a[source_id],
                "review_b": models_b[source_id],
            }
            for source_id in source_ids
        },
        "composite_ranking": {
            "review_a": ranking_a,
            "review_b": ranking_b,
            "same_order": ranking_a == ranking_b,
        },
        "score_agreement": dimensions,
        "severe_error_agreement": severe,
        "severe_error_union": severe_union,
        "best_choice_agreement": best_choice,
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M1 双评审一致性报告",
        "",
        f"评审者 A：`{report['reviewer_a']}`  ",
        f"评审者 B：`{report['reviewer_b']}`  ",
        f"共同图片：{report['image_count']} 张；候选评分对：{report['candidate_rating_pairs']} 对",
        "",
        "## 模型级结果",
        "",
        "| 模型 | A 综合均分 | B 综合均分 | A 严重错误 | B 严重错误 | A 唯一最佳 | B 唯一最佳 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report["model_metrics"].values():
        left = model["review_a"]
        right = model["review_b"]
        lines.append(
            f"| {model['source_name']} | {left['composite_mean']:.2f} | "
            f"{right['composite_mean']:.2f} | {left['severe_error_count']} | "
            f"{right['severe_error_count']} | {left['best_choice_count']} | "
            f"{right['best_choice_count']} |"
        )

    lines.extend(
        [
            "",
            "## 分数一致性",
            "",
            "| 维度 | 完全相同 | 相差不超过 1 分 | 平均绝对分差 | 二次加权 Kappa |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    labels = {"accuracy": "准确性", "completeness": "完整性", "usability": "可用性"}
    for dimension, value in report["score_agreement"].items():
        lines.append(
            f"| {labels[dimension]} | {format_percent(value['exact_agreement_rate'])} | "
            f"{format_percent(value['within_one_rate'])} | "
            f"{value['mean_absolute_difference']:.2f} | "
            f"{value['quadratic_weighted_kappa']:.3f} |"
        )

    best = report["best_choice_agreement"]
    severe = report["severe_error_agreement"]["overall"]
    ranking = report["composite_ranking"]
    lines.extend(
        [
            "",
            "## 离散判断一致性",
            "",
            f"- 最佳候选完全一致：{best['exact_agreement_count']}/{best['image_count']}（{format_percent(best['exact_agreement_rate'])}），Cohen's Kappa = {best['cohen_kappa']:.3f}；",
            f"- 严重错误判断一致：{format_percent(severe['agreement_rate'])}，Cohen's Kappa = {severe['cohen_kappa']:.3f}；",
            f"- 三模型综合均分排序完全一致：{'是' if ranking['same_order'] else '否'}。",
            "",
            "## 严重错误交叉表",
            "",
            "| 模型 | 两人均判严重 | 仅 A 判严重 | 仅 B 判严重 | 两人均判不严重 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for source_id in report["source_ids"]:
        value = report["severe_error_agreement"][source_id]
        lines.append(
            f"| {value['source_name']} | {value['both_positive']} | "
            f"{value['a_only']} | {value['b_only']} | {value['both_negative']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = compare(read_jsonl(args.review_a), read_jsonl(args.review_b))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agreement.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "agreement.md").write_text(markdown(report), encoding="utf-8")
    (output_dir / "severe_error_union.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in report["severe_error_union"]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
