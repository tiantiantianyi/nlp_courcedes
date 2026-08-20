from anima_search.evaluation.domain_transfer import (
    WIKIART_STYLES,
    balanced_sample,
    build_manual_review_rows,
    build_queries,
    build_relevance,
    medical_labels,
)


def test_medical_labels_ignores_simple_negation():
    assert "pneumothorax" not in medical_labels(
        "There is no pneumothorax."
    )
    assert "pneumothorax" in medical_labels(
        "A small right pneumothorax is present."
    )
    assert "pleural_effusion" in medical_labels(
        "There is a moderate pleural effusion."
    )
    assert "no_acute_abnormality" in medical_labels(
        "No acute cardiopulmonary process."
    )


def test_balanced_sample_is_unique_and_deterministic():
    rows = [
        {
            "index": str(index),
            "file": f"{index}.jpg",
            "style_name": style,
        }
        for index, style in enumerate(WIKIART_STYLES * 3)
    ]
    first, groups = balanced_sample(
        rows,
        WIKIART_STYLES,
        lambda row: {row["style_name"]},
        2,
        7,
    )
    second, _ = balanced_sample(
        rows,
        WIKIART_STYLES,
        lambda row: {row["style_name"]},
        2,
        7,
    )
    assert first == second
    assert len(first) == 10
    assert len({row["file"] for row in first}) == 10
    assert all(len(value) == 2 for value in groups.values())


def test_queries_relevance_and_review_have_expected_sizes():
    queries = build_queries()
    records = []
    for style in WIKIART_STYLES:
        for index in range(4):
            records.append(
                {
                    "image_id": f"art-{style}-{index}",
                    "domain": "wikiart",
                    "relative_path": f"art/{style}-{index}.jpg",
                    "labels": [style],
                }
            )
    medical = [
        "pleural_effusion",
        "pulmonary_edema",
        "pneumothorax",
        "atelectasis",
        "no_acute_abnormality",
    ]
    for label in medical:
        for index in range(4):
            records.append(
                {
                    "image_id": f"med-{label}-{index}",
                    "domain": "mimic_cxr",
                    "relative_path": f"med/{label}-{index}.jpg",
                    "labels": [label],
                }
            )
    relevance = build_relevance(queries, records)
    review = build_manual_review_rows(
        queries,
        records,
        relevance,
        seed=9,
    )
    assert len(queries) == 20
    assert all(len(grades) == 4 for grades in relevance.values())
    assert len(review) == 50
    assert sum(row["auto_relevance"] == 2 for row in review) == 30
    assert sum(row["auto_relevance"] == 0 for row in review) == 20
