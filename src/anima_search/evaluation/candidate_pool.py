from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from anima_search.evaluation.manual_set import QUERY_CATEGORIES


_QUERY_NUMBER = re.compile(r"^(.*?)(\d+)$")


def _query_sort_key(row: Mapping[str, object]) -> tuple[str, int, str]:
    query_id = str(row.get("query_id", ""))
    match = _QUERY_NUMBER.fullmatch(query_id)
    if match is None:
        return query_id, -1, query_id
    return match.group(1), int(match.group(2)), query_id


def select_balanced_queries(
    queries: list[dict[str, object]],
    *,
    count: int,
) -> list[dict[str, object]]:
    """Select a deterministic near-equal number of queries from all five categories."""
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(queries):
        raise ValueError(f"requested {count} queries from only {len(queries)} rows")

    by_category = {
        category: sorted(
            [row for row in queries if str(row.get("category")) == category],
            key=_query_sort_key,
        )
        for category in QUERY_CATEGORIES
    }
    if any(not rows for rows in by_category.values()):
        missing = [category for category, rows in by_category.items() if not rows]
        raise ValueError(f"category coverage is incomplete; missing: {missing}")

    base, remainder = divmod(count, len(QUERY_CATEGORIES))
    targets = {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(QUERY_CATEGORIES)
    }
    insufficient = {
        category: (targets[category], len(by_category[category]))
        for category in QUERY_CATEGORIES
        if len(by_category[category]) < targets[category]
    }
    if insufficient:
        raise ValueError(
            f"category coverage cannot satisfy balanced selection: {insufficient}"
        )

    selected = [
        row
        for category in QUERY_CATEGORIES
        for row in by_category[category][: targets[category]]
    ]
    return sorted(selected, key=_query_sort_key)


def _candidate_identity(candidate: object) -> tuple[str, str]:
    if isinstance(candidate, str):
        return candidate.strip(), ""
    if isinstance(candidate, Mapping):
        return (
            str(candidate.get("image_id", "")).strip(),
            str(candidate.get("relative_path", "")).strip(),
        )
    return (
        str(getattr(candidate, "image_id", "")).strip(),
        str(getattr(candidate, "relative_path", "")).strip(),
    )


def build_candidate_pool(
    queries: list[dict[str, object]],
    rankings_by_variant: Mapping[str, Mapping[str, Sequence[object]]],
    source_ids: Mapping[str, str],
    per_variant_k: int,
    candidate_cap: int = 25,
) -> list[dict[str, object]]:
    """Build an auditable first-seen union of retrieval candidates per query."""
    if per_variant_k <= 0:
        raise ValueError("per_variant_k must be positive")
    if candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive")
    if not rankings_by_variant:
        raise ValueError("at least one retrieval variant is required")

    query_ids = [str(row.get("query_id", "")) for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("queries contain duplicate query IDs")
    expected_ids = set(query_ids)
    source_keys = set(source_ids)
    if source_keys != expected_ids:
        raise ValueError(
            "source IDs must cover exactly the selected query IDs; "
            f"missing={sorted(expected_ids - source_keys)}, "
            f"foreign={sorted(source_keys - expected_ids)}"
        )

    for variant, rankings in rankings_by_variant.items():
        ranked_ids = set(rankings)
        foreign = sorted(ranked_ids - expected_ids)
        missing = sorted(expected_ids - ranked_ids)
        if foreign:
            raise ValueError(
                f"variant {variant!r} contains foreign query IDs: {foreign}"
            )
        if missing:
            raise ValueError(f"variant {variant!r} is missing query IDs: {missing}")

    pool: list[dict[str, object]] = []
    for query in queries:
        query_id = str(query["query_id"])
        source_id = str(source_ids[query_id]).strip()
        declared_source = str(query.get("source_image_id", "")).strip()
        if not source_id or source_id != declared_source:
            raise ValueError(
                f"source ID mismatch for {query_id}: {source_id!r} != {declared_source!r}"
            )

        candidates: dict[str, dict[str, object]] = {
            source_id: {
                "image_id": source_id,
                "relative_path": str(query.get("source_relative_path", "")),
                "is_source": True,
                "retrieved_by": [],
                "best_rank": None,
                "grade": None,
                "annotator": "",
                "reviewed": False,
            }
        }
        order = [source_id]
        for variant, rankings in rankings_by_variant.items():
            seen_in_variant: set[str] = set()
            for rank, raw_candidate in enumerate(
                rankings[query_id][:per_variant_k], start=1
            ):
                image_id, relative_path = _candidate_identity(raw_candidate)
                if not image_id:
                    raise ValueError(
                        f"variant {variant!r}/{query_id} contains a blank image ID"
                    )
                if image_id in seen_in_variant:
                    continue
                seen_in_variant.add(image_id)
                candidate = candidates.get(image_id)
                if candidate is None:
                    if len(order) >= candidate_cap:
                        continue
                    candidate = {
                        "image_id": image_id,
                        "relative_path": relative_path,
                        "is_source": False,
                        "retrieved_by": [],
                        "best_rank": None,
                        "grade": None,
                        "annotator": "",
                        "reviewed": False,
                    }
                    candidates[image_id] = candidate
                    order.append(image_id)
                elif not candidate["relative_path"] and relative_path:
                    candidate["relative_path"] = relative_path
                retrieved_by = candidate["retrieved_by"]
                if variant not in retrieved_by:
                    retrieved_by.append(variant)
                best_rank = candidate["best_rank"]
                candidate["best_rank"] = rank if best_rank is None else min(best_rank, rank)

        pool.append(
            {
                "schema_version": "formal-relevance-pool-v1.0",
                "query_id": query_id,
                "text": str(query.get("text", "")),
                "category": str(query.get("category", "")),
                "source_image_id": source_id,
                "source_relative_path": str(query.get("source_relative_path", "")),
                "candidates": [candidates[image_id] for image_id in order],
            }
        )
    return pool
