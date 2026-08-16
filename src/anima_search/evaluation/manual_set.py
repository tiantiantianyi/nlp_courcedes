from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from anima_search.schemas import ManifestItem


QUERY_CATEGORIES = ("simple", "compositional", "negative", "count", "ocr")
RELEVANCE_FIELDS = ("query_id", "image_id", "relevance", "annotator", "note")
_JUDGMENT = re.compile(r"^(?P<image_id>[^:,\s]+)\s*[:：,]\s*(?P<grade>[012])$")


def load_manifest(path: Path) -> list[ManifestItem]:
    items = [
        ManifestItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [item for item in items if item.valid and item.duplicate_of is None]


def sample_manual_tasks(
    manifest: list[ManifestItem],
    count: int = 100,
    seed: int = 20260810,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(manifest):
        raise ValueError(f"requested {count} tasks from only {len(manifest)} valid images")
    shuffled = list(manifest)
    random.Random(seed).shuffle(shuffled)
    selected = sorted(shuffled[:count], key=lambda item: item.image_id)
    return [
        {
            "query_id": f"q{index:03d}",
            "text": "",
            "category": "",
            "source_image_id": item.image_id,
            "source_relative_path": item.relative_path,
            "reviewed": False,
            "annotator": "",
            "note": "",
        }
        for index, item in enumerate(selected, start=1)
    ]


def write_tasks(path: Path, tasks: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_tasks(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"manual query task file does not exist: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_relevance(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELEVANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_relevance_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_judgments(
    text: str,
    *,
    query_id: str,
    annotator: str,
    note: str = "",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in text.replace("；", "\n").replace(";", "\n").splitlines():
        value = raw.strip()
        if not value:
            continue
        match = _JUDGMENT.fullmatch(value)
        if match is None:
            raise ValueError(
                f"invalid relevance judgment {value!r}; use one 'image_id:grade' per line"
            )
        image_id = match.group("image_id")
        if image_id in seen:
            raise ValueError(f"duplicate relevance image ID: {image_id}")
        seen.add(image_id)
        grade = int(match.group("grade"))
        rows.append(
            {
                "query_id": query_id,
                "image_id": image_id,
                "relevance": grade,
                "annotator": annotator.strip(),
                "note": note.strip(),
            }
        )
    return rows


def format_judgments(rows: list[dict[str, object]], query_id: str) -> str:
    return "\n".join(
        f"{row['image_id']}:{row['relevance']}"
        for row in rows
        if str(row.get("query_id")) == query_id
    )


def replace_query_relevance(
    rows: list[dict[str, object]],
    query_id: str,
    replacement: list[dict[str, object]],
) -> list[dict[str, object]]:
    retained = [row for row in rows if str(row.get("query_id")) != query_id]
    return [*retained, *replacement]


def save_review(
    task_path: Path,
    relevance_path: Path,
    *,
    index: int,
    text: str,
    category: str,
    annotator: str,
    note: str,
    judgments: str,
    reviewed: bool,
) -> dict[str, object]:
    tasks = load_tasks(task_path)
    if not 0 <= index < len(tasks):
        raise IndexError(f"task index {index} is outside 0..{len(tasks) - 1}")
    if category not in QUERY_CATEGORIES:
        raise ValueError(f"category must be one of {QUERY_CATEGORIES}")
    if not text.strip():
        raise ValueError("query text must not be empty")
    if not annotator.strip():
        raise ValueError("annotator must not be empty")

    task = dict(tasks[index])
    parsed = parse_judgments(
        judgments,
        query_id=str(task["query_id"]),
        annotator=annotator,
        note=note,
    )
    if reviewed and not parsed:
        raise ValueError("a reviewed query requires at least one positive relevance judgment")
    if reviewed and not any(int(row["relevance"]) == 2 for row in parsed):
        raise ValueError("a reviewed query requires at least one relevance=2 judgment")

    task.update(
        text=text.strip(),
        category=category,
        annotator=annotator.strip(),
        note=note.strip(),
        reviewed=bool(reviewed),
    )
    tasks[index] = task
    rows = replace_query_relevance(
        load_relevance_rows(relevance_path),
        str(task["query_id"]),
        parsed,
    )
    write_tasks(task_path, tasks)
    write_relevance(relevance_path, rows)
    return task


def validate_manual_set(
    tasks: list[dict[str, object]],
    relevance_rows: list[dict[str, object]],
    *,
    expected_count: int | None = 100,
    valid_image_ids: set[str] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    if expected_count is not None and len(tasks) != expected_count:
        errors.append(f"expected {expected_count} queries, found {len(tasks)}")

    query_ids = [str(row.get("query_id", "")) for row in tasks]
    duplicates = sorted({query_id for query_id in query_ids if query_ids.count(query_id) > 1})
    if duplicates:
        errors.append(f"duplicate query IDs: {duplicates[:10]}")

    relevance_by_query: dict[str, list[dict[str, object]]] = {}
    for row in relevance_rows:
        query_id = str(row.get("query_id", ""))
        image_id = str(row.get("image_id", ""))
        try:
            grade = int(row.get("relevance", -1))
        except (TypeError, ValueError):
            grade = -1
        relevance_by_query.setdefault(query_id, []).append(row)
        if query_id not in query_ids:
            errors.append(f"relevance references unknown query: {query_id}")
        if grade not in {0, 1, 2}:
            errors.append(f"{query_id}/{image_id} has invalid relevance {grade}")
        if valid_image_ids is not None and image_id not in valid_image_ids:
            errors.append(f"{query_id} references unknown image: {image_id}")
        if not str(row.get("annotator", "")).strip():
            errors.append(f"{query_id}/{image_id} is missing an annotator")

    category_counts = {category: 0 for category in QUERY_CATEGORIES}
    for row in tasks:
        query_id = str(row.get("query_id", ""))
        category = str(row.get("category", ""))
        if not str(row.get("text", "")).strip():
            errors.append(f"{query_id} has empty query text")
        if category not in QUERY_CATEGORIES:
            errors.append(f"{query_id} has invalid category {category!r}")
        else:
            category_counts[category] += 1
        if not bool(row.get("reviewed", False)):
            errors.append(f"{query_id} is not reviewed")
        if not str(row.get("annotator", "")).strip():
            errors.append(f"{query_id} is missing an annotator")
        judgments = relevance_by_query.get(query_id, [])
        if not judgments:
            errors.append(f"{query_id} has no positive relevance judgments")
        elif not any(int(item["relevance"]) == 2 for item in judgments):
            errors.append(f"{query_id} has no relevance=2 image")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:30])
        suffix = f"\n... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise ValueError(f"manual evaluation set is incomplete:\n{preview}{suffix}")

    return {
        "query_count": len(tasks),
        "relevance_row_count": len(relevance_rows),
        "reviewed_count": sum(bool(row.get("reviewed")) for row in tasks),
        "category_counts": category_counts,
        "annotators": sorted(
            {str(row.get("annotator")) for row in tasks if str(row.get("annotator", "")).strip()}
        ),
    }
