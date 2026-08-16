from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CANDIDATE_RELEVANCE_FIELDS = (
    "query_id",
    "image_id",
    "relevance",
    "annotator",
    "note",
    "reviewed",
)
_GRADE_LINE = re.compile(r"^(?P<image_id>[^:,\s]+)\s*[:：,]\s*(?P<grade>\S+)$")


def load_candidate_pool(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate pool does not exist: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("candidate pool is empty")
    return rows


def load_candidate_relevance(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_candidate_grades(text: str, *, expected_ids: set[str]) -> dict[str, int]:
    grades: dict[str, int] = {}
    for raw_line in text.replace("；", "\n").replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _GRADE_LINE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"invalid candidate grade line {line!r}; use one image_id:grade per line"
            )
        image_id = match.group("image_id")
        if image_id in grades:
            raise ValueError(f"duplicate candidate ID: {image_id}")
        raw_grade = match.group("grade")
        try:
            grade = int(raw_grade)
        except ValueError as exc:
            raise ValueError(f"invalid candidate grade for {image_id}: {raw_grade}") from exc
        if grade not in {0, 1, 2}:
            raise ValueError(f"invalid candidate grade for {image_id}: {grade}")
        grades[image_id] = grade

    actual_ids = set(grades)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing:
        raise ValueError(f"missing candidate IDs: {missing}")
    if unexpected:
        raise ValueError(f"unexpected candidate IDs: {unexpected}")
    return grades


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _candidate_ids(pool_row: Mapping[str, object]) -> list[str]:
    return [
        str(candidate.get("image_id", ""))
        for candidate in list(pool_row.get("candidates", []))
    ]


def _write_candidate_relevance(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_RELEVANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_candidate_review(
    pool: list[dict[str, object]],
    output_path: Path,
    *,
    index: int,
    grades_text: str,
    annotator: str,
    note: str,
    reviewed: bool,
) -> list[dict[str, object]]:
    if not 0 <= index < len(pool):
        raise IndexError(f"candidate review index {index} is outside 0..{len(pool) - 1}")
    annotator = annotator.strip()
    if not annotator:
        raise ValueError("annotator must not be empty")

    pool_row = pool[index]
    query_id = str(pool_row["query_id"])
    candidate_ids = _candidate_ids(pool_row)
    if not candidate_ids or any(not image_id for image_id in candidate_ids):
        raise ValueError(f"candidate pool row {query_id} contains blank or no candidate IDs")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"candidate pool row {query_id} contains duplicate candidate IDs")
    grades = parse_candidate_grades(grades_text, expected_ids=set(candidate_ids))
    source_id = str(pool_row["source_image_id"])
    if grades.get(source_id) != 2:
        raise ValueError(f"source image {source_id} must retain grade 2")

    replacement = [
        {
            "query_id": query_id,
            "image_id": image_id,
            "relevance": grades[image_id],
            "annotator": annotator,
            "note": note.strip(),
            "reviewed": bool(reviewed),
        }
        for image_id in candidate_ids
    ]
    existing = load_candidate_relevance(output_path)
    retained = [row for row in existing if str(row.get("query_id")) != query_id]
    rows = [*retained, *replacement]
    query_order = {str(row["query_id"]): position for position, row in enumerate(pool)}
    candidate_order = {
        (str(row["query_id"]), image_id): position
        for row in pool
        for position, image_id in enumerate(_candidate_ids(row))
    }
    rows.sort(
        key=lambda row: (
            query_order.get(str(row.get("query_id")), len(query_order)),
            candidate_order.get(
                (str(row.get("query_id")), str(row.get("image_id"))), 10**9
            ),
        )
    )
    _write_candidate_relevance(output_path, rows)
    return rows


def review_progress(
    pool: list[dict[str, object]], rows: list[dict[str, object]]
) -> tuple[int, int]:
    by_query: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_query.setdefault(str(row.get("query_id", "")), []).append(row)

    complete = 0
    for pool_row in pool:
        query_id = str(pool_row["query_id"])
        candidate_ids = _candidate_ids(pool_row)
        judgments = by_query.get(query_id, [])
        if len(judgments) != len(candidate_ids):
            continue
        if {str(row.get("image_id", "")) for row in judgments} != set(candidate_ids):
            continue
        if any(not _is_true(row.get("reviewed")) for row in judgments):
            continue
        if any(not str(row.get("annotator", "")).strip() for row in judgments):
            continue
        try:
            grades = {str(row["image_id"]): int(row["relevance"]) for row in judgments}
        except (KeyError, TypeError, ValueError):
            continue
        if any(grade not in {0, 1, 2} for grade in grades.values()):
            continue
        if grades.get(str(pool_row["source_image_id"])) != 2:
            continue
        complete += 1
    return complete, len(pool)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_candidate_contact_sheet(
    pool_row: Mapping[str, object],
    project_root: Path,
    *,
    columns: int = 5,
    tile_size: int = 192,
    label_height: int | None = None,
) -> Image.Image:
    candidates = list(pool_row.get("candidates", []))
    if not 1 <= len(candidates) <= 25:
        raise ValueError(
            f"candidate contact sheet requires at most 25 candidates; received {len(candidates)}"
        )
    label_height = label_height or max(34, tile_size // 4)
    if columns <= 0 or tile_size <= label_height:
        raise ValueError("invalid candidate contact-sheet geometry")

    row_count = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (columns * tile_size, row_count * tile_size), "#111827")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(max(11, tile_size // 13))
    provenance_font = _font(max(9, tile_size // 17))
    image_height = tile_size - label_height
    for index, candidate in enumerate(candidates):
        image_id = str(candidate.get("image_id", ""))
        image_path = project_root / str(candidate.get("relative_path", ""))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"candidate image does not exist: {image_id} -> {image_path}"
            )
        with Image.open(image_path) as source:
            thumbnail = ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (tile_size, image_height),
                method=Image.Resampling.LANCZOS,
            )
        column = index % columns
        row = index // columns
        left = column * tile_size
        top = row * tile_size
        sheet.paste(thumbnail, (left, top + label_height))
        draw.rectangle(
            (left, top, left + tile_size - 1, top + tile_size - 1),
            outline="#22c55e" if bool(candidate.get("is_source")) else "#64748b",
            width=3 if bool(candidate.get("is_source")) else 2,
        )
        source_label = " [来源]" if bool(candidate.get("is_source")) else ""
        draw.text(
            (left + 6, top + 4),
            f"{index + 1:02d} {image_id}{source_label}",
            fill="white",
            font=title_font,
        )
        provenance = ",".join(str(value) for value in candidate.get("retrieved_by", []))
        draw.text(
            (left + 6, top + 5 + title_font.size),
            provenance[:30] or "source pinned",
            fill="#cbd5e1",
            font=provenance_font,
        )
    return sheet
