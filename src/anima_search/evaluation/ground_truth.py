from __future__ import annotations

import csv, json
from pathlib import Path


def load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_relevance(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["query_id"], {})[row["image_id"]] = int(row["relevance"])
    return result
