from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from anima_search.m6.results import M6QueryResult
from anima_search.schemas import SearchResult


def load_m6_query(path: Path, query_id: str) -> M6QueryResult:
    requested = query_id.strip()
    if not requested:
        raise ValueError("query_id must not be empty")
    found: M6QueryResult | None = None
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            result = M6QueryResult.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(
                f"invalid M6 result at line {line_number}: {exc}"
            ) from exc
        if result.query_id in seen:
            raise ValueError(
                f"duplicate M6 query_id at line {line_number}: {result.query_id}"
            )
        seen.add(result.query_id)
        if result.query_id == requested:
            found = result
    if found is None:
        raise ValueError(f"M6 query_id not found: {requested}")
    return found


def select_story_candidates(
    result: M6QueryResult,
    count: int,
) -> list[SearchResult]:
    if not 3 <= count <= 8:
        raise ValueError(f"story selection requires 3 to 8 images; received {count}")
    if len(result.candidates) < count:
        raise ValueError(
            f"M6 result only contains {len(result.candidates)} candidates"
        )
    return [
        SearchResult(
            image_id=item.image_id,
            relative_path=item.relative_path,
            fused_score=item.fused_score,
            rerank_score=item.rerank_score,
            branch_scores=dict(item.branch_scores),
            branch_ranks=dict(item.branch_ranks),
            matched_fields=list(item.matched_fields),
            active_branches=list(item.branch_scores),
            mismatch=list(item.mismatch),
            source="real",
        )
        for item in result.candidates[:count]
    ]
