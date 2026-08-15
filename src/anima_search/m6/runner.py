from __future__ import annotations

import math
from typing import Literal

from anima_search.m6.contract import M5Candidate, M5QueryBatch
from anima_search.m6.results import M6CandidateResult, M6QueryResult
from anima_search.schemas import SearchResult


def _unique_messages(messages: list[str]) -> list[str]:
    return list(dict.fromkeys(message for message in messages if message))


def _result(
    batch: M5QueryBatch,
    ordered: list[SearchResult],
    *,
    method: Literal["pointwise", "listwise"],
    mismatch: list[str],
    candidate_messages: dict[str, list[str]],
) -> M6QueryResult:
    source_by_id: dict[str, M5Candidate] = {
        candidate.image_id: candidate for candidate in batch.candidates
    }
    output_candidates: list[M6CandidateResult] = []
    for rerank_rank, item in enumerate(ordered, start=1):
        source = source_by_id[item.image_id]
        messages = _unique_messages(
            [*item.mismatch, *candidate_messages.get(item.image_id, [])]
        )
        output_candidates.append(
            M6CandidateResult(
                rank=source.rank,
                image_id=source.image_id,
                relative_path=source.relative_path,
                fused_score=source.fused_score,
                branch_scores=dict(source.branch_scores),
                branch_ranks=dict(source.branch_ranks),
                matched_fields=list(source.matched_fields),
                rerank_rank=rerank_rank,
                rerank_score=item.rerank_score,
                mismatch=messages,
            )
        )

    all_messages = _unique_messages(
        [
            *mismatch,
            *(
                message
                for candidate in output_candidates
                for message in candidate.mismatch
            ),
        ]
    )
    return M6QueryResult(
        schema_version="m6-rerank-v1.0",
        source_schema_version=batch.schema_version,
        query_id=batch.query_id,
        query=batch.query,
        category=batch.category,
        split=batch.split,
        fusion_method=batch.fusion_method,
        top_k=batch.top_k,
        annotation_version=batch.annotation_version,
        index_manifest_sha256=batch.index_manifest_sha256,
        config_sha256=batch.config_sha256,
        rerank_method=method,
        degraded=bool(all_messages),
        mismatch=all_messages,
        candidates=output_candidates,
    )


def _hard_fallback(
    batch: M5QueryBatch,
    *,
    method: Literal["pointwise", "listwise"],
    reason: str,
) -> M6QueryResult:
    ordered = batch.to_search_results()
    candidate_messages: dict[str, list[str]] = {}
    for item in ordered:
        item.rerank_score = 0.0
        candidate_messages[item.image_id] = [reason]
    return _result(
        batch,
        ordered,
        method=method,
        mismatch=[reason],
        candidate_messages=candidate_messages,
    )


def rerank_query_batch(
    batch: M5QueryBatch,
    reranker: object,
    *,
    method: Literal["pointwise", "listwise"],
) -> M6QueryResult:
    working = [
        candidate.model_copy(deep=True)
        for candidate in batch.to_search_results()
    ]
    try:
        returned = reranker.rerank(batch.query, working)  # type: ignore[attr-defined]
    except Exception as exc:
        return _hard_fallback(
            batch,
            method=method,
            reason=f"reranker raised {type(exc).__name__}: {exc}",
        )

    if not isinstance(returned, list) or not returned:
        return _hard_fallback(
            batch,
            method=method,
            reason="reranker returned an empty or non-list result",
        )
    if any(not isinstance(item, SearchResult) for item in returned):
        return _hard_fallback(
            batch,
            method=method,
            reason="reranker returned a non-SearchResult candidate",
        )

    source_ids = [candidate.image_id for candidate in batch.candidates]
    source_set = set(source_ids)
    unknown_ids = [
        item.image_id for item in returned if item.image_id not in source_set
    ]
    if unknown_ids:
        return _hard_fallback(
            batch,
            method=method,
            reason=f"reranker returned unknown image_id: {unknown_ids[0]}",
        )

    ordered: list[SearchResult] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    by_id = {item.image_id: item for item in working}
    for item in returned:
        if item.image_id in seen:
            duplicate_ids.append(item.image_id)
            continue
        if item.rerank_score is not None and not math.isfinite(item.rerank_score):
            return _hard_fallback(
                batch,
                method=method,
                reason=f"reranker returned non-finite score for {item.image_id}",
            )
        seen.add(item.image_id)
        ordered.append(item)

    missing_ids = [image_id for image_id in source_ids if image_id not in seen]
    candidate_messages: dict[str, list[str]] = {}
    mismatch: list[str] = []
    if duplicate_ids:
        duplicate_message = (
            f"dropped duplicate IDs: {sorted(set(duplicate_ids))}"
        )
        mismatch.append(duplicate_message)
        for image_id in set(duplicate_ids):
            candidate_messages.setdefault(image_id, []).append(
                duplicate_message
            )
    if missing_ids:
        missing_message = f"appended missing IDs: {missing_ids}"
        mismatch.append(missing_message)
        for image_id in missing_ids:
            item = by_id[image_id]
            item.rerank_score = 0.0
            ordered.append(item)
            candidate_messages.setdefault(image_id, []).append(
                missing_message
            )

    last_error = getattr(reranker, "last_error", None)
    last_degraded_reason = getattr(reranker, "last_degraded_reason", None)
    if last_error:
        mismatch.append(str(last_error))
    if last_degraded_reason:
        mismatch.append(str(last_degraded_reason))

    if len(ordered) != 20:
        return _hard_fallback(
            batch,
            method=method,
            reason=(
                "reranker output could not be repaired to contain every "
                "input candidate exactly once"
            ),
        )
    return _result(
        batch,
        ordered,
        method=method,
        mismatch=mismatch,
        candidate_messages=candidate_messages,
    )
