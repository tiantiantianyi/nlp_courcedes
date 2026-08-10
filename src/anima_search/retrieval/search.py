from anima_search.retrieval.filters import AnnotationFilter
from anima_search.retrieval.fusion import reciprocal_rank_fusion_with_ranks
from anima_search.schemas import ImageAnnotation, SearchQuery, SearchResult


def query_to_document(query: SearchQuery) -> str:
    count = []
    if query.count_target is not None and query.count_value is not None:
        count = [f"{query.count_target}:{query.count_operator or 'eq'}:{query.count_value}"]
    fields = [
        ("主体", query.objects), ("数量", count), ("动作", query.actions),
        ("场景", query.scene), ("时间", query.time_of_day), ("天气", query.weather),
        ("情绪", query.mood), ("颜色", query.colors), ("风格", query.style),
        ("必须", query.required_terms), ("文字", query.ocr_terms),
    ]
    structured = " ".join(f"{name}:{' '.join(values)}" for name, values in fields if values)
    semantic = query.semantic_text.strip() or query.raw_text
    return f"{semantic} {structured}".strip()


class HybridSearcher:
    def __init__(self, annotations: dict[str, ImageAnnotation], bm25: object | None = None,
                 vector: object | None = None, rrf_k: int = 60, image: object | None = None,
                 indexes: dict[str, object] | None = None, aliases: dict | None = None) -> None:
        self.annotations = annotations
        self.rrf_k = rrf_k
        self.indexes = indexes or {
            name: index for name, index in (("image", image), ("text", vector), ("bm25", bm25))
            if index is not None
        }
        self.annotation_filter = AnnotationFilter(aliases)
        self.last_branch_errors: dict[str, str] = {}

    def _filter_ranking(self, ranking: list[tuple[str, float]],
                        query: SearchQuery) -> list[tuple[str, float]]:
        filtered: list[tuple[str, float]] = []
        seen: set[str] = set()
        for image_id, score in ranking:
            if image_id in seen:
                continue
            seen.add(image_id)
            annotation = self.annotations.get(image_id)
            if annotation is not None and self.annotation_filter.evaluate(annotation, query).allowed:
                filtered.append((image_id, score))
        return filtered

    def search(self, query: SearchQuery, candidate_count: int = 50,
               result_count: int = 30) -> list[SearchResult]:
        if candidate_count <= 0 or result_count <= 0:
            return []
        retrieval_query = query_to_document(query)
        rankings: dict[str, list[tuple[str, float]]] = {}
        self.last_branch_errors = {}
        for name, index in self.indexes.items():
            try:
                rankings[name] = self._filter_ranking(
                    index.search(retrieval_query, candidate_count), query
                )
            except Exception as exc:  # one failed branch must not abort other branches
                self.last_branch_errors[name] = f"{type(exc).__name__}: {exc}"
        if not rankings:
            details = "; ".join(f"{name}={error}" for name, error in self.last_branch_errors.items())
            raise RuntimeError(f"all retrieval branches failed: {details}")

        preferred_order = ("image", "text", "bm25")
        active_branches = [name for name in preferred_order if name in rankings]
        active_branches.extend(name for name in rankings if name not in preferred_order)
        results: list[SearchResult] = []
        for image_id, score, branch_scores, branch_ranks in reciprocal_rank_fusion_with_ranks(
            rankings, self.rrf_k
        ):
            annotation = self.annotations.get(image_id)
            if annotation is None:
                continue
            decision = self.annotation_filter.evaluate(annotation, query)
            if not decision.allowed:
                continue
            results.append(SearchResult(
                image_id=image_id,
                relative_path=annotation.relative_path,
                fused_score=score,
                branch_scores=branch_scores,
                branch_ranks=branch_ranks,
                matched_fields=decision.matched_fields,
                evidence=decision.evidence,
                mismatch=decision.mismatch,
                active_branches=active_branches,
            ))
            if len(results) >= result_count:
                break
        return results
