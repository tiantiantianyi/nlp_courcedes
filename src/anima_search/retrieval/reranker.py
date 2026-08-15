from __future__ import annotations

from pathlib import Path

from PIL import Image

from anima_search.annotation.validation import extract_json_object
from anima_search.schemas import SearchResult


def _string_list(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


class VisualReranker:
    def __init__(self, client: object, prompt: str, project_root: Path,
                 rrf_weight: float = 0.35, vlm_weight: float = 0.65,
                 max_new_tokens: int = 128) -> None:
        self.client = client
        self.prompt = prompt
        self.project_root = project_root
        self.rrf_weight = rrf_weight
        self.vlm_weight = vlm_weight
        self.max_new_tokens = max_new_tokens

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []
        max_rrf = max(item.fused_score for item in candidates) or 1.0
        scored: list[tuple[float, SearchResult]] = []
        for item in candidates:
            try:
                with Image.open(self.project_root / item.relative_path) as image:
                    raw = self.client.generate(
                        image.copy(),
                        f"{self.prompt}\n用户查询：{query}\n当前 image_id：{item.image_id}",
                        max_new_tokens=self.max_new_tokens,
                    )
                payload = extract_json_object(raw)
                response_image_id = str(payload.get("image_id", "")).strip()
                if response_image_id and response_image_id != item.image_id:
                    raise ValueError(
                        f"reranker image_id {response_image_id!r} does not match {item.image_id!r}"
                    )
                score = min(100.0, max(0.0, float(payload.get("score", 0.0))))
                item.rerank_score = score
                item.evidence = _string_list(payload.get("evidence"))
                item.mismatch = _string_list(payload.get("mismatch"))
            except Exception as exc:
                item.rerank_score = 0.0
                item.mismatch = [f"视觉重排不可用：{type(exc).__name__}: {exc}"]
                score = 0.0
            combined = (
                self.rrf_weight * (item.fused_score / max_rrf)
                + self.vlm_weight * (score / 100.0)
            )
            scored.append((combined, item))
        return [
            item
            for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].image_id))
        ]
