from __future__ import annotations

import hashlib
from pathlib import Path

from anima_search.schemas import SearchResult


class MockSearchService:
    """Deterministic image-directory search used only for UI and M7 integration tests."""

    def __init__(self, project_root: str | Path, image_dir: str | Path,
                 result_count: int = 8) -> None:
        self.project_root = Path(project_root).resolve()
        raw_image_dir = Path(image_dir)
        self.image_dir = (
            raw_image_dir.resolve() if raw_image_dir.is_absolute()
            else (self.project_root / raw_image_dir).resolve()
        )
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"mock image directory does not exist: {self.image_dir}")
        if result_count <= 0:
            raise ValueError("result_count must be positive")
        self.result_count = result_count
        self.annotations: dict[str, object] = {}
        self.config = {
            "project_root": str(self.project_root),
            "retrieval": {"rerank_default": False, "result_count": result_count},
            "models": {"qwen_vl": "not loaded in mock mode", "stable_diffusion": "not loaded",
                       "embedder": "not loaded"},
            "annotation": {"prompt_version": "unavailable"},
            "runtime": {"mode": "mock"},
        }
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.images = sorted(
            (path for path in self.image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions),
            key=lambda path: path.name.casefold(),
        )
        if not self.images:
            raise ValueError(f"mock image directory contains no supported images: {self.image_dir}")

    def _relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return Path("..", path.relative_to(self.project_root.parent)).as_posix()

    def search(self, query: str, use_reranker: bool = False) -> list[SearchResult]:
        del use_reranker
        digest = hashlib.sha256(query.strip().encode("utf-8")).digest()
        offset = int.from_bytes(digest[:4], "big") % len(self.images)
        ordered = self.images[offset:] + self.images[:offset]
        results = []
        for rank, path in enumerate(ordered[:self.result_count], start=1):
            results.append(SearchResult(
                image_id=f"mock-{path.stem}",
                relative_path=self._relative_path(path),
                fused_score=1.0 / rank,
                branch_scores={"mock": 1.0 / rank},
                branch_ranks={"mock": rank},
                active_branches=["mock"],
                evidence=["模拟模式仅验证界面与接口，不代表检索相关性。"],
            ))
        return results

    def answer_about_image(self, image_id: str, question: str) -> str:
        del image_id, question
        return "模拟模式未加载 VLM；请切换真实模型后进行图片问答。"

    def write_content(self, image_id: str, content_type: str, tone: str) -> dict[str, str]:
        del image_id, content_type, tone
        return {"status": "unavailable", "reason": "模拟模式未加载 VLM。"}

    def generate_image(self, query: str, image_id: str | None = None,
                       seed: int | None = None) -> Path:
        del query, image_id, seed
        raise RuntimeError("模拟模式未加载图像生成模型")
