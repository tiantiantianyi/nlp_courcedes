from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from anima_search.annotation.validation import extract_json_object
from anima_search.schemas import SearchResult


def _label_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def build_contact_sheet(
    candidates: list[SearchResult],
    project_root: Path,
    *,
    columns: int = 5,
    tile_size: int = 192,
    label_height: int = 34,
) -> Image.Image:
    """Build one bounded visual input for a listwise rerank call."""
    if not 1 <= len(candidates) <= 20:
        raise ValueError(f"listwise reranking requires 1-20 candidates; received {len(candidates)}")
    if columns <= 0 or tile_size <= label_height:
        raise ValueError("invalid contact-sheet geometry")

    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "#111827")
    draw = ImageDraw.Draw(sheet)
    font = _label_font(max(12, label_height // 2))
    image_height = tile_size - label_height

    for index, candidate in enumerate(candidates):
        path = project_root / candidate.relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"listwise candidate image does not exist: {candidate.image_id} -> {path}"
            )
        with Image.open(path) as source:
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
            outline="#64748b",
            width=2,
        )
        label = f"{index + 1:02d} {candidate.image_id}"
        draw.text((left + 7, top + 6), label, fill="white", font=font)

    return sheet


class ListwiseVisualReranker:
    """Rank up to 20 candidates with one Qwen-VL call over a contact sheet."""

    def __init__(
        self,
        client: object,
        prompt: str,
        project_root: Path,
        *,
        max_new_tokens: int = 768,
        columns: int = 5,
        tile_size: int = 192,
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.project_root = project_root
        self.max_new_tokens = max_new_tokens
        self.columns = columns
        self.tile_size = tile_size
        self.last_error: str | None = None
        self.last_degraded_reason: str | None = None
        self.last_contact_sheet_size: tuple[int, int] | None = None

    def _instruction(self, query: str, candidates: list[SearchResult]) -> str:
        candidate_map = "\n".join(
            f"{index + 1:02d} -> {item.image_id}"
            for index, item in enumerate(candidates)
        )
        return (
            f"{self.prompt}\n"
            f"用户查询：{query}\n"
            "候选编号与 image_id：\n"
            f"{candidate_map}\n"
            f"ranking 必须包含以上全部 {len(candidates)} 个 image_id，"
            "每个 image_id 恰好一次。"
        )

    def _parse_ranking(
        self,
        raw: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        payload = extract_json_object(raw)
        ranking = payload.get("ranking")
        if not isinstance(ranking, list) or not ranking:
            raise ValueError("listwise response must contain a non-empty ranking array")

        by_id = {item.image_id: item for item in candidates}
        returned_ids: list[str] = []
        scores: dict[str, float] = {}
        duplicate_ids: list[str] = []
        for rank, entry in enumerate(ranking, start=1):
            if isinstance(entry, dict):
                identifier = str(entry.get("image_id", "")).strip()
                score_value = entry.get("score")
            else:
                identifier = str(entry).strip()
                score_value = None
            if not identifier:
                raise ValueError("listwise ranking entry is missing image_id or index")
            image_id = identifier
            if image_id not in by_id and identifier.isdigit():
                candidate_index = int(identifier)
                if 1 <= candidate_index <= len(candidates):
                    image_id = candidates[candidate_index - 1].image_id
            if image_id not in by_id:
                raise ValueError(
                    f"listwise response contains unknown image_id or index {identifier!r}"
                )
            if image_id in scores:
                duplicate_ids.append(image_id)
                continue
            if score_value is None:
                denominator = max(1, len(candidates) - 1)
                score = 100.0 * (len(candidates) - rank) / denominator
            else:
                score = min(100.0, max(0.0, float(score_value)))
            returned_ids.append(image_id)
            scores[image_id] = score

        missing_ids = [
            item.image_id for item in candidates if item.image_id not in scores
        ]
        for image_id in missing_ids:
            returned_ids.append(image_id)
            scores[image_id] = 0.0
        warnings: list[str] = []
        if duplicate_ids:
            warnings.append(f"dropped duplicates: {sorted(set(duplicate_ids))}")
        if missing_ids:
            warnings.append(f"appended missing IDs: {missing_ids}")
        self.last_degraded_reason = "; ".join(warnings) or None

        ranked = [by_id[image_id] for image_id in returned_ids]
        for item in ranked:
            item.rerank_score = scores[item.image_id]
        return ranked

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []
        self.last_error = None
        self.last_degraded_reason = None
        try:
            sheet = build_contact_sheet(
                candidates,
                self.project_root,
                columns=self.columns,
                tile_size=self.tile_size,
            )
            self.last_contact_sheet_size = sheet.size
            raw = self.client.generate(
                sheet,
                self._instruction(query, candidates),
                max_new_tokens=self.max_new_tokens,
            )
            return self._parse_ranking(raw, candidates)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            message = f"视觉重排不可用：Listwise {self.last_error}"
            for item in candidates:
                item.rerank_score = 0.0
                item.mismatch = [*item.mismatch, message]
            return candidates
