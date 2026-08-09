from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    split: Literal["Train", "Val"]
    relative_path: str
    sha256: str
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    size_bytes: int
    valid: bool = True
    error: str | None = None
    duplicate_of: str | None = None


class ImageAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    split: Literal["Train", "Val"]
    relative_path: str
    sha256: str
    duplicate_of: str | None = None
    summary: str = Field(min_length=1)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene: str
    attributes: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    ocr_text: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=3)
    generation_prompt: str
    uncertainty: list[str] = Field(default_factory=list)
    model_version: str
    prompt_version: str
    elapsed_seconds: float | None = None
    prompt_sha256: str = ""
    model_digest: str = ""
    generation_parameters: dict[str, object] = Field(default_factory=dict)
    peak_vram_bytes: int | None = None


class SearchQuery(BaseModel):
    raw_text: str
    semantic_text: str = ""
    query_type: Literal["simple", "compositional", "negative", "count", "ocr"] = "simple"
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)
    required_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    ocr_terms: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    image_id: str
    relative_path: str
    fused_score: float
    rerank_score: float | None = None
    branch_scores: dict[str, float] = Field(default_factory=dict)
    branch_ranks: dict[str, int] = Field(default_factory=dict)
    matched_fields: list[str] = Field(default_factory=list)
    active_branches: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    mismatch: list[str] = Field(default_factory=list)
    source: Literal["real", "generated"] = "real"


class GenerationPrompts(BaseModel):
    positive_prompt: str
    negative_prompt: str
