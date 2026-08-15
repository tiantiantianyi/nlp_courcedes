from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class M6CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rank: int = Field(ge=1, le=20)
    image_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    fused_score: float
    branch_scores: dict[str, float]
    branch_ranks: dict[str, int]
    matched_fields: list[str]
    rerank_rank: int = Field(ge=1, le=20)
    rerank_score: float | None
    mismatch: list[str] = Field(default_factory=list)


class M6QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["m6-rerank-v1.0"]
    source_schema_version: Literal["m5-to-m6-v1.0"]
    query_id: str
    query: str
    category: Literal["simple", "compositional", "negative", "count", "ocr"]
    split: Literal["train", "val"]
    fusion_method: Literal["rrf", "weighted"]
    top_k: Literal[20]
    annotation_version: str
    index_manifest_sha256: str
    config_sha256: str
    rerank_method: Literal["pointwise", "listwise"]
    degraded: bool
    mismatch: list[str]
    candidates: list[M6CandidateResult] = Field(min_length=20, max_length=20)
