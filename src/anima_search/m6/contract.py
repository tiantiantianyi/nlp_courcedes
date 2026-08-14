from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anima_search.schemas import SearchResult


BranchName = Literal["image", "text", "bm25"]


class M5Candidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    rank: int = Field(ge=1, le=20)
    image_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    fused_score: float
    branch_scores: dict[BranchName, float] = Field(min_length=1)
    branch_ranks: dict[BranchName, int] = Field(min_length=1)
    matched_fields: list[str]

    @model_validator(mode="after")
    def validate_branch_keys(self) -> "M5Candidate":
        if set(self.branch_scores) != set(self.branch_ranks):
            raise ValueError("branch keys must match between scores and ranks")
        if any(rank < 1 for rank in self.branch_ranks.values()):
            raise ValueError("branch ranks must be positive")
        return self

    def to_search_result(self) -> SearchResult:
        return SearchResult(
            image_id=self.image_id,
            relative_path=self.relative_path,
            fused_score=self.fused_score,
            branch_scores=dict(self.branch_scores),
            branch_ranks=dict(self.branch_ranks),
            matched_fields=list(self.matched_fields),
            active_branches=list(self.branch_scores),
        )


class M5QueryBatch(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    schema_version: Literal["m5-to-m6-v1.0"]
    query_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    query: str = Field(min_length=1)
    category: Literal["simple", "compositional", "negative", "count", "ocr"]
    split: Literal["train", "val"]
    fusion_method: Literal["rrf", "weighted"]
    top_k: Literal[20]
    annotation_version: str = Field(min_length=1)
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: list[M5Candidate] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def validate_candidate_identity(self) -> "M5QueryBatch":
        if [item.rank for item in self.candidates] != list(range(1, 21)):
            raise ValueError("candidate ranks must be the ordered sequence 1..20")
        image_ids = [item.image_id for item in self.candidates]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("candidate image IDs must be unique within a query")
        return self

    def to_search_results(self) -> list[SearchResult]:
        return [item.to_search_result() for item in self.candidates]
