from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anima_search.m6.contract import BranchName


class M6CandidateResult(BaseModel):
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
    rerank_rank: int = Field(ge=1, le=20)
    rerank_score: float | None
    mismatch: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_branch_metadata(self) -> "M6CandidateResult":
        if set(self.branch_scores) != set(self.branch_ranks):
            raise ValueError("branch keys must match between scores and ranks")
        if any(rank < 1 for rank in self.branch_ranks.values()):
            raise ValueError("branch ranks must be positive")
        return self


class M6QueryResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    schema_version: Literal["m6-rerank-v1.0"]
    source_schema_version: Literal["m5-to-m6-v1.0"]
    query_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    query: str = Field(min_length=1)
    category: Literal["simple", "compositional", "negative", "count", "ocr"]
    split: Literal["train", "val"]
    fusion_method: Literal["rrf", "weighted"]
    top_k: Literal[20]
    annotation_version: str = Field(min_length=1)
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rerank_method: Literal["pointwise", "listwise"]
    degraded: bool
    mismatch: list[str]
    candidates: list[M6CandidateResult] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def validate_output_invariants(self) -> "M6QueryResult":
        image_ids = [item.image_id for item in self.candidates]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("candidate image IDs must be unique within a query")

        rerank_ranks = [item.rerank_rank for item in self.candidates]
        if rerank_ranks != list(range(1, 21)):
            raise ValueError(
                "rerank ranks must match array order as the sequence 1..20"
            )

        source_ranks = [item.rank for item in self.candidates]
        if sorted(source_ranks) != list(range(1, 21)):
            raise ValueError("source ranks must be a permutation of 1..20")

        if self.degraded != bool(self.mismatch):
            raise ValueError(
                "degraded must be true exactly when query mismatch is non-empty"
            )

        query_messages = set(self.mismatch)
        hidden_messages = {
            message
            for candidate in self.candidates
            for message in candidate.mismatch
            if message not in query_messages
        }
        if hidden_messages:
            raise ValueError(
                "candidate mismatch messages must appear in query mismatch"
            )
        return self
