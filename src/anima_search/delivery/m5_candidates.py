from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anima_search.schemas import SearchResult


SCHEMA_VERSION = "m5-to-m6-v1.0"
BranchName = Literal["image", "text", "bm25"]
QueryCategory = Literal["simple", "compositional", "negative", "count", "ocr"]


class M5Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rank: int = Field(ge=1, le=20)
    image_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    fused_score: float
    branch_scores: dict[BranchName, float] = Field(min_length=1)
    branch_ranks: dict[BranchName, int] = Field(min_length=1)
    matched_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_branch_contract(self) -> "M5Candidate":
        if set(self.branch_scores) != set(self.branch_ranks):
            raise ValueError("branch_scores and branch_ranks must use identical keys")
        if any(rank < 1 for rank in self.branch_ranks.values()):
            raise ValueError("branch ranks must be positive")
        if "\\" in self.relative_path:
            raise ValueError("relative_path must use POSIX separators")
        return self


class M5CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["m5-to-m6-v1.0"] = SCHEMA_VERSION
    query_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    query: str = Field(min_length=1)
    category: QueryCategory
    split: Literal["train", "val"]
    fusion_method: Literal["rrf", "weighted"]
    top_k: Literal[20] = 20
    annotation_version: str = Field(min_length=1)
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: list[M5Candidate] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def validate_candidate_contract(self) -> "M5CandidateBatch":
        if self.query != self.query.strip():
            raise ValueError("query must not have leading or trailing whitespace")
        ids = [candidate.image_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate image_id values must be unique per query")
        for expected_rank, candidate in enumerate(self.candidates, start=1):
            if candidate.rank != expected_rank:
                raise ValueError("candidate rank must match its array position")
        return self


def build_m5_candidate_batch(
    *,
    query_id: str,
    query: str,
    category: QueryCategory,
    split: Literal["train", "val"],
    fusion_method: Literal["rrf", "weighted"],
    annotation_version: str,
    index_manifest_sha256: str,
    config_sha256: str,
    results: list[SearchResult],
) -> M5CandidateBatch:
    if len(results) != 20:
        raise ValueError(f"M5 delivery requires exactly 20 candidates; received {len(results)}")
    candidates = [
        M5Candidate(
            rank=rank,
            image_id=result.image_id,
            relative_path=result.relative_path,
            fused_score=result.fused_score,
            branch_scores=result.branch_scores,
            branch_ranks=result.branch_ranks,
            matched_fields=result.matched_fields,
        )
        for rank, result in enumerate(results, start=1)
    ]
    return M5CandidateBatch(
        query_id=query_id,
        query=query.strip(),
        category=category,
        split=split,
        fusion_method=fusion_method,
        annotation_version=annotation_version,
        index_manifest_sha256=index_manifest_sha256,
        config_sha256=config_sha256,
        candidates=candidates,
    )
