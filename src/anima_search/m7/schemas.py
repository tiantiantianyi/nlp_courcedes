from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IntentRoute(BaseModel):
    intent: Literal["search", "qa", "story", "generate"]
    reason: str = ""


class ImageEvidence(BaseModel):
    image_id: str
    relevant: bool = True
    facts: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    used_annotation: bool = False


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    refused: bool = False
    evidence: list[ImageEvidence] = Field(default_factory=list)


class StorySection(BaseModel):
    image_id: str
    subtitle: str
    text: str
    source: Literal["real", "generated"] = "real"
    ai_generated: bool = False


class StoryGap(BaseModel):
    gap_id: str
    after_image_id: str
    before_image_id: str
    reason: str
    generation_prompt: str
    status: Literal["missing", "generated", "failed"] = "missing"
    source: Literal["generated"] = "generated"
    ai_generated: bool = True
    generated_image_id: str | None = None
    relative_path: str | None = None
    error: str | None = None


class VisualStory(BaseModel):
    title: str
    sections: list[StorySection]
    ordered_image_ids: list[str] = Field(default_factory=list)
    ordering_reason: str = ""
    gaps: list[StoryGap] = Field(default_factory=list)
    disclaimer: str = "叙事性表达基于图片可见内容，不代表真实地点、身份或事件经过。"
