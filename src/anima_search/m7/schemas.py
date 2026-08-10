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


class VisualStory(BaseModel):
    title: str
    sections: list[StorySection]
    disclaimer: str = "叙事性表达基于图片可见内容，不代表真实地点、身份或事件经过。"
