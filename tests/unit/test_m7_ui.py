from __future__ import annotations

import json

from anima_search.app.ui import (
    APP_CSS,
    build_app,
    render_annotation_details,
    render_content_details,
    render_evidence_details,
    render_search_details,
    render_story_details,
    render_story_html,
)
from anima_search.m7.schemas import StoryGap, StorySection, VisualStory


def _story() -> VisualStory:
    return VisualStory(
        title="一天的旅程 <script>",
        ordering_reason="按时间与场景排序",
        sections=[
            StorySection(image_id="val-1", subtitle="清晨", text="从公园出发"),
            StorySection(image_id="val-2", subtitle="黄昏", text="抵达海边"),
            StorySection(image_id="val-3", subtitle="夜晚", text="回到城市"),
        ],
        gaps=[
            StoryGap(
                gap_id="gap-generated",
                after_image_id="val-1",
                before_image_id="val-2",
                reason="时间跨度较大",
                generation_prompt="日落过渡",
                status="generated",
            ),
            StoryGap(
                gap_id="gap-missing",
                after_image_id="val-2",
                before_image_id="val-3",
                reason="场景变化明显",
                generation_prompt="夜幕过渡",
            ),
            StoryGap(
                gap_id="gap-failed",
                after_image_id="val-3",
                before_image_id="val-1",
                reason="循环收束",
                generation_prompt="清晨过渡",
                status="failed",
                error="显存不足 <retry>",
            ),
        ],
    )


def test_story_html_distinguishes_real_generated_missing_and_failed_items():
    html = render_story_html(_story())

    assert "🖼 原始图片" in html
    assert "✨ AI 生成 · 已补全" in html
    assert 'class="story-badge missing"' in html
    assert "生成失败 · 可重试" in html
    assert "原始片段 <strong>3</strong>" in html
    assert "检测缺口 <strong>3</strong>" in html
    assert "AI 已补全 <strong>1</strong>" in html
    assert "待补全 <strong>1</strong>" in html
    assert "补全失败 <strong>1</strong>" in html


def test_story_html_escapes_model_and_error_text():
    html = render_story_html(_story())

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "显存不足 &lt;retry&gt;" in html


def test_m7_tab_contains_readable_intro_and_status_legend(tmp_path):
    class Service:
        config = {
            "project_root": str(tmp_path),
            "retrieval": {"rerank_default": False},
            "generation": {"seed": 42},
            "m7": {"fill_gaps_default": False},
            "models": {},
            "annotation": {},
        }
        annotations = {}

    app = build_app(Service())
    config = json.dumps(app.config, ensure_ascii=False, default=str)

    assert "m7-intro" in config
    assert "原始图片" in config
    assert "缺图占位" in config
    assert "AI 生成" in config
    assert all(
        class_name in APP_CSS
        for class_name in (
            ".m7-intro",
            ".story-summary",
            ".story-card-head",
            ".story-position",
            ".story-badge.missing",
        )
    )



def test_readable_cards_present_search_annotation_evidence_story_and_content():
    search_html = render_search_details(
        [
            {
                "image_id": "val-1",
                "fused_score": 0.75,
                "branch_scores": {"image": 0.9, "text": 0.7},
                "branch_ranks": {"image": 1, "text": 2},
                "active_branches": ["image", "text"],
                "matched_fields": ["场景", "主体"],
                "evidence": ["画面中出现城市街道"],
            }
        ]
    )
    annotation_html = render_annotation_details(
        {
            "summary": "雨夜街道",
            "scene": "城市",
            "objects": ["汽车"],
            "object_counts": {"汽车": 2},
            "actions": ["行驶"],
            "ocr_text": ["出口"],
        }
    )
    evidence_html = render_evidence_details(
        {
            "citations": ["val-1"],
            "confidence": 0.8,
            "refused": False,
            "evidence": [
                {
                    "image_id": "val-1",
                    "relevant": True,
                    "facts": ["可以看到两辆汽车"],
                    "uncertainty": ["无法确认具体城市"],
                }
            ],
        }
    )
    story_html = render_story_details(_story())
    content_html = render_content_details(
        {"title": "雨夜归途", "content": "灯光沿街道延伸", "hashtags": ["夜景"]}
    )

    assert "综合匹配 0.7500" in search_html
    assert "画面语义" in search_html
    assert "汽车 × 2" in annotation_html
    assert "置信度 80%" in evidence_html
    assert "无法确认具体城市" in evidence_html
    assert "排序方式" in story_html
    assert "雨夜归途" in content_html
    assert all(
        'class="detail-card"' in html
        for html in (search_html, annotation_html, evidence_html, story_html, content_html)
    )


def test_readable_cards_escape_untrusted_text():
    html = render_annotation_details(
        {"summary": "<script>alert(1)</script>", "scene": "城市", "objects": []}
    )

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
