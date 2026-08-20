from __future__ import annotations

from html import escape
from pathlib import Path

import gradio as gr

from anima_search.app.service import SearchService


APP_CSS = """
:root {
  --anima-ink: #172033;
  --anima-muted: #667085;
  --anima-indigo: #5b5bd6;
  --anima-violet: #8b5cf6;
  --anima-card: rgba(255, 255, 255, 0.88);
}
.gradio-container {
  background:
    radial-gradient(circle at 8% 0%, rgba(139, 92, 246, .12), transparent 30rem),
    radial-gradient(circle at 92% 8%, rgba(59, 130, 246, .11), transparent 28rem),
    #f7f8fc !important;
  color: var(--anima-ink);
  font-family: Inter, "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
}
.anima-shell { max-width: 1440px; margin: 0 auto; padding-bottom: 2rem; }
.anima-hero {
  position: relative;
  overflow: hidden;
  padding: 2.4rem 2.5rem;
  margin: .8rem 0 1.2rem;
  border-radius: 28px;
  color: white;
  background: linear-gradient(125deg, #25265e 0%, #5b5bd6 48%, #8b5cf6 100%);
  box-shadow: 0 24px 70px rgba(67, 56, 202, .22);
}
.anima-hero:after {
  content: "";
  position: absolute;
  width: 19rem;
  height: 19rem;
  right: -5rem;
  top: -9rem;
  border: 1px solid rgba(255,255,255,.35);
  border-radius: 50%;
  box-shadow: 0 0 0 42px rgba(255,255,255,.05), 0 0 0 84px rgba(255,255,255,.04);
}
.anima-kicker {
  font-size: .75rem;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
  opacity: .76;
}
.anima-hero h1 { margin: .35rem 0 .55rem; font-size: clamp(2rem, 5vw, 3.7rem); line-height: 1; }
.anima-hero p { max-width: 47rem; margin: 0; font-size: 1rem; opacity: .88; line-height: 1.7; }
.anima-pills { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.25rem; }
.anima-pill {
  padding: .42rem .72rem;
  border: 1px solid rgba(255,255,255,.24);
  border-radius: 999px;
  background: rgba(255,255,255,.11);
  font-size: .76rem;
  font-weight: 650;
}
.anima-panel {
  padding: .45rem !important;
  border: 1px solid rgba(91, 91, 214, .10) !important;
  border-radius: 22px !important;
  background: var(--anima-card) !important;
  box-shadow: 0 10px 35px rgba(34, 46, 80, .06) !important;
}
.anima-panel > .gr-box, .anima-panel > .block {
  border-color: transparent !important;
}
.m7-intro {
  display: flex;
  align-items: flex-start;
  gap: .9rem;
  margin: .15rem 0 1rem;
  padding: 1rem 1.15rem;
  border: 1px solid #e8e9fb;
  border-radius: 18px;
  background: linear-gradient(135deg, #fbfbff 0%, #f5f3ff 100%);
}
.m7-intro-icon {
  display: grid;
  flex: 0 0 2.4rem;
  width: 2.4rem;
  height: 2.4rem;
  place-items: center;
  border-radius: 13px;
  color: white;
  background: linear-gradient(145deg, #6366f1, #8b5cf6);
  box-shadow: 0 7px 16px rgba(99, 102, 241, .25);
  font-size: 1.15rem;
}
.m7-intro h3 { margin: 0 0 .28rem; color: var(--anima-ink); font-size: 1.08rem; }
.m7-intro p { margin: 0; color: #596275; font-size: .88rem; line-height: 1.65; }
.m7-legend { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .62rem; }
.m7-legend span {
  padding: .22rem .52rem;
  border-radius: 999px;
  color: #596275;
  background: rgba(255,255,255,.7);
  border: 1px solid #e4e5f5;
  font-size: .72rem;
  font-weight: 700;
}
.m7-legend .legend-ai { color: #7e22ce; background: #f3e8ff; border-color: #e9d5ff; }
.m7-legend .legend-gap { color: #6d28d9; background: #ede9fe; border-color: #ddd6fe; }
.story-summary {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin: .95rem 0 1rem;
}
.story-summary-item {
  display: inline-flex;
  align-items: center;
  gap: .35rem;
  padding: .34rem .62rem;
  border-radius: 10px;
  color: #596275;
  background: #f8f9fd;
  border: 1px solid #eaecf5;
  font-size: .76rem;
  font-weight: 650;
}
.story-summary-item strong { color: var(--anima-ink); font-size: .88rem; }
.anima-gallery {
  border: 0 !important;
  border-radius: 20px !important;
  overflow: hidden;
}
.anima-gallery img { transition: transform .25s ease, filter .25s ease; }
.anima-gallery img:hover { transform: scale(1.025); filter: saturate(1.08); }
.anima-section-note { color: var(--anima-muted); font-size: .9rem; line-height: 1.65; }
.story-empty {
  min-height: 25rem;
  display: grid;
  place-items: center;
  padding: 2rem;
  border: 1px dashed rgba(91, 91, 214, .3);
  border-radius: 22px;
  color: var(--anima-muted);
  background: linear-gradient(145deg, rgba(255,255,255,.85), rgba(245,243,255,.75));
  text-align: center;
}
.story-empty strong { display: block; color: var(--anima-ink); font-size: 1.15rem; margin-bottom: .35rem; }
.story-head { margin-bottom: 1.25rem; }
.story-head h2 { margin: 0 0 .35rem; font-size: 1.7rem; }
.story-head p { color: var(--anima-muted); margin: 0; line-height: 1.6; }
.story-timeline { position: relative; padding: .4rem .25rem .4rem 1.45rem; }
.story-timeline:before {
  content: "";
  position: absolute;
  left: .42rem;
  top: .8rem;
  bottom: .8rem;
  width: 2px;
  background: linear-gradient(#6366f1, #c4b5fd);
}
.story-node, .story-gap-card {
  position: relative;
  margin: 0 0 1rem;
  padding: 1rem 1.05rem;
  border-radius: 17px;
  background: white;
  border: 1px solid #eaecf5;
  box-shadow: 0 7px 20px rgba(30, 41, 59, .055);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.story-node:hover, .story-gap-card:hover {
  transform: translateY(-2px);
  border-color: #cfd2ff;
  box-shadow: 0 12px 26px rgba(30, 41, 59, .09);
}
.story-node:before, .story-gap-card:before {
  content: "";
  position: absolute;
  left: -1.36rem;
  top: 1.2rem;
  width: .66rem;
  height: .66rem;
  border-radius: 50%;
  background: #6366f1;
  box-shadow: 0 0 0 4px #eef2ff;
}
.story-gap-card {
  border: 1px dashed #c4b5fd;
  background: linear-gradient(145deg, #fafaff, #f5f3ff);
}
.story-gap-card:before { background: #a78bfa; }
.story-node h3, .story-gap-card h4 { margin: .35rem 0 .4rem; color: var(--anima-ink); }
.story-node p, .story-gap-card p { margin: 0; color: #475467; line-height: 1.65; }
.story-meta { color: #7c8294; font-size: .74rem; font-weight: 650; letter-spacing: .02em; }
.story-card-head { display: flex; align-items: center; justify-content: space-between; gap: .7rem; flex-wrap: wrap; }
.story-position {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.55rem;
  height: 1.55rem;
  margin-right: .42rem;
  border-radius: 50%;
  color: #4338ca;
  background: #eef2ff;
  font-size: .7rem;
  font-weight: 850;
}
.story-ref { color: #8a91a3; font-size: .73rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.story-badge {
  display: inline-flex;
  align-items: center;
  padding: .24rem .52rem;
  border-radius: 999px;
  font-size: .7rem;
  font-weight: 800;
  background: #eef2ff;
  color: #4f46e5;
}
.story-badge.ai { background: #f3e8ff; color: #7e22ce; }
.story-badge.failed { background: #fef2f2; color: #b42318; }
.story-badge.missing { background: #ede9fe; color: #6d28d9; }
.story-gap-card .story-meta { color: #7c3aed; }
.story-gap-card h4 { display: flex; align-items: center; gap: .4rem; }
.story-gap-card h4:before { content: "↔"; color: #8b5cf6; font-size: 1rem; }
.story-disclaimer {
  margin-top: .9rem;
  padding: .8rem 1rem;
  border-radius: 13px;
  background: #f8fafc;
  color: #667085;
  font-size: .78rem;
  line-height: 1.55;
}

.readable-empty {
  padding: 1.1rem 1.2rem;
  border: 1px dashed #cfd2e8;
  border-radius: 15px;
  color: var(--anima-muted);
  background: #fafbff;
  line-height: 1.65;
}
.readable-head { margin: .2rem 0 .9rem; color: #475467; line-height: 1.65; }
.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: .75rem;
}
.detail-card {
  padding: .9rem 1rem;
  border: 1px solid #e7e9f3;
  border-radius: 15px;
  background: #fff;
  box-shadow: 0 5px 16px rgba(30, 41, 59, .045);
}
.detail-card h4 { margin: 0 0 .55rem; color: var(--anima-ink); font-size: .92rem; }
.detail-card p { margin: .2rem 0; color: #596275; line-height: 1.6; }
.detail-card .detail-title { color: var(--anima-ink); font-size: 1rem; font-weight: 780; }
.detail-tags { display: flex; flex-wrap: wrap; gap: .38rem; }
.detail-tag {
  display: inline-flex;
  padding: .24rem .5rem;
  border-radius: 999px;
  color: #4f46e5;
  background: #eef2ff;
  font-size: .72rem;
  font-weight: 700;
}
.detail-tag.neutral { color: #475467; background: #f2f4f7; }
.detail-tag.success { color: #067647; background: #ecfdf3; }
.detail-tag.warning { color: #b54708; background: #fffaeb; }
.detail-list { margin: .25rem 0 0; padding-left: 1.15rem; color: #596275; line-height: 1.65; }
.detail-kv { display: grid; grid-template-columns: auto 1fr; gap: .25rem .65rem; font-size: .8rem; }
.detail-kv dt { color: #7c8294; font-weight: 700; }
.detail-kv dd { margin: 0; color: #344054; }
.detail-notice { margin-top: .65rem; padding: .65rem .75rem; border-radius: 11px; line-height: 1.55; }
.detail-notice.success { color: #067647; background: #ecfdf3; }
.detail-notice.warning { color: #b54708; background: #fffaeb; }
.detail-progress { height: .42rem; margin-top: .45rem; overflow: hidden; border-radius: 999px; background: #eaecf0; }
.detail-progress > span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #6366f1, #8b5cf6); }

.primary-action button { min-height: 46px; font-weight: 750 !important; }
footer { display: none !important; }
@media (max-width: 760px) {
  .anima-hero { padding: 1.7rem 1.35rem; border-radius: 22px; }
  .anima-hero h1 { font-size: 2.25rem; }
  .m7-intro { padding: .85rem .9rem; }
  .story-timeline { padding-left: 1.2rem; }
  .story-node:before, .story-gap-card:before { left: -1.12rem; }
}
"""


EMPTY_STORY_HTML = """
<div class="story-empty">
  <div><strong>你的视觉故事会出现在这里</strong>
  搜索并选择 3–8 张图片，系统将自动排序、识别叙事断点，并按需生成过渡图。</div>
</div>
"""


def render_story_html(story) -> str:
    """Render an auditable M7 timeline without changing the story payload."""
    gap_by_after: dict[str, list[object]] = {}
    for gap in story.gaps:
        gap_by_after.setdefault(gap.after_image_id, []).append(gap)
    generated_count = sum(gap.status == "generated" for gap in story.gaps)
    failed_count = sum(gap.status == "failed" for gap in story.gaps)
    pending_count = len(story.gaps) - generated_count - failed_count
    nodes = []
    for position, section in enumerate(story.sections, start=1):
        source_badge = (
            '<span class="story-badge ai" title="该画面由生成模型创建">✨ AI 生成</span>'
            if section.ai_generated
            else '<span class="story-badge" title="来自检索结果的真实来源图">🖼 原始图片</span>'
        )
        nodes.append(
            '<article class="story-node">'
            '<div class="story-card-head">'
            f'{source_badge}<span class="story-ref">{escape(section.image_id)}</span>'
            '</div>'
            f'<div class="story-meta"><span class="story-position">{position:02d}</span>'
            '叙事片段</div>'
            f'<h3>{escape(section.subtitle)}</h3>'
            f'<p>{escape(section.text)}</p></article>'
        )
        for gap in gap_by_after.get(section.image_id, []):
            if gap.status == "generated":
                badge = (
                    '<span class="story-badge ai" '
                    'title="该画面由生成模型创建">✨ AI 生成 · 已补全</span>'
                )
                detail = "过渡图已生成；下方影像序列会继续保留醒目的 AI 标识。"
            elif gap.status == "failed":
                badge = '<span class="story-badge failed">生成失败 · 可重试</span>'
                detail = escape(gap.error or "请稍后重试或检查生成服务。")
            else:
                badge = '<span class="story-badge missing">缺图占位 · 待补全</span>'
                detail = "勾选“自动检测并补全缺图”后可使用智能补图。"
            nodes.append(
                '<article class="story-gap-card">'
                '<div class="story-card-head">'
                f'{badge}<span class="story-ref">{escape(gap.gap_id)}</span>'
                '</div>'
                '<div class="story-meta">叙事连续性检测</div>'
                '<h4>叙事过渡画面</h4>'
                f'<p>{escape(gap.reason)}<br>{detail}</p></article>'
            )
    summary = (
        '<div class="story-summary">'
        f'<span class="story-summary-item">原始片段 <strong>{len(story.sections)}</strong></span>'
        f'<span class="story-summary-item">检测缺口 <strong>{len(story.gaps)}</strong></span>'
        f'<span class="story-summary-item">AI 已补全 <strong>{generated_count}</strong></span>'
        f'<span class="story-summary-item">待补全 <strong>{pending_count}</strong></span>'
    )
    if failed_count:
        summary += (
            f'<span class="story-summary-item">补全失败 <strong>{failed_count}</strong></span>'
        )
    summary += '</div>'
    return (
        '<section class="story-head">'
        f'<h2>{escape(story.title)}</h2>'
        f'<p>{escape(story.ordering_reason)}</p></section>'
        f'{summary}'
        f'<div class="story-timeline">{"".join(nodes)}</div>'
        f'<div class="story-disclaimer">{escape(story.disclaimer)}</div>'
    )



def _payload(value) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _text_items(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _tag_html(values, *, empty: str = "暂无", kind: str = "") -> str:
    items = _text_items(values)
    css_class = f"detail-tag {kind}".strip()
    if not items:
        return f'<span class="detail-tag neutral">{escape(empty)}</span>'
    return "".join(
        f'<span class="{css_class}">{escape(item)}</span>' for item in items
    )


def _list_html(values, *, empty: str = "暂无") -> str:
    items = _text_items(values)
    if not items:
        return f'<p>{escape(empty)}</p>'
    return '<ul class="detail-list">' + "".join(
        f'<li>{escape(item)}</li>' for item in items
    ) + '</ul>'


def _empty_detail(title: str, message: str) -> str:
    return (
        '<div class="readable-empty">'
        f'<strong>{escape(title)}</strong><br>{escape(message)}</div>'
    )


def _format_score(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render_search_details(results) -> str:
    if not results:
        return _empty_detail("暂无匹配依据", "完成一次搜索后，这里会解释每张图片为什么被找到。")
    branch_labels = {"image": "画面语义", "text": "描述语义", "bm25": "关键词"}
    cards = []
    for position, result in enumerate(results, start=1):
        item = _payload(result)
        branch_scores = item.get("branch_scores") or {}
        branch_ranks = item.get("branch_ranks") or {}
        active = _text_items(item.get("active_branches")) or list(branch_scores)
        branch_rows = []
        for branch in active:
            score = _format_score(branch_scores.get(branch))
            rank = branch_ranks.get(branch)
            rank_text = f"第 {rank} 名" if rank is not None else "未进入该路结果"
            branch_rows.append(
                f'<dt>{escape(branch_labels.get(branch, branch))}</dt>'
                f'<dd>相似度 {score} · {escape(rank_text)}</dd>'
            )
        rerank = item.get("rerank_score")
        rerank_text = (
            f'<span class="detail-tag success">智能排序 {_format_score(rerank, 1)}</span>'
            if rerank is not None else ""
        )
        mismatch = _text_items(item.get("mismatch"))
        warning = (
            '<div class="detail-notice warning"><strong>需要注意：</strong>'
            + "；".join(escape(value) for value in mismatch)
            + '</div>'
            if mismatch else ""
        )
        cards.append(
            '<article class="detail-card">'
            f'<div class="detail-title">#{position} · {escape(str(item.get("image_id", "未知图片")))}</div>'
            '<div class="detail-tags" style="margin:.45rem 0;">'
            f'<span class="detail-tag">综合匹配 {_format_score(item.get("fused_score"))}</span>{rerank_text}'
            '</div>'
            f'<dl class="detail-kv">{"".join(branch_rows)}</dl>'
            '<h4 style="margin-top:.75rem;">匹配线索</h4>'
            f'{_tag_html(item.get("matched_fields"), empty="综合语义匹配", kind="success")}'
            f'{_list_html(item.get("evidence"), empty="系统通过整体画面语义找到此图片。")}'
            f'{warning}</article>'
        )
    return (
        f'<p class="readable-head">共展示 {len(cards)} 个结果。综合匹配分数仅用于排序，数值越高表示与当前查询越接近。</p>'
        f'<div class="detail-grid">{"".join(cards)}</div>'
    )


def render_annotation_details(annotation) -> str:
    item = _payload(annotation)
    if not item:
        return _empty_detail("尚未选择图片", "请先在“探索相册”中单击一张结果图片。")
    if item.get("status"):
        return _empty_detail("图片信息不可用", str(item["status"]))
    object_counts = item.get("object_counts") or {}
    objects = _text_items(item.get("objects"))
    counted_objects = [
        f"{name} × {object_counts[name]}" if name in object_counts else name for name in objects
    ]
    for name, count in object_counts.items():
        if name not in objects:
            counted_objects.append(f"{name} × {count}")
    groups = [
        ("主要物体", counted_objects, "未识别到明确物体"),
        ("动作", item.get("actions"), "未识别到明确动作"),
        ("画面属性", item.get("attributes"), "暂无补充属性"),
        ("空间关系", item.get("spatial_relations"), "暂无明确空间关系"),
        ("风格", item.get("style"), "暂无明确风格"),
        ("氛围", item.get("mood"), "暂无明确氛围"),
        ("主要颜色", item.get("colors"), "暂无主要颜色"),
        ("画面文字", item.get("ocr_text"), "未检测到清晰文字"),
    ]
    cards = [
        '<article class="detail-card">'
        '<h4>图片概览</h4>'
        f'<p class="detail-title">{escape(str(item.get("summary") or "暂无图片描述"))}</p>'
        '<div class="detail-tags" style="margin-top:.6rem;">'
        f'{_tag_html(item.get("scene"), empty="场景未知", kind="success")}</div></article>'
    ]
    cards.extend(
        '<article class="detail-card">'
        f'<h4>{escape(title)}</h4><div class="detail-tags">'
        f'{_tag_html(values, empty=empty)}</div></article>'
        for title, values, empty in groups
    )
    uncertainty = _text_items(item.get("uncertainty"))
    notice = (
        '<div class="detail-notice warning"><strong>不确定信息：</strong>'
        + "；".join(escape(value) for value in uncertainty)
        + '</div>' if uncertainty else
        '<div class="detail-notice success">当前理解结果没有额外的不确定性提示。</div>'
    )
    return f'<div class="detail-grid">{"".join(cards)}</div>{notice}'


def render_evidence_details(answer_result) -> str:
    item = _payload(answer_result)
    if not item:
        return _empty_detail("暂无引用依据", "提出关于所选图片的问题后，这里会显示逐图依据。")
    refused = bool(item.get("refused"))
    confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
    state_class = "warning" if refused else "success"
    state_text = "证据不足，系统已避免推测" if refused else "回答已通过所选图片证据约束"
    cards = []
    for evidence in item.get("evidence") or []:
        detail = _payload(evidence)
        relevant = bool(detail.get("relevant", True))
        cards.append(
            '<article class="detail-card">'
            f'<div class="detail-title">{escape(str(detail.get("image_id", "未知图片")))}</div>'
            '<div class="detail-tags" style="margin:.45rem 0;">'
            f'<span class="detail-tag {"success" if relevant else "neutral"}">'
            f'{"提供了有效依据" if relevant else "与问题关系较弱"}</span></div>'
            '<h4>可直接确认</h4>'
            f'{_list_html(detail.get("facts"), empty="没有提取到可直接确认的事实")}'
            '<h4 style="margin-top:.65rem;">仍需谨慎</h4>'
            f'{_list_html(detail.get("uncertainty"), empty="没有额外的不确定性提示")}'
            '</article>'
        )
    citation_tags = _tag_html(item.get("citations"), empty="无引用", kind="success")
    return (
        f'<div class="detail-notice {state_class}"><strong>{state_text}</strong><br>'
        f'置信度 {confidence:.0%}<div class="detail-progress"><span style="width:{confidence * 100:.1f}%"></span></div></div>'
        f'<h4 style="margin:.85rem 0 .45rem;">使用的图片引用</h4><div class="detail-tags">{citation_tags}</div>'
        f'<div class="detail-grid" style="margin-top:.8rem;">{"".join(cards)}</div>'
    )


def render_story_details(story) -> str:
    item = _payload(story)
    if not item:
        return _empty_detail("暂无故事详情", "完成视觉故事编排后，这里会说明排序与缺图处理。")
    ordered_ids = item.get("ordered_image_ids") or [
        _payload(section).get("image_id") for section in item.get("sections") or []
    ]
    gap_cards = []
    status_labels = {"generated": "AI 已补全", "failed": "补全失败", "missing": "等待补全"}
    status_classes = {"generated": "success", "failed": "warning", "missing": "neutral"}
    for gap in item.get("gaps") or []:
        detail = _payload(gap)
        status = str(detail.get("status") or "missing")
        error = detail.get("error")
        error_html = (
            f'<div class="detail-notice warning">{escape(str(error))}</div>' if error else ""
        )
        gap_cards.append(
            '<article class="detail-card">'
            f'<div class="detail-title">{escape(str(detail.get("after_image_id", "?")))} → '
            f'{escape(str(detail.get("before_image_id", "?")))}</div>'
            '<div class="detail-tags" style="margin:.45rem 0;">'
            f'<span class="detail-tag {status_classes.get(status, "neutral")}">'
            f'{escape(status_labels.get(status, status))}</span></div>'
            f'<p>{escape(str(detail.get("reason") or "检测到叙事过渡"))}</p>{error_html}</article>'
        )
    gap_html = (
        f'<div class="detail-grid">{"".join(gap_cards)}</div>' if gap_cards else
        '<div class="detail-notice success">当前图片序列衔接自然，没有检测到需要补全的过渡。</div>'
    )
    return (
        '<article class="detail-card">'
        '<h4>排序方式</h4>'
        f'<p>{escape(str(item.get("ordering_reason") or "按照所选图片顺序编排"))}</p>'
        '<h4 style="margin-top:.7rem;">图片顺序</h4>'
        f'<div class="detail-tags">{_tag_html(ordered_ids, empty="暂无图片")}</div>'
        '</article>'
        f'<h4 style="margin:.9rem 0 .5rem;">过渡画面处理</h4>{gap_html}'
        f'<div class="detail-notice warning">{escape(str(item.get("disclaimer") or "故事内容仅基于图片可见信息。"))}</div>'
    )


def render_content_details(content) -> str:
    item = _payload(content)
    if not item:
        return _empty_detail("暂无创作内容", "选择图片并生成文案后，结果会以易读卡片显示。")
    if item.get("error"):
        return _empty_detail("暂时无法生成", str(item["error"]))
    labels = {
        "title": "标题", "content": "正文", "text": "正文", "copy": "文案",
        "story": "故事", "hashtags": "推荐标签", "tags": "推荐标签",
    }
    cards = []
    for key, value in item.items():
        title = labels.get(str(key), str(key).replace("_", " ").title())
        if isinstance(value, dict):
            body = '<dl class="detail-kv">' + "".join(
                f'<dt>{escape(str(name))}</dt><dd>{escape(str(detail))}</dd>'
                for name, detail in value.items()
            ) + '</dl>'
        elif isinstance(value, (list, tuple, set)):
            body = f'<div class="detail-tags">{_tag_html(value)}</div>'
        else:
            body = f'<p>{escape(str(value))}</p>'
        cards.append(f'<article class="detail-card"><h4>{escape(title)}</h4>{body}</article>')
    return f'<div class="detail-grid">{"".join(cards)}</div>'


def build_app(service: SearchService) -> gr.Blocks:
    root = Path(service.config["project_root"]).resolve()
    generation_seed = int(service.config.get("generation", {}).get("seed", 42))
    fill_gaps_default = bool(
        service.config.get("m7", {}).get("fill_gaps_default", False)
    )

    def result_path(relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else (root / path).resolve()

    def run_search(query: str, rerank: bool):
        results = service.search(query, rerank)
        gallery = []
        for item in results:
            caption = f"{item.image_id} · 融合 {item.fused_score:.4f}"
            if item.rerank_score is not None:
                caption += f" · VLM {item.rerank_score:.0f}"
            gallery.append((str(result_path(item.relative_path)), caption))
        image_ids = [item.image_id for item in results]
        selected = image_ids[:min(3, len(image_ids))]
        return (
            gallery,
            render_search_details(results),
            results,
            gr.CheckboxGroup(choices=image_ids, value=selected),
        )

    def select_result(results, event: gr.SelectData):
        if not results or not isinstance(event.index, int) or event.index >= len(results):
            return "", None, {}
        item = results[event.index]
        annotation = service.annotations.get(item.image_id)
        annotation_payload = (
            annotation
            if annotation is not None
            else {"status": "当前图片没有可展示的理解信息"}
        )
        return (
            item.image_id,
            str(result_path(item.relative_path)),
            render_annotation_details(annotation_payload),
        )

    def answer(image_id: str, question: str):
        return service.answer_about_image(image_id, question) if image_id else "请先选择图片。"

    def write(image_id: str, content_type: str, tone: str):
        if not image_id:
            return _empty_detail("请先选择图片", "返回“探索相册”并单击一张结果图片。")
        try:
            return render_content_details(service.write_content(image_id, content_type, tone))
        except Exception as exc:
            return _empty_detail("文案生成失败", f"{type(exc).__name__}: {exc}")

    def generate(query: str, image_id: str, seed: int):
        return str(service.generate_image(query, image_id or None, int(seed)))

    def answer_with_evidence(results, selected_ids, question: str):
        if not results:
            return "请先搜索图片。", _empty_detail("暂无引用依据", "请先搜索并选择图片。")
        try:
            answer_result = service.answer_with_evidence(
                question,
                results,
                list(selected_ids or []),
                top_k=3,
            )
        except Exception as exc:
            return (
                f"回答失败：{type(exc).__name__}: {exc}",
                _empty_detail("回答失败", str(exc)),
            )
        citation_text = "、".join(
            f"[img_{image_id}]" for image_id in answer_result.citations
        ) or "无"
        markdown = (
            f"{answer_result.answer}\n\n"
            f"**引用：** {citation_text}　"
            f"**置信度：** {answer_result.confidence:.2f}　"
            f"**拒答：** {'是' if answer_result.refused else '否'}"
        )
        return markdown, render_evidence_details(answer_result)

    def create_story(
        results,
        selected_ids,
        theme: str,
        tone: str,
        fill_gaps: bool,
        seed: int,
    ):
        if not results:
            return (
                EMPTY_STORY_HTML,
                [],
                _empty_detail("暂无故事详情", "请先搜索并选择图片。"),
            )
        try:
            story = service.create_visual_story(
                results,
                list(selected_ids or []),
                theme=theme,
                tone=tone,
                fill_gaps=bool(fill_gaps),
                seed=int(seed),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return (
                f'<div class="story-empty"><div><strong>故事生成失败</strong>'
                f'{escape(error)}</div></div>',
                [],
                _empty_detail("故事生成失败", str(exc)),
            )

        by_id = {item.image_id: item for item in results}
        gap_by_after: dict[str, list[object]] = {}
        for gap in story.gaps:
            gap_by_after.setdefault(gap.after_image_id, []).append(gap)
        gallery = []
        for section in story.sections:
            item = by_id.get(section.image_id)
            if item is not None:
                gallery.append(
                    (
                        str(result_path(item.relative_path)),
                        f"🖼️ 原图 · {section.image_id} · {section.subtitle}",
                    )
                )
            for gap in gap_by_after.get(section.image_id, []):
                if gap.status == "generated" and gap.relative_path:
                    generated_path = result_path(gap.relative_path)
                    if generated_path.is_file():
                        gallery.append(
                            (
                                str(generated_path),
                                f"✨ AI 生成 · {gap.gap_id} · {gap.reason}",
                            )
                        )
        return render_story_html(story), gallery, render_story_details(story)

    with gr.Blocks(
        title="忆见 AskAlbum · 智能相册",
        fill_width=True,
    ) as app:
        selected_id = gr.State("")
        search_results = gr.State([])
        with gr.Column(elem_classes=["anima-shell"]):
            gr.HTML(
                """
                <header class="anima-hero">
                  <div class="anima-kicker">AI-Powered Photo Discovery</div>
                  <h1>忆见 <span style="font-weight: 500; opacity: .82;">AskAlbum</span></h1>
                  <p>让自然语言成为相册的入口：检索、理解、证据问答，
                  再把零散影像自动编排为可追溯的视觉故事。</p>
                  <div class="anima-pills">
                    <span class="anima-pill">智能检索</span>
                    <span class="anima-pill">精准排序</span>
                    <span class="anima-pill">图片问答</span>
                    <span class="anima-pill">视觉故事</span>
                  </div>
                </header>
                """
            )
            with gr.Tabs():
                with gr.Tab("探索相册"):
                    with gr.Column(elem_classes=["anima-panel"]):
                        with gr.Row():
                            query = gr.Textbox(
                                label="用自然语言寻找图片",
                                placeholder="例如：不要人物，寻找冷色调的雨夜城市",
                                scale=7,
                            )
                            use_rerank = gr.Checkbox(
                                value=bool(
                                    service.config["retrieval"].get(
                                        "rerank_default", False
                                    )
                                ),
                                label="启用智能排序",
                                scale=2,
                            )
                            search_button = gr.Button(
                                "开始探索",
                                variant="primary",
                                scale=1,
                                elem_classes=["primary-action"],
                            )
                    gallery = gr.Gallery(
                        label="检索结果",
                        columns=4,
                        rows=2,
                        height=570,
                        object_fit="cover",
                        elem_classes=["anima-gallery"],
                    )
                    with gr.Accordion("查看匹配依据", open=False):
                        result_details = gr.HTML(
                            value=_empty_detail(
                                "暂无匹配依据",
                                "完成一次搜索后，这里会解释每张图片为什么被找到。",
                            )
                        )

                with gr.Tab("图片详情"):
                    with gr.Row():
                        selected_image = gr.Image(
                            label="当前图片",
                            height=500,
                            elem_classes=["anima-panel"],
                        )
                        with gr.Column(elem_classes=["anima-panel"]):
                            gr.Markdown("### 图片理解信息")
                            annotation_details = gr.HTML(
                                value=_empty_detail(
                                    "尚未选择图片",
                                    "请先在“探索相册”中单击一张结果图片。",
                                )
                            )
                            question = gr.Textbox(
                                label="关于这张图片的问题",
                                placeholder="例如：画面中的主要物体是什么？",
                            )
                            answer_button = gr.Button("基于图片回答", variant="primary")
                            answer_text = gr.Textbox(label="回答", lines=4)
                            answer_button.click(
                                answer,
                                [selected_id, question],
                                answer_text,
                            )

                with gr.Tab("视觉故事"):
                    gr.HTML(
                        """
                        <section class="m7-intro">
                          <div class="m7-intro-icon">✦</div>
                          <div>
                            <h3>把检索结果编排成可追溯的视觉故事</h3>
                            <p>系统先根据时间与场景自动排序，再检查相邻画面的叙事连续性。
                            原图与 AI 生成的过渡图始终使用不同标识，来源清晰可见。</p>
                            <div class="m7-legend" aria-label="故事状态图例">
                              <span>🖼 原始图片</span>
                              <span class="legend-gap">缺图占位</span>
                              <span class="legend-ai">✨ AI 生成</span>
                            </div>
                          </div>
                        </section>
                        """
                    )
                    gr.Markdown(
                        "从当前结果选择图片：证据问答支持 1–3 张；视觉故事支持 "
                        "3–8 张，并按时间与场景自动排序。",
                        elem_classes=["anima-section-note"],
                    )
                    candidate_ids = gr.CheckboxGroup(
                        choices=[],
                        label="选择图片",
                        info="每次搜索后自动选择前三张，也可以手动调整。",
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=4, elem_classes=["anima-panel"]):
                            gr.Markdown("### 图片问答")
                            grounded_question = gr.Textbox(
                                label="关于所选图片的问题",
                                placeholder="例如：这些图片中哪些出现了车辆？",
                                lines=2,
                            )
                            grounded_button = gr.Button(
                                "生成带引用回答",
                                variant="secondary",
                            )
                            grounded_markdown = gr.Markdown()
                            with gr.Accordion("查看引用依据", open=False):
                                grounded_details = gr.HTML(
                                    value=_empty_detail(
                                        "暂无引用依据",
                                        "提出关于所选图片的问题后，这里会显示逐图依据。",
                                    )
                                )
                            grounded_button.click(
                                answer_with_evidence,
                                [search_results, candidate_ids, grounded_question],
                                [grounded_markdown, grounded_details],
                            )
                            gr.Markdown("### 故事设置")
                            story_theme = gr.Textbox(
                                value="图文游记",
                                label="故事主题",
                            )
                            story_tone = gr.Dropdown(
                                ["自然", "治愈", "纪实", "幽默"],
                                value="自然",
                                label="叙事语气",
                            )
                            fill_story_gaps = gr.Checkbox(
                                value=fill_gaps_default,
                                label="自动检测并补全缺图",
                                info="启用后会使用智能生成能力补充过渡画面。",
                            )
                            story_seed = gr.Number(
                                value=generation_seed,
                                precision=0,
                                label="创作编号（相同编号可复现）",
                            )
                            story_button = gr.Button(
                                "自动编排视觉故事",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )
                        with gr.Column(scale=7, elem_classes=["anima-panel"]):
                            story_timeline = gr.HTML(value=EMPTY_STORY_HTML)
                            story_gallery = gr.Gallery(
                                label="故事影像序列",
                                columns=3,
                                height=360,
                                object_fit="cover",
                                elem_classes=["anima-gallery"],
                            )
                            with gr.Accordion("查看故事详情", open=False):
                                story_details = gr.HTML(
                                    value=_empty_detail(
                                        "暂无故事详情",
                                        "完成视觉故事编排后，这里会说明排序与缺图处理。",
                                    )
                                )
                    story_button.click(
                        create_story,
                        [
                            search_results,
                            candidate_ids,
                            story_theme,
                            story_tone,
                            fill_story_gaps,
                            story_seed,
                        ],
                        [story_timeline, story_gallery, story_details],
                    )

                with gr.Tab("创作工作台"):
                    with gr.Row():
                        with gr.Column(elem_classes=["anima-panel"]):
                            content_type = gr.Dropdown(
                                [("标题", "title"), ("分享文案", "moments"), ("短故事", "story")],
                                value="moments",
                                label="内容类型",
                            )
                            tone = gr.Dropdown(
                                [("正式", "formal"), ("幽默", "humorous"), ("治愈", "healing")],
                                value="healing",
                                label="语气",
                            )
                            content_button = gr.Button("生成文案", variant="secondary")
                            content_details = gr.HTML(
                                value=_empty_detail(
                                    "暂无创作内容",
                                    "选择图片并生成文案后，结果会以易读卡片显示。",
                                )
                            )
                            content_button.click(
                                write,
                                [selected_id, content_type, tone],
                                content_details,
                            )
                        with gr.Column(elem_classes=["anima-panel"]):
                            generation_query = gr.Textbox(
                                label="补图需求",
                                placeholder="例如：保留雨夜街道，增加电影感霓虹灯",
                                lines=2,
                            )
                            seed = gr.Number(
                                value=generation_seed,
                                precision=0,
                                label="创作编号（相同编号可复现）",
                            )
                            generation_button = gr.Button(
                                "生成补充图片",
                                variant="primary",
                            )
                            generated_image = gr.Image(
                                label="✨ AI 生成图片",
                                height=420,
                            )
                            generation_button.click(
                                generate,
                                [generation_query, selected_id, seed],
                                generated_image,
                            )

            search_button.click(
                run_search,
                [query, use_rerank],
                [gallery, result_details, search_results, candidate_ids],
            )
            gallery.select(
                select_result,
                [search_results],
                [selected_id, selected_image, annotation_details],
            )
    return app
