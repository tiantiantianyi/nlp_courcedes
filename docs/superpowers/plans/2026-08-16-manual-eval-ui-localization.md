# Manual Evaluation UI Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make manual relevance annotation safer by displaying Chinese query categories, the read-only source image ID, and Zhang Tianyi as the default annotator.

**Architecture:** Keep all persisted category values and the manual-evaluation JSONL/CSV schema unchanged. Add a small task-to-form adapter used by the Gradio load callback, while localized `(label, value)` dropdown choices affect presentation only.

**Tech Stack:** Python 3.11, Gradio 6, pytest, Conda `vlm-course`.

## Global Constraints

- Chinese labels are display-only; persisted values remain `simple`, `compositional`, `negative`, `count`, and `ocr`.
- `source_image_id` is read-only and comes from the task JSONL; never infer or fabricate it.
- Blank tasks display `张添翼`; a saved non-empty annotator is preserved.
- Blank judgments default to `<source_image_id>:2`; existing judgments are preserved.
- Do not auto-review, auto-save, generate query text, or score other candidate images.
- Do not modify the manual-evaluation JSONL/CSV schema or M3–M7 pipeline behavior.
- Use `conda run -n vlm-course` for Python commands.

---

### Task 1: Localize and clarify the Gradio annotation form

**Files:**
- Modify: `scripts/launch_eval_annotator.py`
- Modify: `tests/unit/test_launch_eval_annotator.py`

**Interfaces:**
- Consumes: task rows containing `query_id`, `source_image_id`, `source_relative_path`, `text`, `category`, `annotator`, `note`, and `reviewed`.
- Produces: `_task_form_values(task: dict[str, object], rows: list[dict[str, object]], project_root: Path) -> tuple[object, ...]` and Gradio dropdown values that remain compatible with `save_review`.

- [ ] **Step 1: Add failing adapter and UI-configuration tests**

Add literal expectations for the user-visible behavior:

```python
def test_task_form_values_exposes_source_id_and_defaults_annotator(tmp_path: Path) -> None:
    task = {
        "query_id": "q001",
        "source_image_id": "val-2007",
        "source_relative_path": "../Val/2007.jpg",
        "text": "",
        "category": "",
        "annotator": "",
        "note": "",
        "reviewed": False,
    }
    values = launch_eval_annotator._task_form_values(task, [], tmp_path)
    assert values[2] == "val-2007"
    assert values[5] == "张添翼"
    assert values[7] == "val-2007:2"


def test_task_form_values_preserves_saved_annotator(tmp_path: Path) -> None:
    task = {
        "query_id": "q001",
        "source_image_id": "val-2007",
        "source_relative_path": "../Val/2007.jpg",
        "text": "猫",
        "category": "simple",
        "annotator": "已有标注者",
        "note": "",
        "reviewed": True,
    }
    values = launch_eval_annotator._task_form_values(task, [], tmp_path)
    assert values[5] == "已有标注者"


def test_category_choices_have_chinese_labels_and_stable_values() -> None:
    assert launch_eval_annotator.CATEGORY_CHOICES == [
        ("简单查询", "simple"),
        ("组合查询", "compositional"),
        ("否定查询", "negative"),
        ("数量查询", "count"),
        ("文字识别查询", "ocr"),
    ]


def test_task_form_values_preserves_existing_judgments(tmp_path: Path) -> None:
    task = {
        "query_id": "q001",
        "source_image_id": "val-2007",
        "source_relative_path": "../Val/2007.jpg",
        "text": "猫",
        "category": "simple",
        "annotator": "张添翼",
        "note": "",
        "reviewed": True,
    }
    rows = [{
        "query_id": "q001",
        "image_id": "val-2010",
        "relevance": "1",
    }]
    values = launch_eval_annotator._task_form_values(task, rows, tmp_path)
    assert values[7] == "val-2010:1"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q \
  tests/unit/test_launch_eval_annotator.py
```

Expected: collection succeeds, then tests fail because `_task_form_values` and `CATEGORY_CHOICES` do not exist.

- [ ] **Step 3: Implement the minimal form adapter and localized components**

In `scripts/launch_eval_annotator.py`, define:

```python
CATEGORY_CHOICES = [
    ("简单查询", "simple"),
    ("组合查询", "compositional"),
    ("否定查询", "negative"),
    ("数量查询", "count"),
    ("文字识别查询", "ocr"),
]
DEFAULT_ANNOTATOR = "张添翼"


def _task_form_values(
    task: dict[str, object],
    rows: list[dict[str, object]],
    project_root: Path,
) -> tuple[object, ...]:
    image_path = project_root / str(task["source_relative_path"])
    query_id = str(task["query_id"])
    judgments = format_judgments(rows, query_id) or f"{task['source_image_id']}:2"
    return (
        str(image_path),
        query_id,
        str(task["source_image_id"]),
        str(task.get("text", "")),
        str(task.get("category", "")) or None,
        str(task.get("annotator", "")).strip() or DEFAULT_ANNOTATOR,
        str(task.get("note", "")),
        judgments,
        bool(task.get("reviewed", False)),
    )
```

Make `load_position` reuse this adapter while preserving its position, progress, and status outputs. Add `source_image_id = gr.Textbox(label="来源图 ID", interactive=False)` immediately after `query_id`, add it to the callback outputs in the matching position, and construct the category input with:

```python
category = gr.Dropdown(CATEGORY_CHOICES, label="查询类别")
```

- [ ] **Step 4: Run focused tests and compile**

Run:

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q \
  tests/unit/test_launch_eval_annotator.py tests/unit/test_manual_eval_set.py
conda run -n vlm-course python -m compileall -q scripts/launch_eval_annotator.py
```

Expected: all tests pass and compileall exits 0.

- [ ] **Step 5: Restart and verify the real first task**

Restart the service on port 7862 with the proxy isolation already required by the local environment:

```bash
env -u ALL_PROXY -u all_proxy PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n vlm-course python \
  scripts/launch_eval_annotator.py \
  --config configs/default.yaml \
  --queries evaluation/manual_val_50/queries.jsonl \
  --relevance evaluation/manual_val_50/relevance.csv --port 7862
```

Use `gradio_client.Client(...).predict(1, api_name="/load_position")` and the page config to verify `q001`, `val-2007`, `张添翼`, a non-empty image path, and Chinese dropdown labels with English values.

- [ ] **Step 6: Commit**

```bash
git add scripts/launch_eval_annotator.py tests/unit/test_launch_eval_annotator.py
git commit -m "feat: localize manual evaluation form"
```
