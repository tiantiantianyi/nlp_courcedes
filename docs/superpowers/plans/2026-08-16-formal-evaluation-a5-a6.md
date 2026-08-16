# Formal Evaluation, A5, and A6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the two reviewed 50-query batches into an auditable 100-query evaluation set, collect candidate-level 0/1/2 qrels for a balanced 50-query subset, and produce formal A5 and A6 quality/resource results.

**Architecture:** Keep the two source batches immutable and create a merged tracked evaluation dataset. Build candidate pools through the existing A5 service matrix, annotate the pool through a dedicated contact-sheet UI, then feed the same qrels into the existing retrieval evaluator and a newly wired reranker-quality path. Operational benchmarks and formal quality metrics stay separate but share query IDs and candidate identities.

**Tech Stack:** Python 3.11, Pydantic, Gradio, Pillow, FAISS, existing `anima_search` services, pytest, JSONL/CSV/LaTeX artifacts.

## Global Constraints

- Develop on `main`; the user explicitly approved direct development on `main`.
- Do not modify M0--M5 model/index implementations or canonical Qwen3.5 annotations.
- Preserve both source query directories and their audit notes.
- Formal qrels use exactly grades `0`, `1`, and `2`; every reviewed query retains at least one source-image grade `2`.
- Report single-positive Recall/MRR over all 100 queries separately from multi-grade nDCG over the balanced 50-query candidate pool.
- Never infer quality from model scores, source-image identity, or mock data.
- Use `env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for pytest because the host exports an incompatible ROS plugin path and `socks://` proxy.
- Do not commit course images, model weights, FAISS indexes, or large runtime artifacts.

---

### Task 1: Preserve explicit zero-grade judgments

**Files:**
- Modify: `src/anima_search/evaluation/manual_set.py`
- Modify: `tests/unit/test_manual_eval_set.py`
- Modify: `tests/unit/test_launch_eval_annotator.py`

**Interfaces:**
- Consumes: `parse_judgments(text, query_id, annotator, note)` and `validate_manual_set(...)`.
- Produces: relevance rows containing grades `0/1/2`; validation still requires at least one grade `2` per reviewed query.

- [ ] **Step 1: Replace the zero-dropping test with a failing preservation test**

```python
def test_parse_judgments_preserves_zero_and_rejects_duplicates():
    rows = parse_judgments(
        "val-1:2\nval-2:0\nval-3：1",
        query_id="q001",
        annotator="human-a",
    )
    assert [(row["image_id"], row["relevance"]) for row in rows] == [
        ("val-1", 2),
        ("val-2", 0),
        ("val-3", 1),
    ]
```

- [ ] **Step 2: Add a failing validation test for a reviewed query with 2/1/0 rows**

```python
def test_validate_manual_set_accepts_explicit_zero_with_positive_source():
    tasks = [{
        "query_id": "q001", "text": "城市", "category": "simple",
        "source_image_id": "val-1", "source_relative_path": "../Val/1.jpg",
        "reviewed": True, "annotator": "human-a", "note": "",
    }]
    rows = [
        {"query_id": "q001", "image_id": "val-1", "relevance": 2, "annotator": "human-a", "note": ""},
        {"query_id": "q001", "image_id": "val-2", "relevance": 0, "annotator": "human-a", "note": ""},
    ]
    assert validate_manual_set(tasks, rows, expected_count=1)["relevance_row_count"] == 2
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest \
  tests/unit/test_manual_eval_set.py tests/unit/test_launch_eval_annotator.py -q
```

Expected: the zero-preservation assertion fails because `parse_judgments` currently drops grade 0.

- [ ] **Step 4: Implement the minimal contract change**

Delete the `if grade == 0: continue` branch. Change `validate_manual_set` from `grade not in {1, 2}` to `grade not in {0, 1, 2}`. Keep both reviewed-query checks: at least one row and at least one grade `2`.

- [ ] **Step 5: Run focused and full tests**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest \
  tests/unit/test_manual_eval_set.py tests/unit/test_launch_eval_annotator.py -q
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q
```

Expected: focused tests pass and the full count is at least 216 passed.

- [ ] **Step 6: Commit**

```bash
git add src/anima_search/evaluation/manual_set.py \
  tests/unit/test_manual_eval_set.py tests/unit/test_launch_eval_annotator.py
git commit -m "fix: preserve zero-grade relevance judgments"
```

### Task 2: Merge the two reviewed batches without overwriting sources

**Files:**
- Create: `src/anima_search/evaluation/formal_set.py`
- Create: `scripts/prepare_formal_eval.py`
- Create: `tests/unit/test_formal_set.py`
- Create: `evaluation/formal_val_100/queries.jsonl`
- Create: `evaluation/formal_val_100/relevance.csv`
- Create: `evaluation/formal_val_100/README.md`

**Interfaces:**
- Consumes: `load_tasks`, `load_relevance_rows`, `validate_manual_set`.
- Produces: `merge_reviewed_sets(query_paths, relevance_paths, expected_count) -> tuple[list[dict], list[dict], dict]` and a CLI-written 100-query dataset.

- [ ] **Step 1: Write failing merge tests**

Cover four cases: successful q001/q002 merge, duplicate query ID rejection, duplicate source image rejection, and relevance for an unknown query rejection. Assert output order follows numeric query ID and source files are byte-identical before/after.

```python
tasks, rows, summary = merge_reviewed_sets(
    [left_queries, right_queries], [left_relevance, right_relevance], expected_count=2
)
assert [row["query_id"] for row in tasks] == ["q001", "q002"]
assert summary["query_count"] == 2
```

- [ ] **Step 2: Run the merge tests and verify RED**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest tests/unit/test_formal_set.py -q
```

Expected: import failure for `anima_search.evaluation.formal_set`.

- [ ] **Step 3: Implement `merge_reviewed_sets`**

The function loads every pair, validates each batch with its own length, rejects duplicate query/source IDs across batches, validates the combined set with `expected_count`, and returns a summary containing query count, relevance row count, category counts, annotators, and source paths. It never calls a writer.

- [ ] **Step 4: Implement the CLI**

CLI arguments:

```text
--queries evaluation/manual_val_50/queries.jsonl evaluation/manual_val_50_assisted/queries.jsonl
--relevance evaluation/manual_val_50/relevance.csv evaluation/manual_val_50_assisted/relevance.csv
--output-dir evaluation/formal_val_100
--expected-count 100
```

Use `write_tasks` and `write_relevance`, then write `merge_report.json` with input SHA-256 values.

- [ ] **Step 5: Run tests and create the formal set**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest tests/unit/test_formal_set.py -q
conda run -n vlm-course python scripts/prepare_formal_eval.py \
  --queries evaluation/manual_val_50/queries.jsonl evaluation/manual_val_50_assisted/queries.jsonl \
  --relevance evaluation/manual_val_50/relevance.csv evaluation/manual_val_50_assisted/relevance.csv \
  --output-dir evaluation/formal_val_100 --expected-count 100
conda run -n vlm-course python scripts/validate_manual_eval_set.py \
  --queries evaluation/formal_val_100/queries.jsonl \
  --relevance evaluation/formal_val_100/relevance.csv --expected-count 100
```

Expected: 100 reviewed queries and 100 initial source-image grade-2 rows.

- [ ] **Step 6: Commit only the formal source data and implementation**

```bash
git add src/anima_search/evaluation/formal_set.py scripts/prepare_formal_eval.py \
  tests/unit/test_formal_set.py evaluation/manual_val_50 \
  evaluation/manual_val_50_assisted evaluation/formal_val_100
git commit -m "feat: prepare reviewed 100-query evaluation set"
```

### Task 3: Build a balanced candidate pool for graded relevance

**Files:**
- Create: `src/anima_search/evaluation/candidate_pool.py`
- Create: `scripts/build_relevance_pool.py`
- Create: `tests/unit/test_candidate_pool.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `a5_ablation_matrix()`, `create_service(config, split, branches, fusion_method)`, formal query JSONL.
- Produces: `build_candidate_pool(queries, rankings_by_variant, source_ids, per_variant_k) -> list[dict]` and `artifacts/evaluation/formal/relevance_pool.jsonl`.

- [ ] **Step 1: Write failing pure-function tests**

Assert deterministic first-seen union order, source-image pinning, variant provenance, duplicate removal, category-balanced selection, and rejection when a ranking has a foreign query ID.

```python
pool = build_candidate_pool(
    queries=[query],
    rankings_by_variant={"clip_only": {"q001": ["val-2", "val-1"]}},
    source_ids={"q001": "val-1"},
    per_variant_k=2,
)
assert [row["image_id"] for row in pool[0]["candidates"]] == ["val-1", "val-2"]
```

- [ ] **Step 2: Run tests and verify RED**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest tests/unit/test_candidate_pool.py -q
```

- [ ] **Step 3: Implement pool construction and schema**

Each query row contains `schema_version=formal-relevance-pool-v1.0`, query metadata, source image, and candidates with `image_id`, `relative_path`, `is_source`, `retrieved_by`, `best_rank`, `grade=null`, `annotator=""`, and `reviewed=false`.

- [ ] **Step 4: Implement CLI retrieval collection**

Defaults: select 50 queries balanced over available categories, `per_variant_k=5`, candidate cap 25. Instantiate each A5 service sequentially, collect rankings with reranking disabled, release encoders between variants, and save a JSON summary with counts by category/variant.

- [ ] **Step 5: Run pool generation**

```bash
conda run -n vlm-course python scripts/build_relevance_pool.py \
  --queries evaluation/formal_val_100/queries.jsonl \
  --config configs/default.yaml --split val \
  --graded-query-count 50 --per-variant-k 5 --candidate-cap 25 \
  --output artifacts/evaluation/formal/relevance_pool.jsonl
```

Expected: 50 category-stratified query rows; each contains its source image and no duplicate candidate IDs.

- [ ] **Step 6: Commit implementation, not runtime pool**

```bash
git add src/anima_search/evaluation/candidate_pool.py \
  scripts/build_relevance_pool.py tests/unit/test_candidate_pool.py .gitignore
git commit -m "feat: build formal multi-retriever relevance pool"
```

### Task 4: Add a candidate contact-sheet annotation UI

**Files:**
- Create: `src/anima_search/evaluation/candidate_review.py`
- Create: `scripts/launch_candidate_annotator.py`
- Create: `tests/unit/test_candidate_review.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `formal-relevance-pool-v1.0` JSONL and Val manifest/image paths.
- Produces: candidate contact sheets and `artifacts/evaluation/formal/candidate_relevance.csv` with all `0/1/2` rows.

- [ ] **Step 1: Write failing state and validation tests**

Test `parse_candidate_grades`, exact candidate coverage, duplicate rejection, grade domain, source grade 2, atomic save, and progress count. A query is complete only when every displayed candidate has an explicit grade.

```python
grades = parse_candidate_grades("val-1:2\nval-2:0", expected_ids={"val-1", "val-2"})
assert grades == {"val-1": 2, "val-2": 0}
```

- [ ] **Step 2: Run tests and verify RED**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest tests/unit/test_candidate_review.py -q
```

- [ ] **Step 3: Implement contact-sheet rendering and atomic saves**

Render at most 25 numbered 192 px tiles, label each tile with image ID and retrieval provenance, and save via temporary-file replacement. Never alter the source pool.

- [ ] **Step 4: Implement Gradio UI**

Show query text/category/source ID, contact sheet, one grade line per candidate, annotator, reviewed checkbox, progress, previous/next, and save. Default annotator is `张添翼`; existing saved rows retain their annotator.

- [ ] **Step 5: Run tests and launch locally**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest tests/unit/test_candidate_review.py -q
env -u ALL_PROXY -u all_proxy conda run -n vlm-course \
  python scripts/launch_candidate_annotator.py \
  --pool artifacts/evaluation/formal/relevance_pool.jsonl \
  --output artifacts/evaluation/formal/candidate_relevance.csv --port 7864
```

Expected: UI opens at `http://127.0.0.1:7864`; formal experiments remain blocked until all 50 pool rows are reviewed.

- [ ] **Step 6: Commit**

```bash
git add src/anima_search/evaluation/candidate_review.py \
  scripts/launch_candidate_annotator.py tests/unit/test_candidate_review.py README.md
git commit -m "feat: add graded candidate relevance review UI"
```

### Task 5: Validate and assemble final qrels

**Files:**
- Create: `src/anima_search/evaluation/qrels_validation.py`
- Create: `scripts/finalize_formal_qrels.py`
- Create: `tests/unit/test_qrels_validation.py`

**Interfaces:**
- Consumes: 100 source-positive rows, 50-query candidate pool, candidate review CSV.
- Produces: `artifacts/evaluation/formal/val_queries.jsonl`, `val_relevance.csv`, and `qrels_validation.json`.

- [ ] **Step 1: Write failing validation tests**

Reject missing candidate judgments, extra candidate IDs, duplicate rows, source grade other than 2, invalid annotator, and a graded subset without all five categories. Verify zero rows survive final CSV output.

- [ ] **Step 2: Run tests and verify RED**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest tests/unit/test_qrels_validation.py -q
```

- [ ] **Step 3: Implement finalization**

All 100 queries retain source positives. The reviewed 50-query subset replaces its source-only rows with complete candidate judgments. Report separate `single_positive_query_count=50` and `graded_pool_query_count=50`.

- [ ] **Step 4: Run finalization after human review**

```bash
conda run -n vlm-course python scripts/finalize_formal_qrels.py \
  --queries evaluation/formal_val_100/queries.jsonl \
  --source-relevance evaluation/formal_val_100/relevance.csv \
  --pool artifacts/evaluation/formal/relevance_pool.jsonl \
  --candidate-relevance artifacts/evaluation/formal/candidate_relevance.csv \
  --output-dir artifacts/evaluation/formal
```

Expected: `valid=true`, 100 queries, 50 complete graded-pool queries, and no unknown IDs.

- [ ] **Step 5: Commit code and lightweight validation summary**

```bash
git add src/anima_search/evaluation/qrels_validation.py \
  scripts/finalize_formal_qrels.py tests/unit/test_qrels_validation.py
git commit -m "feat: validate formal candidate-level qrels"
```

### Task 6: Run and freeze A5 formal results

**Files:**
- Modify: `scripts/run_ablation.py`
- Modify: `src/anima_search/evaluation/ablation.py`
- Modify: `tests/unit/test_ablation.py`
- Create: `docs/A5_FORMAL_RESULTS_2026-08-16.md`

**Interfaces:**
- Consumes: finalized formal queries/qrels and existing A5 service matrix.
- Produces: per-variant details plus aggregate JSON/CSV/LaTeX split into all-100 single-positive metrics and graded-50 nDCG metrics.

- [ ] **Step 1: Write failing split-report tests**

Assert each of five variants contains `all_queries`, `graded_queries`, category metrics, failure rate, and latency. Reject a graded query without complete candidate-pool judgments.

- [ ] **Step 2: Run tests and verify RED**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest tests/unit/test_ablation.py -q
```

- [ ] **Step 3: Implement split aggregation and provenance**

Add query-set SHA-256, qrels SHA-256, config SHA-256, index manifest SHA-256, actual variant settings, and runtime timestamp to the JSON. Keep the current five-row LaTeX table and add a graded-subset table.

- [ ] **Step 4: Run A5**

```bash
conda run -n vlm-course python scripts/run_ablation.py \
  --queries artifacts/evaluation/formal/val_queries.jsonl \
  --relevance artifacts/evaluation/formal/val_relevance.csv \
  --output-dir artifacts/evaluation/formal/a5
```

Expected: five variants, zero unreported failures, JSON/CSV/LaTeX outputs, and no `quality_claim=none_without_reviewed_relevance`.

- [ ] **Step 5: Write evidence-backed result notes and commit code/docs**

Document actual numbers, category behavior, failures, limits, and whether RRF or weighted wins. Do not prestate the winner.

```bash
git add scripts/run_ablation.py src/anima_search/evaluation/ablation.py \
  tests/unit/test_ablation.py docs/A5_FORMAL_RESULTS_2026-08-16.md
git commit -m "feat: report formal A5 retrieval quality"
```

### Task 7: Wire A6 baseline/pointwise/listwise quality into the Top-20 benchmark

**Files:**
- Modify: `src/anima_search/evaluation/rerank_quality.py`
- Modify: `tests/unit/test_rerank_quality.py`
- Modify: `scripts/benchmark_listwise_top20.py`
- Modify: `tests/unit/test_listwise_benchmark.py`
- Create: `docs/A6_FORMAL_RESULTS_2026-08-16.md`

**Interfaces:**
- Consumes: query IDs, fixed Top-20 candidates, pointwise records, listwise records, formal relevance.
- Produces: per-query/per-repeat baseline, pointwise, listwise MRR/nDCG@10 plus aggregate quality/resource table.

- [ ] **Step 1: Add failing CLI-quality tests**

Use fake candidate sets and rerank records. Assert `--relevance` is required when `--quality-output` is supplied, pointwise records are converted to one order per repeat with baseline-order ties/failures, listwise candidate sets must match baseline, and only query IDs present in qrels are scored.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest \
  tests/unit/test_rerank_quality.py tests/unit/test_listwise_benchmark.py -q
```

- [ ] **Step 3: Implement CLI integration**

Add `--relevance`, `--quality-output`, and `--graded-only`. Store `baseline_image_ids` in each per-query payload. For each repeat, derive pointwise order with `rank_pointwise_scores`, read listwise `ranked_image_ids`, call `evaluate_rerank_orders`, then aggregate. Replace the static quality claim only when qrels were actually loaded and scored.

- [ ] **Step 4: Run a one-query quality smoke**

```bash
conda run -n vlm-course python scripts/benchmark_listwise_top20.py \
  --queries artifacts/evaluation/formal/val_queries.jsonl \
  --relevance artifacts/evaluation/formal/val_relevance.csv \
  --config configs/benchmark_8gb.yaml --branches image text bm25 \
  --top-k 20 --query-limit 1 --repeats 1 \
  --output artifacts/evaluation/formal/a6/smoke.json \
  --quality-output artifacts/evaluation/formal/a6/smoke_quality.json
```

Expected: all three orders contain the same 20 IDs and quality metrics are finite.

- [ ] **Step 5: Run the balanced graded subset**

Run the same command with `--graded-only --query-limit 50`. Record cold/warm latency, peak VRAM, failures, listwise degradation, MRR, and nDCG@10. If runtime is excessive, run category-stratified batches and merge only after schema validation.

- [ ] **Step 6: Write evidence-backed results and commit**

```bash
git add src/anima_search/evaluation/rerank_quality.py \
  tests/unit/test_rerank_quality.py scripts/benchmark_listwise_top20.py \
  tests/unit/test_listwise_benchmark.py docs/A6_FORMAL_RESULTS_2026-08-16.md
git commit -m "feat: evaluate formal A6 reranker quality"
```

### Task 8: Phase verification and handoff to M2

**Files:**
- Modify: `README.md`
- Modify: `docs/M6_M7_POST_ANNOTATION_INTEGRATION_2026-08-15.md`
- Create: `docs/FORMAL_EVALUATION_HANDOFF_2026-08-16.md`

**Interfaces:**
- Consumes: Tasks 1--7 code and artifacts.
- Produces: exact reproducibility commands, frozen hashes, result boundaries, and M2/report inputs.

- [ ] **Step 1: Run full verification**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q
conda run -n vlm-course python -m compileall -q src scripts tests
conda run -n vlm-course python scripts/validate_manual_eval_set.py \
  --queries evaluation/formal_val_100/queries.jsonl \
  --relevance evaluation/formal_val_100/relevance.csv --expected-count 100
git diff --check
```

- [ ] **Step 2: Audit claims against artifacts**

Record exact query counts, qrels counts by grade, five A5 variants, three A6 methods, failures, degradation, latency, VRAM, and SHA-256 values. State the second-reviewer gap until another teammate completes it.

- [ ] **Step 3: Update documentation and commit**

```bash
git add README.md docs/M6_M7_POST_ANNOTATION_INTEGRATION_2026-08-15.md \
  docs/FORMAL_EVALUATION_HANDOFF_2026-08-16.md
git commit -m "docs: freeze formal retrieval evaluation evidence"
```

- [ ] **Step 4: Confirm clean scoped status**

```bash
git status --short
git log -8 --oneline
```

Expected: no uncommitted Phase-A source/docs changes; ignored runtime artifacts remain local.
