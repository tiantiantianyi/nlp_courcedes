# M3–M7 Selective Migration and Full Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Selectively migrate the teammate's M3–M5 implementation, preserve the hardened M6/M7, rebuild the real local pipeline, verify the completed M0–M2 upstream, produce the proposal evidence, and push the verified `main` branch to GitHub.

**Architecture:** M0–M2 remain read-only upstream inputs. M3 imports canonical Qwen3.5 annotations and builds image/text/BM25 indexes; M4 parses and filters queries; M5 exports a strict Top-20 contract plus a retrieval-config snapshot; M6 validates and reranks all 12 queries; M7 creates three auditable stories. Shared files are reconciled behavior-by-behavior instead of being overwritten by the teammate's old M6/M7 patch.

**Tech Stack:** Python 3.11, Conda `vlm-course`, Pydantic 2, PyTorch, FAISS, Chinese-CLIP, BGE, BM25, Qwen3-VL-2B, Stable Diffusion, Gradio, pytest, Git.

## Global Constraints

- Work directly on `main`, as explicitly requested by the user.
- Teammate source is read-only at `/home/tiantiantianyi/Desktop/nlp/nlp_courcedes_delivery_20260815/nlp_courcedes`.
- M0–M2 implementations are not rewritten; only their files, hashes, counts, and M3 compatibility are verified.
- Original images and canonical annotations are read-only.
- Never apply the teammate's complete patch; it contains obsolete M6/M7 files.
- Preserve `src/anima_search/m6/`, `scripts/run_m6_from_m5.py`, `src/anima_search/m7/canonical_annotations.py`, `src/anima_search/m7/m6_bridge.py`, and `scripts/run_m7_from_m6.py` unless a task explicitly names a tested interface change.
- Use `apply_patch` for manual edits; use path-filtered `git apply` only for exact new-file additions from the verified teammate patch.
- Use `conda run -n vlm-course` for every Python command.
- Do not commit models, FAISS indexes, original images, original annotations, or generated caches.
- Do not use `git push --force` or any history-rewriting push.
- Report teammate M3–M5 implementation as teammate work; report selective migration, interface work, M6/M7, experiments, and verification as Zhang Tianyi's work.

## Recommended Execution Schedule on the 8 GB RTX 4060 Laptop

This plan is the complete path, but Tasks 1–16 are not an honest one-day promise. Code migration can usually finish in one focused day; human relevance review, full Train indexing, 50-query pointwise reranking, Stable Diffusion gap filling, and missing teammate evidence are duration or external gates. Use this order:

- Morning, 08:30–09:00: Task 1; freeze the baseline and upstream hashes.
- Morning, 09:00–12:00: Tasks 2–6; migrate M3–M5 and correct the M5→M6 snapshot contract with focused tests and commits.
- Afternoon, 13:30–14:30: Task 7; run the full automated gate. Do not start expensive experiments while it is red.
- Afternoon, 14:30–15:00: Task 8; rebuild manifests and import canonical Qwen3.5 annotations.
- Afternoon and evening: Task 9 Val first, then Train. The three branches are sequential on 8 GB VRAM; do not run Qwen or Stable Diffusion concurrently.
- While CPU/BM25 work or human review is possible without occupying CUDA: start Task 11. This is separate relevance annotation, not a repeat of M1 image annotation.
- After Val indexes are green: Task 10, then Task 12, then Task 13. These are the formal M5→M6→M7 chain.
- After Task 11 passes: Task 14. A5 is moderate; A6 pointwise Top-20 over 50 queries is the longest local GPU job and is suitable for an overnight run.
- Last: Tasks 15–16; freeze evidence, audit ownership, reconcile the remote, and push normally.

At the end of the first day, a valid checkpoint is Tasks 1–10 complete with Task 9 Train indexing or Task 11 human review still running. Do not relabel those long or external gates as completed merely to fit one calendar day.


## File Responsibility Map

- `src/anima_search/adapters/annotation.py`: canonical v1.3 → project annotation adapter.
- `scripts/import_m1_qwen35.py`: read-only canonical import into M3 artifacts.
- `src/anima_search/indexing/*`: image/text/BM25 persistence and manifest contracts.
- `scripts/build_indexes.py`: real three-branch M3 build CLI.
- `src/anima_search/retrieval/{query_parser,filters,terms,search,fusion}.py`: M4/M5 behavior.
- `src/anima_search/delivery/m5_candidates.py`: M5 builder backed by the canonical M6 input model.
- `scripts/export_m5_candidates.py`: formal 12-query export and config snapshot.
- `src/anima_search/m6/interface_validation.py`: independent snapshot/manifest/artifact validation.
- `scripts/{validate_m5_m6_interface,run_m6_from_m5}.py`: formal M5→M6 CLIs.
- `scripts/run_m7_from_m6.py`: one-query real M7 story CLI.
- `docs/M3_M5_QWEN35_INTEGRATION.md`: teammate-work handoff record.
- `docs/M3_M7_FINAL_INTEGRATION_2026-08-16.md`: final evidence and proposal matrix.

---

### Task 1: Freeze the baseline and audit M0–M2 inputs

**Files:**
- Create: `docs/M0_M2_UPSTREAM_ACCEPTANCE_2026-08-16.md`
- Read: `/home/tiantiantianyi/Desktop/nlp/M1_clean_annotations_v1.3/*`
- Read: `artifacts/manifests/{train,val}.jsonl`

**Interfaces:**
- Consumes: teammate-completed M0–M2 delivery.
- Produces: immutable upstream counts and hashes used by every later experiment.

- [ ] **Step 1: Confirm the repository baseline**

Run:

```bash
cd /home/tiantiantianyi/Desktop/nlp/nlp_courcedes
git status --short --branch
git log --oneline -5
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q
```

Expected: clean `main`; the pre-migration suite reports 183 passed or a larger already-recorded count.

- [ ] **Step 2: Verify the canonical package without changing it**

Run:

```bash
cd /home/tiantiantianyi/Desktop/nlp/M1_clean_annotations_v1.3
sha256sum -c SHA256SUMS
wc -l qwen3.5_9b_annotations.jsonl
jq '.valid, .schema_version // .version, .record_count // .counts' verification.json
```

Expected: every declared hash is OK; the Qwen3.5 file contains 2362 non-empty records.

- [ ] **Step 3: Record the upstream acceptance**

Use `apply_patch` to create the acceptance document with these exact fields:

```markdown
# M0–M2 Upstream Acceptance

- Ownership: teammate-provided; not implemented by Zhang Tianyi
- Canonical schema: v1.3
- Manifest images: Train 2000, Val 369
- Qwen3.5 valid annotations: 2362
- Expected imported coverage: Train 1993, Val 369
- Missing Train numeric IDs: 48, 649, 764, 899, 1155, 1217, 1918
- Integrity: record the SHA256SUMS command result and verification.json result
- Consumer boundary: M3 may read these files but must not rewrite them
```

- [ ] **Step 4: Commit the audit**

```bash
git add docs/M0_M2_UPSTREAM_ACCEPTANCE_2026-08-16.md
git commit -m "docs: accept verified M0 M2 upstream"
```

---

### Task 2: Migrate the canonical v1.3 adapter and import CLI

**Files:**
- Modify: `tests/unit/test_annotation_adapter.py`
- Create: `tests/unit/test_import_m1_qwen35.py`
- Modify: `src/anima_search/adapters/annotation.py`
- Modify: `src/anima_search/adapters/__init__.py`
- Create: `scripts/import_m1_qwen35.py`
- Modify: `configs/default.yaml`
- Create: `docs/M3_M5_QWEN35_INTEGRATION.md`

**Interfaces:**
- Consumes: `adapt_annotation(payload: dict, manifest: ManifestItem) -> ImageAnnotation`.
- Produces: `import_annotations(source: Path, artifacts: Path, require_complete: bool = False) -> dict`.

- [ ] **Step 1: Add the canonical identity regression test**

Port the teammate test `test_adapts_qwen35_canonical_v13_and_checks_manifest_identity` and add this import-level assertion:

```python
import json
from pathlib import Path

from anima_search.schemas import ManifestItem
from scripts.import_m1_qwen35 import import_annotations


def _canonical_record(numeric_id: str, digest: str) -> dict[str, object]:
    return {
        "image_id": numeric_id,
        "processed_sha256": digest,
        "source_model_id": "Qwen/Qwen3.5-9B",
        "annotation": {
            "scene": {"primary_type": "general"},
            "capture_visual": {},
            "entities": [],
            "ocr": [],
            "relations": [],
            "event": {},
            "subjective": {},
            "captions": {"short_zh": f"测试图像 {numeric_id}"},
            "uncertainties": [],
        },
    }


def test_import_preserves_manifest_identity_and_reports_missing(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    manifests = artifacts / "manifests"
    manifests.mkdir(parents=True)
    train = ManifestItem(
        image_id="train-1", split="Train", relative_path="../Train/1.jpg",
        sha256="a" * 64, size_bytes=10,
    )
    val = ManifestItem(
        image_id="val-2", split="Val", relative_path="../Val/2.jpg",
        sha256="b" * 64, size_bytes=20,
    )
    (manifests / "train.jsonl").write_text(
        train.model_dump_json() + "\n", encoding="utf-8"
    )
    (manifests / "val.jsonl").write_text(
        val.model_dump_json() + "\n", encoding="utf-8"
    )
    source = tmp_path / "qwen35.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                _canonical_record("1", "a" * 64),
                _canonical_record("2", "b" * 64),
            )
        ) + "\n",
        encoding="utf-8",
    )

    report = import_annotations(source, artifacts)

    assert report["annotation_version"] == "qwen35-canonical-v1.3"
    assert report["imported"] == {"train": 1, "val": 1}
    assert report["missing_image_ids"] == []
    assert report["failures"] == []
```

- [ ] **Step 2: Verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_annotation_adapter.py tests/unit/test_import_m1_qwen35.py
```

Expected: collection fails because `scripts.import_m1_qwen35` does not exist, or the canonical identity case fails.

- [ ] **Step 3: Apply only the new import CLI from the teammate patch**

```bash
git apply --check \
  --include=scripts/import_m1_qwen35.py \
  /home/tiantiantianyi/Desktop/nlp/nlp_courcedes_delivery_20260815/0001-Integrate-Qwen3.5-annotations-and-complete-M3-M7-del.patch
git apply \
  --include=scripts/import_m1_qwen35.py \
  /home/tiantiantianyi/Desktop/nlp/nlp_courcedes_delivery_20260815/0001-Integrate-Qwen3.5-annotations-and-complete-M3-M7-del.patch
```

Use `apply_patch` to reconcile the adapter rather than replacing it. Preserve all current fields and add the canonical nested mappings for `scene`, `capture_visual`, `entities`, `ocr`, `relations`, `event`, `subjective`, `captions`, and `uncertainties`. Reject a `processed_sha256` mismatch.

- [ ] **Step 4: Set the portable source path**

In `configs/default.yaml`, set:

```yaml
annotation:
  source: ../M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl
  prompt_version: qwen35-canonical-v1.3
  allow_missing: true
```

- [ ] **Step 5: Verify GREEN**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_annotation_adapter.py tests/unit/test_import_m1_qwen35.py
conda run -n vlm-course python -m compileall -q \
  src/anima_search/adapters scripts/import_m1_qwen35.py
```

- [ ] **Step 6: Commit**

```bash
git add configs/default.yaml docs/M3_M5_QWEN35_INTEGRATION.md \
  scripts/import_m1_qwen35.py src/anima_search/adapters \
  tests/unit/test_annotation_adapter.py tests/unit/test_import_m1_qwen35.py
git commit -m "feat: migrate canonical Qwen3.5 import"
```

---

### Task 3: Reconcile M3 three-branch indexing

**Files:**
- Modify: `tests/unit/test_indexing.py`
- Modify: `tests/integration/test_build_indexes_cli.py`
- Modify: `src/anima_search/indexing/index_manifest.py`
- Modify: `src/anima_search/indexing/{bm25_index,documents,faiss_io,image_vector_index,vector_index}.py`
- Modify: `scripts/build_indexes.py`
- Modify: `src/anima_search/app/factory.py`

**Interfaces:**
- Consumes: imported `ImageAnnotation` JSONL.
- Produces: `write_index_manifest(path: Path, *, split: str, image_ids: list[str], annotation_path: Path, annotation_version: str, branches: dict[str, dict[str, Any]], config_digest: str = "", image_records: list[dict[str, str]] | None = None) -> dict[str, Any]` and loadable image/text/BM25 indexes.

- [ ] **Step 1: Add the manifest provenance test**


```python
def test_index_manifest_persists_portable_image_records(tmp_path):
    payload = write_index_manifest(
        tmp_path / "manifest.json",
        split="val",
        image_ids=["val-2002"],
        annotation_path=tmp_path / "val.jsonl",
        annotation_version="qwen35-canonical-v1.3",
        branches={"image": {"record_count": 1}},
        config_digest="a" * 64,
        image_records=[{
            "image_id": "val-2002",
            "relative_path": "../Val/2002.jpg",
            "sha256": "b" * 64,
        }],
    )
    assert payload["image_records"][0]["relative_path"] == "../Val/2002.jpg"
```

- [ ] **Step 2: Verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_indexing.py::test_index_manifest_persists_portable_image_records
```

Expected: `write_index_manifest` rejects `image_records`.

- [ ] **Step 3: Reconcile the M3 code behavior by behavior**

Use the teammate snapshot only as a source for portable `image_records`, encoder type/options, finite-vector checks, identical annotation/index ID ordering, CUDA release after each branch, runtime `annotations.json`, and formal failure when any requested branch fails. Preserve all current helpers and do not replace shared files wholesale.

The final manifest writer must add `image_records` to the existing payload without changing the meaning of `config_digest`; every `relative_path` must be POSIX and project-relative.

- [ ] **Step 4: Verify GREEN**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_indexing.py tests/unit/test_image_only.py \
  tests/integration/test_build_indexes_cli.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_indexes.py src/anima_search/indexing \
  src/anima_search/app/factory.py tests/unit/test_indexing.py \
  tests/unit/test_image_only.py tests/integration/test_build_indexes_cli.py
git commit -m "feat: reconcile M3 three-branch indexing"
```

---

### Task 4: Reconcile M4 structured query behavior

**Files:**
- Modify: `tests/unit/test_query_parser.py`
- Modify: `tests/unit/test_filters.py`
- Modify: `tests/unit/test_scene_router.py`
- Modify: `src/anima_search/retrieval/{query_parser,filters,terms,openai_compatible}.py`
- Modify: `src/anima_search/routing/scene_router.py`
- Modify: `configs/{retrieval_aliases,scene_routing}.yaml`
- Modify: `scripts/verify_m4_query_parser.py`

**Interfaces:**
- Consumes: natural-language query and configured aliases.
- Produces: deterministic `StructuredQuery`, requested/effective backend metadata, fallback error, elapsed seconds, and hard/soft filter evidence.

- [ ] **Step 1: Port and run the five exact teammate regressions**

Port these functions from the teammate test snapshot: `test_negative_exception_does_not_treat_drone_as_no_people`, `test_generator_scalar_fields_are_normalized_and_cannot_invent_exclusions`, `test_single_character_alias_does_not_match_drone_object`, `test_ocr_term_is_a_hard_filter_and_evidence`, and `test_scene_router_validates_category_and_vector_contracts`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_query_parser.py tests/unit/test_filters.py \
  tests/unit/test_scene_router.py tests/unit/test_openai_compatible.py
```

Expected: only missing teammate behavior is red. If all five cases already pass, do not alter that production component.

- [ ] **Step 2: Reconcile only the failing behavior**

Keep current local-Qwen and OpenAI-compatible API behavior. Explicit negative, required, OCR, and count constraints remain hard; generated scalar/list normalization must not invent exclusions; one-character aliases require token-safe matching.

- [ ] **Step 3: Add deterministic 12-query file mode**

Add these parser arguments and require at least one of `--query` or `--queries-file`:

```python
parser.add_argument("--queries-file", type=Path)
parser.add_argument("--output", type=Path, required=True)
```

Accept input field `query` or `text`. Write a JSON array with `query_id`, `requested_backend`, `effective_backend`, `fallback_error`, `elapsed_seconds`, and `parsed`.

- [ ] **Step 4: Verify and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_query_parser.py tests/unit/test_filters.py \
  tests/unit/test_scene_router.py tests/unit/test_service_m4_backends.py
git add configs/retrieval_aliases.yaml configs/scene_routing.yaml \
  scripts/verify_m4_query_parser.py src/anima_search/retrieval \
  src/anima_search/routing tests/unit/test_query_parser.py \
  tests/unit/test_filters.py tests/unit/test_scene_router.py
git commit -m "feat: reconcile M4 structured query routing"
```

---

### Task 5: Add the canonical M5 exporter without duplicating the interface schema

**Files:**
- Create: `src/anima_search/delivery/{__init__,m5_candidates}.py`
- Create: `scripts/export_m5_candidates.py`
- Create: `tests/unit/test_m5_delivery.py`
- Modify: `tests/integration/test_m3_m5_pipeline.py`

**Interfaces:**
- Consumes: `list[SearchResult]`.
- Produces: `build_m5_candidate_batch(*, query_id: str, query: str, category: Literal["simple", "compositional", "negative", "count", "ocr"], split: Literal["train", "val"], fusion_method: Literal["rrf", "weighted"], annotation_version: str, index_manifest_sha256: str, config_sha256: str, results: list[SearchResult]) -> M5QueryBatch`.

- [ ] **Step 1: Write three RED tests**

Create 20 `SearchResult` fixtures and assert that the builder returns the canonical `anima_search.m6.contract.M5QueryBatch`, ranks are exactly 1–20, all branch fields are retained, and no `rerank_score` field is serialized. Add separate tests that 19 results raise `ValueError` matching `exactly 20` and duplicated image IDs raise `ValueError` matching `must be unique`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_m5_delivery.py
```

Expected: import fails because `anima_search.delivery` does not exist.

- [ ] **Step 2: Implement the thin canonical adapter**

Do not migrate the teammate duplicate Pydantic schema. Build only the existing M6 contract:

```python
from typing import Literal

from anima_search.m6.contract import M5QueryBatch
from anima_search.schemas import SearchResult


def build_m5_candidate_batch(
    *,
    query_id: str,
    query: str,
    category: Literal["simple", "compositional", "negative", "count", "ocr"],
    split: Literal["train", "val"],
    fusion_method: Literal["rrf", "weighted"],
    annotation_version: str,
    index_manifest_sha256: str,
    config_sha256: str,
    results: list[SearchResult],
) -> M5QueryBatch:
    if len(results) != 20:
        raise ValueError("M5 delivery requires exactly 20 results")
    return M5QueryBatch.model_validate({
        "schema_version": "m5-to-m6-v1.0",
        "query_id": query_id,
        "query": query,
        "category": category,
        "split": split,
        "fusion_method": fusion_method,
        "top_k": 20,
        "annotation_version": annotation_version,
        "index_manifest_sha256": index_manifest_sha256,
        "config_sha256": config_sha256,
        "candidates": [
            {
                "rank": rank,
                "image_id": result.image_id,
                "relative_path": result.relative_path,
                "fused_score": result.fused_score,
                "branch_scores": result.branch_scores,
                "branch_ranks": result.branch_ranks,
                "matched_fields": result.matched_fields,
            }
            for rank, result in enumerate(results, start=1)
        ],
    })
```

- [ ] **Step 3: Migrate the exporter**

Port `scripts/export_m5_candidates.py` from the teammate snapshot and import the canonical builder above. Preserve atomic replacement, 12-query loading, full-split branch recall, soft-positive fallback, original branch scores/ranks, and `m5_retrieval_config.snapshot.json`. The snapshot hash becomes each row `config_sha256`.

- [ ] **Step 4: Verify and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_m5_delivery.py tests/integration/test_m3_m5_pipeline.py \
  tests/unit/test_m6_contract.py
git add src/anima_search/delivery scripts/export_m5_candidates.py \
  tests/unit/test_m5_delivery.py tests/integration/test_m3_m5_pipeline.py
git commit -m "feat: export canonical M5 Top-20 deliveries"
```

---

### Task 6: Correct the M5 config-snapshot contract at the M6 boundary

**Files:**
- Modify: `src/anima_search/m6/interface_validation.py`
- Modify: `scripts/{validate_m5_m6_interface,run_m6_from_m5}.py`
- Modify: `tests/unit/test_m5_m6_interface_validation.py`
- Modify: `tests/integration/test_validate_m5_m6_cli.py`
- Modify: `tests/unit/test_m6_offline_cli.py`

**Interfaces:**
- Consumes: M5 JSONL, M5 retrieval snapshot, index manifest, annotation artifact, and image roots.
- Produces: `validate_interface_file(input_path: Path, project_root: Path, train_dir: Path, val_dir: Path, index_manifest_path: Path, config_snapshot_path: Path) -> tuple[list[M5QueryBatch], InterfaceValidationReport]`.

- [ ] **Step 1: Add RED semantic tests**

In the existing `_fixture_paths` test fixture, create `m5_retrieval_config.snapshot.json`, set the batch `config_sha256` to `sha256_file(snapshot)`, deliberately set manifest `config_digest` to another 64-character digest, refresh `index_manifest_sha256`, and assert validation succeeds. Then modify one byte of the snapshot and assert issue code `E_MANIFEST_MISMATCH`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_m5_m6_interface_validation.py -k config_snapshot
```

Expected: `config_snapshot_path` is not accepted or the validator incorrectly compares with `config_digest`.

- [ ] **Step 2: Implement the corrected boundary**

Hash `config_snapshot_path` once and compare every batch `config_sha256` with that digest. Remove the comparison between `batch.config_sha256` and manifest `config_digest`; keep `config_digest` only as index-build provenance. Add required `--m5-config-snapshot` to both CLIs and include the snapshot in output-alias protection.

- [ ] **Step 3: Verify and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_m5_m6_interface_validation.py \
  tests/integration/test_validate_m5_m6_cli.py \
  tests/unit/test_m6_offline_cli.py
git add src/anima_search/m6/interface_validation.py \
  scripts/validate_m5_m6_interface.py scripts/run_m6_from_m5.py \
  tests/unit/test_m5_m6_interface_validation.py \
  tests/integration/test_validate_m5_m6_cli.py tests/unit/test_m6_offline_cli.py
git commit -m "fix: validate the M5 retrieval snapshot"
```

---

### Task 7: Run the post-migration automated gate

**Files:** Modify only a Task 2–6 file when its focused regression identifies the owner.

- [ ] **Step 1: Run the full suite and static checks**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q
conda run -n vlm-course python -m compileall -q src scripts tests
git diff --check
git status --short
```

Expected: zero failures, compileall exit 0, and no migration debris. For each failure, rerun the smallest file with `-vv`, add or preserve a regression, patch minimally, commit in its owning Task 2–6, and rerun this full gate. Task 7 creates no standalone commit.

---

### Task 8: Regenerate manifests and import real canonical annotations

**Generated and ignored:** `artifacts/manifests/{train,val}.jsonl`, `artifacts/annotations/{train,val}.qwen35-canonical-v1.3.jsonl`, and `artifacts/annotations/qwen35_canonical_v1.3_import_report.json`.

- [ ] **Step 1: Build manifests and import**

```bash
conda run -n vlm-course python scripts/build_manifest.py \
  --config configs/default.yaml
conda run -n vlm-course python scripts/import_m1_qwen35.py \
  --config configs/default.yaml \
  --source /home/tiantiantianyi/Desktop/nlp/M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl
```

- [ ] **Step 2: Verify exact counts**

```bash
wc -l artifacts/annotations/train.qwen35-canonical-v1.3.jsonl \
  artifacts/annotations/val.qwen35-canonical-v1.3.jsonl
jq ".imported, .missing_image_ids, .failures" \
  artifacts/annotations/qwen35_canonical_v1.3_import_report.json
```

Expected: Train 1993, Val 369, missing numeric IDs `48,649,764,899,1155,1217,1918`, and no failures. Do not commit generated annotations.

---

### Task 9: Build and validate real M3 indexes

**Generated and ignored:** `artifacts/indexes/{val,train}/`.

- [ ] **Step 1: Build and inspect Val first**

```bash
conda run -n vlm-course python scripts/build_indexes.py \
  --config configs/default.yaml --split Val --branches image,text,bm25
jq "{split,record_count,active_branches,annotation_path,annotation_sha256,config_digest}" \
  artifacts/indexes/val/manifest.json
```

Expected: 369 records in all three branches and no `branch_failures`.

- [ ] **Step 2: Build Train**

```bash
conda run -n vlm-course python scripts/build_indexes.py \
  --config configs/default.yaml --split Train --branches image,text,bm25
```

Expected: 1993 records per branch; no missing record is synthesized.

- [ ] **Step 3: Run and save M3–M5 verification**

```bash
mkdir -p artifacts/evaluation
conda run -n vlm-course python scripts/verify_m3_m5.py \
  --config configs/default.yaml --split val \
  > artifacts/evaluation/m3_m5_verification.txt
```

Expected: exit 0 and JSON `status=passed` with image, text, and BM25 active.

---

### Task 10: Run real M4 and export the formal M5 handoff

**Generated:** M4 rule/Qwen JSON, 12-row M5 JSONL, retrieval snapshot, and validation report under `artifacts/evaluation/`.

- [ ] **Step 1: Run deterministic M4 and one local-Qwen smoke**

```bash
conda run -n vlm-course python scripts/verify_m4_query_parser.py \
  --config configs/default.yaml --backend rules \
  --queries-file configs/m6_benchmark_queries.jsonl \
  --output artifacts/evaluation/m4_rules_12.json
conda run -n vlm-course python scripts/verify_m4_query_parser.py \
  --config configs/benchmark_8gb.yaml --backend local_qwen \
  --query "雨夜城市街道，没有人物" \
  --output artifacts/evaluation/m4_local_qwen_smoke.json
```

Expected: 12 deterministic rule rows. The Qwen smoke must show either a real local-Qwen parse or an explicit rules fallback with error.

- [ ] **Step 2: Export and validate 12×20 M5**

```bash
conda run -n vlm-course python scripts/export_m5_candidates.py \
  --queries configs/m6_benchmark_queries.jsonl \
  --config configs/default.yaml --split val \
  --output artifacts/evaluation/m5_to_m6_candidates.jsonl
conda run -n vlm-course python scripts/validate_m5_m6_interface.py \
  --input artifacts/evaluation/m5_to_m6_candidates.jsonl \
  --m5-config-snapshot artifacts/evaluation/m5_retrieval_config.snapshot.json \
  --project-root . --train-dir ../Train --val-dir ../Val \
  --index-manifest artifacts/indexes/val/manifest.json \
  --report artifacts/evaluation/m5_validation_report.json
jq "{valid,query_count,candidate_count,issues}" \
  artifacts/evaluation/m5_validation_report.json
```

Expected: `valid=true`, 12 queries, 240 candidates, and zero issues.

---

### Task 11: Complete the human relevance gate

**Files:** Preserve `evaluation/manual_val/*`; create `evaluation/manual_val_50/*` only if needed; generate `artifacts/evaluation/active_manual_eval_dir.txt`.

The Qwen3.5 M1 annotations are model image descriptions, not retrieval relevance judgments. They cannot replace this gate.

- [ ] **Step 1: Select a complete 100-query set or initialize 50 separate tasks**

```bash
mkdir -p artifacts/evaluation
if conda run -n vlm-course python scripts/validate_manual_eval_set.py \
  --queries evaluation/manual_val/queries.jsonl \
  --relevance evaluation/manual_val/relevance.csv \
  --manifest artifacts/manifests/val.jsonl --expected-count 100
then
  printf "%s\n" evaluation/manual_val > \
    artifacts/evaluation/active_manual_eval_dir.txt
else
  if test ! -e evaluation/manual_val_50/queries.jsonl && \
    test ! -e evaluation/manual_val_50/relevance.csv
  then
    conda run -n vlm-course python scripts/create_manual_eval_tasks.py \
      --config configs/default.yaml --count 50 \
      --output-dir evaluation/manual_val_50
  fi
  printf "%s\n" evaluation/manual_val_50 > \
    artifacts/evaluation/active_manual_eval_dir.txt
fi
```

- [ ] **Step 2: Review every selected task in Gradio**

```bash
M3M7_EVAL_DIR=$(sed -n "1p" artifacts/evaluation/active_manual_eval_dir.txt)
conda run -n vlm-course python scripts/launch_eval_annotator.py \
  --config configs/default.yaml \
  --queries "$M3M7_EVAL_DIR/queries.jsonl" \
  --relevance "$M3M7_EVAL_DIR/relevance.csv" --port 7862
```

Write each query from the source image, choose category, enter annotator, add `image_id:grade`, and tick reviewed.

- [ ] **Step 3: Validate the selected set**

```bash
M3M7_EVAL_DIR=$(sed -n "1p" artifacts/evaluation/active_manual_eval_dir.txt)
case "$M3M7_EVAL_DIR" in
  evaluation/manual_val) M3M7_EXPECTED_COUNT=100 ;;
  evaluation/manual_val_50) M3M7_EXPECTED_COUNT=50 ;;
  *) echo "unexpected evaluation directory"; exit 1 ;;
esac
conda run -n vlm-course python scripts/validate_manual_eval_set.py \
  --queries "$M3M7_EVAL_DIR/queries.jsonl" \
  --relevance "$M3M7_EVAL_DIR/relevance.csv" \
  --manifest artifacts/manifests/val.jsonl \
  --expected-count "$M3M7_EXPECTED_COUNT"
```

No MRR/NDCG quality claim is allowed before this succeeds.

---

### Task 12: Add M6 runtime metrics and run all 12 queries

**Files:** Modify `scripts/run_m6_from_m5.py` and `tests/unit/test_m6_offline_cli.py`; generate M6 results, metrics, and validation report.

- [ ] **Step 1: Test and implement a metrics sidecar**

Add `--metrics-output`. In a dry-run CLI test, monkeypatch `validate_interface_file` to return `_batch()` and a valid one-query report; assert metrics contain query ID `dry-q001`, nonnegative elapsed seconds, and `degraded=true`. Time each `rerank_query_batch` with `time.perf_counter`; record method, mismatch, degraded, and `client.last_generation_metadata.get("peak_vram_bytes")`. Aggregate count, degraded count, total/mean seconds, and peak VRAM. Protect the metrics path from every input and output alias.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_m6_offline_cli.py tests/unit/test_m6_runner.py
git add scripts/run_m6_from_m5.py tests/unit/test_m6_offline_cli.py
git commit -m "feat: record auditable M6 runtime metrics"
```

- [ ] **Step 2: Run formal listwise M6**

```bash
conda run -n vlm-course python scripts/run_m6_from_m5.py \
  --input artifacts/evaluation/m5_to_m6_candidates.jsonl \
  --m5-config-snapshot artifacts/evaluation/m5_retrieval_config.snapshot.json \
  --output artifacts/evaluation/m6_rerank_results.jsonl \
  --metrics-output artifacts/evaluation/m6_runtime_metrics.json \
  --validation-report artifacts/evaluation/m6_validation_report.json \
  --config configs/benchmark_8gb.yaml \
  --index-manifest artifacts/indexes/val/manifest.json \
  --train-dir ../Train --val-dir ../Val --method listwise
```

Expected: 12 rows; each has exactly 20 unique candidates. Every model failure must retain full M5 order and explicit degraded reason.

---

### Task 13: Run three M7 stories and one real AI gap fill

**Generated:** `artifacts/evaluation/m7_stories/{m6-q01,m6-q03,m6-q09}.json` plus generated gap image.

- [ ] **Step 1: Run q01 and q09 without gaps**

```bash
for M3M7_QUERY in m6-q01 m6-q09
do
  conda run -n vlm-course python scripts/run_m7_from_m6.py \
    --m6-results artifacts/evaluation/m6_rerank_results.jsonl \
    --query-id "$M3M7_QUERY" --select-count 5 \
    --annotations ../M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl \
    --train-manifest artifacts/manifests/train.jsonl \
    --val-manifest artifacts/manifests/val.jsonl \
    --config configs/benchmark_8gb.yaml \
    --output "artifacts/evaluation/m7_stories/$M3M7_QUERY.json" \
    --theme "可追溯视觉故事" --tone "自然" --seed 20260816
done
```

- [ ] **Step 2: Run q03 with Stable Diffusion gap filling**

```bash
conda run -n vlm-course python scripts/run_m7_from_m6.py \
  --m6-results artifacts/evaluation/m6_rerank_results.jsonl \
  --query-id m6-q03 --select-count 5 \
  --annotations ../M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl \
  --train-manifest artifacts/manifests/train.jsonl \
  --val-manifest artifacts/manifests/val.jsonl \
  --config configs/benchmark_8gb.yaml \
  --output artifacts/evaluation/m7_stories/m6-q03.json \
  --theme "雨夜城市转场" --tone "电影感" --seed 20260816 --fill-gaps
```

- [ ] **Step 3: Validate and capture UI evidence**

For all three JSON files, require schema `m7-story-v1.0`, 3–8 selected IDs, equal selected/ordered ID sets, and section order equal to `ordered_image_ids`. For q03 require at least one gap with `status=generated`, `source=generated`, and `ai_generated=true`; an empty or failed gap is incomplete.

```bash
conda run -n vlm-course python scripts/launch_app.py \
  --config configs/benchmark_8gb.yaml --split val
```

Save `docs/assets/final_m3_m7/search_results.png`, `story_order.png`, and `ai_gap_marker.png`.

---

### Task 14: Complete A5–A7 and audit A1–A4/A8/A9

**Files:** Create `src/anima_search/evaluation/rerank_quality.py`, `tests/unit/test_rerank_quality.py`, and `docs/PROPOSAL_EXPERIMENT_ACCEPTANCE_2026-08-16.md`; modify `scripts/benchmark_listwise_top20.py`.

- [ ] **Step 1: Run A5 on the validated human set**

```bash
M3M7_EVAL_DIR=$(sed -n "1p" artifacts/evaluation/active_manual_eval_dir.txt)
conda run -n vlm-course python scripts/run_ablation.py \
  --config configs/default.yaml \
  --queries "$M3M7_EVAL_DIR/queries.jsonl" \
  --relevance "$M3M7_EVAL_DIR/relevance.csv" \
  --split val --output-dir artifacts/evaluation/a5
```

Expected: CLIP-only, text-only, BM25-only, RRF three-way, and weighted three-way rows with Recall/MRR/mAP/NDCG@10 and latency. This is the formal RRF versus normalized-weighted comparison.

- [ ] **Step 2: Add tested A6 quality calculation**

Implement:

```python
def evaluate_rerank_orders(
    baseline_ids: list[str],
    pointwise_ids: list[str],
    listwise_ids: list[str],
    relevance: dict[str, int],
) -> dict[str, dict[str, float]]:
    expected = set(baseline_ids)
    if (
        len(expected) != len(baseline_ids)
        or set(pointwise_ids) != expected
        or set(listwise_ids) != expected
    ):
        raise ValueError("all reranker variants must contain the same candidate IDs")
    orders = {
        "baseline": baseline_ids,
        "pointwise": pointwise_ids,
        "listwise": listwise_ids,
    }
    return {
        method: {
            "mrr": reciprocal_rank(ids, relevance),
            "ndcg@10": ndcg_at_k(ids, relevance, 10),
        }
        for method, ids in orders.items()
    }
```

Add a unit test with baseline `a,b,c`, pointwise `b,a,c`, listwise `c,b,a`, relevance `b:2,c:1`; assert baseline MRR 0.5 and both rerankers MRR 1.0. Assert mismatched candidate sets raise. Add required `--relevance` to the benchmark, reconstruct baseline from candidate order, pointwise by descending finite score with original-rank tie-break, listwise from `ranked_image_ids`, and aggregate mean MRR/NDCG@10.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_rerank_quality.py tests/unit/test_listwise_benchmark.py \
  tests/unit/test_rerank_benchmark.py
git add src/anima_search/evaluation/rerank_quality.py \
  scripts/benchmark_listwise_top20.py tests/unit/test_rerank_quality.py
git commit -m "feat: evaluate A6 reranker quality"
```

- [ ] **Step 3: Run A6 over 50 human queries**

```bash
M3M7_EVAL_DIR=$(sed -n "1p" artifacts/evaluation/active_manual_eval_dir.txt)
conda run -n vlm-course python scripts/benchmark_listwise_top20.py \
  --queries "$M3M7_EVAL_DIR/queries.jsonl" \
  --relevance "$M3M7_EVAL_DIR/relevance.csv" \
  --config configs/benchmark_8gb.yaml --split val \
  --branches image text bm25 --top-k 20 --query-limit 50 --repeats 1 \
  --output artifacts/evaluation/a6_pointwise_listwise_top20.json
```

Expected: 50 rows of identical Top-20 candidate sets plus baseline/pointwise/listwise quality, latency, VRAM, failure, and degradation summaries. This is the longest local job and may run overnight.

- [ ] **Step 4: Run A7 paired resources**

```bash
conda run -n vlm-course python scripts/benchmark_image_encoders.py \
  --config configs/benchmark_8gb.yaml --split Val --limit 64 \
  --chinese-batch-size 4 --jina-batch-size 1 --jina-dim 512 \
  --output artifacts/a7_encoder_comparison.json
```

Report time, latency, peak VRAM, index size, and model size. Do not claim A7 quality superiority without encoder-specific relevance evidence.

- [ ] **Step 5: Audit all proposal experiments**

Create an A1–A9 table. For teammate-owned A1–A4/A8/A9, cite exact path, SHA-256, sample count, and metric; otherwise mark `缺少可核验产物`. A9 must separate real scenes, abstract art, Chinese classical paintings, and medical images and state: `医学影像实验仅用于观察模型跨领域行为，不用于任何诊断用途。`

```bash
git add docs/PROPOSAL_EXPERIMENT_ACCEPTANCE_2026-08-16.md \
  docs/assets/final_m3_m7
git commit -m "docs: record proposal experiment evidence"
```

---

### Task 15: Update the integration and Zhang Tianyi personal reports

**Files:** Create `docs/M3_M7_FINAL_INTEGRATION_2026-08-16.md`; create or modify `docs/personal_report/张添翼_U202315231_个人报告.tex`; generate `docs/personal_report/张添翼_U202315231_个人报告.pdf`.

- [ ] **Step 1: Write evidence with correct ownership**

The integration report must contain exact test count/commands, M3 counts/dimensions, M4 backend/fallback, M5 12/240 validation, M6 latency/VRAM/degradation, three M7 stories and AI marker, A1–A9 status, and every missing external item. The personal report must say `在队友实现基础上完成迁移与联调` for M3–M5, while selective migration, M5→M6 contract, M6/M7, experiments, and evidence are Zhang Tianyi work.

- [ ] **Step 2: Scan ownership language**

```bash
rg -n "我实现|本人实现|独立完成|完成了 M3|完成了 M4|完成了 M5" \
  docs/M3_M7_FINAL_INTEGRATION_2026-08-16.md \
  docs/personal_report/张添翼_U202315231_个人报告.tex
```

Every match must be supported by a personal commit; rewrite teammate M3–M5 claims.

- [ ] **Step 3: Compile and inspect the named report**

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd \
  docs/personal_report/张添翼_U202315231_个人报告.tex
pdfinfo docs/personal_report/张添翼_U202315231_个人报告.pdf
pdftotext docs/personal_report/张添翼_U202315231_个人报告.pdf -
```

Expected: author 张添翼, student ID U202315231, extractable Chinese, no undefined references, and no unsupported completion claim.

- [ ] **Step 4: Commit**

```bash
git add docs/M3_M7_FINAL_INTEGRATION_2026-08-16.md \
  docs/personal_report/张添翼_U202315231_个人报告.tex \
  docs/personal_report/张添翼_U202315231_个人报告.pdf
git commit -m "docs: report verified M3 M7 integration"
```

---

### Task 16: Final verification, remote reconciliation, and GitHub push

- [ ] **Step 1: Run final gates**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q
conda run -n vlm-course python -m compileall -q src scripts tests
git diff --check
git status --short
git diff --name-only origin/main...HEAD
git ls-files | rg "(^|/)(models|artifacts/indexes|Train|Val)/|\.faiss$|\.safetensors$|\.bin$"
```

Expected: zero test failures; clean tree; no tracked model, index, raw image, or original annotation.

- [ ] **Step 2: Restore authentication and reconcile remote**

```bash
ssh -T git@github.com
git fetch origin
git log --oneline --left-right --graph origin/main...main
```

If remote has commits, inspect `git diff --stat main...origin/main`, merge with `git merge --no-edit origin/main`, and rerun Step 1. Never force-push.

- [ ] **Step 3: Push and verify hashes**

```bash
git push origin main
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
git status --short --branch
```

Expected: local and remote hashes match and branch is neither ahead nor behind.

## Execution Checkpoints

1. Stop after Task 7 if the automated suite is red.
2. Stop after Task 8 unless import is exactly Train 1993 and Val 369.
3. Stop after Task 9 if any requested branch is missing or degraded.
4. Stop after Task 10 unless M5 is 12 queries, 240 candidates, zero issues.
5. Stop after Task 12 if any M6 row violates Top-20 identity.
6. Do not claim quality while Task 11 is incomplete.
7. Do not claim all A1–A9 complete when teammate evidence is missing.
8. Do not push while tests, large-file audit, or remote reconciliation fails.
