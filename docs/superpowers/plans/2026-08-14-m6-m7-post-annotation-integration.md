# M6/M7 标注到达后联调 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 M1--M5 实现的前提下，严格校验队友导出的 M5 Top-20，完成 M6 离线重排输出，并让 M7 只读消费 Qwen3.5 canonical v1.3 标注与 M6 结果。

**Architecture:** 新建独立的 `anima_search.m6` 包承载跨模块契约、文件校验和重排结果，不把接口字段塞进通用 schema，也不调用检索服务。M7 在自己的包内增加 canonical v1.3 只读映射和 M6 结果桥接；M1--M5 文件和原始交付数据始终只读。

**Tech Stack:** Python 3.11、Pydantic 2、Pillow、pytest、现有 Qwen3-VL/ModelManager、JSONL、Conda `vlm-course`。

## Global Constraints

- 只修改 M6/M7 包、对应脚本、测试和文档；不得修改 M1--M5 实现。
- 不修改 `src/anima_search/adapters/annotation.py`、`scripts/build_indexes.py`、`src/anima_search/retrieval/fusion.py` 或原始标注文件。
- M5 输入契约固定为 `docs/M5_TO_M6_INTERFACE_V1.md` 的 `m5-to-m6-v1.0`。
- 一条查询必须包含恰好 20 个唯一候选，原始名次严格为 1--20。
- M6 不重新检索、不重算 M5 分数、不覆盖输入 JSONL。
- canonical v1.3 转换仅供 M7 使用，不输出供 M3--M5 建库的通用标注快照。
- 没有人工 relevance judgments 时，只报告接口、资源、稳定性和降级证据，不声明排序质量提升。
- 模型、索引、原图、原始 JSONL 和运行输出继续位于 Git 忽略目录，不进入提交。
- 每项生产代码必须先看到对应测试因缺少行为而失败，再写最小实现。

---

## File Map

- Create: `src/anima_search/m6/__init__.py` — M6 包公开入口。
- Create: `src/anima_search/m6/contract.py` — `m5-to-m6-v1.0` Pydantic 契约与 `SearchResult` 映射。
- Create: `src/anima_search/m6/interface_validation.py` — JSONL、跨行、manifest、路径和图片校验。
- Create: `src/anima_search/m6/results.py` — `m6-rerank-v1.0` 输出模型。
- Create: `src/anima_search/m6/runner.py` — 对已校验 batch 执行 pointwise/listwise 并保持 M5 证据。
- Create: `scripts/validate_m5_m6_interface.py` — 队友交付的严格校验 CLI。
- Create: `scripts/run_m6_from_m5.py` — 不加载检索编码器的 M6 离线重排 CLI。
- Create: `src/anima_search/m7/canonical_annotations.py` — Qwen3.5 canonical v1.3 到 M7 所需 `ImageAnnotation` 的只读映射。
- Modify: `src/anima_search/m7/story_planner.py` — 精确处理 canonical `time_of_day`，不改变其他模块。
- Create: `src/anima_search/m7/m6_bridge.py` — 从 M6 输出选择 3--8 张真实图片。
- Create: `scripts/run_m7_from_m6.py` — M6 输出到 M7 故事/补图的 CLI。
- Create: `tests/unit/test_m5_m6_contract.py`。
- Create: `tests/unit/test_m5_m6_interface_validation.py`。
- Create: `tests/integration/test_validate_m5_m6_cli.py`。
- Create: `tests/unit/test_m6_runner.py`。
- Create: `tests/unit/test_m7_canonical_annotations.py`。
- Modify: `tests/unit/test_story_planner.py`。
- Create: `tests/unit/test_m7_m6_bridge.py`。
- Create: `docs/M6_M7_POST_ANNOTATION_INTEGRATION_2026-08-14.md` — 实测记录与剩余依赖。

---

### Task 1: M5→M6 契约模型与无损 SearchResult 映射

**Files:**
- Create: `src/anima_search/m6/__init__.py`
- Create: `src/anima_search/m6/contract.py`
- Test: `tests/unit/test_m5_m6_contract.py`

**Interfaces:**
- Consumes: 单行 JSON 对象，schema 为 `m5-to-m6-v1.0`。
- Produces: `M5Candidate`, `M5QueryBatch`, `M5Candidate.to_search_result()`, `M5QueryBatch.to_search_results()`。

- [ ] **Step 1: 写合法 Top-20、分支键和 rank 的失败测试**

在 `tests/unit/test_m5_m6_contract.py` 中构造一个固定工厂：

```python
def candidate(rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "image_id": f"val-{2001 + rank}",
        "relative_path": f"../Val/{2001 + rank}.jpg",
        "fused_score": 1.0 / rank,
        "branch_scores": {"image": 0.9 / rank, "text": 0.8 / rank},
        "branch_ranks": {"image": rank, "text": rank + 1},
        "matched_fields": ["scene"],
    }


def batch_payload() -> dict[str, object]:
    return {
        "schema_version": "m5-to-m6-v1.0",
        "query_id": "m6-q001",
        "query": "夜晚的城市街道",
        "category": "simple",
        "split": "val",
        "fusion_method": "rrf",
        "top_k": 20,
        "annotation_version": "qwen35-canonical-v1.3",
        "index_manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "candidates": [candidate(rank) for rank in range(1, 21)],
    }
```

测试必须断言：

```python
def test_valid_batch_maps_m5_evidence_without_reordering():
    batch = M5QueryBatch.model_validate(batch_payload())
    results = batch.to_search_results()
    assert [item.image_id for item in results] == [f"val-{n}" for n in range(2002, 2022)]
    assert results[0].fused_score == 1.0
    assert results[0].branch_scores == {"image": 0.9, "text": 0.8}
    assert results[0].branch_ranks == {"image": 1, "text": 2}
    assert results[0].matched_fields == ["scene"]


@pytest.mark.parametrize("count", [19, 21])
def test_batch_requires_exactly_twenty_candidates(count: int):
    payload = batch_payload()
    payload["candidates"] = [candidate(rank) for rank in range(1, count + 1)]
    with pytest.raises(ValidationError):
        M5QueryBatch.model_validate(payload)


def test_branch_score_and_rank_keys_must_match():
    payload = batch_payload()
    payload["candidates"][0]["branch_ranks"] = {"image": 1}
    with pytest.raises(ValidationError, match="branch keys"):
        M5QueryBatch.model_validate(payload)


def test_candidate_rank_sequence_and_image_ids_are_unique():
    payload = batch_payload()
    payload["candidates"][1]["rank"] = 1
    payload["candidates"][1]["image_id"] = payload["candidates"][0]["image_id"]
    with pytest.raises(ValidationError):
        M5QueryBatch.model_validate(payload)
```

- [ ] **Step 2: 运行测试并确认因 M6 契约模块不存在而失败**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course \
  python -m pytest -q tests/unit/test_m5_m6_contract.py
```

Expected: collection fails with `ModuleNotFoundError: anima_search.m6`。

- [ ] **Step 3: 实现最小契约模型**

`src/anima_search/m6/contract.py` 定义：

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anima_search.schemas import SearchResult

BranchName = Literal["image", "text", "bm25"]


class M5Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
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
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
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
        ids = [item.image_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate image IDs must be unique within a query")
        return self

    def to_search_results(self) -> list[SearchResult]:
        return [item.to_search_result() for item in self.candidates]
```

`src/anima_search/m6/__init__.py` 只导出 `M5Candidate` 和 `M5QueryBatch`。

- [ ] **Step 4: 运行定向测试并确认通过**

Run: Task 1 Step 2 的命令。  
Expected: all tests in `test_m5_m6_contract.py` pass。

- [ ] **Step 5: 提交契约模型**

```bash
git add src/anima_search/m6 tests/unit/test_m5_m6_contract.py
git commit -m "feat: add strict M5 to M6 contract"
```

---

### Task 2: 严格 JSONL、manifest、路径和图片校验

**Files:**
- Create: `src/anima_search/m6/interface_validation.py`
- Test: `tests/unit/test_m5_m6_interface_validation.py`

**Interfaces:**
- Consumes: `input_path`, `project_root`, `train_dir`, `val_dir`, `index_manifest_path`。
- Produces: `validate_interface_file(...) -> tuple[list[M5QueryBatch], InterfaceValidationReport]`。
- Reads: 索引 manifest 及其同目录 `annotations.json`，不修改两者。

- [ ] **Step 1: 写多错误收集和合法文件的失败测试**

测试用 Pillow 在 `tmp_path/Val` 创建 20 张 8×8 JPEG，写入对应 `annotations.json` 和
schema v2 index manifest。测试至少包含：

```python
def test_valid_file_returns_batches_and_zero_issues(tmp_path: Path):
    paths = fixture_paths(tmp_path)
    batches, report = validate_interface_file(**paths)
    assert report.valid
    assert report.query_count == 1
    assert report.candidate_count == 20
    assert report.issues == []
    assert len(batches) == 1


def test_validator_collects_duplicate_query_path_and_branch_errors(tmp_path: Path):
    paths = fixture_paths(tmp_path, duplicate_query=True, wrong_split_path=True,
                          branch_key_mismatch=True)
    batches, report = validate_interface_file(**paths)
    assert batches == []
    assert not report.valid
    assert {issue.code for issue in report.issues} >= {
        "E_DUPLICATE_QUERY_ID", "E_PATH_SPLIT", "E_BRANCH_KEYS",
    }


def test_validator_rejects_manifest_hash_and_candidate_path_mismatch(tmp_path: Path):
    paths = fixture_paths(tmp_path, wrong_manifest_hash=True,
                          catalog_path_mismatch=True)
    _, report = validate_interface_file(**paths)
    assert {issue.code for issue in report.issues} >= {
        "E_MANIFEST_MISMATCH",
    }
```

Fixture 中 manifest 的 `image_ids_sha256` 使用现有
`anima_search.indexing.index_manifest.image_ids_digest()` 产生，避免复制 M3 算法。

- [ ] **Step 2: 运行测试并确认因校验模块不存在而失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course \
  python -m pytest -q tests/unit/test_m5_m6_interface_validation.py
```

Expected: import fails for `anima_search.m6.interface_validation`。

- [ ] **Step 3: 实现报告模型和完整扫描**

生产接口固定为：

```python
class InterfaceIssue(BaseModel):
    code: str
    line: int | None = None
    query_id: str | None = None
    message: str


class InterfaceValidationReport(BaseModel):
    valid: bool
    query_count: int
    candidate_count: int
    issues: list[InterfaceIssue]


def validate_interface_file(
    input_path: Path,
    project_root: Path,
    train_dir: Path,
    val_dir: Path,
    index_manifest_path: Path,
) -> tuple[list[M5QueryBatch], InterfaceValidationReport]:
    ...
```

实现必须按以下顺序执行：

1. 对每个非空行运行严格 `json.loads`；通过 `parse_constant` 拒绝 NaN/Inf；
2. 用 `M5QueryBatch.model_validate` 校验结构，并把 Pydantic 错误映射到文档稳定错误码；
3. 收集跨行重复 `query_id`；
4. 用 `sha256_file(index_manifest_path)` 核对每行 `index_manifest_sha256`；
5. 读取 manifest，核对 `split`、`annotation_version`、`config_digest`；
6. 读取同目录 `annotations.json`，建立 `image_id -> relative_path` 目录；
7. 用 annotations ID 顺序核对 manifest 的 `record_count` 和 `image_ids_sha256`；
8. 验证候选 ID 存在且 `relative_path` 与目录一致；
9. 将路径解析到 `project_root`，消解符号链接后检查位于对应 split 根目录；
10. 用 `Image.open(path).verify()` 检查图片可解码；
11. 若存在任何 issue，返回空 batch 列表；否则返回所有 batch。

用一个小函数保证路径边界：

```python
def _resolve_candidate_path(project_root: Path, relative_path: str) -> Path:
    if "\\" in relative_path or Path(relative_path).is_absolute():
        raise ValueError("candidate path must be a relative POSIX path")
    return (project_root / relative_path).resolve()
```

- [ ] **Step 4: 增加非有限数值、损坏图片和 19 候选测试**

JSON parser 测试直接写入包含 `NaN` 的原始字符串；损坏图片测试写入 `b"not-jpeg"`；
候选数量测试删除一项。分别断言 `E_NONFINITE_SCORE`、`E_IMAGE_DECODE` 和
`E_CANDIDATE_COUNT`。

- [ ] **Step 5: 运行定向测试并确认通过**

Run: Task 2 Step 2 的命令。  
Expected: all validation tests pass。

- [ ] **Step 6: 提交文件校验器**

```bash
git add src/anima_search/m6/interface_validation.py \
  tests/unit/test_m5_m6_interface_validation.py
git commit -m "feat: validate M5 candidate deliveries"
```

---

### Task 3: M5→M6 校验命令行

**Files:**
- Create: `scripts/validate_m5_m6_interface.py`
- Create: `tests/integration/test_validate_m5_m6_cli.py`

**Interfaces:**
- Consumes: `docs/M5_TO_M6_INTERFACE_V1.md` 第 10 节规定的参数。
- Produces: UTF-8 JSON 校验报告；合法返回 0，非法返回 1。

- [ ] **Step 1: 写 CLI 退出码失败测试**

使用 subprocess 对临时合法/非法交付文件运行脚本，断言：

```python
assert valid.returncode == 0
assert json.loads(valid_report.read_text())["valid"] is True
assert invalid.returncode == 1
assert json.loads(invalid_report.read_text())["valid"] is False
```

- [ ] **Step 2: 运行测试并确认脚本缺失**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course \
  python -m pytest -q tests/integration/test_validate_m5_m6_cli.py
```

Expected: subprocess returns 2 because script file does not exist。

- [ ] **Step 3: 实现 CLI**

参数固定为：

```python
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--project-root", type=Path, required=True)
parser.add_argument("--train-dir", type=Path, required=True)
parser.add_argument("--val-dir", type=Path, required=True)
parser.add_argument("--index-manifest", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
```

CLI 调用 `validate_interface_file`，用 `model_dump_json(indent=2)` 写报告，打印查询数、候选数
和错误数。报告有 issue 时 `raise SystemExit(1)`，否则正常返回。

- [ ] **Step 4: 运行 CLI 定向测试并确认通过**

Run: Task 3 Step 2 的命令。  
Expected: valid and invalid cases pass。

- [ ] **Step 5: 提交校验 CLI**

```bash
git add scripts/validate_m5_m6_interface.py \
  tests/integration/test_validate_m5_m6_cli.py
git commit -m "feat: add M5 delivery validation CLI"
```

---

### Task 4: M6 重排输出模型和无损离线 runner

**Files:**
- Create: `src/anima_search/m6/results.py`
- Create: `src/anima_search/m6/runner.py`
- Create: `tests/unit/test_m6_runner.py`

**Interfaces:**
- Consumes: 已通过 Task 2 的 `M5QueryBatch` 和现有 reranker 对象。
- Produces: `M6QueryResult`，schema 固定为 `m6-rerank-v1.0`。

- [ ] **Step 1: 写反向排序仍保留 M5 证据的失败测试**

Fake reranker 反转 20 个 `SearchResult` 并写入 100 到 81 的分数。测试断言：

```python
result = rerank_query_batch(batch, ReverseReranker(), method="listwise")
assert result.schema_version == "m6-rerank-v1.0"
assert [item.rerank_rank for item in result.candidates] == list(range(1, 21))
assert result.candidates[0].image_id == batch.candidates[-1].image_id
assert result.candidates[0].rank == 20
assert result.candidates[0].branch_scores == batch.candidates[-1].branch_scores
assert result.candidates[0].branch_ranks == batch.candidates[-1].branch_ranks
assert not result.degraded
```

再写两个降级测试：第一个 reranker 返回一个重复 ID 并遗漏另一个 ID，断言 runner 去重后按
M5 原始顺序把遗漏 ID 补在末尾、`degraded=true`，且 mismatch 同时记录 dropped duplicate 和
appended missing；第二个 reranker 返回未知 ID，断言 runner 恢复完整 M5 顺序并记录硬降级。

- [ ] **Step 2: 运行测试并确认结果模块不存在**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course \
  python -m pytest -q tests/unit/test_m6_runner.py
```

Expected: import error for `anima_search.m6.runner`。

- [ ] **Step 3: 实现 M6 输出模型**

`results.py` 定义 `M6CandidateResult`，完整复制 M5 候选字段（包括原字段名 `rank`），并新增：

```python
rerank_rank: int
rerank_score: float | None
mismatch: list[str] = Field(default_factory=list)
```

`M6QueryResult` 字段固定为：

```python
schema_version: Literal["m6-rerank-v1.0"]
source_schema_version: Literal["m5-to-m6-v1.0"]
query_id: str
query: str
category: str
split: Literal["train", "val"]
fusion_method: Literal["rrf", "weighted"]
top_k: Literal[20]
annotation_version: str
index_manifest_sha256: str
config_sha256: str
rerank_method: Literal["pointwise", "listwise"]
degraded: bool
mismatch: list[str]
candidates: list[M6CandidateResult] = Field(min_length=20, max_length=20)
```

- [ ] **Step 4: 实现 `rerank_query_batch`**

签名固定为：

```python
def rerank_query_batch(
    batch: M5QueryBatch,
    reranker: object,
    *,
    method: Literal["pointwise", "listwise"],
) -> M6QueryResult:
    ...
```

实现先深拷贝 `batch.to_search_results()` 再调用 reranker。完整有效时按 reranker 顺序输出；
出现重复 ID 时保留第一次并记录 dropped duplicate，出现遗漏 ID 时按 M5 原始顺序补到末尾并
设置 `rerank_score=0.0`。返回未知 ID、空数组、抛出异常或无法解析时使用完整 M5 顺序硬回退。
所有降级都在 batch 和受影响候选 `mismatch` 中记录；`last_error` 或
`last_degraded_reason` 非空也必须反映到 `degraded`。输出顶层和候选必须保留 M5 原字段及
原值，不能用重排结果覆盖 `rank`、`fused_score`、`branch_scores` 或 `branch_ranks`。

- [ ] **Step 5: 运行定向测试并确认通过**

Run: Task 4 Step 2 的命令。  
Expected: all M6 runner tests pass。

- [ ] **Step 6: 提交 M6 runner**

```bash
git add src/anima_search/m6/results.py src/anima_search/m6/runner.py \
  tests/unit/test_m6_runner.py
git commit -m "feat: rerank validated M5 candidate batches"
```

---

### Task 5: 不加载检索编码器的 M6 离线命令

**Files:**
- Create: `scripts/run_m6_from_m5.py`
- Test: `tests/unit/test_m6_offline_cli.py`

**Interfaces:**
- Consumes: 已通过校验的 M5 JSONL、config、index manifest。
- Produces: `m6_reranked_results.jsonl`；绝不覆盖输入。

- [ ] **Step 1: 写输出路径保护与 dry-run 失败测试**

将 CLI 逻辑拆为 `main(argv: list[str] | None = None) -> int`。测试 monkeypatch
`validate_interface_file` 返回一个合法 batch，并断言：

```python
with pytest.raises(ValueError, match="must differ from input"):
    main(["--input", str(path), "--output", str(path), "--dry-run", ...])

assert main([..., "--dry-run"]) == 0
assert json.loads(output.read_text().splitlines()[0])["schema_version"] == "m6-rerank-v1.0"
```

dry-run 使用保持 M5 顺序的 deterministic reranker，只验证读取和输出契约，并在结果中写入
`degraded=true`、`mismatch=["dry-run: Qwen3-VL was not invoked"]`，不能伪装成真实重排。

- [ ] **Step 2: 运行测试并确认脚本模块不存在**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course \
  python -m pytest -q tests/unit/test_m6_offline_cli.py
```

Expected: import error for `scripts.run_m6_from_m5`。

- [ ] **Step 3: 实现 CLI 参数和模型生命周期**

参数包含：

```text
--input --output --validation-report --config --index-manifest
--train-dir --val-dir --method {pointwise,listwise}
--query-limit --dry-run
```

非 dry-run 模式：

1. 读取 config 和现有 Qwen3-VL 路径；
2. 直接构造 `QwenVLClient`，不调用 `create_service`，因此不会加载 M3--M5 索引或检索编码器；
3. listwise 使用现有 `ListwiseVisualReranker`；pointwise 使用现有 `VisualReranker`；
4. 每完成一条查询立即 append 一行结果并 flush；
5. `finally` 中调用 client 的 `unload()`（若存在）；
6. 输入与输出 resolve 后相同立即拒绝。

- [ ] **Step 4: 运行定向测试并确认通过**

Run: Task 5 Step 2 的命令。  
Expected: path guard and dry-run tests pass。

- [ ] **Step 5: 提交离线 M6 CLI**

```bash
git add scripts/run_m6_from_m5.py tests/unit/test_m6_offline_cli.py
git commit -m "feat: run M6 from offline M5 deliveries"
```

---

### Task 6: M7 专用 canonical v1.3 只读标注映射

**Files:**
- Create: `src/anima_search/m7/canonical_annotations.py`
- Modify: `src/anima_search/m7/story_planner.py`
- Create: `tests/unit/test_m7_canonical_annotations.py`
- Modify: `tests/unit/test_story_planner.py`

**Interfaces:**
- Consumes: `qwen3.5_9b_annotations.jsonl` 和现有 Train/Val manifest JSONL。
- Produces: `load_canonical_m7_annotations(...) -> dict[str, ImageAnnotation]`，仅供 M7。

- [ ] **Step 1: 写真实字段映射和哈希错误的失败测试**

使用最小 canonical 记录覆盖：

```python
record = {
    "image_id": "2002",
    "processed_sha256": "a" * 64,
    "source_model_id": "Qwen/Qwen3.5-9B",
    "normalizer_version": "m1-normalize-v1.0.0",
    "repairs_applied": 1,
    "lossy_repairs": False,
    "annotation": {
        "scene": {"primary_type": "street_urban", "secondary_types": [],
                  "media_type": "natural_image", "sub_type_zh": "城市街道",
                  "environment": "outdoor"},
        "capture_visual": {"time_of_day": "night", "weather": "clear",
                           "lighting": "artificial", "viewpoint": "eye_level",
                           "shot_scale": "wide", "blur_level": "none"},
        "entities": [{"entity_id": "e1", "entity_type": "vehicle",
                      "name_zh": "汽车", "count": 2, "count_exact": True,
                      "attributes": {"colors_zh": ["黑色"], "materials_zh": [],
                                     "states_zh": ["行驶中"], "action_zh": "行驶",
                                     "attire_zh": []}}],
        "ocr": [{"text_raw": "便利店"}],
        "relations": [],
        "event": {"summary_zh": "汽车在夜间街道行驶"},
        "subjective": {"mood_terms_zh": ["安静"], "palette_terms_zh": ["冷色"]},
        "captions": {"short_zh": "夜间街道。", "dense_zh": "黑色汽车驶过夜间城市街道。"},
        "uncertainties": [],
    },
}
```

断言输出 key 为 `val-2002`，并检查 summary、scene、objects、object_counts、actions、colors、
mood、OCR、`time_of_day:night`、model/prompt version。把 hash 改成错误值后必须抛出
`ValueError`，不能静默接收。

- [ ] **Step 2: 写 `dawn_dusk` 不被误判为 dawn 的失败测试**

在 `test_story_planner.py` 增加：

```python
def test_canonical_dawn_dusk_uses_explicit_twilight_bucket():
    item = annotation("twilight", time="dawn_dusk", scene="海边")
    assert time_bucket(item) == (4, "晨昏")
```

Expected before implementation: 当前逻辑因字符串包含 `dawn` 返回 `(0, "黎明")`。

- [ ] **Step 3: 运行测试并确认预期失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q \
  tests/unit/test_m7_canonical_annotations.py tests/unit/test_story_planner.py
```

Expected: canonical loader import fails；twilight 断言失败为黎明。

- [ ] **Step 4: 实现 M7 局部映射**

函数签名：

```python
def load_canonical_m7_annotations(
    annotation_path: Path,
    manifest_paths: list[Path],
    *,
    split: Literal["Train", "Val"] | None = None,
) -> dict[str, ImageAnnotation]:
    ...
```

manifest 通过去掉 `train-`/`val-` 前缀建立数字 ID 映射。每条 canonical 记录必须唯一、存在于
manifest 且 `processed_sha256` 相同。映射规则：

- `captions.dense_zh` → summary，空时依次回退 short/event；
- `scene.sub_type_zh` → scene，空时用 primary type；
- `entities[].name_zh` 去重 → objects；
- 同名实体全部 `count_exact=true` 且 count 非空时求和 → object_counts，否则不写该对象计数；
- 实体 `action_zh` 去重 → actions；
- `capture_visual` 写成 `key:value` attributes；
- 主场景、environment、实体 states/materials 同样写入 attributes；
- 主观 palette 加实体 colors → colors；主观 mood → mood；
- `ocr[].text_raw` → ocr_text；uncertainty note → uncertainty；
- summary、short caption、scene 组成恰好三条 `search_queries`，只满足内部 schema，不用于正式检索评测；
- `source_model_id` → model_version；prompt_version 固定 `canonical-v1.3`；
- repairs 元数据放入 `generation_parameters`，不修改原始 JSONL。

- [ ] **Step 5: 精确处理 canonical 时间枚举**

在 `time_bucket()` 开头先解析 `annotation.attributes` 中精确的 `time_of_day:`：

```python
canonical = {
    "night": (5, "夜晚"),
    "dawn_dusk": (4, "晨昏"),
}
```

`day` 和 `unknown` 不强行映射到早晨/中午，继续依赖 caption 中更具体的时间证据；这避免把
宽泛白天字段伪装成精确时间。精确枚举处理后再执行原有文本匹配。

- [ ] **Step 6: 运行定向测试并确认通过**

Run: Task 6 Step 3 的命令。  
Expected: loader and all story planner tests pass。

- [ ] **Step 7: 提交 M7 canonical 读取能力**

```bash
git add src/anima_search/m7/canonical_annotations.py \
  src/anima_search/m7/story_planner.py \
  tests/unit/test_m7_canonical_annotations.py tests/unit/test_story_planner.py
git commit -m "feat: load canonical annotations for M7"
```

---

### Task 7: M6→M7 选择桥接、真实数据验收与报告

**Files:**
- Create: `src/anima_search/m7/m6_bridge.py`
- Create: `scripts/run_m7_from_m6.py`
- Create: `tests/unit/test_m7_m6_bridge.py`
- Create: `docs/M6_M7_POST_ANNOTATION_INTEGRATION_2026-08-14.md`

**Interfaces:**
- Consumes: `M6QueryResult` JSONL、canonical 标注、manifest、现有 M7Service。
- Produces: 3--8 张按 M6 顺序选择的 `SearchResult` 和 `VisualStory` JSON。

- [ ] **Step 1: 写 query 选择和 3--8 张边界的失败测试**

固定接口：

```python
def load_m6_query(path: Path, query_id: str) -> M6QueryResult: ...

def select_story_candidates(
    result: M6QueryResult,
    count: int,
) -> list[SearchResult]: ...
```

测试断言找到唯一 query、保持 M6 rerank 顺序、保留相对路径，并对 count=2/9、重复 query ID、
缺失 query ID 抛出明确 ValueError。

- [ ] **Step 2: 运行测试并确认桥接模块不存在**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course \
  python -m pytest -q tests/unit/test_m7_m6_bridge.py
```

Expected: import error for `anima_search.m7.m6_bridge`。

- [ ] **Step 3: 实现桥接和 M7 CLI**

`run_m7_from_m6.py` 参数固定为：

```text
--m6-results --query-id --select-count --annotations
--train-manifest --val-manifest --config --output --fill-gaps
```

CLI 读取指定 M6 query，选择 3--8 张，加载仅对应 split 的 canonical 标注，构造现有
`M7Service`。先调用 `create_story`；`--fill-gaps` 为真时才调用现有缺图生成桥接。输出写入新
JSON，包含 ordered IDs、sections、gaps、每个 gap 的 `source=generated` 和
`ai_generated=true`，禁止覆盖 M6 文件。

- [ ] **Step 4: 运行桥接测试并确认通过**

Run: Task 7 Step 2 的命令。  
Expected: all bridge tests pass。

- [ ] **Step 5: 用真实 Qwen3.5 Val 标注做只读覆盖验收**

在仓库根目录运行：

```bash
PYTHONPATH=src conda run -n vlm-course python -c '
from pathlib import Path
from anima_search.m7.canonical_annotations import load_canonical_m7_annotations
items = load_canonical_m7_annotations(
    Path("../M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl"),
    [Path("artifacts/manifests/train.jsonl"), Path("artifacts/manifests/val.jsonl")],
    split="Val",
)
assert len(items) == 369
print({"val_annotations": len(items)})
'
```

Expected: `{'val_annotations': 369}`，且不产生仓库文件。

- [ ] **Step 6: 在队友交付 M5 文件前完成 dry-run；交付后执行真实 M6/M7**

交付前：

```bash
conda run -n vlm-course python scripts/run_m6_from_m5.py \
  --input artifacts/evaluation/m5_to_m6_candidates.jsonl \
  --output artifacts/evaluation/m6_dry_run.jsonl \
  --validation-report artifacts/evaluation/m5_to_m6_validation.json \
  --config configs/benchmark_8gb.yaml \
  --index-manifest artifacts/indexes/val/manifest.json \
  --train-dir ../Train --val-dir ../Val --method listwise --dry-run
```

如果输入尚未交付，记录为外部依赖，不创建伪造的正式结果。交付后去掉 `--dry-run`，先加
`--query-limit 1` 运行单查询，再运行约定查询集。随后用 `run_m7_from_m6.py` 选择 3 张做
故事联调；补图默认关闭，确认故事后再显式加 `--fill-gaps`。

- [ ] **Step 7: 写阶段报告并明确证据边界**

`docs/M6_M7_POST_ANNOTATION_INTEGRATION_2026-08-14.md` 必须记录：

- 接口文件校验摘要和 hash；
- M6 查询数、Top-20、模型调用数、延迟、峰值显存、硬失败和部分降级；
- M7 canonical Val 覆盖、有效时间桶数量、故事 ordered IDs、gap 数和 AI 标识；
- 使用的 commit/config/model；
- M5 文件未交付时哪些步骤只完成代码/测试；
- 没有 qrels 时不声明 pointwise/listwise 质量优劣。

- [ ] **Step 8: 运行完整回归和静态检查**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n vlm-course python -m pytest -q
conda run -n vlm-course python -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass，compileall 与 diff check 退出码为 0。

- [ ] **Step 9: 提交 M6→M7 桥接和报告**

```bash
git add src/anima_search/m7/m6_bridge.py scripts/run_m7_from_m6.py \
  tests/unit/test_m7_m6_bridge.py \
  docs/M6_M7_POST_ANNOTATION_INTEGRATION_2026-08-14.md
git commit -m "feat: integrate validated M6 results with M7"
```

---

## Final Verification Checklist

- [ ] `git diff --name-only <base>..HEAD` 不包含 M1--M5 实现文件。
- [ ] M5 输入文件 hash 在 M6 前后相同。
- [ ] 合法 Top-20 完整映射；非法输入严格阻断。
- [ ] M6 输出每个输入 ID 恰好一次，M5 原始证据未改变。
- [ ] M6 硬失败与部分降级可区分。
- [ ] M7 canonical loader 对 Val 返回 369 条且哈希一致。
- [ ] M7 故事仅选择 3--8 张，保持 M6 选择顺序作为输入。
- [ ] 生成图始终包含 `source=generated`、`ai_generated=true`。
- [ ] 无人工 qrels 时报告不包含排序质量提升声明。
- [ ] 完整测试、compileall、`git diff --check` 通过。
