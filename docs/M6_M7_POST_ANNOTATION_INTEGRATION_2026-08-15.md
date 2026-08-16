# M6/M7 标注到达后联调阶段报告（2026-08-15）

## 1. 范围与个人工作边界

本阶段只完善本人负责的 M6、M7 联调，不修改 M1--M5 的标注、建库、检索或融合实现。
Qwen3.5-9B 是本阶段读取的标注来源；M6/M7 运行时使用的视觉语言模型仍是项目配置中的
Qwen3-VL-2B-Instruct。两者用途不同：

- Qwen3.5-9B canonical v1.3：提供已有图片的结构化标注；
- Qwen3-VL-2B-Instruct：执行 M6 视觉重排和 M7 图片证据提取/故事生成；
- Stable Diffusion：仅在 M7 显式启用缺图补全时加载。

本阶段没有把队友完成的 M1--M5 实现或标注生产工作记为个人成果。

## 2. 已完成内容

### 2.1 M5→M6 严格接口

按照 docs/M5_TO_M6_INTERFACE_V1.md 实现：

- m5-to-m6-v1.0 Top-20 Pydantic 契约；
- 一查询一行、恰好 20 个唯一候选、rank 1--20；
- branch_scores/branch_ranks 必填且键集合一致；
- JSON、跨行 query ID、manifest、候选路径、split 边界和图片解码完整扫描；
- 任意接口错误都会阻断整批数据，不覆盖 M5 输入；
- 独立校验 CLI 和 JSON 验收报告。

### 2.2 M6 离线重排

- 新增 m6-rerank-v1.0 输出模型，保留 M5 顶层字段和候选原字段；
- 正常输出按 Qwen3-VL 重排顺序写入 rerank_rank；
- 重复 ID 去重，遗漏 ID 按 M5 原顺序补回并记入 mismatch；
- 未知 ID、空/非法结果或异常触发完整 M5 顺序硬回退；
- dry-run 明确写入 degraded=true 和 dry-run: Qwen3-VL was not invoked；
- 离线 CLI 直接构造 Qwen client，不加载 CLIP、BM25 或 M3--M5 索引。

### 2.3 M7 canonical v1.3 只读映射

新增仅供 M7 使用的映射，不修改通用 adapter，也不产出 M3--M5 建库标注：

- dense/short/event caption → summary 与查询种子；
- scene、entities、可靠计数、action、capture、state/material；
- palette/entity colors、mood、OCR、uncertainty；
- manifest image ID、路径和图片 SHA-256 严格绑定；
- Qwen3.5 repairs 元数据保存在 generation_parameters；
- dawn_dusk 精确映射为“晨昏”，不再因为包含 dawn 被误判为“黎明”；
- day 不伪装成早晨或中午，缺乏更具体证据时保留为未知时间桶。

### 2.4 M6→M7 故事桥接

- 从 M6 JSONL 中严格选择唯一 query；
- 只允许选择 3--8 张图片，并保持 M6 候选顺序作为故事输入；
- M7 根据 canonical 时间和场景特征执行故事内自动排序；
- 输出 ordered image IDs、sections 和 gaps；
- 生成 gap 固定带有 source=generated 与 ai_generated=true；
- M7 CLI 不创建检索索引或加载检索编码器；缺图补全默认关闭。

## 3. 数据验收证据

### 3.1 Qwen3.5 canonical v1.3

| 项目 | 结果 |
|---|---:|
| canonical 文件 SHA-256 | 167e6b890d802bcaad7f9e76dae171f4a1981c8d6b01abeb0c9a6e8d622fb341 |
| Val manifest SHA-256 | 65d4c5e4aebef734303fc60ba963d029651f877c68569e8929d19408a546fb68 |
| Val index manifest SHA-256 | 5af8ac9b3d183889a22876d9d0810842be42e61e6fe0ed368cd83ea88ceb7490 |
| Val manifest 记录 | 369 |
| canonical 成功映射 | 369 |
| 图片 SHA-256 错配 | 0 |

真实 Val 映射后的精确时间桶统计：

| 时间桶 | 数量 |
|---|---:|
| 夜晚 | 51 |
| 晨昏 | 21 |
| 未知/宽泛白天 | 297 |

“未知/宽泛白天”不是标注丢失：M7 有意不把 canonical 的 day 强制解释成早晨或中午。

### 3.2 自动化测试

| 测试范围 | 结果 |
|---|---:|
| 开发前完整基线 | 122 passed |
| M6 契约、校验、runner、CLI 定向回归 | 20 passed |
| M7 loader、故事、桥接、CLI 定向回归 | 24 passed |
| 实现后完整回归 | 153 passed |
| 真实 Qwen3.5 Val 只读覆盖 | 369/369 |

关键实现提交：

- 93767e2：M6/M7 标注到达后实施计划；
- 4f7d76d：M5→M6 严格契约；
- e977317、96810d8：完整扫描校验器与 CLI；
- 847b299、54f544a：M6 runner 与离线 CLI；
- 94250c1：M7 canonical loader 与时间桶修复；
- 82478ca：M6→M7 选择桥接和故事 CLI。

## 4. 正式 M6/M7 实验状态

截至 2026-08-15，artifacts/evaluation 中不存在双方约定名称和 schema 的
m5_to_m6_candidates.jsonl。目录中的 m5_fusion_20.json 与早期 M6 JSON 文件不是
m5-to-m6-v1.0 正式交付，因此没有将其静默转换或冒充正式输入。

因此当前正式实验状态为：

| 项目 | 当前状态 |
|---|---|
| 正式 M5 接口查询数 | 0（等待 Top-20 JSONL 交付） |
| 正式 M6 模型调用数 | 0 |
| 正式 M6 延迟/峰值显存 | 未运行，不能报告 |
| 正式 M6 硬失败/部分降级 | 代码和测试已覆盖，真实数据待交付 |
| 正式 M7 ordered IDs/gap 数 | 等待正式 M6 输出 |
| 正式缺图生成 | 默认关闭，等待故事确认后再显式运行 |

没有人工 qrels/relevance judgments，现阶段不声明 pointwise 或 listwise 带来排序质量提升。

## 5. M5 Top-20 到达后的验收流程

### 5.1 校验接口

~~~bash
conda run -n vlm-course python scripts/validate_m5_m6_interface.py \
  --input artifacts/evaluation/m5_to_m6_candidates.jsonl \
  --m5-config-snapshot artifacts/evaluation/m5_retrieval_config.snapshot.json \
  --project-root . \
  --train-dir ../Train \
  --val-dir ../Val \
  --index-manifest artifacts/indexes/val/manifest.json \
  --report artifacts/evaluation/m5_to_m6_validation.json
~~~

只有报告 valid=true、错误数为 0 时才能继续。

### 5.2 无模型 dry-run

~~~bash
conda run -n vlm-course python scripts/run_m6_from_m5.py \
  --input artifacts/evaluation/m5_to_m6_candidates.jsonl \
  --m5-config-snapshot artifacts/evaluation/m5_retrieval_config.snapshot.json \
  --output artifacts/evaluation/m6_dry_run.jsonl \
  --validation-report artifacts/evaluation/m5_to_m6_validation.json \
  --config configs/benchmark_8gb.yaml \
  --index-manifest artifacts/indexes/val/manifest.json \
  --train-dir ../Train --val-dir ../Val \
  --method listwise --dry-run
~~~

dry-run 只验证读取、Top-20 和输出契约，所有结果必须明确标为 degraded。

### 5.3 单查询真实 M6 冒烟

~~~bash
conda run -n vlm-course python scripts/run_m6_from_m5.py \
  --input artifacts/evaluation/m5_to_m6_candidates.jsonl \
  --m5-config-snapshot artifacts/evaluation/m5_retrieval_config.snapshot.json \
  --output artifacts/evaluation/m6_reranked_results.jsonl \
  --validation-report artifacts/evaluation/m5_to_m6_validation.json \
  --config configs/benchmark_8gb.yaml \
  --index-manifest artifacts/indexes/val/manifest.json \
  --train-dir ../Train --val-dir ../Val \
  --method listwise --query-limit 1
~~~

单查询通过后再运行约定查询集，并记录查询数、模型调用数、每查询延迟、峰值显存、
硬失败和部分降级。

### 5.4 M7 故事联调

~~~bash
conda run -n vlm-course python scripts/run_m7_from_m6.py \
  --m6-results artifacts/evaluation/m6_reranked_results.jsonl \
  --query-id '<正式 query_id>' \
  --select-count 3 \
  --annotations ../M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl \
  --train-manifest artifacts/manifests/train.jsonl \
  --val-manifest artifacts/manifests/val.jsonl \
  --config configs/benchmark_8gb.yaml \
  --output artifacts/evaluation/m7_story.json
~~~

先检查自动排序、sections、gaps 和事实边界；确认故事后才追加 --fill-gaps。生成图必须保留
AI 标识，不能与真实数据集图片混淆。

## 6. 下一验收点

队友需要交付：

- UTF-8 m5-to-m6-v1.0 JSONL；
- 每查询恰好 20 个唯一候选；
- 与候选对应的 Val index manifest 和检索配置 digest；
- 双方约定的 query 总数和 query ID 清单。

交付后依次执行接口校验、dry-run、单查询真实 M6、约定查询集 M6、三图 M7 故事、
可选缺图补全，并把实测资源和输出证据追加到本报告。

## 7. 2026-08-16 正式运行追加记录

本节是第 4--6 节所述等待项到达后的追加记录，不改写 2026-08-15 当时的历史状态。
正式 M5 输入、M6 结果、运行指标和三个 M7 故事均位于本地 `artifacts/`；原始图片、
模型、索引和生成图片不提交 GitHub。

### 7.1 M5→M6 与正式 listwise 运行

| 项目 | 实测结果 |
|---|---:|
| M5 接口查询数 | 12 |
| M5 候选数 | 240（每查询 Top-20） |
| 接口问题数 | 0，`valid=true` |
| M6 方法 | Qwen3-VL-2B-Instruct listwise |
| M6 输出 | 12 行；每行 20 个唯一候选 |
| 总耗时 | 110.968 秒 |
| 平均每查询耗时 | 9.247 秒 |
| 峰值 CUDA 显存 | 4,405,989,376 B（约 4.10 GiB） |
| 无降级结果 | 4/12 |
| 部分降级结果 | 8/12（66.7%） |
| 硬失败/完整 M5 回退 | 0/12 |

8 条部分降级均来自模型 listwise 输出中的重复或遗漏 ID。处理严格遵循
`docs/M5_TO_M6_INTERFACE_V1.md` 第 9 节：重复 ID 去重，遗漏 ID 按 M5 原始顺序补到
末尾，并写入 `degraded=true` 与具体 `mismatch`；没有未知 ID、空数组、无效 JSON 或
异常导致的整批硬回退。因部分降级率较高，报告必须同时给出该数值，不能描述成
“12 条全部无降级成功”。

证据文件：

- `artifacts/evaluation/m5_to_m6_candidates.jsonl`；
- `artifacts/evaluation/m6_validation_report.json`；
- `artifacts/evaluation/m6_rerank_results.jsonl`；
- `artifacts/evaluation/m6_runtime_metrics.json`。

### 7.2 三个 M7 正式故事

| Query | 选择图片数 | 故事内 gap | 补图结果 | 关键验收 |
|---|---:|---:|---|---|
| `m6-q01` | 5 | 2 | 未请求生成，保留 missing 占位 | sections 顺序与 ordered IDs 一致 |
| `m6-q03` | 5 | 1 | 1 张生成成功 | `source=generated`、`ai_generated=true` |
| `m6-q09` | 5 | 2 | 未请求生成，保留 missing 占位 | sections 顺序与 ordered IDs 一致 |

三个输出均为 `m7-story-v1.0`，selected IDs 与 ordered IDs 集合相同，且 section 顺序
严格等于 ordered IDs。`m6-q03` 的转场为 `val-2044 → val-2078`，生成图片为
`artifacts/generated/generated-20260816.png`（430,028 B）；故事 JSON 中保留
`status=generated`、`source=generated` 和 `ai_generated=true`，不会与真实 Val 图片混淆。

证据文件：

- `artifacts/evaluation/m7_stories/m6-q01.json`；
- `artifacts/evaluation/m7_stories/m6-q03.json`；
- `artifacts/evaluation/m7_stories/m6-q09.json`；
- `artifacts/generated/generated-20260816.png`。

### 7.3 当前可声明范围

本次运行完成了 M6 的 Top-20 listwise 资源与稳定性测试，以及 M7 的自动选图、故事内
排序、缺图检测、真实补图和 AI 来源标识。M6/M7 定向回归为 49 passed。

人工 relevance 仍在标注，因此这些结果不能用于声明 listwise 相对 M5 baseline 的 MRR、
NDCG@10 或检索质量提升。A5/A6 的质量对照必须等人工查询与相关性判断通过完整性校验后
再运行；该等待项不影响上述 M6/M7 工程联调与资源测试结论。
