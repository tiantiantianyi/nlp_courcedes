# M3-M5 Qwen3.5 标注接入与交付

日期：2026-08-15

范围：只修改 M3-M5。M6/M7 代码和运行行为保持不变，联通面是 M5 输出的
`m5-to-m6-v1.0` Top-20 JSONL。

## 1. 数据源与覆盖

主标注固定为：

```text
../M1_results_package/annotations/M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl
```

真实导入结果：

| 项目 | 数量 |
|---|---:|
| 图片 manifest | 2369 |
| Qwen3.5 合法标注 | 2362 |
| Train 导入 | 1993 |
| Val 导入 | 369 |
| 转换失败 | 0 |

缺失 Train 数字 ID：`48`、`649`、`764`、`899`、`1155`、`1217`、`1918`。
这些记录不会自动填空，也不会回退到其他模型。

## 2. Schema 适配

`adapt_annotation()` 将 canonical v1.3 映射到 M3 的 `ImageAnnotation`：

- `captions.dense_zh` 进入摘要和生成上下文；
- `entities` 进入对象、精确数量、动作、颜色、材质和状态；
- `scene` 与 `capture_visual` 进入场景、时间、天气、光照和视角；
- `ocr`、`relations`、`event`、`subjective` 和 `uncertainties` 分别保留；
- 英文枚举保留为 `*_code`，同时生成中文检索值，例如
  `night -> 夜晚`、`rain -> 雨天`、`street_urban -> 城市街道`；
- manifest 提供规范 `train-/val-` ID、split、POSIX 相对路径和图片哈希；
- `processed_sha256` 与 manifest 不一致时阻断导入。

## 3. M3 构建

```bash
python scripts/build_manifest.py --config configs/default.yaml
python scripts/import_m1_qwen35.py --config configs/default.yaml
python scripts/build_indexes.py --config configs/default.yaml --split Train --branches image,text,bm25
python scripts/build_indexes.py --config configs/default.yaml --split Val --branches image,text,bm25
```

新索引 manifest 除原有记录数、标注版本、模型摘要和配置摘要外，还包含
`image_records`，用于核验每个 `image_id` 的 `relative_path` 和图片 SHA-256。

`run_m3_m5_pipeline.py` 在配置了 `annotation.source` 时自动执行上述导入，不再重新调用
旧标注模型。`annotation.allow_missing=true` 只允许上游缺失；重复 ID、额外 ID、路径、
哈希或版本错误仍会阻断。

## 4. M4-M5 检索优化

Qwen3.5 时间/天气枚举已经转换为查询解析器使用的中文规范词，避免 hybrid 模式把正确
候选误判为不匹配。数量只使用 `count_exact=true` 的实体计数；不可靠数量不会被伪造成
精确过滤依据。

M5 继续保留每个候选的原始 `branch_scores`、过滤后 `branch_ranks`、`fused_score` 和
`matched_fields`。RRF 与归一化加权融合均保持现有实现。

## 5. Top-20 导出

```bash
python scripts/export_m5_candidates.py \
  --queries configs/m6_benchmark_queries.jsonl \
  --config configs/default.yaml \
  --split val \
  --output artifacts/evaluation/m5_to_m6_candidates.jsonl
```

导出器保证：

- schema 固定为 `m5-to-m6-v1.0`；
- 一个查询一行，查询 ID 唯一；
- 每行恰好 20 个唯一候选，M5 名次连续为 1-20；
- 分数必须有限，分支只能是 `image/text/bm25`；
- `branch_scores` 和 `branch_ranks` 键集合完全一致；
- 不包含 `rerank_score` 或其他 M6 字段；
- index manifest 和有效检索配置快照的 SHA-256 写入每条记录；
- 原子写入新文件，不覆盖或修改上游标注。

正式导出默认使用配置中的三路索引。工程冒烟可显式指定：

```bash
python scripts/export_m5_candidates.py \
  --queries configs/m6_benchmark_queries.jsonl \
  --split val --branches bm25 \
  --output artifacts/evaluation/m5_to_m6_candidates.bm25-smoke.jsonl
```

## 6. 已执行验收

- Qwen3.5 真实导入：2362 条成功，0 条转换失败；
- Train 三路真实建库：1993 条，`image/text/bm25` 均成功；
- Val 三路真实建库：369 条，`image/text/bm25` 均成功；
- image 与 text 索引均为 512 维，两个 split 的 FAISS 记录数均与 manifest 一致；
- 12 个查询的正式 M5 交付：12 行、240 个 Top-20 候选；
- 严格接口校验：0 个错误，240 个候选引用的 130 张唯一图片全部可解码；
- 真实 Qwen3-VL 2B listwise Top-20 冒烟：20 个 ID 完整且唯一，无降级；
- Train、Val 端到端三路检索验证均通过；
- 自动化测试：`128 passed`；`compileall` 通过。

上述验收证明三路工程链路和 M5--M6 接口达到交付要求。由于人工审核的 relevance
judgments 仍未完成，本结果不声明 Recall、nDCG 或排序质量相对基线有统计提升。
