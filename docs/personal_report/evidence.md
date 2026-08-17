# 个人报告证据台账

本文件用于约束个人报告的贡献归属与实验数字。它不是团队成果清单，也不将共同工作
记为个人独立成果。

## 队友已有基线

| 声明 | 归属 | 证据 | 报告允许措辞 |
|---|---|---|---|
| 原始 M3--M5 多模态检索流水线 | 队友 | `222f232` | 队友已经提交多路索引、检索与 RRF 基线；本人以此为开发起点。 |

## 本人扩展提交

| 提交 | 本人扩展内容 | 证据文件 | 报告允许措辞 |
|---|---|---|---|
| `4457459` | 补全无标注开发模式、image-only、评测门禁和阶段报告 | `LOCAL_NO_ANNOTATION_PLAN.md`、`src/anima_search/app/mock_service.py`、`src/anima_search/indexing/image_only.py` | 在队友基线上补全无标注条件下的开发与验证路径。 |
| `9c5d54e` | 扩展 M0 路由、人工评测工具、M6 多查询基准和 M7 UI 桥接 | `src/anima_search/evaluation/manual_set.py`、`src/anima_search/routing/scene_router.py` | 扩展评测准备、路由和交互联调能力。 |
| `33551fe` | 补全任意图片目录的一键流水线与录屏指南 | `run.py`、`src/anima_search/pipeline/directory.py` | 补全隐藏测试集与新目录的可复现入口。 |
| `6550040` | 扩展 M5 归一化加权融合并与 RRF 对照 | `src/anima_search/retrieval/fusion.py`、`scripts/compare_fusion_methods.py` | 在原有 RRF 基线上加入可配置加权融合和工程对照。 |
| `c4d329c` | 补全 M6 Top-20 listwise 重排、降级逻辑和 8GB 基准 | `src/anima_search/retrieval/listwise_reranker.py`、`scripts/benchmark_listwise_top20.py` | 补全 listwise 重排并验证本地 8GB 可行性。 |
| `0c54bf8` | 扩展 M7 自动故事排序、缺图检测、补图与 AI 标识 | `src/anima_search/m7/story_planner.py`、`src/anima_search/app/ui.py` | 扩展 M7 故事与生成链路并完成真实联调。 |
| `223f3e0` | 补全 M4 rules/local Qwen/API 三后端与回退 | `src/anima_search/retrieval/openai_compatible.py`、`src/anima_search/retrieval/query_parser.py` | 补全查询理解后端并验证结构化输出契约。 |
| `a8f3a05` | 扩展 A7 jina-clip-v2、RoPE 修复、NaN 防护与资源对比 | `src/anima_search/indexing/image_vector_index.py`、`scripts/benchmark_image_encoders.py` | 扩展第二图像编码器并修复非有限向量问题。 |
| `bfc2c1c` | 迁移队友 Qwen3.5 canonical adapter 与只读导入 CLI | `src/anima_search/adapters/annotation.py`、`scripts/import_m1_qwen35.py` | 在队友标注和 M3--M5 代码基础上完成选择性迁移与接口联调；不声明标注生产归属。 |
| `fc4e7c6` | 补全 Qwen3.5 全量三路索引一致性与 manifest 验证 | `scripts/build_indexes.py`、`src/anima_search/indexing/index_manifest.py` | 验证 Train 1993、Val 369 的 image/text/BM25 全量索引。 |
| `e7f40da`--`557216b` | 规定并验证 M5→M6 Top-20 只读接口，补全离线 listwise 与资源指标 | `docs/M5_TO_M6_INTERFACE_V1.md`、`scripts/run_m6_from_m5.py` | 完成接口、异常门禁和 M6 联调，不把 M5 原始实现计为个人工作。 |
| `94250c1`、`82478ca`、`9451b78` | 补全 M7 canonical 映射、M6→M7 桥接与三个真实故事 | `src/anima_search/m7/canonical_annotations.py`、`scripts/run_m7_from_m6.py` | 完成 M6/M7 联调、故事排序、补图和 AI provenance 验证。 |
| `c64a958` | 修复正式 qrels 中零等级判断被丢弃的问题 | `src/anima_search/evaluation/ground_truth.py` | 保留人工明确判断的 0 等级，避免只剩正例。 |
| `6deb435` | 合并两批已审核查询为正式 100 条集合 | `evaluation/formal_val_100/` | 完成个人审核集整理；来源图正例不替代候选级 qrels。 |
| `7401f86` | 构建五类均衡、五个 A5 变体的 50 查询候选池 | `scripts/build_relevance_pool.py` | 准备候选级人工审核输入，不自动推断相关等级。 |
| `d5422a1` | 增加候选级 0/1/2 图片审核页面 | `scripts/launch_candidate_annotator.py` | 补全人工审核工具与逐图完成门禁。 |
| `ded5d2a` | 增加 qrels 冻结与完整性验证 | `scripts/finalize_formal_qrels.py` | 只有 50 个 graded query 全部完成才允许正式实验。 |
| `a784a5e` | 补全 A5 五变体正式结果与来源哈希输出 | `scripts/run_ablation.py` | 代码与 dry-run 就绪；候选 qrels 未冻结时不产出最终表。 |
| `6b2401a` | 补全 A6 固定候选 baseline/pointwise/listwise 质量接口 | `scripts/benchmark_listwise_top20.py` | 单查询 source-positive smoke 仅验证指标链路，不能替代 50 查询 graded 结论。 |
| `c9873fb`、`260fb4c` | 补全部署契约与最终 Demo 手册 | `docs/DEPLOY.md`、`docs/FINAL_DEMO_RUNBOOK.md` | 完成可复现部署和演示准备，不改变队友模块归属。 |

## 可引用的实测数字

| 模块 | 测试范围 | 数值 | 结论边界 |
|---|---|---|---|
| M4 | 本地 Qwen，3 类查询 | 3/3 完成；首条冷启动 10.085 s；后两条热调用均值约 2.697 s | 接口与结构化输出联调，不是查询理解准确率。 |
| M5 | 20 图、17 条有效结构化标注、12 查询、Top-8 | 重合率 92.71%；共同结果平均位次变化 0.446；RRF 10.58 ms；weighted 6.62 ms | 小样工程对照，不是检索质量证据。 |
| M6 | 3 查询、Top-20 | pointwise 44.377 s/query、4.039 GiB；listwise 9.143 s/query、4.092 GiB；1/3 部分降级 | 资源与可靠性证据；没有 relevance，不能比较排序质量。 |
| M7 | 本地 Qwen3-VL-2B + SD 1.5 FP16 | 故事链路和补图真实运行；生成图带 AI 标识 | 证明功能可运行，不证明故事质量或风格一致性。 |
| A7 Chinese-CLIP | 64 图、512 维 | 建库 6.060 s；峰值 0.394 GiB；热查询 2.251 ms | 资源证据，不是质量证据。 |
| A7 jina-clip-v2 | 64 图、512 维 | 建库 10.510 s；峰值 2.577 GiB；热查询 47.701 ms | 资源证据，不是质量证据。 |
| Qwen3.5 上游输入 | 队友交付 canonical v1.3 | 2362 条；Train 1993、Val 369 可导入 | 标注生产归属队友；本人只做只读验收、迁移和联调。 |
| M3 全量索引 | Qwen3.5 canonical v1.3 | Train 1993、Val 369；image/text/BM25 三分支均为 512/512/sparse | 证明全量链路已建成，不等于检索质量。 |
| M5→M6 | 12 查询、每条 Top-20 | 240 候选，接口 valid=true；listwise 110.968 s，4.10 GiB；8/12 部分降级，0 硬失败 | 资源与稳定性证据，必须披露 66.7% 部分降级。 |
| M7 正式故事 | 3 查询、每个 5 图 | 3 个故事；q03 的 1 个 gap 真实生成，含双重 AI 标识 | 功能与 provenance 证据，不是故事质量分数。 |
| 正式查询准备 | 100 条来源正例 + 50 条均衡候选池 | 五类为 28/26/14/14/18；候选池五类各 10，平均 10.8 候选；`candidate_relevance.csv` 已开始写入人工 0/1/2 判断 | 候选级审核正在进行；必须完成 50 个 `graded_query_ids` 并通过完整性校验后，才能冻结 qrels。 |
| A6 固定候选 smoke | q001、Top-20、1 repeat | pointwise 37.800 s/4.04 GiB；listwise 9.158 s/4.09 GiB；4.128×；均无失败/降级 | 仅来源图正例的一查询链路检查；MRR/nDCG 数字不是最终质量结论。 |
| 回归测试 | 当前完整仓库 | 248 passed | 证明当前自动化回归通过，不替代正式数据实验。 |

## 共同依赖与禁止声明

- M1 Qwen3.5 全量结构化标注由队友生产；本人只做只读验收、adapter 迁移与下游联调。
- 人工 query、relevance、参考描述和 Arena 投票由团队共同承担。
- Train 1993、Val 369 的 M3 三路索引已经执行并验证，但原始 M3--M5 基线仍归属队友。
- 100 条查询和来源图正例已审核；50 查询候选池的候选级审核已经开始，但尚未逐图完成和冻结。
- A5/A6 的最终 Recall、MRR、mAP、nDCG 尚未产生；q001 source-positive smoke 不得作为最终表。
- 禁止用 mock、image-only、20 图或 64 图资源实验声明检索质量优劣。

## 提交到报告章节映射

| 报告章节 | 主要证据 |
|---|---|
| 个人边界与基线 | `222f232`、`4457459` |
| 无标注与任意目录流程 | `4457459`、`33551fe` |
| M5 | `6550040`、`docs/M5_FUSION_COMPARISON_2026-08-11.md` |
| M6 | `c4d329c`、`docs/M6_LISTWISE_TOP20_2026-08-11.md` |
| M7 | `0c54bf8`、`docs/M7_AUTO_STORY_UI_2026-08-11.md` |
| M4 | `223f3e0`、`docs/M4_QUERY_BACKENDS_2026-08-11.md` |
| A7 | `a8f3a05`、`docs/A7_JINA_CLIP_COMPARISON_2026-08-11.md` |
| 标注到达后的迁移联调 | `bfc2c1c`、`fc4e7c6`、`docs/M3_M5_QWEN35_INTEGRATION.md` |
| M5→M6 与三个 M7 故事 | `e7f40da`--`9451b78`、`docs/M6_M7_POST_ANNOTATION_INTEGRATION_2026-08-15.md` |
| 正式评测准备 | `c64a958`、`6deb435`、`7401f86`、`d5422a1`、`ded5d2a` |
| A5/A6 正式接口与 smoke | `a784a5e`、`6b2401a`；正式表等待候选 qrels |
| 部署与 Demo | `c9873fb`、`260fb4c` |
