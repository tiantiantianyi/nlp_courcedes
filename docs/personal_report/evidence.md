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

## 可引用的实测数字

| 模块 | 测试范围 | 数值 | 结论边界 |
|---|---|---|---|
| M4 | 本地 Qwen，3 类查询 | 3/3 完成；首条冷启动 10.085 s；后两条热调用均值约 2.697 s | 接口与结构化输出联调，不是查询理解准确率。 |
| M5 | 20 图、17 条有效结构化标注、12 查询、Top-8 | 重合率 92.71%；共同结果平均位次变化 0.446；RRF 10.58 ms；weighted 6.62 ms | 小样工程对照，不是检索质量证据。 |
| M6 | 3 查询、Top-20 | pointwise 44.377 s/query、4.039 GiB；listwise 9.143 s/query、4.092 GiB；1/3 部分降级 | 资源与可靠性证据；没有 relevance，不能比较排序质量。 |
| M7 | 本地 Qwen3-VL-2B + SD 1.5 FP16 | 故事链路和补图真实运行；生成图带 AI 标识 | 证明功能可运行，不证明故事质量或风格一致性。 |
| A7 Chinese-CLIP | 64 图、512 维 | 建库 6.060 s；峰值 0.394 GiB；热查询 2.251 ms | 资源证据，不是质量证据。 |
| A7 jina-clip-v2 | 64 图、512 维 | 建库 10.510 s；峰值 2.577 GiB；热查询 47.701 ms | 资源证据，不是质量证据。 |
| 回归测试 | A7 提交时完整仓库 | 122 passed | 证明当时自动化测试通过，不代表正式数据实验完成。 |

## 共同依赖与禁止声明

- M1 全量结构化标注由队友生产，不能写成本人完成。
- 人工 query、relevance、参考描述和 Arena 投票由团队共同承担。
- 标注冻结后的全量 M3 三路索引尚未执行。
- A5--A7、M5/M6 的 Recall、MRR、mAP、nDCG 正式结果尚未产生。
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
