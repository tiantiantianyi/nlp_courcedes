# 课程报告可视化与证据清单

本文档用于指导团队报告、张添翼个人报告和最终答辩 PPT 的图表准备。所有图表都必须标注数据范围、日期、硬件和“工程证据/正式质量指标”的边界，不能把 mock、image-only 或 source-positive smoke 当作正式检索质量。

## 一、建议报告中的核心图表

| 编号 | 图表 | 目的 | 数据来源/状态 | 放置位置 |
|---|---|---|---|---|
| Fig.1 | M0–M7 系统数据流图 | 说明顺序流水线、M5 并行 fan-out/fan-in、M4 路由 | 可依据技术方案绘制，已完成 | 团队报告 Method |
| Fig.2 | M0–M7 完成度矩阵 | 区分代码完成、资源验证、正式评测 | `docs/assets/stage_report/proposal_alignment.png`；需注明为阶段图 | 团队报告 Introduction/Discussion |
| Fig.3 | M3 三路索引结构图 | 解释 image/text/BM25 三分支如何进入 M5 | manifest 与 M5 文档 | 团队报告 Method |
| Fig.4 | M5 RRF 与 weighted 排名对照 | 说明两种融合输出大体重合但排序可变化 | `artifacts/evaluation/m5_fusion_20.json` | 团队报告 Experiment |
| Fig.5 | M6 pointwise/listwise 资源柱状图 | 展示 8GB 可行性与 listwise 调用次数下降 | `artifacts/evaluation/m6_runtime_metrics.json` 与 q001 smoke | 团队/个人报告 Results |
| Fig.6 | M6 降级类型堆叠柱状图 | 诚实呈现 12 条查询中 4 条无降级、8 条部分降级 | `artifacts/evaluation/m6_runtime_metrics.json` | 团队报告 Limitations |
| Fig.7 | M7 故事时间线 | 展示原图、missing gap、AI 生成图的顺序和 provenance | `artifacts/evaluation/m7_stories/m6-q01.json`, `m6-q03.json`, `m6-q09.json` | 团队/个人报告 M7 |
| Fig.8 | A7 编码器资源对照 | 比较 Chinese-CLIP 与 Jina 的冷启动、显存和查询时延 | `artifacts/a7_encoder_comparison_64.json` | 团队/个人报告 A7 |
| Fig.9 | 正式评测门禁流程图 | 解释 100 条来源正例、50 条候选池、0/1/2 qrels、A5/A6 | `evaluation/formal_val_100/README.md` | 团队报告 Experiment |
| Fig.10 | 查询类别分布图 | 展示 100 条查询覆盖 simple/compositional/negative/count/OCR | `artifacts/evaluation/formal_val_100/merge_report.json` | 团队报告 Dataset |

## 二、标注完成后必须新增的正式图表

在 `candidate_relevance.csv` 完成并通过 `finalize_formal_qrels.py` 后，优先生成以下图表：

1. 各类别的 0/1/2 标签分布堆叠柱状图；
2. 五种 M5 方法的 Recall@1、Recall@5、MRR、nDCG@10 分组柱状图；
3. M6 baseline、pointwise、listwise 的同一 qrels 指标对照；
4. 按查询类别拆分的 M6 nDCG@10 箱线图；
5. M6 每查询延迟与是否 degraded 的散点图；
6. A7 两种编码器在同一 qrels 下的质量—资源 Pareto 图；
7. 正式评测的失败/回退类型统计图。

这些图表必须使用同一批冻结 qrels，图注中写明：查询数量、候选数量、硬件、模型版本和 qrels SHA-256。若只完成部分查询，图表标题必须写“阶段性/非正式”，不能写“最终结果”。

## 三、建议加入报告的截图

- M7 页面首页：展示搜索框、结果图库和 M3–M7 功能标签；
- M7 故事页：展示自动排序后的时间线；
- 缺图补全：同时展示原图、missing gap 和 `AI 生成` 标签；
- M5→M6 接口验收 JSON：展示 `valid=true`、20 个唯一候选和 manifest 校验；
- 候选标注页面：展示来源图绿色边框和 0/1/2 输入格式；
- Docker/Gradio 启动成功页面：展示服务监听和可访问地址。

截图不得包含 API Key、绝对用户目录、未授权模型文件路径或其他隐私信息。候选标注截图只用于说明工具，不应展示尚未公开的完整数据集。

## 四、图表生成与检查命令

已有阶段图表：

```bash
conda run -n vlm-course python scripts/generate_stage_report_figures.py \
  --output-dir docs/assets/stage_report
```

生成报告前检查：

```bash
git diff --check
env -u ALL_PROXY -u all_proxy \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q
```

LaTeX 报告检查：

```bash
env -u ALL_PROXY -u all_proxy \
  conda run -n vlm-report tectonic \
  --keep-logs --keep-intermediates '张添翼_U202315231_个人报告.tex'
```

## 五、每张图的统一图注模板

> 图 X：……。数据范围为……，运行环境为 RTX 4060 Laptop 8GB，模型/配置为……。该图反映工程资源、接口稳定性或正式质量中的……；在人工 relevance judgments 尚未冻结时，不将其解释为检索质量提升。

## 六、贡献归属要求

团队报告可以展示整个系统的图表，但必须在正文或图注附近说明贡献边界：

- 队友：M0–M2、Qwen3.5 canonical 标注、原始 M3–M5 三路索引和 RRF 基线；
- 张添翼：基线迁移与一致性验收、M5 weighted 对照、M5→M6 接口、M6 listwise、M7 故事/补图、M4 后端、A7、评测门禁、部署与 Demo；
- 正式人工 qrels 和最终质量结果：团队共同实验，不应归为单个人的独立成果。
