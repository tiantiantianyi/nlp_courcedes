# 团队报告最终评测结果补充

候选级人工 qrels 已完成冻结，正式结果可替换团队报告中“等待 qrels”的状态描述。

## qrels

- 100 条查询，其中 50 条为五类均衡 graded queries；
- 540 条候选判断；
- 等级分布：0=204、1=202、2=184；
- `qrels_validation.json`：`valid=true`；
- 单审核者：张添翼。

## A5：混合召回

| 方法 | Recall@5 | Recall@10 | MRR | mAP | nDCG@10 | graded nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP-only | 0.720 | 0.783 | 0.858 | 0.704 | 0.784 | 0.795 |
| text-only | 0.721 | 0.774 | 0.840 | 0.677 | 0.762 | 0.729 |
| BM25-only | 0.633 | 0.691 | 0.816 | 0.623 | 0.711 | 0.614 |
| RRF 三路 | 0.733 | 0.804 | 0.858 | 0.707 | 0.788 | 0.756 |
| weighted 三路 | **0.754** | **0.842** | **0.870** | **0.744** | **0.820** | **0.817** |

在当前 qrels 和固定配置上，weighted 三路融合优于 RRF；该结论不应泛化为未公开测试集上的必然优势。

## A6：固定候选池重排

候选池共 50 条查询、540 个候选，候选数量为 1–20 不等长集合。

| 方法 | MRR | nDCG@10 | 失败/降级 | 资源 |
|---|---:|---:|---|---|
| M5 baseline | 1.000 | 0.953 | — | 固定候选原序 |
| pointwise | 0.987 | 0.923 | 2/540 失败 | 平均 2.395 s/候选；峰值约 4.06 GiB |
| listwise | 0.982 | 0.903 | 0/50 硬失败；13/50 部分降级 | 平均 5.744 s/查询；峰值约 4.10 GiB |

pointwise 总耗时为 1293.217 s，listwise 总耗时为 287.192 s，listwise 相对 pointwise 加速 4.503 倍。当前实验不能声称 VLM 重排提升了排序质量；更准确的结论是 listwise 显著减少了调用成本，但仍需解决重复/遗漏 ID 的部分降级问题。

## 报告图表更新

正式结果应新增：

1. A5 五方法 Recall@5/10、MRR、nDCG@10 分组柱状图；
2. A6 三方法 MRR/nDCG@10 柱状图，并用阴影标注“单审核者 qrels”；
3. A6 pointwise/listwise 总耗时与平均耗时对照图；
4. listwise 13/50 部分降级类型统计图；
5. 五类查询的 graded nDCG@10 分类柱状图。

原始证据文件：

- `artifacts/evaluation/formal/qrels_validation.json`；
- `artifacts/evaluation/formal/a5/a5_formal_results.json`；
- `artifacts/evaluation/formal/a6/formal_candidate_pool.json`；
- `artifacts/evaluation/formal/a6/formal_candidate_pool_quality.json`。
