# M5 融合方法工程对照

日期：2026-08-11
硬件：RTX 4060 Laptop 8GB
环境：Conda `vlm-course`

## 1. 目的与边界

本实验为 M5 增加“分支内 min-max 归一化后加权求和”，并与原有 RRF 做代码级、
真实索引级对照。当前没有经过人工审核的 relevance judgments，因此结果只用于验证：

- 两种融合方法都能在三路真实索引上运行；
- 排名是否发生变化；
- 查询延迟是否处于同一数量级；
- 分支原始分数、名次和降级信息是否保留。

本文不据此判断哪种方法的 Recall、MRR、mAP 或 nDCG 更高。

## 2. 实现

默认方法仍为 RRF，避免改变已有系统行为。配置可切换：

```yaml
retrieval:
  fusion_method: rrf  # rrf 或 weighted
  fusion_weights:
    image: 0.4
    text: 0.4
    bm25: 0.2
```

weighted 模式先对每个查询、每个检索分支的候选分数独立做 min-max 归一化，再按
有效分支权重加权。某个分支不可用时仅在仍然有效的分支间重新归一化权重。

A5 消融矩阵现在包含五项：CLIP-only、text-only、BM25-only、三路 RRF 和三路
归一化加权融合。正式质量指标仍受人工审核门禁保护。

## 3. 本地小样

- 输入：20 张 Val 图片；
- 有效结构化标注：17 条；
- 标注失败：3 条，原因为必填字段或有效 JSON 缺失，不是 OOM；
- 索引：image、text、BM25 三路均构建成功；
- 查询：12 条，覆盖 simple、compositional、negative、count、OCR；
- 返回数量：Top-8。

原始模型、索引和逐查询 JSON 位于 `artifacts/`，由 Git 忽略，不提交仓库。

## 4. 结果

| 指标 | 结果 |
|---|---:|
| 查询数 | 12 |
| 平均 Top-8 重合率 | 92.71% |
| 共同结果平均位次变化 | 0.446 |
| RRF 平均延迟 | 10.58 ms |
| weighted 平均延迟 | 6.62 ms |

12 条查询均成功返回结果。这个小样中两种方法的大部分 Top-8 候选相同，但内部顺序
有可观察变化。由于样本很小且计时未做多轮统计，延迟差值仅作为冒烟结果，不主张
weighted 一定比 RRF 更快。

## 5. 复现

已有结构化标注时构建三路索引：

```bash
conda run -n vlm-course python scripts/build_indexes.py \
  --config artifacts/directory_runs/m5-fusion-20/configs/runtime.yaml \
  --split Val --branches image,text,bm25
```

运行融合对照：

```bash
conda run -n vlm-course python scripts/compare_fusion_methods.py \
  --config artifacts/directory_runs/m5-fusion-20/configs/runtime.yaml \
  --queries configs/m6_benchmark_queries.jsonl \
  --top-k 8 \
  --output artifacts/evaluation/m5_fusion_20.json
```

质量检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q
conda run -n vlm-course python -m compileall -q src scripts tests
git diff --check
```

本次结果：`93 passed`，语法检查和差异检查通过。

## 6. 正式实验仍需完成

1. 接收冻结后的完整标注并重建三路索引；
2. 由组员人工审核查询和 relevance；
3. 使用相同候选池运行五组 A5；
4. 报告总体及分类 Recall@K、MRR、mAP、nDCG@10 和多轮延迟；
5. 不根据 Val 最终结果反复调权重。
