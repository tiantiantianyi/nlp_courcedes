# 正式 100 条检索评测集

本目录由两份已审核的 50 条查询集合并生成，源文件保持不变。

## 内容

- `queries.jsonl`：100 条人工审核查询及其类别、来源图和审核信息。
- `relevance.csv`：当前每条查询的来源图 `relevance=2` 判断。
- `merge_report.json`：输入路径、SHA-256、类别数量和标注者汇总。

合并时按查询 ID 排序，并拒绝跨批次重复查询 ID、重复来源图 ID，以及指向未知查询的相关性记录。

## 来源与边界

- `q001`–`q050` 来自 `evaluation/manual_val_50/`。
- `q051`–`q100` 来自 `evaluation/manual_val_50_assisted/`；文本由 AI 辅助起草，张添翼已逐条观察原图并人工审核。
- 100 条记录当前均为 `reviewed=true`，标注者为张添翼。
- 每条查询至少保留一张人工确认的来源图正例（等级 2）。
- 当前文件可用于来源图单正例 Recall/MRR 评测，但还不能支持严格的多等级 nDCG 结论。

这些数据没有从检索分数自动推断相关性。正式多等级评测还需对平衡抽取的 50 条查询之候选池逐图标注 `0/1/2`。

## 复现

```bash
conda run -n vlm-course python scripts/prepare_formal_eval.py \
  --queries evaluation/manual_val_50/queries.jsonl evaluation/manual_val_50_assisted/queries.jsonl \
  --relevance evaluation/manual_val_50/relevance.csv evaluation/manual_val_50_assisted/relevance.csv \
  --output-dir evaluation/formal_val_100 --expected-count 100
```

## 校验

```bash
conda run -n vlm-course python scripts/validate_manual_eval_set.py \
  --queries evaluation/formal_val_100/queries.jsonl \
  --relevance evaluation/formal_val_100/relevance.csv --expected-count 100
```

2026-08-16 校验结果：100 条查询、100 条相关性记录、100 条均已审核。

## 后续

1. 从五类查询中平衡选取 50 条并运行 A5 各检索变体，构建去重候选池。
2. 对候选池中的每张图显式标注 `0/1/2`，不得用模型分数代替人工判断。
3. 合并候选级 qrels 后，分别报告全部 100 条的来源图指标和 50 条候选池的多等级 nDCG。
