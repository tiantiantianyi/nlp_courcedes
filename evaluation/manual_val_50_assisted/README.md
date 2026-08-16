# 第二批 50 条人工审核数据（AI 辅助起草）

本目录保存 `q051`–`q100`，与 `evaluation/manual_val_50/` 中已经人工完成的第一批 50 条相互独立。查询最初由 Codex 辅助起草，已于 2026-08-16 由张添翼逐条观察原图并完成人工审核。

- 五类查询各 10 条：`simple`、`compositional`、`negative`、`count`、`ocr`。
- 50 张来源图均未在第一批中使用。
- 50 条查询均为 `annotator=张添翼`、`reviewed=true`。
- 每条来源图的相关性经人工确认后标为 `2`。
- 正式候选池产生后，仍需补标来源图之外的候选图片为 `0/1/2`。

如需再次复核，可打开本批次的审核页面：

```bash
env -u ALL_PROXY -u all_proxy \
  conda run -n vlm-course \
  python scripts/launch_eval_annotator.py \
  --queries evaluation/manual_val_50_assisted/queries.jsonl \
  --relevance evaluation/manual_val_50_assisted/relevance.csv \
  --port 7863
```

如需修改，逐条核对原图、查询文本、类别和来源图相关性后保存当前任务。

校验本批次：

```bash
conda run -n vlm-course python scripts/validate_manual_eval_set.py \
  --queries evaluation/manual_val_50_assisted/queries.jsonl \
  --relevance evaluation/manual_val_50_assisted/relevance.csv \
  --expected-count 50
```
