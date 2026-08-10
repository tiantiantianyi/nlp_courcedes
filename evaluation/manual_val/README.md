# Val 人工检索评测工作区

这里包含 100 条从 Val 原图抽样得到的人工任务：

- `queries.jsonl`：查询文本、类别、审核状态和来源图片。
- `relevance.csv`：人工相关性判断，等级 2=高度相关、1=部分相关。
- 来源图片只用于提示人工撰写查询，不是自动 relevance 标签。

启动标注界面：

```bash
env -u ALL_PROXY -u all_proxy \
  conda run -n vlm-course \
  python scripts/launch_eval_annotator.py
```

每条任务必须：

1. 直接观察原图撰写自然查询，不能复制自动 caption。
2. 选择 `simple/compositional/negative/count/ocr` 类别。
3. 填写真实标注者名称。
4. 每行用 `image_id:grade` 填相关性，例如 `val-2002:2`。
5. 至少包含一个 relevance=2 的图片后才能勾选“已人工审核”。

完整性检查：

```bash
conda run -n vlm-course python scripts/validate_manual_eval_set.py
```

在 100 条任务全部由人完成前，校验脚本应当失败；这是为了阻止空白或自动种子被用于正式 Recall/MRR/nDCG。正式实验前还应汇集 CLIP、text、BM25 和 RRF 的候选池，让人工补充除来源图之外的相关图片。
