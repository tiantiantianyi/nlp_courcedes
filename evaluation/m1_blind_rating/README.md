# M1 三模型简化盲评工具

这套工具在同一批 50 张图片上匿名比较 Qwen3.5-9B、InternVL3.5-14B 和 Qwen3-VL-8B-Instruct 的清洗后标注。每张图片显示候选 A、B、C，只评准确性、完整性、整体可用性、严重错误和本图最佳结果。

旧的 `m1_annotation_audit` 是逐字段精细审计；本目录是独立的简化盲评，不会读取或修改旧评审记录。

## 启动正式任务

在仓库根目录执行：

```bash
python evaluation/m1_blind_rating/blind_rating_server.py \
  --rating-dir <workspace>/artifacts/m1_blind_rating \
  --workspace-root <workspace> \
  --host 127.0.0.1 \
  --port 8766
```

正式服务使用 8766，避免和旧精细评审的 8765 冲突。Windows PowerShell 建立 SSH 转发：

```powershell
ssh -N -L 18766:127.0.0.1:8766 <user>@<server>
```

浏览器打开 `http://127.0.0.1:18766`。固定使用同一个评审者名称，程序会按名称隔离记录并恢复进度。

## 填写方式

每张图片同时显示三份匿名结果。每份填写：

- 准确性 1–5 分；
- 完整性 1–5 分；
- 整体可用性 1–5 分；
- 是否存在严重错误。

最后选择候选 A、B、C、并列或全部不合格。切换图片前，网页会保存已经填写的草稿；也可以手动点击“保存草稿”。提交后默认锁定，点击“重新打开本图”可以修改。

评分锚点：1 分表示基本不可用，3 分表示主体基本正确但需要明显修改，5 分表示主要内容准确完整且几乎可以直接使用。

## 导出结果

```bash
python evaluation/m1_blind_rating/export_blind_rating.py \
  --rating-dir <workspace>/artifacts/m1_blind_rating \
  --reviewer '<网页中使用的评审者名称>'
```

多人独立评审时，为避免覆盖前一位评审者的结果，指定单独输出目录：

```bash
python evaluation/m1_blind_rating/export_blind_rating.py \
  --rating-dir <workspace>/artifacts/m1_blind_rating \
  --reviewer '<网页中使用的评审者名称>' \
  --output-dir <workspace>/artifacts/m1_blind_rating/reviewer_exports/<评审者目录>
```

输出文件：

```text
exports/reviews.jsonl  逐图评分、A/B/C 映射和揭盲结果
reports/metrics.json   机器可读汇总
reports/metrics.md     便于课程报告引用的表格
```

只有已经提交的图片会进入统计，草稿不会。

两位评审者都导出后，可以比较模型排序、逐项分差、严重错误和最佳候选的一致性：

```bash
python evaluation/m1_blind_rating/compare_reviewers.py \
  --review-a <第一位评审者的 exports/reviews.jsonl> \
  --review-b <第二位评审者的 exports/reviews.jsonl> \
  --output-dir <一致性报告目录>
```

## 重新生成任务

正式 50 张任务已经生成，一般不需要重新抽样。如需从当前三份清洗结果重建一个新目录：

```bash
python external/nlp_courcedes/evaluation/m1_blind_rating/build_blind_rating_tasks.py \
  --manifest artifacts/m1_full/images_train_val_2369.jsonl \
  --qwen35 artifacts/m1_postprocess/qwen35_9b/normalized_annotations.jsonl \
  --internvl35 artifacts/m1_postprocess/internvl35_14b/normalized_annotations.jsonl \
  --qwen3-vl-8b artifacts/m1_postprocess/qwen3_vl_8b_instruct/normalized_annotations.jsonl \
  --output-dir <新的空目录>
```

固定随机种子下，程序会从三模型共同有效的 2286 张图片中选择 40 张普通分层样本和 10 张高差异样本，train/val 为 42/8。

盲评期间不要打开任务目录中的 `rating_tasks.jsonl`，其中包含真实模型名称，会破坏盲化。

## 开发检查

```bash
python -m unittest discover \
  -s evaluation/m1_blind_rating \
  -p 'test_*.py' \
  -v

node --check evaluation/m1_blind_rating/web/app.js
```

工具只依赖 Python 3.9+ 标准库，不需要安装 Flask、Gradio 或 Node 包。服务默认只监听 `127.0.0.1`，应通过 SSH 转发访问，不要直接暴露到公网。
