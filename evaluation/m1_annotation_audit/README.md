# M1 标注融合人工评测工具

这套工具用于在同一批 50 张图片上，对比以下三份视觉标注：

- Qwen3.5-9B 的规范化结果；
- InternVL3.5-14B 的规范化结果；
- M1 融合结果。

三份结果会在网页中随机显示为 A、B、C，评审者不知道各自来源。每张图先只看原图填写人工事实，再检查三份候选，减少被模型答案带偏的风险。评审记录写入 SQLite，可中断后继续；完成后可导出逐图记录和汇总指标。

本目录只包含代码，不包含原图、模型标注、冻结任务或人工评审数据库。这些内容不应提交到 Git。

## 1. 最省事的用法：在现有服务器评审

如果你和 M1 开发者使用同一台服务器，不需要重新生成 50 张任务。正式任务已经位于：

```text
/home/xiaobo.xia/JiafengWu/agent/artifacts/m1_human_audit/v0_1
```

服务当前可能已在 `tmux` 会话 `m1-audit-ui` 中运行。先检查：

```bash
tmux list-sessions
curl http://127.0.0.1:8765/api/health
```

若服务没有运行，在本仓库根目录执行：

```bash
python evaluation/m1_annotation_audit/m1_audit_server.py \
  --audit-dir /home/xiaobo.xia/JiafengWu/agent/artifacts/m1_human_audit/v0_1 \
  --workspace-root /home/xiaobo.xia/JiafengWu/agent \
  --host 127.0.0.1 \
  --port 8765
```

`--workspace-root` 必须是任务中图片相对路径的起点。现有任务记录的是 `data/train/...`，所以这里应指向 `/home/xiaobo.xia/JiafengWu/agent`，不能指向本仓库。

在 Windows PowerShell 建立 SSH 本地转发：

```powershell
ssh -N -L 8765:127.0.0.1:8765 xiaobo.xia@<服务器地址>
```

保持窗口运行，然后浏览器打开：

```text
http://127.0.0.1:8765
```

如果本机 8765 端口被占用，可以使用 `-L 18765:127.0.0.1:8765`，再打开 `http://127.0.0.1:18765`。

## 2. 评审步骤

1. 输入固定的评审者名称。以后继续时必须使用完全相同的名称。
2. 第一阶段只看原图，填写场景、环境、主要实体和清晰 OCR。
3. 保存人工事实后，网页才显示盲化的候选 A、B、C。
4. 逐份检查实体、数量、主要实体覆盖、OCR、关系和 caption。
5. 可以随时保存草稿；确认完整后提交本图。
6. 完成 50 张后运行导出命令。

评审期间不要打开正式任务目录中的 `audit_tasks.jsonl`，里面保存了 A/B/C 的真实来源，会破坏盲评。

### 判断口径

- 主要实体：只写对主体、事件或检索有明显作用的对象。
- 实体存在性：颜色、数量或位置错了，不等于实体本身不存在；这些问题应分开判断。
- 数量：只有肉眼能稳定数清时才判断对错，否则选“不可精确计数”。
- OCR：文字只对了一部分时选“部分正确”并填写人工转写；图片看不清时选“原图不可读”。
- 关系：主语、宾语和方向都正确才算正确。
- Caption 新增事实：caption 写出了原图没有依据的身份、地点、人物关系、动作、数量或文字。
- Caption 评分：1 表示严重不可用，3 表示大体可用但有明显问题，5 表示几乎无需修改。
- 隐私：仅凭外观推断姓名、具体身份、职业、民族、国籍、宗教或健康状况，应标记为敏感身份推断。

本轮课程设计采用单人 50 张评审。请先确定由谁完成正式评审，再始终使用该人的评审者名称。程序可以隔离保存多个名称的数据，但不同人的结果不会自动合并，也不应混在同一份单人指标中。

## 3. 导出结果

在本仓库根目录执行：

```bash
python evaluation/m1_annotation_audit/export_m1_audit.py \
  --audit-dir /home/xiaobo.xia/JiafengWu/agent/artifacts/m1_human_audit/v0_1 \
  --reviewer '<网页中使用的评审者名称>'
```

只有已经提交的图片会进入统计，草稿不会。输出位于任务目录：

```text
exports/reviews.jsonl   逐图人工评审记录
reports/metrics.json    机器可读指标
reports/metrics.md      可直接查阅的汇总表
```

报告同时给出实体 precision 和显著实体 recall，以及计数、OCR、关系、caption 和隐私指标。每项都保留有效分母，避免把“没有可评项目”误写成 0 分。

## 4. 在另一台机器重新生成任务

只有拿不到现有服务器任务时才需要这一步。准备四类输入：

| 参数 | 内容 |
|---|---|
| `--manifest` | 图片 manifest，包含 `image_id`、split、图片相对路径、尺寸和 SHA-256 |
| `--qwen` | Qwen 规范化记录 JSONL |
| `--internvl` | InternVL 规范化记录 JSONL |
| `--fusion-dir` | 包含 `annotations.jsonl` 和 `disagreements.jsonl` 的 M1 融合目录 |

生成固定随机种子、风险分层的 50 张任务：

```bash
python evaluation/m1_annotation_audit/build_m1_audit_tasks.py \
  --manifest <images_manifest.jsonl> \
  --qwen <qwen_normalization_records.jsonl> \
  --internvl <internvl_normalization_records.jsonl> \
  --fusion-dir <m1_fusion_dir> \
  --output-dir <audit_dir>
```

输出目录必须不存在或为空。程序会生成：

| 文件 | 用途 |
|---|---|
| `sample_manifest.jsonl` | 50 张样本的身份、抽样层和风险标签 |
| `audit_tasks.jsonl` | 三份候选的冻结快照和来源信息 |
| `audit_manifest.json` | 随机种子、输入输出哈希和分层统计 |
| `reviews.sqlite3` | 启动网页服务后创建的评审数据库 |

启动服务时，`--workspace-root` 应指向能解析 `sample_manifest.jsonl` 中 `processed_path` 的目录。

## 5. 样本设计和结论边界

50 张图片来自 train，并按候选缺失、OCR、实体/计数、关系、场景冲突、低质量图片和高一致样本分层抽取。同一张图同时比较 Qwen、InternVL 和 Fusion，不是每种方案各抽 50 张，因此总共只需要看 50 张图。

这是风险分层抽样，适合检查融合规则容易出错的地方，不代表完整数据集的自然错误率。正式报告需要说明：只有一名评审者、样本量较小、没有评审者间一致性；当差异仅来自一两张图片时，不应下“稳定优于”的结论。

## 6. 文件说明

```text
m1_audit_common.py       数据校验、盲化映射和指标计算
build_m1_audit_tasks.py  风险分层抽样并冻结 50 张评审任务
m1_audit_server.py       本地网页服务和 SQLite 读写
export_m1_audit.py       导出人工记录并计算三种方案的指标
m1_audit_web/            浏览器页面、样式和交互代码
test_m1_audit.py         数据校验、存储、盲化和指标回归测试
```

工具只依赖 Python 3.9+ 标准库，不需要安装 Flask、Gradio 或 Node 包。默认只监听 `127.0.0.1`，不要改成 `0.0.0.0` 直接暴露到公网。

## 7. 开发检查

在仓库根目录执行：

```bash
python -m unittest discover \
  -s evaluation/m1_annotation_audit \
  -p 'test_*.py' \
  -v

node --check evaluation/m1_annotation_audit/m1_audit_web/app.js
```

原图、`audit_tasks.jsonl`、`reviews.sqlite3`、导出结果和报告均属于本地实验数据，不应提交到仓库。
