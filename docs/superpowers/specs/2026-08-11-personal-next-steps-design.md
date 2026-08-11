# 个人下一步阶段门控工作设计

日期：2026-08-11

状态：设计已由用户确认，范围以个人可执行任务为主，同时定义队友标注交付门禁、
标注到达后的联调、正式实验和报告流程。

## 1. 背景与基线

当前仓库已经完成 M3–M7 在“无正式标注”条件下可实现的主要工程内容：

- M3：多路索引接口、image-only 建库和索引元数据；
- M4：规则、本地 Qwen、OpenAI-compatible API 三种查询解析后端；
- M5：RRF 与归一化加权融合的代码级对照；
- M6：pointwise/listwise 重排、Top-20 和 8GB 显存基准；
- M7：带引用回答、自动故事排序、缺图检测、生成图片和 AI 标识；
- A7：Chinese-CLIP 与 jina-clip-v2 适配及 64 图资源对比；
- 工程基线：提交 `a8f3a05`，完整测试 `122 passed`。

这些结果证明本地工程链路可运行，但不能代替正式标注、人工 relevance judgments 或
课程报告中的质量结论。

## 2. 目标

建立一套不依赖标注完成日期的个人执行流程：等待期间持续产出；队友交付标注时先经过
自动与人工门禁；先用 200 张小样完成端到端联调，再接入全量数据；最后完成 A5–A7、
M5/M6 的正式评测以及个人报告、图表和演示材料。

完成标准不是“脚本能启动”，而是每个阶段都有固定输入、固定命令、可检查产物、明确
通过条件和失败后的回退路径。

## 3. 范围

### 3.1 本设计覆盖

- 当前个人能够独立推进的代码、评测、部署、文档和演示工作；
- 队友 M1 标注文件的交付格式与验收清单；
- 人工 query/relevance 评测集的独立制作流程；
- 200 张小样与全量标注的接入顺序；
- M3–M7、A5–A7 的联调和实验依赖；
- RTX 4060 Laptop 8GB 下的串行模型调度；
- 结果冻结、图表、个人报告、演示和最终仓库检查。

### 3.2 本设计不覆盖

- 替队友重新执行 M1 全量标注；
- 在没有人工真值时制造或推断正式质量指标；
- 在个人分支中擅自修改已经冻结的标注 schema；
- 同时常驻 Qwen-VL、Stable Diffusion 和大型图像编码器；
- 把模型、原图、FAISS 索引、实验 JSON 或 API Key 提交到 Git。

## 4. 必须区分的两类“标注”

### 4.1 M1 图片结构化标注

生产者是队友，消费者是 M2、M3、M5、M6 和 M7。正式文件进入：

```text
artifacts/annotations/train.caption_verified_v4.jsonl
artifacts/annotations/val.caption_verified_v4.jsonl
```

适配器接受现有扁平 schema 或提案的嵌套 schema。每条记录最终必须能够解析出：

- `image_id`、`split`、`relative_path`、`sha256`；
- 非空的 `summary`，或嵌套格式中的 `caption_dense/caption_short`；
- objects、scene、OCR、颜色、情绪等允许为空的结构化字段；
- `model_version` 与 `prompt_version`；
- 与 manifest 一致的图片身份和路径。

### 4.2 检索 relevance judgments

这类真值不由 M1 标注自动产生，个人现在即可基于 Val 原图开始制作。工作区是：

```text
evaluation/manual_val/queries.jsonl
evaluation/manual_val/relevance.csv
```

查询类别固定为 `simple/compositional/negative/count/ocr`。每条审核查询必须有真实
`annotator`，至少一个 relevance=2 的图片；relevance=1 表示部分相关。来源图片只用于
提示人工写 query，不得自动视为全部相关答案，也不得复制自动 caption 生成 query。

只有这两类数据都满足门禁，才能报告 Recall@K、MRR、mAP 和 nDCG@10。

## 5. 阶段门控架构

```text
阶段 0：冻结当前工程基线
  ↓
阶段 1：等待标注期间的并行工作
  ↓
门禁 A：队友 M1 标注交付验收
  ↓
阶段 2：200 张小样端到端联调
  ↓
门禁 B：小样稳定性与数据质量验收
  ↓
阶段 3：全量标注冻结与 Train/Val 建库
  ↓
阶段 4：正式检索、融合、重排与编码器实验
  ↓
阶段 5：结果冻结、失败分析和图表
  ↓
阶段 6：个人报告、演示与最终交付
```

后续阶段不得绕过前一门禁。某阶段失败时保留原始输入和失败报告，只修复失败点后重跑，
不在同一结果目录内手工覆盖数据。

## 6. 各阶段设计

### 6.1 阶段 0：冻结工程基线

记录 Git 提交、环境版本、GPU、模型目录指纹、测试数量和现有实测报告。完整测试、
compileall、Pixi lock 检查和 `git diff --check` 必须通过。该记录用于区分后续的数据错误、
环境错误和代码回归。

输出是一个只读基线记录和干净工作树；没有通过基线检查时不接入队友数据。

### 6.2 阶段 1：等待标注期间

阶段 1 并行推进五条个人工作线：

1. 人工 query/relevance：用现有 Gradio 标注界面逐条完成 100 条 Val 查询，并先
   标注来源图和仅凭原图人工发现的相关图片；text/BM25/RRF 候选池补充推迟到阶段 4；
2. 部署与可复现：补齐 `Dockerfile`、`docs/DEPLOY.md`、无 GPU 降级路径和干净环境
   dry-run，满足提案的部署交付要求；
3. 报告框架：先写 Method 中 M3–M7、实验设置、资源表和 Limitations，不预填正式
   质量数值；
4. Demo 预演：固定三个查询、一个故事、一个缺图补全案例和一个安全回退案例，检查
   录屏顺序与 8GB 串行加载；
5. M4 外部 API：有免费 Key 时完成一次真实调用并记录模型、延迟、回退与限流；没有
   Key 时保留现有本地 Qwen 结果，不把 API 调用列为主链路阻塞。

这些任务均不依赖 M1 图片结构化标注。人工 relevance 是正式实验的独立前置条件，
不能等队友 M1 完成后才开始。

### 6.3 门禁 A：队友标注交付验收

收到文件后先复制到隔离的导入目录并生成校验报告，不立即覆盖正式 annotations。门禁
检查以下内容：

- JSONL 每个非空行都是独立合法 JSON；
- Train/Val 数量与有效、非重复 manifest 一致；
- `image_id` 唯一，且集合、split、relative_path、sha256 与 manifest 一致；
- Windows 路径已规范化为 POSIX `/`，不存在盘符绝对路径；
- 扁平或嵌套 schema 能被 `adapt_annotation()` 全量解析；
- caption/summary 非空，字段类型稳定，数值 count 非负；
- `model_version`、`prompt_version` 存在且全批次可追溯；
- 空值率、uncertain_fields 比例、OCR 命中率和解析失败率被汇总；
- 随机抽查 30 张，并额外检查人物数量、OCR、夜晚/天气等高风险字段；
- 不把 image-only 占位 annotations 或自动 query seed 当作正式标注。

任何 ID、路径、split 或 schema 错误都会阻断接入。语义质量问题形成带 image_id 的
返修清单，由队友修订并递增标注版本，不在消费者侧悄悄改写。

### 6.4 阶段 2：200 张小样联调

从通过门禁 A 的版本中确定性抽取 Train/Val 合计 200 张，使用隔离 workspace 运行：

```text
正式 annotations → M3 image/text/BM25 → M4 查询解析
→ M5 RRF/加权融合 → M6 pointwise/listwise → M7 回答/故事/补图
```

联调同时覆盖 simple、compositional、negative、count 和 OCR 查询。验收维度包括索引
记录数一致、三路候选非空、硬过滤正确、融合结果可追溯、重排安全回退、引用图片存在、
生成图带 AI 标识，以及整个流程不超过 8GB 显存。

小样结果单独保存，不与历史 image-only 产物混用。

### 6.5 门禁 B：小样通过条件

小样必须满足：

- annotations、三个索引和 index manifest 的 image_id 集合一致；
- 五类查询各至少一条端到端成功；
- M5 两种融合均返回稳定 Top-K；
- M6 Top-20 listwise 没有硬失败，降级时保留原始排名和错误原因；
- M7 回答引用全部可解析，故事顺序可解释，生成图片带 provenance；
- 无 NaN/Inf 向量、CUDA OOM、缺图路径或静默 schema 回退；
- 自动测试与编译检查仍通过。

只有门禁 B 全部通过，才允许构建全量索引。

### 6.6 阶段 3：全量标注冻结与建库

将通过门禁的数据复制为正式文件，记录 annotation version、文件 SHA256、模型指纹、
配置快照和构建时间。先构建 Val 并执行冒烟，再构建 Train；image、text、BM25 分支
顺序执行，避免模型同时占用显存。

全量产物必须写入新的版本目录或在运行前归档旧版本。索引构建完成后重新验证记录数、
ID 集合、维度、非有限向量和可加载性。

### 6.7 阶段 4：正式实验

正式实验前，先汇集 CLIP、text、BM25 和 RRF 的候选池，由人工补充来源图之外的相关
图片并再次运行评测集校验。随后使用冻结的 annotations、同一份人工 query/qrels、
同一硬件和固定随机种子运行：

- A5：CLIP-only、text-only、BM25-only、RRF 三路、归一化加权融合；
- M5：RRF 与归一化加权融合的质量、延迟和失败率对照；
- A6/M6：无重排、pointwise、listwise Top-20；
- A7：Chinese-CLIP ViT-B/16 与 jina-clip-v2 512 维；
- 分层结果：simple、compositional、negative、count、OCR；
- 指标：Recall@K、MRR、mAP、nDCG@10、平均/P50/P95 延迟和失败率。

每次只改变一个消融变量。资源实验与质量实验分表；不同编码器的相似度绝对值不直接
比较。没有通过人工评测集校验时，评测脚本必须失败而不是输出假指标。

### 6.8 阶段 5：结果冻结与分析

确定最终实验配置后停止调参，冻结原始 JSON、CSV、LaTeX 表和配置快照。图表从原始
结果脚本化生成，不在图片编辑软件中修改数值。失败分析至少覆盖否定、数量、OCR、
视觉相似但语义不相关、重排回退和标注错误六类案例。

报告同时保留负结果，例如某类查询融合没有提升、Jina 资源成本更高、listwise 延迟
过大。Limitations 明确数据规模、人工标注偏差、生成模型污染和 8GB 硬件限制。

### 6.9 阶段 6：个人报告与演示

个人报告以“在队友 M3–M5 代码基础上的工程完善和正式验证”为叙事主线，区分已有代码、
个人新增、队友标注依赖和共同结果。正文覆盖架构、关键实现、A5–A7、M6/M7、失败分析、
资源限制和个人贡献；附录给出命令、配置和 prompt。

演示采用固定脚本：环境与数据版本 → 普通检索 → 复合/否定查询 → 重排开关 → 带引用
回答 → 自动故事 → 缺图补全与 AI 标识 → 失败回退。录屏前用同一提交和冻结数据完整
彩排一次，不在演示中临时下载模型。

## 7. 文件职责与预期新增文档

- `docs/superpowers/plans/2026-08-11-personal-next-steps.md`：最终逐步执行计划；
- `docs/DEPLOY.md`：Linux/Windows、Conda/Pixi、GPU/CPU 降级和故障排查；
- `Dockerfile`：可复现服务镜像，不内置模型、数据或密钥；
- `evaluation/manual_val/*`：人工 query 与 relevance 真值；
- `artifacts/annotations/*`：正式 M1 标注，Git 忽略；
- `artifacts/evaluation/*`：原始实验结果，Git 忽略；
- `docs/results/`：由冻结结果生成、允许提交的汇总表与图；
- `docs/FINAL_PERSONAL_REPORT.md`：个人报告正文草稿；
- `docs/DEMO_SCRIPT.md`：演示步骤、预期画面和失败回退。

## 8. 错误处理与回退原则

- 数据错误：停止接入，输出 image_id 级返修清单；
- 环境错误：先复现基线测试，再改依赖，不在实验中升级版本；
- CUDA OOM：降低 batch、卸载当前模型、清理缓存并串行重跑，不降低正式实验图片集合；
- M6 失败：记录失败并回退原始融合排名，不伪造 VLM 分数；
- M7 生成失败：保留检索故事，明确标记未生成，不把原图冒充生成图；
- API 失败：记录状态码/超时/重试并回退 rules，不记录或提交 API Key；
- relevance 不完整：拒绝运行正式评测，只允许资源和工程冒烟。

## 9. 验证策略

每个实施任务遵循测试先行和小提交：先写或确认失败检查，再做最小修改，运行定向测试，
最后运行完整回归。全局完成检查固定为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m compileall -q src scripts tests
pixi lock --check
git diff --check
git status --short
```

GPU 任务额外记录 `nvidia-smi`、模型 dtype、batch size、峰值 CUDA 分配和运行时间。

## 10. 成功标准

当以下条件同时成立，个人下一步工作才算完成：

1. 等待期五条工作线有可审核产物；
2. 队友 M1 标注通过门禁 A；
3. 200 张小样通过门禁 B；
4. 全量 Train/Val 三路索引与冻结标注一致；
5. 100 条人工 query/qrels 通过校验；
6. A5–A7 与 M5/M6 正式实验可复现并保留原始结果；
7. 报告中的每个结论都能追溯到配置和结果文件；
8. Demo 在 8GB 显存上按固定脚本完成且存在降级路径；
9. 模型、数据、索引、Key 和本地产物没有进入 Git；
10. 完整测试、编译、锁文件和 Git 检查全部通过。
