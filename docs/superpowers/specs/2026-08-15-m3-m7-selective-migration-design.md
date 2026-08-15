# M3–M7 选择性迁移与真实联调设计

日期：2026-08-15  
负责人：张添翼（U202315231）

## 1. 目标

将队友交付包中的 M3–M5 实现选择性迁移到当前 `main`，保留并兼容当前已经加固的
M6/M7，不整体应用包含旧版 M6/M7 的交付 patch。使用本机 Qwen3.5 canonical v1.3
标注、Train/Val 图片和本地模型重建真实链路，最终形成可复现的 M3–M7 代码、正式
M5→M6 接口产物、真实 M6 结果、三组 M7 故事和阶段报告证据。

## 2. 验收范围

- M0–M2：视为队友已完成的只读上游，不迁移、不重做；最终只核对数据清单、canonical
  v1.3 标注哈希和 M3 消费兼容性，不将其计入张添翼的个人实现。
- M3：Train/Val 的 Chinese-CLIP、BGE、BM25 三路索引均可构建并通过 manifest 校验。
- M4：规则解析后端必须通过全部 12 个基准查询；本地 Qwen 模式至少完成一次真实冒烟。
- M5：全部 12 个查询导出 `m5-to-m6-v1.0`，每行严格包含 20 个唯一候选。
- M6：全部 12 个查询完成 listwise Top-20 重排，记录耗时、峰值显存、降级状态和顺序完整性。
- M7：选择 3 个代表性查询生成故事；至少 1 个故事执行缺图补全并带有 AI 生成标识。
- UI 与报告：保存 M7 页面截图、结构化实验结果和与课程计划的对应说明。
- 自动化：迁移后的全量测试、compileall、接口校验和 Git 空白检查全部通过。

## 3. 合并策略

采用选择性迁移，而不是 `git am` 整体交付 patch：

1. 队友独占的 M3–M5 新文件按功能单元迁移，例如 annotation adapter、M5 delivery、
   索引实现、导出脚本和对应测试。
2. 双方都修改过的共享文件逐段合并，包括配置、schema、indexing、retrieval、factory
   和 CLI；不得以文件覆盖方式丢弃当前行为。
3. 当前 `src/anima_search/m6/`、M6 CLI、M7 canonical bridge 和 M7 CLI 作为保留基线。
4. 队友 patch 中旧版 M6/M7、旧报告和会覆盖个人工作的内容不迁移。
5. 每个功能单元独立执行红—绿测试循环并单独提交，便于回退和审查。

## 4. 数据与产物边界

### 4.1 只读输入

- `/home/tiantiantianyi/Desktop/nlp/M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl`
- `/home/tiantiantianyi/Desktop/nlp/Train/`
- `/home/tiantiantianyi/Desktop/nlp/Val/`
- 队友源码快照：
  `/home/tiantiantianyi/Desktop/nlp/nlp_courcedes_delivery_20260815/nlp_courcedes/`

原始标注和图片不得被修改。

### 4.2 本地生成产物

- `artifacts/manifests/{train,val}.jsonl`
- `artifacts/annotations/{train,val}.qwen35-canonical-v1.3.jsonl`
- `artifacts/indexes/{train,val}/`
- `artifacts/evaluation/m5_to_m6_candidates.jsonl`
- `artifacts/evaluation/m5_retrieval_config.snapshot.json`
- `artifacts/evaluation/m5_validation_report.json`
- `artifacts/evaluation/m6_rerank_results.jsonl`
- `artifacts/evaluation/m6_validation_report.json`
- `artifacts/evaluation/m7_stories/`

大型索引、模型缓存和生成图片保留在本地且不提交 Git。小型 JSON、CSV、Markdown 和
必要截图是否提交由 `.gitignore` 与报告复现需求共同决定，但不得提交原始数据副本。

## 5. 模块设计

### 5.1 M3：标注适配与三路索引

canonical v1.3 adapter 将 Qwen3.5 记录映射为项目 `ImageAnnotation`，同时保留中文检索
字段、英文枚举、实体数量、OCR、关系、事件、主观描述和不确定性。导入时用数字 ID
连接 Train/Val manifest，并校验 `processed_sha256` 与图片 manifest 哈希。

索引层分别构建：

- Chinese-CLIP 图像向量索引；
- BGE 中文文本向量索引；
- BM25 稀疏索引。

index manifest 必须保存记录数、ID 摘要、标注版本、标注产物路径及哈希、分支元数据、
建库配置摘要和逐图片记录。所有持久化路径使用以项目根为基准的 POSIX 相对路径，避免
Windows 绝对路径进入正式交付。

### 5.2 M4：结构化查询与过滤

迁移规则解析、别名规范化、结构化过滤和场景路由。正式批量实验使用可复现的规则后端；
本地 Qwen3-VL-2B 只用于真实联调冒烟，不改变 12 查询基准的确定性结果。硬约束用于否定
词、必要词和 OCR 等明确条件；可能导致结果不足 20 的正向条件允许记录化软化回退。

### 5.3 M5：融合与正式导出

M5 保留 RRF 与归一化加权融合两个实现，正式主结果使用配置冻结的融合方法，并生成两者
代码级对照。导出器对每个查询保存原始 `branch_scores`、过滤后的 `branch_ranks`、
`fused_score` 与 `matched_fields`，不得写入任何 M6 字段。

正式导出包含 12 行、240 个候选，并同时产生只读检索配置快照。写入采用临时文件替换，
避免产生半成品 JSONL。

### 5.4 M5→M6 接口修正

接口文档规定 `config_sha256` 是 M5 有效检索配置快照的 SHA-256；index manifest 中的
`config_digest` 是建库配置摘要，两者语义不同，不得直接比较。

M6 校验入口新增明确的 M5 配置快照输入：

- `config_sha256` 与该快照文件的实际哈希比较；
- `index_manifest_sha256` 与所给 index manifest 比较；
- manifest 的 `config_digest` 仅作为建库溯源元数据保留；
- manifest 声明的 annotation artifact 必须存在且通过哈希、记录数和 ID 摘要校验。

### 5.5 M6：真实 listwise 重排

使用本地 Qwen3-VL-2B 对 12 个 Top-20 候选执行 listwise 重排。每个查询输出严格保留
20 个唯一 ID，并记录 M5 原 rank、M6 rank、模型分数、mismatch 和 degraded。模型异常、
未知 ID、非有限分数或输出无法修复时，必须恢复该查询的完整 M5 顺序，不能静默产生伪结果。

运行时逐查询落盘并记录耗时与峰值显存，避免单个失败丢失整个批次。正式报告区分成功、
可修复降级和硬回退。

### 5.6 M7：三组真实故事

从 M6 结果中选择 3 个具有代表性的查询，每个故事选择 3–8 张图片。M7 保持 M6 排序
来源可追溯，并用 canonical 标注生成标题、段落和图片顺序。至少一个故事启用 Stable
Diffusion 缺图补全；所有生成资产必须同时具有 `source="generated"` 和
`ai_generated=true`，页面显示清晰的 AI 生成标识。

UI 验收覆盖自动选图、排序、故事内容、缺图状态、生成标识和错误提示，并保存截图用于
个人报告。

## 6. 数据流

```text
Qwen3.5 canonical + Train/Val images
  → manifest/hash validation
  → canonical adapter
  → M3 image/text/BM25 indexes
  → M4 structured query and filters
  → M5 fusion and strict Top-20 export
  → M5 config snapshot + index manifest validation
  → M6 Qwen3-VL listwise rerank
  → M7 candidate selection and story generation
  → optional Stable Diffusion gap asset with AI marker
  → UI screenshots, metrics and report
```

## 7. 错误处理与可恢复性

- 原始图片、标注和正式输入全部只读；所有输出先检查路径别名。
- 导入阶段对缺失的 7 个 Train 标注明确记录，不使用其他模型静默补齐。
- 任一索引分支失败时，正式三路构建失败；单分支模式只能标记为 smoke/degraded。
- M5 不足 20 个候选时先执行受控软化；仍不足则阻断该查询导出。
- M6 模型失败时硬回退并标记 degraded，不将 dry-run 当作正式实验。
- M7 缺少 canonical 标注时阻断；生成失败时保留 planned/failed gap，不伪造 generated。
- 每个阶段输出报告包含输入哈希、记录数、错误列表和运行参数，支持断点复核。

## 8. 测试与验证设计

迁移遵循测试驱动：先加入能暴露当前缺失行为或接口冲突的测试并确认失败，再迁移最小代码
使其通过。测试层次如下：

1. 单元测试：adapter、索引序列化、查询解析、过滤、融合、M5 schema、M6/M7 bridge。
2. 集成测试：小型图片 fixture 的导入→建库→检索→导出→M6→M7。
3. 接口测试：正式 12 行 JSONL、配置快照、index manifest、annotation artifact 与图片路径。
4. 真实冒烟：三路 Val 检索、本地 Qwen M4、单查询 M6、单故事 M7 gap。
5. 正式实验：12 查询 M5/M6 和 3 故事 M7。
6. 最终验证：全量 pytest、compileall、`git diff --check`、产物哈希与工作区审计。

## 9. Git 与职责归属

继续在用户明确指定的 `main` 开发，但每个迁移单元独立提交。提交信息区分 `test`、
`feat`、`fix`、`docs`。队友原有 M3–M5 工作在报告中标注为“队友实现/本人迁移与联调”；
张添翼的个人贡献只包括选择性迁移、接口协调、M6/M7 实现与真实联调、实验验证和报告证据，
不得把队友的原始实现写成本人独立完成。

所有实现和实验验证完成后，将本地 `main` 与远程状态进行显式比较。只有工作区干净、
全量验证通过且推送目标确认为 `origin/main` 时才执行 push；不得使用会覆盖远程历史的
force push。若远程包含未合并提交，先 fetch、审查差异并正常合并，再重新运行全量验证。

## 10. 完成定义

只有同时满足下列条件才可宣称 M3–M7 完成：

- 选择性迁移后的源码与当前 M6/M7 无未解决冲突；
- M0–M2 上游 manifest、canonical 标注哈希及 M3 消费兼容性通过只读验收；
- 全量测试及静态验证通过；
- Train/Val 三路索引完成并有可核验 manifest；
- 12 查询正式 M5 文件通过严格接口校验；
- 12 查询 M6 结果完成且所有降级均有明确记录；
- 3 个 M7 故事完成，其中至少 1 个具有真实 AI 补图和标识；
- 报告明确区分队友工作、本人工作、真实结果、模拟结果和未完成项；
- Git 工作区干净，未提交模型、索引、原始图片或原始标注；
- 本地 `main` 已通过非强制 push 同步到 GitHub `origin/main`。
