# Anima M3-M5 实现方案

日期：2026-08-08  
负责人：M3-M5 检索组（待填写）  
项目目录：`C:\Users\下弦月\Desktop\NLP\anima`  
硬件边界：RTX 3070 Ti Laptop GPU，8 GB 显存；Windows 本地运行  
目标：在 M1/M2 最终标注冻结前，用模拟标注完成 M3 多路索引、M4 查询理解和 M5 混合召回框架；正式标注到达后只重建索引，不改检索接口。

## 1. 范围与非目标

本方案覆盖：

- M3：图像向量、标注文本向量、BM25/OCR 三路索引。
- M4：中文自然语言查询解析、否定条件和结构化过滤条件提取。
- M5：三路并行召回、过滤、RRF 融合、结果解释与降级。
- M3-M5 的单元测试、集成测试、检索评测和消融实验。

本阶段不实现：

- M1 标注生成和 M2 标注质量验证。
- M6 VLM 视觉重排，只预留稳定接口。
- M7 问答、游记和文生图，只提供可供 M7 消费的检索结果。
- 大模型训练。BGE 微调属于可选扩展，不作为 M3-M5 第一版的交付门槛。

## 2. 当前代码基础与缺口

现有代码可直接复用：

| 能力 | 当前文件 | 状态 |
|---|---|---|
| 标注转检索文档 | `src/anima_search/indexing/documents.py` | 已有基础实现 |
| BM25 索引 | `src/anima_search/indexing/bm25_index.py` | 已有基础实现 |
| BGE 文本向量索引 | `src/anima_search/indexing/vector_index.py` | 已有基础实现 |
| RRF | `src/anima_search/retrieval/fusion.py` | 已有基础实现 |
| 查询解析 | `src/anima_search/retrieval/query_parser.py` | 有 LLM 解析和失败回退 |
| 混合搜索 | `src/anima_search/retrieval/search.py` | 当前仅 BM25 + 文本向量两路 |
| 建索引和搜索命令 | `scripts/build_indexes.py`、`scripts/search_cli.py` | 已有入口 |

第一版必须补齐的缺口：

1. 当前不是原方案要求的三路召回，缺少 CLIP 图像索引。
2. 当前只在拼接后的文档上做字符串包含过滤，不能可靠处理“不要人物”等同义表达。
3. `SearchResult` 只保存分支原始分数，没有分支名次、命中字段、活动分支和过滤过程。
4. 查询解析失败后只保留原始文本，否定条件会丢失；需要规则解析作为稳定底座。
5. 缺少索引版本清单和输入一致性检查，可能出现标注、模型和索引版本不匹配。
6. 现有测试只覆盖 RRF 的一个简单性质，没有覆盖建索引、过滤、三路融合和降级。

## 3. 目标架构

```text
M1/M2 annotations.jsonl
          |
          v
   文档构建与字段归一化
          |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
  CLIP 图像向量索引    BGE 文本向量索引      BM25 稀疏索引
      image_id             image_id             image_id
          |                  |                  |
          +------------------+------------------+
                             |
用户查询 -> M4 查询解析 -> 三路 Top-K -> 结构化过滤 -> RRF -> Top-N
                                                        |
                                                        v
                                              SearchResult[] -> M6/M7
```

约束：三种编码模型不会同时用于批量建索引。图片和文本向量将离线生成并缓存，在线阶段只编码一条查询，FAISS 和 BM25 均在 CPU 上执行。

## 4. 输入契约

### 4.1 M1/M2 必须提供的字段

现有 `ImageAnnotation` 将继续作为输入 Schema。M3-M5 至少依赖：

```text
image_id
split
relative_path
sha256
summary
objects
actions
scene
attributes
spatial_relations
style
mood
colors
ocr_text
uncertainty
prompt_version
model_version
```

约定：

- `image_id` 在 Train、Val 内唯一。
- `relative_path` 必须能从项目根目录解析到真实图片。
- M2 若输出清洗版本，应保持同一 Schema，只增加版本信息，不在 M3-M5 中写 M2 专用逻辑。
- `search_queries` 不进入 Val 检索文档，避免用模型生成的 query 反向评测同一份标注造成泄漏。
- 重复图、损坏图和 M2 判定不可用的记录应在建索引前排除，并记录原因。

### 4.2 查询契约

计划扩展 `SearchQuery`：

```python
class SearchQuery(BaseModel):
    raw_text: str
    semantic_text: str
    query_type: Literal["simple", "compositional", "negative", "count", "ocr"]
    objects: list[str]
    actions: list[str]
    scene: list[str]
    mood: list[str]
    colors: list[str]
    style: list[str]
    required_terms: list[str]
    excluded_terms: list[str]
    ocr_terms: list[str]
```

所有字段提供默认值，使旧调用方仍可只传 `raw_text`。

### 4.3 输出契约

计划扩展 `SearchResult`，保证 M6/M7 能解释和复用结果：

```python
class SearchResult(BaseModel):
    image_id: str
    relative_path: str
    fused_score: float
    branch_scores: dict[str, float]
    branch_ranks: dict[str, int]
    matched_fields: list[str]
    evidence: list[str]
    mismatch: list[str]
    active_branches: list[str]
    source: Literal["real", "generated"] = "real"
```

`fused_score` 只表达 RRF 排名分数，不与各分支原始分数直接比较。

## 5. M3：多路索引设计

### 5.1 文档构建

为不同分支生成不同输入：

| 分支 | 索引内容 | 设计理由 |
|---|---|---|
| 图像 CLIP | 原始图片 | 捕捉未被标注文本覆盖的视觉语义 |
| BGE 文本 | `summary + objects + actions + scene + attributes + relations + style + mood + colors + OCR` | 处理长复合中文查询 |
| BM25 | 对象、场景、颜色、OCR 等明确词项 | 支持 OCR、专名和精确关键词命中 |

第一版 BM25 使用 jieba。对象、场景和 OCR 将保留字段标签，防止所有文本被压成不可解释的单段字符串。字段加权作为可选实验，不在第一版手工调参。

### 5.2 图像向量索引

新增 `ImageVectorIndex`，接口与现有 `VectorIndex` 对齐：

```python
build(image_ids, image_paths, batch_size)
search(text_query, limit)
save(directory)
load(directory)
```

默认模型计划使用本地 Chinese-CLIP ViT-B/16。若模型暂未准备，系统允许 `image` 分支显式关闭，以 BM25 + BGE 两路完成开发；最终 M3 验收前必须补上图像分支。

8 GB 显存策略：

- 图像编码从 `batch_size=8` 开始冒烟测试，根据峰值显存上调或下调。
- 使用 FP16 推理，输出立即转为 CPU `float32` 并写入 FAISS。
- 不使用 GPU FAISS，数据量只有 2369 张，`IndexFlatIP` 足够。
- 图像嵌入和 BGE 嵌入分别构建，避免两套编码器同时占用显存。

### 5.3 文本向量索引

复用 `VectorIndex`，默认使用本地 `bge-small-zh-v1.5`：

- 文档和查询使用同一模型。
- 向量 L2 归一化，FAISS 使用 `IndexFlatIP`。
- 默认批量大小从 32 开始；发生 OOM 时降为 16 或 8。
- 首版不微调，先建立稳定 baseline。

### 5.4 索引版本与文件结构

计划输出：

```text
artifacts/indexes/<split>/
├── manifest.json
├── annotations.json
├── bm25.pkl
├── text/
│   ├── vectors.faiss
│   └── metadata.json
└── image/
    ├── vectors.faiss
    └── metadata.json
```

`manifest.json` 至少记录：

```text
split
record_count
annotation_version
annotation_sha256
schema_version
active_branches
model_paths/model_digests
vector_dimensions
normalization
build_parameters
config_digest
```

加载索引时将检查记录数量、ID 顺序、向量维度和标注版本；不一致时直接报错，不静默使用旧索引。

## 6. M4：查询理解设计

M4 使用“规则优先、LLM 可选、失败可降级”的本地方案。

### 6.1 解析顺序

1. 规范化全角/半角、空格和常见同义词。
2. 用确定性规则提取否定表达、OCR 引号词、颜色、时间、场景和数量。
3. 可选调用本地 Qwen3-VL-2B 的文本生成接口补充结构化字段。
4. 合并规则与模型结果；显式否定规则的优先级最高。
5. LLM 不可用或 JSON 非法时，保留规则结果和原始语义文本。

### 6.2 同义词与否定条件

新增可版本化词表，例如：

```yaml
aliases:
  人物: [人, 男人, 女人, 行人, 游客, 儿童]
  汽车: [轿车, 小汽车, 车辆]
negative_patterns:
  - "不要{term}"
  - "没有{term}"
  - "无{term}"
```

过滤原则：

- 否定条件只排除“标注明确确认存在”的候选。
- 标注没有提到某对象，不等于图片中一定没有该对象。
- `uncertainty` 涉及相关字段时，不把该字段作为强制硬过滤证据。
- 所有过滤都返回命中字段和原因，供错误分析使用。

## 7. M5：三路召回与 RRF

### 7.1 在线检索流程

计划固定第一版参数：

```yaml
retrieval:
  candidate_count: 50
  result_count: 8
  rrf_k: 60
  enabled_branches: [image, text, bm25]
```

执行顺序：

1. M4 生成 `SearchQuery`。
2. 图像、文本和 BM25 分支各返回 Top-50。
3. 对每个分支执行明确的结构化排除过滤，并重新形成有效候选名次。
4. 使用 RRF 合并有效排名，不直接对原始分数做加权相加。
5. 取融合 Top-8，生成命中字段、分支名次和证据。
6. 将结果交给 M6；若 M6 未启用，则直接作为最终检索结果交给 M7。

RRF 定义：

```text
RRF(image) = sum(1 / (k + rank_branch(image)))
k = 60
```

排序稳定性规则：先按 RRF 降序，再按 `image_id` 升序，保证相同输入可复现。

### 7.2 降级策略

| 故障 | 降级行为 |
|---|---|
| CLIP 模型或图像索引不可用 | 使用 BGE + BM25，两路 RRF |
| BGE 不可用 | 使用 CLIP + BM25 |
| BM25 文件损坏 | 使用两个向量分支 |
| M4 LLM 解析失败 | 使用规则解析和原始查询 |
| 任一分支查询失败 | 记录错误并继续其他分支，不返回空白页面 |
| 全部分支失败 | 返回明确错误，不伪造结果 |

响应中必须记录 `active_branches`，避免二路降级结果被误当作三路实验结果。

## 8. 计划修改的文件

| 文件 | 工作 |
|---|---|
| `configs/default.yaml` | 增加 image/text embedder、活动分支、批量大小和词表路径 |
| `configs/retrieval_aliases.yaml` | 新增同义词、否定模式和字段映射 |
| `src/anima_search/schemas.py` | 扩展查询和结果契约，保持默认值兼容 |
| `src/anima_search/indexing/documents.py` | 分离 dense/sparse 文档构建，并明确排除评测泄漏字段 |
| `src/anima_search/indexing/image_vector_index.py` | 新增 CLIP 图片索引 |
| `src/anima_search/indexing/index_manifest.py` | 新增索引版本和一致性检查 |
| `src/anima_search/retrieval/query_parser.py` | 增加规则解析、同义词和合并策略 |
| `src/anima_search/retrieval/filters.py` | 新增结构化过滤与过滤解释 |
| `src/anima_search/retrieval/fusion.py` | 保留 RRF，并补充分支名次输出 |
| `src/anima_search/retrieval/search.py` | 接入三路召回、过滤、降级和解释 |
| `scripts/build_indexes.py` | 构建三路索引，支持 `--limit` 和 `--branches` |
| `scripts/search_cli.py` | 展示活动分支、分支名次、命中字段和错误 |
| `tests/fixtures/retrieval/` | 固定小图、模拟标注和人工查询 |
| `tests/unit/test_indexing.py` | 索引和版本测试 |
| `tests/unit/test_query_parser.py` | 查询解析和失败降级测试 |
| `tests/unit/test_filters.py` | 否定/必选过滤测试 |
| `tests/unit/test_retrieval.py` | 三路 RRF、稳定排序和分支降级测试 |
| `tests/integration/test_m3_m5_pipeline.py` | 从标注到搜索结果的最小闭环 |

## 9. 分阶段实施

### Phase 0：契约和 fixture

- 冻结 M1/M2 输入字段和 M5 输出字段。
- 从 Train 选择 20 张覆盖人物、车辆、室内、街景、食物、OCR、昼夜的图片。
- 手工建立 `tests/fixtures/retrieval/annotations.jsonl`。
- 编写 15-20 条查询，至少覆盖简单、组合、否定和 OCR。

完成条件：fixture 能通过 Pydantic 校验，查询和预期相关图片由两名成员确认。

### Phase 1：完成 M3 两路基线

- 完善 BM25 和 BGE 文本索引。
- 加入索引 manifest、一致性检查和可重复构建测试。
- 先用 mock encoder 跑单元测试，再用真实 BGE 对 20 张 fixture 冒烟。

完成条件：BM25-only 和 BGE-only 均能返回确定性结果，重复构建 ID 顺序一致。

### Phase 2：补齐 CLIP 图像分支

- 实现 `ImageVectorIndex`。
- 对 20 张 fixture 完成图像编码和文本搜图。
- 记录峰值显存、构建耗时和单查询耗时。

完成条件：不发生 OOM，图像分支能独立检索，保存后重载结果一致。

### Phase 3：完成 M4

- 实现规则解析、同义词展开、否定模式和 OCR 提取。
- 接入可选本地 LLM，并保证失败回退。
- 保存解析结果用于调试和评测。

完成条件：“不要人物，寻找冷色调的雨夜城市”等固定查询解析正确；无模型时仍可检索。

### Phase 4：完成 M5

- 三路并行召回。
- 过滤后重排分支名次，再执行 RRF。
- 返回分支分数、名次、命中字段和活动分支。
- 验证任一分支失败时仍能降级搜索。

完成条件：20 张 fixture 的端到端测试通过，结果可直接交给 M6/M7。

### Phase 5：评测与冻结

- 在 Train 内部开发集选择规则和候选数量，不查看 Val 最终结果调参。
- 冻结 Val 人工 query 和 0/1/2 相关度标注。
- 运行单路、两路、三路和 M4 开关消融。
- 冻结配置、模型摘要和索引摘要。

完成条件：结果表、逐查询明细和失败案例齐全，实验可重复运行。

## 10. 计划命令

以下命令将在相应参数实现后使用：

```powershell
# 单元测试
pixi run python -m pytest tests\unit\test_indexing.py tests\unit\test_query_parser.py tests\unit\test_filters.py tests\unit\test_retrieval.py -q

# 20 张图片冒烟构建
pixi run python scripts\build_indexes.py --config configs\default.yaml --split Train --limit 20 --branches image,text,bm25

# 基础三路搜索
pixi run python scripts\search_cli.py "不要人物，寻找冷色调的雨夜城市" --split train

# M3-M5 集成测试
pixi run python -m pytest tests\integration\test_m3_m5_pipeline.py -q

# 正式构建
pixi run python scripts\build_indexes.py --config configs\default.yaml --split Train --branches image,text,bm25
pixi run python scripts\build_indexes.py --config configs\default.yaml --split Val --branches image,text,bm25

# 冻结评测集上的检索评测
pixi run python scripts\evaluate_retrieval.py --config configs\default.yaml --queries artifacts\evaluation\val_queries.jsonl --relevance artifacts\evaluation\val_relevance.csv
```

## 11. 测试与实验设计

### 11.1 单元测试

必须覆盖：

- 文档字段顺序固定，`search_queries` 不进入 Val 检索文档。
- BM25、文本向量、图像向量的保存与重载。
- 索引 ID、维度或版本不一致时拒绝加载。
- RRF 手算样例、重复分支命中和稳定排序。
- 简单、组合、否定、OCR、空查询的解析。
- LLM 不可用、超时、非法 JSON 时规则回退。
- 同义词过滤和 `uncertainty` 保护。
- 单分支失败、两路降级和全部失败。

单元测试不得下载模型或执行真实 GPU 推理，统一使用假编码器和临时索引。

### 11.2 GPU 冒烟测试

真实模型只在显式冒烟命令中加载：

1. BGE 编码 20 条文档并执行一次查询。
2. Chinese-CLIP 编码 20 张图片并执行一次文本搜图。
3. 分别记录模型加载时间、索引时间、单查询时间和峰值显存。
4. 两个模型分开运行，确认没有隐式同时常驻。

### 11.3 检索评测

| 实验 | 目的 | 公平性控制 |
|---|---|---|
| BM25-only | 精确词 baseline | 同一 corpus、query、Top-K |
| BGE-only | 文本语义 baseline | 同一 corpus、query、Top-K |
| CLIP-only | 纯图文 baseline | 同一 corpus、query、Top-K |
| BM25 + BGE | 验证现有两路系统 | 相同 RRF 参数 |
| CLIP + BGE + BM25 | 主方法 | 相同候选数和结果数 |
| 三路但关闭 M4 过滤 | 衡量查询理解贡献 | 仅切换解析/过滤开关 |
| 三路并开启 M4 | 完整 M3-M5 | 冻结配置和索引 |

主指标：

- Recall@1/5/10
- MRR
- nDCG@10
- mAP

工程指标：

- 平均查询延迟和 P95 查询延迟
- 三个分支各自耗时
- 索引构建时间和磁盘大小
- 峰值显存
- 降级发生次数

泄漏控制：

- Train 用于开发和规则调整；Val 只用于冻结后的最终评测。
- Val query 必须人工改写和确认，不能直接复制 `search_queries`。
- 所有方法使用同一 query、相关度文件、候选数量和指标实现。
- 不根据 Val 最终结果修改 RRF 参数或同义词表。

结果稳定性：

- 索引顺序按 `image_id` 固定。
- fixture 和抽样使用配置中的固定 seed `20260802`。
- 无随机模型参与基础检索；若 M4 启用 LLM，解析结果必须缓存并在各消融组复用。

## 12. 预期产物

```text
artifacts/indexes/train/
artifacts/indexes/val/
artifacts/evaluation/retrieval_metrics.json
artifacts/evaluation/retrieval_details.csv
artifacts/evaluation/retrieval_failures.jsonl
artifacts/evaluation/ablation_results.csv
artifacts/evaluation/hardware_profile.json
```

建议报告图表：

```text
report/figures/retrieval_ablation.pdf
report/figures/retrieval_by_query_type.pdf
report/figures/latency_quality_tradeoff.pdf
report/figures/retrieval_failure_cases.pdf
```

所有数值在实验运行后填写，不在实施前预设结果。

## 13. 验收标准

M3-M5 完成需要同时满足：

1. 正式标注可通过单条命令构建三路索引，且每个索引包含相同有效 `image_id` 集合。
2. 20 张 fixture 的简单、组合、否定和 OCR 查询均能完成端到端检索。
3. 任一分支不可用时系统能明确降级，结果中准确记录活动分支。
4. 相同索引和查询重复执行时结果顺序完全一致。
5. 建索引和在线检索均不发生 8 GB 显存 OOM。
6. 基础检索的 P95 延迟在模型已加载条件下达到可交互水平，实际数值记录在 `hardware_profile.json`。
7. 冻结 Val 集上完成全部 baseline 和消融，报告 Recall、MRR、nDCG、mAP、延迟和显存。
8. 若三路 RRF 未超过最佳单路 baseline，不隐藏结果；优先检查图像编码器、文档字段和 query 分类，并保留最佳单路作为产品降级配置。

## 14. 风险与处理

| 风险 | 检测 | 处理 |
|---|---|---|
| M1 Schema 后续变化 | Pydantic 或索引 manifest 校验失败 | 通过适配层迁移，不在检索逻辑中散布字段兼容代码 |
| Chinese-CLIP 尚未准备 | 图像分支冒烟失败 | 先完成两路框架；最终验收前补本地图像模型 |
| 8 GB 显存 OOM | GPU 冒烟和峰值显存记录 | 降低 batch，FP16，模型分阶段加载，FAISS 放 CPU |
| 中文分词导致精确词召回差 | OCR/专名失败案例 | 增加用户词典或字符级补充，不直接修改 Val query |
| 否定过滤误杀 | 按 query 类型分析失败案例 | 只排除明确存在字段，对 uncertainty 保守处理 |
| 标注缺失导致硬过滤漏召回 | 过滤前后 Recall 对比 | 普通条件优先软检索，只有明确规则使用硬过滤 |
| 三路原始分数不可比 | 单路分数分布差异 | 使用 RRF，不直接相加原始分数 |
| 循环评测 | query 与标注 `search_queries` 重合 | Val 人工改写、冻结后评测、保存泄漏检查报告 |
| M4 LLM 输出不稳定 | 相同查询解析不一致 | 规则优先、温度 0、缓存解析结果、失败回退 |

## 15. 组员交接清单

- [ ] M1/M2 负责人确认 `ImageAnnotation` 字段和最终文件路径。
- [ ] 检索负责人提交三路索引、manifest、查询解析、过滤和 RRF 代码。
- [ ] 评测负责人冻结 Val query 与 0/1/2 相关度文件。
- [ ] 每次正式实验保存配置摘要、模型摘要、索引摘要和代码提交 ID。
- [ ] 向 M6/M7 负责人提供 `SearchQuery`、`SearchResult` 示例和降级语义。
- [ ] README 补充三路建索引、搜索、评测和常见 OOM 命令。

## 16. 转写为最终报告

实验完成后：

1. 将本方案的目标架构和模块接口整理为 Method 中的 M3-M5 小节。
2. 将单路、两路、三路和 M4 开关实验写入 Result 的检索消融表。
3. 将按查询类型的效果、延迟/显存和失败案例写入 Discussion。
4. 明确说明所有方法共享同一冻结 Val 评测集，未使用生成 `search_queries` 作为最终 query。
5. 如主方法没有在所有查询类型上占优，重点分析其在组合、否定和 OCR 查询上的适用边界。
