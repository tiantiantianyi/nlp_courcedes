# M5 → M6 Top-20 候选交付接口 v1.0

状态：已确认，可作为 M5 与 M6 的联调契约
Schema 版本：`m5-to-m6-v1.0`
适用范围：M5 混合召回/融合输出 → M6 Qwen3-VL pointwise/listwise 重排输入

## 1. 目的与责任边界

本接口把 M5 和 M6 解耦。M5 负责检索与融合，并为每个查询导出按融合名次排列的
Top-20 候选；M6 只读取、校验和重排这些候选，不重新检索，也不改写 M5 的分数、名次
或原始文件。

```text
M5 三路召回与融合
        ↓ m5_to_m6_candidates.jsonl
M6 接口校验器
        ↓ 合法的逐查询 Top-20
Qwen3-VL pointwise/listwise 重排
        ↓ m6_reranked_results.jsonl
M7 问答、故事与缺图补全
```

责任划分：

- M5 负责 `query`、融合方法、Top-20、`fused_score`、三路原始分数和三路名次。
- M5 不调用 Qwen3-VL，不写入 `rerank_score`。
- M6 不重新运行 M3--M5，不根据 `fused_score` 二次排序。
- M6 保留全部 M5 字段，只新增重排分数、重排名次和降级信息。
- M7 只消费 M6 输出，不回写 M5 输入文件。

## 2. 文件格式

交付文件名建议为：

```text
artifacts/evaluation/m5_to_m6_candidates.jsonl
```

文件编码必须为 UTF-8，使用 JSONL 格式：

- 一行是一个完整 JSON 对象；
- 一个查询对应一行；
- 空行可以忽略；
- 不允许 JSON 注释、尾随逗号、`NaN`、`Infinity` 或 `-Infinity`；
- 同一文件内 `query_id` 必须唯一；
- 每一行必须包含恰好 20 个候选。

M5 交付时建议同时提供只读的索引 manifest 和检索配置快照，使 M6 能核对本接口中的
`index_manifest_sha256` 与 `config_sha256`。模型权重、图片和索引二进制文件不通过 Git
提交。

## 3. 顶层字段

| 字段 | 类型 | 必填 | 约束 |
|---|---|---:|---|
| `schema_version` | string | 是 | 固定为 `m5-to-m6-v1.0` |
| `query_id` | string | 是 | 文件内唯一；建议匹配 `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` |
| `query` | string | 是 | 原始中文查询；去除首尾空白后非空 |
| `category` | string | 是 | `simple`、`compositional`、`negative`、`count`、`ocr` 之一 |
| `split` | string | 是 | 小写 `train` 或 `val` |
| `fusion_method` | string | 是 | `rrf` 或 `weighted` |
| `top_k` | integer | 是 | 固定为 `20` |
| `annotation_version` | string | 是 | M5 建库时使用的标注版本，非空 |
| `index_manifest_sha256` | string | 是 | 对应索引 manifest 文件的 64 位小写十六进制 SHA-256 |
| `config_sha256` | string | 是 | 对应 M5 检索配置快照的 64 位小写十六进制 SHA-256 |
| `candidates` | array | 是 | 长度恰好为 20；元素结构见第 4 节 |

顶层不接受未在本表中定义的额外字段。若需要扩展字段，应提升 `schema_version`，不能在
v1.0 中静默改变语义。

## 4. 候选字段

| 字段 | 类型 | 必填 | 约束 |
|---|---|---:|---|
| `rank` | integer | 是 | M5 融合名次；当前行内依次为 1--20 |
| `image_id` | string | 是 | 当前查询内唯一；必须存在于对应索引 manifest |
| `relative_path` | string | 是 | 项目根目录相对的 POSIX 路径 |
| `fused_score` | number | 是 | M5 最终融合分数；必须为有限数值 |
| `branch_scores` | object | 是 | 有效分支到原始分数的映射；至少一个键 |
| `branch_ranks` | object | 是 | 有效分支到原始名次的映射；键集合必须与 `branch_scores` 完全相同 |
| `matched_fields` | array[string] | 是 | M5 已命中的结构化字段；没有时传空数组 |

分支约束：

- 分支名只能是 `image`、`text`、`bm25`；
- `branch_scores` 中的值必须为有限数值；
- `branch_ranks` 中的值必须是大于等于 1 的整数；
- 某一路没有召回该图片时，两个字典都不写该分支；
- 不得用分数 0 或名次 0 伪造“该分支未召回”；
- `branch_scores` 与 `branch_ranks` 的键集合不同即为接口错误。

候选数组约束：

```text
candidates[i].rank == i + 1
```

M6 按数组顺序和 `rank` 接收 M5 候选，不重新根据 `fused_score` 排序。

## 5. 图片路径与文件安全

当前项目图片位于项目仓库相邻的 `Train`、`Val` 目录，因此合法路径示例为：

```text
../Train/0.jpg
../Val/2002.jpg
```

路径必须满足：

1. 使用 `/`，禁止 Windows 反斜杠；
2. 禁止绝对路径；
3. 以项目根目录解析并消解符号链接后，必须位于配置指定的 split 图片目录；
4. `split=val` 的候选只能解析到 Val 目录，`split=train` 同理；
5. 文件必须存在、是普通文件并可由 Pillow 解码；
6. `image_id` 必须存在于同一索引 manifest，且路径应与 manifest 一致。

`../Val/2002.jpg` 中的 `..` 是当前仓库布局允许的父目录引用，但解析后的最终路径不得逃逸
配置指定的 Val 根目录。

## 6. 完整合法示例

下面是一行完整的合法结构示例。哈希和分数仅用于说明接口，不代表实验结果。

```json
{"schema_version":"m5-to-m6-v1.0","query_id":"m6-q001","query":"夜晚的城市街道","category":"simple","split":"val","fusion_method":"rrf","top_k":20,"annotation_version":"qwen35-canonical-v1.3","index_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","config_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","candidates":[{"rank":1,"image_id":"val-2002","relative_path":"../Val/2002.jpg","fused_score":0.04918,"branch_scores":{"image":0.81,"text":0.76,"bm25":4.12},"branch_ranks":{"image":1,"text":2,"bm25":7},"matched_fields":["scene","time_of_day"]},{"rank":2,"image_id":"val-2003","relative_path":"../Val/2003.jpg","fused_score":0.04865,"branch_scores":{"image":0.79,"text":0.74},"branch_ranks":{"image":2,"text":4},"matched_fields":["scene"]},{"rank":3,"image_id":"val-2004","relative_path":"../Val/2004.jpg","fused_score":0.04791,"branch_scores":{"image":0.77,"text":0.78,"bm25":3.50},"branch_ranks":{"image":4,"text":1,"bm25":12},"matched_fields":["scene","time_of_day"]},{"rank":4,"image_id":"val-2005","relative_path":"../Val/2005.jpg","fused_score":0.04682,"branch_scores":{"image":0.75,"bm25":5.21},"branch_ranks":{"image":5,"bm25":3},"matched_fields":["scene"]},{"rank":5,"image_id":"val-2006","relative_path":"../Val/2006.jpg","fused_score":0.04631,"branch_scores":{"image":0.74,"text":0.71},"branch_ranks":{"image":6,"text":8},"matched_fields":[]},{"rank":6,"image_id":"val-2007","relative_path":"../Val/2007.jpg","fused_score":0.04590,"branch_scores":{"image":0.73,"text":0.72,"bm25":3.91},"branch_ranks":{"image":7,"text":6,"bm25":9},"matched_fields":["time_of_day"]},{"rank":7,"image_id":"val-2008","relative_path":"../Val/2008.jpg","fused_score":0.04512,"branch_scores":{"text":0.70,"bm25":5.03},"branch_ranks":{"text":9,"bm25":4},"matched_fields":["scene"]},{"rank":8,"image_id":"val-2009","relative_path":"../Val/2009.jpg","fused_score":0.04487,"branch_scores":{"image":0.71,"text":0.69},"branch_ranks":{"image":9,"text":10},"matched_fields":[]},{"rank":9,"image_id":"val-2010","relative_path":"../Val/2010.jpg","fused_score":0.04433,"branch_scores":{"image":0.70,"text":0.68,"bm25":3.20},"branch_ranks":{"image":10,"text":11,"bm25":16},"matched_fields":["scene"]},{"rank":10,"image_id":"val-2011","relative_path":"../Val/2011.jpg","fused_score":0.04395,"branch_scores":{"image":0.69,"bm25":4.80},"branch_ranks":{"image":11,"bm25":5},"matched_fields":["time_of_day"]},{"rank":11,"image_id":"val-2012","relative_path":"../Val/2012.jpg","fused_score":0.04341,"branch_scores":{"image":0.68,"text":0.67},"branch_ranks":{"image":12,"text":13},"matched_fields":[]},{"rank":12,"image_id":"val-2013","relative_path":"../Val/2013.jpg","fused_score":0.04288,"branch_scores":{"text":0.66,"bm25":4.41},"branch_ranks":{"text":14,"bm25":6},"matched_fields":["scene"]},{"rank":13,"image_id":"val-2014","relative_path":"../Val/2014.jpg","fused_score":0.04237,"branch_scores":{"image":0.67,"text":0.65,"bm25":3.65},"branch_ranks":{"image":13,"text":15,"bm25":11},"matched_fields":[]},{"rank":14,"image_id":"val-2015","relative_path":"../Val/2015.jpg","fused_score":0.04192,"branch_scores":{"image":0.66,"text":0.64},"branch_ranks":{"image":14,"text":17},"matched_fields":["scene"]},{"rank":15,"image_id":"val-2016","relative_path":"../Val/2016.jpg","fused_score":0.04148,"branch_scores":{"image":0.65,"bm25":3.42},"branch_ranks":{"image":15,"bm25":14},"matched_fields":[]},{"rank":16,"image_id":"val-2017","relative_path":"../Val/2017.jpg","fused_score":0.04099,"branch_scores":{"image":0.64,"text":0.63,"bm25":3.08},"branch_ranks":{"image":16,"text":18,"bm25":18},"matched_fields":["time_of_day"]},{"rank":17,"image_id":"val-2018","relative_path":"../Val/2018.jpg","fused_score":0.04052,"branch_scores":{"text":0.62,"bm25":3.77},"branch_ranks":{"text":19,"bm25":10},"matched_fields":["scene"]},{"rank":18,"image_id":"val-2019","relative_path":"../Val/2019.jpg","fused_score":0.04007,"branch_scores":{"image":0.63,"text":0.61},"branch_ranks":{"image":18,"text":20},"matched_fields":[]},{"rank":19,"image_id":"val-2020","relative_path":"../Val/2020.jpg","fused_score":0.03961,"branch_scores":{"image":0.62,"bm25":2.91},"branch_ranks":{"image":19,"bm25":20},"matched_fields":["scene"]},{"rank":20,"image_id":"val-2021","relative_path":"../Val/2021.jpg","fused_score":0.03910,"branch_scores":{"image":0.61,"text":0.60,"bm25":2.84},"branch_ranks":{"image":20,"text":21,"bm25":22},"matched_fields":[]}]}
```

## 7. 非法示例

### 7.1 Top-20 数量错误

```json
{"top_k":20,"candidates":[{"rank":1,"image_id":"val-2002"}]}
```

错误：声明 `top_k=20`，但候选数组只有 1 个元素。

### 7.2 分支键不一致

```json
{"branch_scores":{"image":0.81,"text":0.76},"branch_ranks":{"image":1}}
```

错误：`text` 出现在 `branch_scores`，但没有出现在 `branch_ranks`。

### 7.3 候选重复

```json
{"candidates":[{"rank":1,"image_id":"val-2002"},{"rank":2,"image_id":"val-2002"}]}
```

错误：同一查询的 `image_id` 重复。

### 7.4 路径与 split 不一致

```json
{"split":"val","relative_path":"../Train/0.jpg"}
```

错误：Val 查询的候选解析到 Train 目录。

## 8. 接口校验行为

校验器采用“完整扫描、严格阻断”：

1. 逐行读取文件并收集全部错误和行号；
2. 单行错误不阻止继续检查后续行；
3. 扫描结束后，只要存在一个接口错误，命令返回非零；
4. 有接口错误时，整批数据不进入正式 M6；
5. 校验器不修改、补写或重排 M5 原文件。

稳定错误类别：

| 错误码 | 含义 |
|---|---|
| `E_JSON_PARSE` | JSON 语法错误 |
| `E_SCHEMA_VERSION` | schema 版本错误 |
| `E_REQUIRED_FIELD` | 缺少必填字段 |
| `E_UNKNOWN_FIELD` | 出现 v1.0 未定义字段 |
| `E_DUPLICATE_QUERY_ID` | 文件内 query ID 重复 |
| `E_TOP_K` | `top_k` 不是 20 |
| `E_CANDIDATE_COUNT` | 候选数量不是 20 |
| `E_RANK_SEQUENCE` | 名次不连续或与数组位置不一致 |
| `E_DUPLICATE_IMAGE_ID` | 当前查询内图片 ID 重复 |
| `E_BRANCH_KEYS` | 分数和名次的分支键不一致 |
| `E_BRANCH_NAME` | 出现未知分支名 |
| `E_NONFINITE_SCORE` | 出现 NaN/Inf 或非数值分数 |
| `E_PATH_FORMAT` | 绝对路径、反斜杠或非法路径 |
| `E_PATH_SPLIT` | 路径不位于声明 split 的数据目录 |
| `E_IMAGE_MISSING` | 图片不存在或不是普通文件 |
| `E_IMAGE_DECODE` | Pillow 无法解码图片 |
| `E_MANIFEST_MISMATCH` | image ID、路径或 manifest 哈希不一致 |

## 9. M6 模型输出降级规则

接口错误与模型错误必须区分。只有接口校验完全通过后才进入 Qwen3-VL：

- Qwen3-VL 返回未知 ID、空数组或无效 JSON：保留完整 M5 顺序，标记硬降级；
- 返回重复 ID：去重并记录 `mismatch`；
- 遗漏 ID：按 M5 原始顺序补到末尾；
- 每个输入候选在 M6 输出中必须恰好出现一次；
- 降级结果必须写入 `degraded=true` 和具体原因；
- M6 只能新增 `rerank_score`、`rerank_rank`、`degraded`、`mismatch`；
- M6 输出另存为新文件，禁止覆盖 M5 输入。

## 10. 验收命令

M6 侧校验器实现后，使用以下命令验收队友交付：

```bash
conda run -n vlm-course python scripts/validate_m5_m6_interface.py \
  --input artifacts/evaluation/m5_to_m6_candidates.jsonl \
  --m5-config-snapshot artifacts/evaluation/m5_retrieval_config.snapshot.json \
  --project-root . \
  --train-dir ../Train \
  --val-dir ../Val \
  --index-manifest artifacts/indexes/val/manifest.json \
  --report artifacts/evaluation/m5_to_m6_validation.json
```

通过条件：

- 命令退出码为 0；
- 错误数为 0；
- 所有行的 schema 版本均为 `m5-to-m6-v1.0`；
- 每行恰好 20 个候选；
- 查询总数与双方约定一致；
- 候选总数等于查询数乘以 20；
- 全部图片存在并可解码；
- 无重复 ID、非法数值、路径或 manifest 错配。

接口通过后，先选择一条查询生成 contact sheet 并完成 M6 冒烟，再执行完整 M6 性能与
降级测试。

## 11. 发给 M5 同学的最短交付清单

- [ ] 输出 UTF-8 JSONL，一查询一行；
- [ ] 每行 `schema_version=m5-to-m6-v1.0`；
- [ ] 每行恰好 20 个候选，名次连续为 1--20；
- [ ] 每个候选含唯一 `image_id`、合法 `relative_path` 和有限 `fused_score`；
- [ ] `branch_scores` 与 `branch_ranks` 必填且键集合一致；
- [ ] 未召回分支从两个字典同时省略；
- [ ] 提供标注版本、索引 manifest 哈希和配置哈希；
- [ ] 不写 `rerank_score`，不调用 Qwen3-VL；
- [ ] 同时提供对应的索引 manifest 和配置快照供验收；
- [ ] 原始交付文件只读保留，后续修订使用新文件或新版本号。
