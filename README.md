# 忆见 AskAlbum 视觉语言相册检索系统

本仓库实现nlp课程设计中的：多路索引、结构化查询理解、RRF 混合召回、可选 VLM pointwise 重排，以及带图片引用的问答和视觉故事。正式标注尚未冻结时，也可以使用 mock 或 image-only 模式继续开发。

视觉重排默认关闭，避免在普通搜索时意外加载 Qwen-VL。模拟数据和 image-only 占位记录只用于工程验证，不能作为课程报告的正式检索指标。

## 1. 三种运行模式

| 模式 | 原图 | 正式标注 | 索引 | 模型 | 用途 |
|---|---:|---:|---:|---:|---|
| mock | 需要 | 不需要 | 不需要 | 不需要 | UI、接口和 M7 联调 |
| image-only | 需要 | 不需要 | image 分支 | Chinese-CLIP | 无标注图文检索 |
| full annotation | 需要 | 需要 | image/text/BM25 | 对应编码器，可选 Qwen | 正式检索和评测 |

### 任意图片目录一键运行

根目录的 `run.py` 和 `scripts/run_all.sh` 对应技术方案要求的隐藏测试集入口。默认 `full` 模式按顺序执行：

```text
目录扫描 → manifest → M0 场景路由 → M1 场景专用标注
→ M3 image/text/BM25 索引 → M4–M7 可运行服务
```

先查看执行计划，不加载模型或写产物：

```bash
python run.py --input_dir /absolute/path/to/images --dry-run --launch
```

无正式标注时只构建 Chinese-CLIP 图像索引：

```bash
python run.py \
  --input_dir /absolute/path/to/images \
  --mode image-only --launch
```

完整自动标注和三路索引：

```bash
python run.py --input_dir /absolute/path/to/images --mode full --launch
```

中断后使用相同参数继续：

```bash
python run.py --input_dir /absolute/path/to/images --mode full --launch --resume
```

小规模冒烟可增加 `--limit 5`。默认工作区位于 `artifacts/directory_runs/`，也可以用 `--workspace` 指定隔离目录。程序不会覆盖已有运行状态；必须显式使用 `--resume` 或换一个工作区。

等价 shell 入口：

```bash
scripts/run_all.sh --input_dir /absolute/path/to/images --mode full --launch
```

M7 阶段演示录制步骤见 [`docs/M7_RECORDING_GUIDE.md`](docs/M7_RECORDING_GUIDE.md)。

## 2. 环境：Pixi 或 Conda 二选一

Pixi 可以理解为“Conda 环境管理 + 锁文件 + 项目任务”。`pixi.toml` 是环境定义，`pixi.lock` 是精确版本锁；只有执行 `pixi install` 后生成 `.pixi/`，才表示本机真正安装了项目环境。

本仓库同时提供：

- `pixi.toml` / `pixi.lock`：同时锁定 Linux 与 Windows，适合已使用 Pixi 的队友。
- `environment.yml`：适合本机 Linux 或已经熟悉 Conda 的成员。

两者不需要同时安装，选择一种即可。Pixi 配置覆盖 `linux-64` 与 `win-64`；本机已经创建好的 Conda 环境仍可继续使用。

### 2.1 Conda（当前 Linux 机器推荐）

```bash
conda env create -f environment.yml
conda activate vlm-course
python -m pytest -q
```

环境已经创建过时：

```bash
conda env update -n vlm-course -f environment.yml --prune
conda activate vlm-course
```

Windows 的 Anaconda Prompt 使用相同命令；PowerShell 需要先执行一次 `conda init powershell`。

### 2.2 Pixi（Windows 队友）

```powershell
pixi install
pixi run test
```

常用命令：

```powershell
pixi run python scripts/search_cli.py "雨夜城市" --split val
pixi run app
pixi run evaluate
```

Pixi 通常不需要 `activate`；`pixi run ...` 会自动在项目环境中执行。

## 3. 配置数据和模型路径

默认配置在 `configs/default.yaml`：

```yaml
data:
  train_dir: ../Train
  val_dir: ../Val
  artifacts_dir: artifacts
models:
  qwen_vl: Qwen--Qwen3-VL-2B-Instruct/snapshots/master
  stable_diffusion: stablediffusion
  embedder: models/bge-small-zh-v1.5
  image_embedder: models/chinese-clip-vit-base-patch16
  jina_clip_v2: models/jina-clip-v2
```

路径规则：

- 配置中的相对路径始终相对于仓库根目录解析。
- JSON/JSONL 中保存 POSIX 风格相对路径，例如 `../Train/1.jpg`，不要保存盘符绝对路径。
- Windows 示例：数据可放在仓库同级的 `Train\` 和 `Val\`，程序仍保存 `/` 分隔形式。
- Linux 示例：当前仓库若为 `/home/user/nlp_courcedes`，默认数据目录是 `/home/user/Train` 和 `/home/user/Val`。
- 不要提交原图、模型权重、`.pixi/`、`.venv/` 或本地实验输出。

需要的模型：

| 功能 | 模型 | 是否必需 |
|---|---|---|
| image 检索 | Chinese-CLIP ViT-B/16 | image-only/full image 分支必需 |
| A7 图像编码器对照 | jina-clip-v2 | 仅切换 A7 编码器时需要 |
| text 检索 | BGE small zh | full text 分支必需 |
| BM25 | 无神经网络模型 | full BM25 分支必需 |
| M6/M7 问答 | Qwen3-VL 2B | 仅重排、问答和故事需要 |
| 缺图生成 | Stable Diffusion | 仅生成图片需要 |

8GB 显存下让 Qwen-VL 与 Stable Diffusion 串行加载，不要同时常驻。

### M4 查询理解后端

retrieval.query_parser_backend 支持三种值：

| 值 | 作用 | 额外条件 |
|---|---|---|
| rules | 确定性槽位抽取，默认值 | 无 |
| local_qwen | 本地 Qwen3-VL 做语义改写和软字段补充 | 本地 Qwen 模型 |
| openai_compatible | 调用 OpenAI-compatible 免费 API | API Key 环境变量 |

直接验证规则或本地 Qwen：

    python scripts/verify_m4_query_parser.py --backend rules \
      --queries-file configs/m6_benchmark_queries.jsonl \
      --output artifacts/evaluation/m4_rules_12.json
    python scripts/verify_m4_query_parser.py --backend local_qwen \
      --query "雨夜城市街道，没有人物" \
      --output artifacts/evaluation/m4_local_qwen_smoke.json

免费 API 示例使用 SiliconFlow 兼容接口。不要把 Key 写入 YAML 或 Git：

    export SILICONFLOW_API_KEY=你的密钥
    python scripts/verify_m4_query_parser.py --backend openai_compatible \
      --query "雨夜城市街道，没有人物" \
      --output artifacts/evaluation/m4_api_smoke.json

若 API 缺 Key、超时、限流或输出不合法，查询会自动回退到规则解析；硬否定、数量、
时间、天气和 OCR 条件始终由确定性规则保护，避免 LLM 猜测变成错误硬过滤。

## 4. 模式一：mock（立即可运行）

只需要 Train 或 Val 原图：

```bash
python scripts/launch_mock_app.py --split val --port 7860
```

也可以显式指定目录：

```bash
python scripts/launch_mock_app.py --image-dir /absolute/path/to/Val --port 7860
```

打开 `http://127.0.0.1:7860`。结果顺序是确定性的模拟顺序，只验证 UI 和 `SearchResult` 契约，不代表相关性。

## 5. 模式二：image-only（无需正式标注）

先扫描原图生成 manifest：

```bash
python scripts/build_manifest.py --config configs/default.yaml
```

再构建一个 split 的 Chinese-CLIP 图像索引：

```bash
python scripts/build_image_only_index.py --config configs/default.yaml --split Val
```

查询时明确只启用 image 分支：

```bash
python scripts/search_cli.py "至少三辆汽车的雨夜街道" --split val --branches image
```

该模式生成的最小 annotations 快照会标记为 `image-only-manifest-v1`。它只是服务契约占位数据，不是 M1/M2 正式标注，不能用于 text/BM25 或正式指标。

## 6. 模式三：full annotation

把正式标注放到：

```text
artifacts/annotations/train.caption_verified_v4.jsonl
artifacts/annotations/val.caption_verified_v4.jsonl
```

构建三路索引：

```bash
python scripts/build_indexes.py --config configs/default.yaml --split Train --branches image,text,bm25
python scripts/build_indexes.py --config configs/default.yaml --split Val --branches image,text,bm25
```

运行检索和 UI：

```bash
python scripts/search_cli.py "不要人物，寻找冷色调的雨夜城市" --split val
python scripts/launch_app.py --split val --port 7860
```

只有显式增加 `--rerank` 或勾选 UI 中的 Qwen3-VL 开关时，才加载视觉重排模型：

```bash
python scripts/search_cli.py "雨夜城市" --split val --rerank
```

## 7. M6 pointwise/listwise 基准

M6 不需要 relevance 标注，但需要可查询的索引、候选原图和 Qwen-VL。原有
pointwise Top-3/5 基准：

```bash
python scripts/benchmark_reranker.py "雨夜城市" \
  --split val --top-k 3 --repeats 3 \
  --output artifacts/evaluation/reranker_top3.jsonl
```

Top-20 pointwise 与单张 contact sheet listwise 对照：

```bash
python scripts/benchmark_listwise_top20.py \
  --config configs/benchmark_8gb.yaml \
  --top-k 20 --query-limit 3 --repeats 1
```

输出包括模型调用次数、查询延迟、硬失败率、部分回退率和 CUDA 峰值显存。
没有人工 relevance judgments 时不评价排序质量提升。RTX 4060 Laptop 8GB 的
实跑结果和边界见
[`docs/M6_LISTWISE_TOP20_2026-08-11.md`](docs/M6_LISTWISE_TOP20_2026-08-11.md)。

## 8. 正式评测和 A5 消融

先生成评测种子，再由未参与标注模块的成员人工改写、分类、补充 relevance，并把 `reviewed` 改为 `true`：

```bash
python scripts/create_eval_set.py --config configs/default.yaml --count 100
```

程序会拒绝未审核、`auto_seed` 或缺少 relevance 的查询。正式评测：

```bash
python scripts/evaluate_retrieval.py \
  --queries artifacts/evaluation/val_queries.jsonl \
  --relevance artifacts/evaluation/val_relevance.csv
```

输出总体与 query category 分组的 Recall@K、MRR、mAP、nDCG@10、平均/P50/P95 延迟和失败率，并写出 JSON、CSV、LaTeX 与失败明细。

A5 的五组实验是 CLIP-only、text-only、BM25-only、三路 RRF 和三路归一化加权融合：

```bash
python scripts/run_ablation.py --dry-run
python scripts/run_ablation.py \
  --queries artifacts/evaluation/val_queries.jsonl \
  --relevance artifacts/evaluation/val_relevance.csv
```

当前两批已审核查询可合并为只含来源图正例的正式 100 条集合，再从五类中各选
10 条，通过五个 A5 变体构建候选池：

```bash
conda run -n vlm-course python scripts/prepare_formal_eval.py \
  --queries evaluation/manual_val_50/queries.jsonl evaluation/manual_val_50_assisted/queries.jsonl \
  --relevance evaluation/manual_val_50/relevance.csv evaluation/manual_val_50_assisted/relevance.csv \
  --output-dir evaluation/formal_val_100 --expected-count 100

env -u ALL_PROXY -u all_proxy conda run -n vlm-course \
  python scripts/build_relevance_pool.py \
  --queries evaluation/formal_val_100/queries.jsonl \
  --graded-query-count 50 --per-variant-k 5 --candidate-cap 25 \
  --output artifacts/evaluation/formal/relevance_pool.jsonl
```

候选池生成后，在本机打开候选级 `0/1/2` 标注页：

```bash
env -u ALL_PROXY -u all_proxy conda run -n vlm-course \
  python scripts/launch_candidate_annotator.py \
  --pool artifacts/evaluation/formal/relevance_pool.jsonl \
  --output artifacts/evaluation/formal/candidate_relevance.csv --port 7864
```

绿色边框来源图必须为 2；其余候选不会被默认标成 0，必须逐图判断。只有当前
查询的每个候选都有显式等级、标注者非空且勾选“已逐图审核”，该查询才计入完成
进度。`artifacts/evaluation/` 默认不提交 Git，避免把中间运行产物误当成冻结结论。

没有正式 relevance 文件时只能完成代码和 dry-run，不能生成课程报告结论。

在正式 relevance 尚未完成时，可运行仅比较排名差异与延迟的工程对照：

```bash
python scripts/compare_fusion_methods.py \
  --queries configs/m6_benchmark_queries.jsonl \
  --top-k 8
```

该命令报告 Top-K 重合率、共同结果平均位次变化和两种融合延迟，不输出或暗示
Recall、MRR、mAP、nDCG 等质量结论。20 张本地小样的实跑记录见
[`docs/M5_FUSION_COMPARISON_2026-08-11.md`](docs/M5_FUSION_COMPARISON_2026-08-11.md)。

## 9. A7 Chinese-CLIP / jina-clip-v2 对照

配置 `retrieval.image_encoder_type` 可取 `chinese_clip` 或 `jina_clip_v2`。Jina
适配器支持 32/64/128/256/512/768/1024 Matryoshka 截断维度，并把编码器类型、
维度选项和模型指纹写入索引元数据。8GB 配置建议 Jina batch size 1。

本地模型齐全时运行 64 张配对资源测试：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/benchmark_image_encoders.py \
  --limit 64 --output artifacts/a7_encoder_comparison_64.json
```

该脚本只比较模型大小、建库速度、峰值显存、向量维度、索引大小和热查询延迟。
没有人工 relevance judgments 时，Top-5 仅用于检查向量有效和检索链路可用，不能据此
声称任一编码器质量更好。RTX 4060 Laptop 8GB 的实测方法、资源表和限制见
[`docs/A7_JINA_CLIP_COMPARISON_2026-08-11.md`](docs/A7_JINA_CLIP_COMPARISON_2026-08-11.md)。

## 10. 测试与检查

```bash
python -m pytest -q
python -m compileall -q src scripts tests
git status --short
```

测试不下载模型，单元测试使用 fake client、临时图片和内存索引。真实 GPU 冒烟与正式实验需要另行准备本地模型和数据。

## 11. 部署

项目支持 Conda、锁定 Pixi 环境和 NVIDIA GPU Docker/Compose。容器不包含课程图片、模型、索引、生成结果或 API Key；目录检查、挂载权限、8GB 设置、Windows 注意事项和恢复命令见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

```bash
docker build --check .
docker compose config --quiet
mkdir -p artifacts/generated
docker compose up --build
```

无标注阶段的 1–8 实施状态见 `LOCAL_NO_ANNOTATION_PLAN.md`。
