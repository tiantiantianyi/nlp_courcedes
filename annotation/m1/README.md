# 结构化视觉标注流水线

本目录保存课程项目实际使用的结构化视觉标注代码。它覆盖图片清单生成、本地视觉语言模型推理、原始回答审计、保守格式修复、统一校验和第三方结果导入。人工质量评测位于 [`evaluation/m1_blind_rating`](../../evaluation/m1_blind_rating/README.md)。

代码不包含原图、模型权重、API 密钥、全量标注结果或人工评审数据库。这些产物应保存在 Git 忽略的工作目录中。

## 目录内容

| 文件或目录 | 作用 |
|---|---|
| `specification/prompt_v1.md` | 三种模型共享的系统提示词和任务提示词 |
| `specification/schemas/` | 原始回答、统一处理结果和外层候选记录的 JSON Schema |
| `specification/examples/` | 合法标注对象和候选记录示例 |
| `build_m1_image_manifest.py` | 扫描训练集、验证集并生成带尺寸与摘要的图片清单 |
| `run_local_vlm_manifest.py` | 本地 Qwen3.5 或 InternVL3.5 的稳定推理入口 |
| `run_local_vlm_multigpu.sh` | 按图片分片，在多张 GPU 上并行启动推理 |
| `summarize_m1_local_run.py` | 汇总分片结果并检查运行覆盖率 |
| `audit_m1_full_runs.py` | 审计原始回答的 JSON 和字段状态 |
| `normalize_m1_candidates.py` | 执行可确定的机械修复并记录每项变更 |
| `merge_m1_normalized_overrides.py` | 合并截断样本等定向重跑结果 |
| `verify_m1_postprocess.py` | 联合检查图片覆盖、Schema 和跨字段一致性 |
| `import_qwen3_vl_8b_local_run.py` | 导入另一条标注支线交付的原始回答 |

## 环境

统一处理和测试只需要：

```bash
python -m pip install jsonschema pillow
```

本地模型推理还需要与模型兼容的 `torch`、`transformers`、`accelerate` 和模型仓库声明的依赖。可选约束解码后端分别需要 `lm-format-enforcer` 或 `xgrammar`。模型支持对 `transformers` 版本较敏感，建议使用模型官方说明中的版本并在小清单上先做冒烟测试。

## 1. 生成图片清单

以下示例假设原图位于 `<workspace>/data/train` 和 `<workspace>/data/val`，所有输出写入 `<workspace>/artifacts`：

```bash
python annotation/m1/build_m1_image_manifest.py \
  --project-root <workspace> \
  --train-dir <workspace>/data/train \
  --val-dir <workspace>/data/val \
  --output <workspace>/artifacts/m1/images.jsonl
```

使用 `--help` 查看当前脚本的完整参数。清单中的图片路径相对于 `--project-root` 解析，因此换机器时无需修改记录内容。

## 2. 运行本地模型

单进程或自行分片时直接调用：

```bash
python annotation/m1/run_local_vlm_manifest.py \
  --project-root <workspace> \
  --model-path <local-model-directory> \
  --model-id Qwen/Qwen3.5-9B \
  --processor-family qwen35 \
  --manifest <workspace>/artifacts/m1/images.jsonl \
  --prompt annotation/m1/specification/prompt_v1.md \
  --schema annotation/m1/specification/schemas/annotation_payload.schema.json \
  --candidate-schema annotation/m1/specification/schemas/candidate_record.schema.json \
  --output-dir <workspace>/artifacts/m1/qwen35 \
  --max-new-tokens 8192
```

InternVL3.5 使用 `--processor-family internvl35` 和相应的本地模型目录。推理默认关闭采样，以减少同一条件下的随机差异。

多卡入口按图片清单分片，每张 GPU 运行一个独立进程：

```bash
PROJECT_ROOT=<workspace> \
VLM_GPU_IDS="0 1 2 3" \
bash annotation/m1/run_local_vlm_multigpu.sh \
  <local-model-directory> \
  Qwen/Qwen3.5-9B \
  qwen35 \
  <workspace>/artifacts/m1/images.jsonl \
  <workspace>/artifacts/m1/qwen35
```

中断后使用相同参数重启会读取现有结果；设置 `RETRY_FAILED=1` 可重新处理失败记录。输出目录包含原始回答、逐图候选记录、分片日志和运行汇总。

## 3. 统一处理

先根据原始字段 Schema 生成清洗后 Schema：

```bash
python annotation/m1/build_m1_canonical_schema.py \
  --source annotation/m1/specification/schemas/annotation_payload.schema.json \
  --output <workspace>/artifacts/m1/annotation_payload.canonical.json
```

再对每套模型结果独立处理：

```bash
python annotation/m1/normalize_m1_candidates.py \
  --run-dir <workspace>/artifacts/m1/qwen35 \
  --schema <workspace>/artifacts/m1/annotation_payload.canonical.json \
  --output-dir <workspace>/artifacts/m1/qwen35_normalized
```

处理只执行能够由代码确定的修复，例如固定取值映射、无效引用移除、坐标越界置空和数量可靠性配对。程序不会补写场景、对象、图片文字或事件等视觉事实。包含信息删除或精度降低的记录会进入 `review_queue.jsonl`。

三套结果完成后运行联合校验：

```bash
python annotation/m1/verify_m1_postprocess.py \
  --manifest <workspace>/artifacts/m1/images.jsonl \
  --schema annotation/m1/specification/schemas/annotation_payload.canonical-v1.3.schema.json \
  --normalized Qwen3.5-9B=<workspace>/artifacts/m1/qwen35_normalized \
  --normalized InternVL3.5-14B=<workspace>/artifacts/m1/internvl35_normalized \
  --normalized Qwen3-VL-8B-Instruct=<workspace>/artifacts/m1/qwen3_vl_normalized \
  --output <workspace>/artifacts/m1/verification.json
```

## 4. 质量评测

统一处理后的三套候选交给匿名评审工具：

```text
evaluation/m1_blind_rating/
```

该工具负责同图抽样、A/B/C 匿名排列、网页评分、结果揭盲和评审一致性统计。评审数据库、冻结任务和导出结果属于实验产物，不提交到 Git。

## 测试

```bash
python -m unittest discover -s annotation/m1 -p 'test_*.py' -v
python -m unittest discover -s evaluation/m1_blind_rating -p 'test_*.py' -v
node --check evaluation/m1_blind_rating/web/app.js
```

测试覆盖 JSON Schema 构造、保守清洗、盲评任务、评分保存、结果导出和双评审一致性统计。
