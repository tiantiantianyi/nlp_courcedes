# AskAlbum M3--M7 最终 Demo 操作手册

本文固定演示环境、五类查询、M6/M7 操作顺序和三分钟录屏节奏。查询来自 `evaluation/formal_val_100/queries.jsonl` 中已审核的正式 100 条集合，并复制到 `configs/final_demo_queries.jsonl`。`expected_source_image_id` 只用于现场核对来源正例是否出现，不等价于候选级人工 qrels，也不能单独证明最终 Recall、MRR 或 nDCG。

## 1. 演示边界

最终 Demo 覆盖：M3 三路索引读取、M4 五类查询理解、M5 融合结果、M6 可选视觉重排，以及 M7 引用问答、拒答、3--8 图自动排序、缺图补全与 AI 标识。M0--M2 不在张添翼的个人演示归属范围内；M3--M5 应表述为“在队友代码基础上完成迁移、接口打通与验证”。

演示结果是一次可复现运行记录。正式 A5/A6 质量结论必须等候候选级 `0/1/2` qrels 冻结后，从正式 JSON 产物读取。

## 2. 演示前环境检查

在仓库根目录执行。以下命令只读数据、模型、索引与 GPU 状态：

```bash
test -d ../Val
test -f artifacts/indexes/val/annotations.json
test -f artifacts/indexes/val/manifest.json
test -d models/chinese-clip-vit-base-patch16
test -d models/bge-small-zh-v1.5
test -d Qwen--Qwen3-VL-2B-Instruct/snapshots/master
test -d stablediffusion
nvidia-smi
```

检查固定查询 JSONL 可解析、正好五条且五类各一条；该命令只读：

```bash
conda run -n vlm-course python -c \
  "import json,pathlib; p=pathlib.Path(\"configs/final_demo_queries.jsonl\"); rows=[json.loads(x) for x in p.read_text(encoding=\"utf-8\").splitlines() if x.strip()]; assert len(rows)==5; assert {r[\"category\"] for r in rows}=={\"simple\",\"compositional\",\"negative\",\"count\",\"ocr\"}; print(rows)"
```

运行目录流水线 dry-run。它只输出计划，不建索引、不加载模型、不创建工作区：

```bash
conda run -n vlm-course python run.py --input_dir ../Val --dry-run --launch
```

确认 `artifacts/generated/` 可写；该命令只会创建空目录：

```bash
mkdir -p artifacts/generated
```

关闭通知和无关窗口，浏览器缩放设为 110%--125%。不要在录屏中展示 API Key、代理、私人路径或通知。

## 3. 启动与预热

推荐使用 8GB 配置启动；服务只在本机监听，普通检索默认不启用 Qwen 重排：

```bash
env -u ALL_PROXY -u all_proxy \
  conda run -n vlm-course python scripts/launch_app.py \
  --config configs/benchmark_8gb.yaml --split val --host 127.0.0.1 --port 7860
```

访问 <http://127.0.0.1:7860/>。另一个终端执行只读健康检查：

```bash
curl --fail --silent --show-error http://127.0.0.1:7860/ >/dev/null
```

录屏前预热一次：搜索“雨夜城市街道”，勾选 Qwen3-VL 视觉重排并执行；随后在 M7 中选前三张，问“这些图片中能直接确认哪些共同场景？”。最后取消重排。预热会加载模型和产生 CUDA 缓存，但不改索引；不要把预热延迟当作热启动延迟。

## 4. 五类固定检索

依次在“探索相册”输入下表查询。前四个保持视觉重排关闭，组合查询可以额外演示一次勾选重排后的次序变化。每次展开“查看融合分数与匹配解释”，确认结果结构完整。

| 顺序 | 类别 | Query ID | 查询文本 | 预期来源图 |
|---:|---|---|---|---|
| 1 | 简单 | q007 | 长颈鹿进食 | val-2033 |
| 2 | 组合 | q001 | 一只在窗边趴着的表情严肃的猫 | val-2007 |
| 3 | 否定 | q072 | 没有人物的乡间田野和远山 | val-2340 |
| 4 | 计数 | q082 | 桌上摆着七个烤生蚝 | val-2138 |
| 5 | OCR | q093 | 写有“库仑定律”的物理笔记 | val-2111 |

检索验收：

- 页面返回 1--8 张图片，无 Error 弹窗。
- 预期来源图若出现，记录其可见排名；若未进入 Top-8，如实记录，不临时换查询。
- 结果 JSON 每项具有 `image_id`、`relative_path`、`fused_score`、`branch_scores`、`branch_ranks`、`matched_fields`、`active_branches`、`evidence`、`mismatch`、`source`。
- 开启 M6 后还应具有 `rerank_score`；未开启时该字段为 null 是正常行为。
- 不把融合分数或 VLM 自评分解释成相关性标签。

## 5. 候选选择与带引用问答

1. 在 q001 或“雨夜城市街道”的结果中打开“视觉故事 · M7”。
2. “当前故事候选”默认勾选前三张；证据问答只保留 1--3 张。
3. 输入有依据问题：“这些图片中能直接确认哪些共同场景或物体？”
4. 点击“生成带引用回答”，展开“逐图证据与引用 JSON”。
5. 确认正文引用只使用当前选中图片，格式为 `[img_val-xxxx]`。

问答 JSON 应具有：`answer`、`citations`、`confidence`、`refused`、`evidence`；逐图 evidence 应具有 `image_id`、`relevant`、`facts`、`uncertainty`、`used_annotation`。

随后演示拒答：

```text
请给出这些照片拍摄的准确日期、摄影者姓名和相机序列号。
```

预期 `refused=true`，或回答明确说明证据不足且不编造具体值；`citations` 不得包含未选图片。若模型产生无法核对的具体身份、日期或设备编号，将其记录为失败案例，不要口头包装为正确。

## 6. 视觉故事：先不补图

1. 搜索“雨夜城市街道”，进入 M7。
2. 在当前候选中选择 5 张；故事允许 3--8 张，超出范围应由页面阻止。
3. 主题填“雨夜城市转场”，语气选“纪实”，Seed 保持 `20260802`。
4. 取消“自动检测并补全缺图”，点击“自动编排视觉故事”。
5. 展开“查看故事结构 JSON”。

无补图验收：

- `ordered_image_ids` 与所选 ID 集合一致，允许顺序变化。
- `sections` 顺序与 `ordered_image_ids` 一致，每节的 `source=real`、`ai_generated=false`。
- 时间线显示“原始图片”，并展示 `ordering_reason`。
- 若识别出 gap，其 `status=missing`、`source=generated`、`ai_generated=true`，但尚无生成图片。
- `disclaimer` 明确叙事不代表真实地点、身份或事件经过。

## 7. 视觉故事：缺图补全与 AI 标识

保持同一批图片、主题、语气和 Seed，只勾选“自动检测并补全缺图”后再次编排。Qwen 与 Stable Diffusion 会串行切换，8GB 显存下不要同时触发其他重排或生成请求。

已验证的备选案例是查询“雨夜城市街道”，五张候选 `val-2078`、`val-2322`、`val-2044`、`val-2199`、`val-2070`；2026-08-16 本机实跑在 `val-2044` 与 `val-2078` 之间检测到场景转场并完成补图。当前索引排序变化时，只在这五张确实出现在本次结果中时选择它们，不允许把旧产物冒充本次结果。

补图验收：

- 生成成功的 gap 同时具有 `status=generated`、`source=generated`、`ai_generated=true`、`generated_image_id`、`relative_path`。
- 时间线徽标和“故事影像序列”标题都显示“✨ AI 生成”，原图仍显示“原始图片”。
- 生成文件仅写入 `artifacts/generated/`，不进入 Val 数据集和 Git。
- 生成失败时保留 `status=failed` 与 `error`，原故事仍可查看；不得用原图伪装成生成图。
- 如果本次没有检测到 gap，应如实展示空 gaps；改用已验证五图序列再试，但不通过降低阈值伪造演示。

## 8. 三分钟录屏时间线

| 时间 | 画面与操作 | 口播重点 |
|---|---|---|
| 0:00--0:15 | 首页和“实验信息” | 本机 4060 Laptop 8GB；三路检索 + 可选 Qwen 重排 + M7 |
| 0:15--0:55 | 快速粘贴五类固定查询 | 五类来自审核集合；展示 source ID 与融合解释，不宣称最终指标 |
| 0:55--1:20 | q001 开关重排前后 | M6 只重排固定候选；分数不是人工 relevance |
| 1:20--1:50 | M7 有依据问答与拒答 | 引用白名单、逐图证据、证据不足不编造 |
| 1:50--2:20 | 5 图故事，不补图 | 3--8 图自动排序、原图来源和免责声明 |
| 2:20--2:50 | 同一故事启用补图 | Qwen/SD 串行；生成图醒目标注 AI，不混入原图 |
| 2:50--3:00 | JSON 与总结 | 展示 provenance 字段；正式 A5/A6 等候 qrels 冻结 |

若补图耗时较长，录屏前完成同配置预热，但保留真实等待过程的一部分，不剪辑成“瞬时完成”。总时长建议 2:45--3:30。

## 9. 结束检查与留证

停止服务用 `Ctrl+C`，不会删除任何文件。录屏后逐项确认：

- 视频可播放，查询、图片 ID、引用和 AI 徽标清晰。
- 没有泄露密钥、私人通知或无关路径。
- 没有把队友 M3--M5 原始实现写成张添翼个人完成。
- 没有把来源正例、模型自评分或工程 smoke 写成最终 nDCG。
- 记录实际 Git commit、配置名、索引 manifest hash、查询 ID、是否预热、是否重排和生成 Seed。
- 截图优先保存首页、五类结果、问答拒答、故事 JSON、AI 补图标识五类画面；图片和视频不默认提交 Git。

建议录像文件名：`张添翼_U202315231_AskAlbum_Final_Demo_2026-08-16.webm`。
