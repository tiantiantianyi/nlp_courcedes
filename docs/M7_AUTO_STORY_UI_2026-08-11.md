# M7 自动故事与缺图补全实测报告

日期：2026-08-11  
环境：RTX 4060 Laptop 8GB，Conda vlm-course，Gradio 6.22  
模型：Qwen3-VL-2B-Instruct；缺图生成使用本地 Stable Diffusion v1.5 FP16

## 1. 结论

M7 的三个输出头已经具备代码实现和本地可运行链路：

1. 带当前图片引用的证据问答与无依据拒答；
2. 3–8 张图片的自动排序、分段故事和来源追踪；
3. 叙事缺口检测、Qwen prompt 构造、Stable Diffusion 补图以及
   source=generated、ai_generated=true 的显式标识。

这表示 M7 的工程主链路已完成，但不能等价于“课程实验全部完成”。正式标注上的
排序覆盖率、故事人工质量评分和生成图风格一致性仍需在队友数据冻结后评测。
当前补图模型是适合 8GB 本机联调的 SD 1.5，不是提案点名的 SDXL/FLUX，因此属于
功能等价的阶段实现，最终报告应如实说明模型差异。

## 2. 与技术提案逐项对照

| 提案 M7 要求 | 本次实现 | 状态 |
|---|---|---|
| top-k 图文问答，引用格式 [img_id] | 已有引用白名单校验、逐图证据与显式拒答 | 已完成代码与真实 Qwen 联调 |
| 用户选择 3–8 张图片 | 服务层和 UI 同时校验选择数量 | 已完成 |
| 按 time_of_day 与场景相似度排序 | 早晨→中午→下午→黄昏→夜晚；同时间段用场景特征 Jaccard 贪心邻接 | 已完成；正式标注覆盖率待测 |
| 生成带小标题的图文游记 | Qwen 生成 title/sections，章节 ID 必须保持自动排序结果 | 已完成 |
| 发现无实拍图的叙事节点 | 检测跨时间段跳跃或低相似场景突变，默认最多两个缺口 | 已完成 |
| 文生图补全并标注“AI 生成” | Qwen→SD 串行生成；schema、时间线和 Gallery 均区分真实图与生成图 | 已完成本地实测 |
| 交互 Demo | Gradio Soft Theme、自定义相册 CSS、左侧控制与右侧故事时间线 | HTTP 冒烟通过 |

## 3. 自动排序与缺图规则

story_planner.py 对每张候选图提取时间桶和场景特征。时间字段来自 scene、
summary 与 attributes，因此既兼容正式 schema，也兼容当前适配后的标注。
当没有任何可用标注时保留用户选择顺序，不伪造拍摄时间。

缺图节点满足以下任一条件：

- 相邻图片的已知时间桶跨度至少为两个阶段；
- 两张图的场景名称不同，且对象、颜色、氛围、风格等特征的 Jaccard 相似度低于
  配置阈值 0.15。

每个缺口保存前后图片 ID、原因、生成 prompt、状态、生成文件路径和错误信息。
生成失败不会吞掉原故事，而会把状态改为 failed 并在页面显示原因。

## 4. 本地真实验证

### 4.1 Qwen 故事链路

真实查询“从早晨到夜晚的城市与自然风景”得到三张候选图：

    input_ids   = [val-2005, val-2250, val-2089]
    ordered_ids = [val-2005, val-2250, val-2089]
    sections    = 3
    gaps        = 0

这组三张当前标注没有明确时间字段，所以服务正确保留选择顺序并报告
“缺少明确时间字段”。该结果只证明真实 Qwen 故事接口可运行，不证明排序质量；
早晨、黄昏、夜晚重排与缺口生成另由带完整字段的集成测试验证。

### 4.2 真实缺图生成

为隔离验证生成桥接，使用三个真实检索结果并注入一个明确的过渡缺口，串行执行
Qwen prompt builder 与本地 Stable Diffusion：

    gap_id             = gap-real-01
    status             = generated
    source             = generated
    ai_generated       = true
    generated_image_id = generated-20260811
    relative_path      = artifacts/generated/generated-20260811.png
    seed               = 20260811
    output             = PNG 512×512 RGB
    steps              = 30

原始生成图片和 JSON 元数据位于 Git 忽略的 artifacts/generated/。这次运行验证
工程链路，不作为生成质量或风格一致性分数。

真实联调还发现 Qwen 偶尔不遵守 SD prompt 的 JSON 输出约束。构造器现已执行一次
重试，仍失败时使用长度受控的确定性 prompt，避免可选 LLM 格式错误阻断补图。

### 4.3 UI 与测试

真实 benchmark_8gb.yaml 配置启动 Gradio 后：

    GET http://127.0.0.1:7862/ → HTTP 200
    response size              → 92,479 bytes
    registered components      → 66
    registered interactions    → 7

M7 定向测试为 13 passed；加入 prompt builder 测试后，完整仓库回归为
110 passed。py_compile 与
git diff --check 均通过。

## 5. 8GB 显存策略

- 检索结束后主动释放 Chinese-CLIP encoder；
- Qwen 只负责证据、故事与英文 SD prompt；
- 进入 Stable Diffusion session 前卸载 Qwen；
- SD 使用 FP16 safetensors、attention slicing、VAE slicing 与 CPU offload；
- 模型目录和生成产物不进入 Git。

## 6. 剩余正式工作

1. 队友标注冻结后统计 time_of_day 覆盖率，并在真实图片上检查自动排序；
2. 对至少 20 个故事做连贯性、事实性、引用正确性和生成图风格一致性人工评分；
3. 如最终严格要求 SDXL/FLUX，替换 SD 1.5 并重新记录速度与显存；
4. 按 docs/M7_RECORDING_GUIDE.md 录制 2.5–3.5 分钟演示。
