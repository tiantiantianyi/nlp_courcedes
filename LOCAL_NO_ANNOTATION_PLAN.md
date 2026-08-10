# 无正式标注条件下的本地开发计划

目标：在 `annotations.jsonl` 尚未冻结时，完成可运行的 M7、纯图像检索和 M3–M7 接口；正式标注到达后仅替换或增强数据源，不重写上层应用。

状态：1–8 的代码与自动化测试已完成。正式 relevance 尚未提供，因此第 7 项只能验证工具和 dry-run，不能产出课程报告中的真实指标。

约束：

- 模拟数据只用于工程测试，不用于报告中的正式指标。
- M7 在标注缺失时按需读取原图；标注存在时优先复用标注。
- 本地 RTX 4060 Laptop 8GB 上，大模型与 Stable Diffusion 串行加载。
- 所有图片通过 `image_id + relative_path` 传递，不使用 Windows 绝对路径。
- 环境可选 Pixi 或 Conda；当前 Linux 机器推荐 `environment.yml`。

## 1. M7 无标注核心流程

- [x] 定义意图、图片证据、带引用回答和图文故事的数据契约。
- [x] 实现逐图证据提取、证据汇总、引用校验和证据不足拒答。
- [x] 实现用户选择 3–8 张图片后的分段故事生成。
- [x] 标注缺失时读取原图；标注存在时把标注作为补充上下文。

验收：不提供 annotation 也能完成“候选图片 → 带引用回答/图文故事”。

## 2. Mock SearchService

- [x] 从 Train/Val 目录生成确定性的模拟搜索结果。
- [x] 提供与真实 `SearchService` 一致的 `search()` 和图片解析接口。
- [x] 增加 mock 演示入口，确保没有索引和模型也能启动 UI。

验收：只保留原始图片时，Gradio 可启动并展示搜索结果。

## 3. 标注 Schema 适配器

- [x] 支持技术方案中的嵌套 schema。
- [x] 支持当前 `caption_verified_v4` 扁平 schema。
- [x] 统一转换为 `ImageAnnotation`，处理空字段、对象数量、OCR 和路径。
- [x] 补充转换器单元测试。

验收：两种 schema 均可进入 M3 文档构建和 M7 上下文。

## 4. Image-only CLIP 索引

- [x] 增加基于 manifest 的 image-only 构建入口。
- [x] 无 annotation 时从 manifest 获取 `image_id` 和路径。
- [x] 保存最小 annotations 快照，保持搜索服务的结果契约。
- [x] 禁止把占位快照作为 text/BM25 正式标注。

验收：只有原图和 manifest 时可构建、加载并查询图像索引。

## 5. M4 查询理解补全

- [x] 提取数量目标、数值和 `eq/gte/lte` 运算符。
- [x] 增加 `time_of_day` 与 `weather` 字段。
- [x] 增加 `soft/hard/hybrid` 正向过滤模式。
- [x] 保持否定、OCR 和 LLM 失败回退能力。

验收：“至少三辆汽车的雨夜街道”能产生可执行的结构化查询。

## 6. M6 pointwise 重排验证

- [x] 增加独立 benchmark 脚本，支持 top-k、重复次数和 JSONL 输出。
- [x] 记录单候选延迟、总延迟、失败率和 CUDA 峰值显存。
- [x] 完善重排分数、引用图片 ID 和异常降级测试。

验收：可在 top-3/top-5 候选上生成可追溯的本地基准记录。

## 7. 评估与消融流水线

- [x] 增加 P50/P95、失败率和按 query category 分组统计。
- [x] 将 A5 消融升级为四种检索路数组合的可执行评测。
- [x] 输出 JSON、CSV 和 LaTeX 表格。
- [x] 拒绝使用未人工审核、`auto_seed` 或缺少 relevance 的 query。

验收：提供正式 relevance 后，一条命令可完成 A5 基础消融并汇总结果。当前只完成代码验收，真实数值待队友交付人工 relevance 后运行。

## 8. 工程收尾

- [x] 视觉重排默认关闭，与配置和 README 一致。
- [x] 记录 Pixi/Conda 选择及 mock/image-only/full 三种模式。
- [x] 改善缺失模型、图片、索引的错误消息。
- [x] 补充 `.env`、`.venv` 和本地运行产物忽略规则。
- [x] 运行完整测试、编译检查和 Git 状态检查。

验收：干净环境能够明确选择 mock、image-only 或 full 三种模式。
