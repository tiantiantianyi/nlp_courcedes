# AskAlbum 课程设计完整交付规范

日期：2026-08-16  
依据：`../VLM_Final_Project_Technical_Proposal.md` 与当前 `nlp_courcedes` 实现  
开发分支：`main`（用户已明确批准直接在 `main` 开发）

## 1. 目标与完成定义

本阶段把已有 M3--M7 工程骨架和 Qwen3.5 canonical 标注推进为可正式提交、可复跑、可审计的课程设计。完成状态必须同时满足：

1. 对任意图片目录存在可 dry-run、可恢复的一键流水线；
2. M0--M7 核心闭环具有真实数据产物，不用 mock 结果冒充正式实验；
3. 正式检索结果来自人工审核查询和候选级相关性判断；
4. 报告中的每个质量或性能结论都能追溯到 JSON/CSV/LaTeX 产物；
5. 代码、环境、部署、Demo 和个人/团队报告均可在新机器上按文档复现；
6. 个人报告只记录张添翼实际完成或联调的 M6/M7、评测与系统收尾工作，不占用队友 M0--M5 的实现成果。

课程硬性交付优先于 Proposal 的加分项。核心交付全部通过后，才执行高成本的全量扩展实验。

## 2. 范围分解

### 2.1 子项目 A：正式检索评测与 A5/A6

把 `evaluation/manual_val_50/` 与 `evaluation/manual_val_50_assisted/` 合并为 100 条正式 query。保留每条查询的人工文本、类别、来源图、审核者与审计说明，不覆盖两批原始文件。

正式 qrels 必须支持 `0/1/2`：

- `2`：高度相关，直接满足查询；
- `1`：部分相关，只满足部分条件；
- `0`：不相关，是候选池中的显式负例。

当前人工页面丢弃 0 分记录、校验器拒绝 0 分，与 Proposal 的三级相关性冲突。本阶段先以测试驱动修复该契约。随后从 CLIP、text、BM25、RRF、加权融合以及 M6 候选中取并集，生成候选级标注任务。只有来源图 `2` 的 query 可以做单正例 Recall/MRR 冒烟，但不能作为多相关 nDCG 的最终结论。

A5 对同一 100 条 query、同一 qrels 运行五组：CLIP-only、text-only、BM25-only、RRF、归一化加权融合。A6 对同一 Top-20 候选运行 baseline、pointwise、listwise，并输出 MRR、nDCG@10、延迟、峰值显存、失败率和降级率。`rerank_quality.py` 必须接入真实 CLI，不能只停留在未跟踪单元测试。

### 2.2 子项目 B：M2 无参考验证

M2 与已有 schema/provenance 校验分离，新增三个边界清晰的组件：

1. `object_grounding`：输入原图和标注对象，输出每个对象的最高检测置信度；
2. `chair`：纯函数计算 CHAIR_i、CHAIR_s、覆盖率与阈值敏感性；
3. `echoback`：调用现有 Stable Diffusion 生成接口和独立视觉编码器，按固定 seed 计算回译相似度。

模型未安装时 CLI 必须给出明确错误或 `--dry-run` 计划；不允许静默生成伪分数。真实实验先在人工审核的 50 张子集运行，资源允许再扩大。所有生成图片保留 `source=generated` 与 `ai_generated=true`。

### 2.3 子项目 C：扩展实验与人工评估

实验分为必须完成和资源允许后完成：

- 必须完成：A1 三种 prompt、A2 路由开关、A5、A6、A7 资源与质量对照、A9 小规模领域迁移；
- 资源允许：A3 三次自洽、A4 多模型融合、A8 三阈值敏感性、自动 prompt 进化；
- 团队人工协作：参考描述、VQA 校验、Arena 盲投。

无法满足原计划规模时，报告必须给出实际样本数和限制，不能把小样冒充完整实验。医学影像只讨论迁移失效，并明确声明不用于诊断。

### 2.4 子项目 D：Demo、报告与部署交付

最终 Demo 使用 full annotation 与正式索引，覆盖：

1. 简单/组合/否定/数量/OCR 查询；
2. 带图片引用且无证据时拒答的问答；
3. 3--8 图自动排序的视觉故事；
4. 缺图补全和醒目的 AI 生成标识。

仓库必须补齐 `docs/DEPLOY.md`、`Dockerfile`、最终 README 状态、环境检查命令和演示脚本。团队报告采用 LaTeX 学术结构且正文不超过 20 页。张添翼个人报告不是从零新建：旧版位于 `.worktrees/personal-report` 的 `report/zhang-tianyi` 分支，已有 6 页 PDF、LaTeX 正文与参考文献。最终阶段应先把该分支的报告文件安全合入 `main`，再更新旧版的阶段性数字和未完成结论；文件名继续包含姓名与学号，并只陈述本人工作。PPT 和不超过 3 分钟视频为加分项，在核心交付之后制作。

## 3. 数据流与产物

正式实验数据流固定为：

```text
两批人工 query
  -> 合并且校验的 100-query 集
  -> 多检索分支候选池
  -> 人工 0/1/2 qrels
  -> A5 五组检索结果
  -> M5 Top-20
  -> A6 baseline/pointwise/listwise
  -> M7 问答与故事
  -> JSON/CSV/LaTeX/图表
  -> 团队报告与个人报告
```

正式文件放在 `artifacts/evaluation/formal/`，本地模型、图片和大体积中间索引继续由 `.gitignore` 排除。仓库提交轻量的 schema、配置、汇总 JSON/CSV/LaTeX、报告源码和必要截图；不提交原始课程图片、模型权重或生成缓存。

## 4. 失败处理与真实性约束

- 任何 query 缺少人工审核、来源图 `2` 或候选 qrels 时，正式评测立即拒绝运行；
- M5/M6 候选集合不一致时不得计算排序质量；
- listwise 重复/遗漏保持现有可审计降级，不得隐藏 66.7% 的历史部分降级率；
- 缺 API Key 时 M4 允许回退到 rules，但报告不能写成真实免费 API 联调完成；
- M2 模型缺失时记录为资源阻塞，不生成随机或模型标注替代值；
- AI 辅助起草的数据必须保留起草说明；正式人工结论只由实际审核者署名；
- 另一个团队成员应独立复核至少 50 条 query，满足 Proposal 的双人构造意图。

## 5. 测试与验收

每个子项目采用测试先行，并使用以下总验收：

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q
conda run -n vlm-course python -m compileall -q src scripts tests
python run.py --input_dir ../Val --dry-run --launch
```

当前基线是 214 tests passed。新增功能不得降低该基线。正式产物验收还必须检查：100 条 query、所有 query 有候选级 qrels、A5 五组齐全、A6 三组齐全、M2 报告无伪分数、Demo 四条路径可操作、LaTeX 可编译、Git 工作区只包含有意提交内容。

## 6. 非目标与边界

- 不重写队友已经验收的 M0--M5 核心实现；只通过公开接口消费并补足评测/交付；
- 不把 7B+ 模型全参训练作为完成条件；如需满足“训练”展示，使用已有轻量检索器训练脚本和小规模可复现实验；
- 不为 2369 张数据引入不必要的向量数据库或分布式系统；
- 不在核心实验完成前花时间重做 UI 视觉主题、PPT 动画或全量 prompt 进化。
