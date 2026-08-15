# M6 Top-20 Listwise 重排与 8GB 性能测试

日期：2026-08-11
硬件：RTX 4060 Laptop 8GB
环境：Conda `vlm-course`
模型：Qwen3-VL-2B-Instruct，FP16

## 1. 实现结论

M6 现在同时支持两种重排方式：

- pointwise：每张候选图片单独调用一次 Qwen-VL；
- listwise：把最多 20 张候选缩放并拼成一张带编号和 image_id 的 contact sheet，
  一次调用返回整体排序。

默认配置仍为 `pointwise`，因此不会改变已有搜索行为。设置
`retrieval.rerank_method: listwise` 后，SearchService 会切换到 listwise。

20 张候选的 contact sheet 为 5 列、每格 192 像素，原始画布为 960×768。
`QwenVLClient` 继续遵守 `runtime.max_image_pixels`，在 8GB benchmark 配置中
限制为 262,144 像素。检索编码器在加载 Qwen 前释放，避免 Chinese-CLIP 与
Qwen 同时常驻显存。

## 2. 输出校验与降级

模型输出必须是 JSON 对象，`ranking` 中只能出现当前候选的 image_id，或可安全
映射到当前候选的 01–20 编号。

本地校验规则：

1. 未知 ID、越界编号、空数组或无效 JSON 是硬失败；
2. 重复 ID 去重；
3. 遗漏 ID 按原检索顺序追加到末尾，并赋 0 分；
4. 硬失败和“部分回退”分别统计；
5. 无论模型输出如何，成功返回时都必须包含输入候选各一次。

这样不会把 18/20 的模型输出静默当成完整排序，也不会因为少数尾部遗漏导致整个
搜索页面不可用。

## 3. 测试设置

- 候选来源：Val image-only Chinese-CLIP 索引；
- 查询数：3；
- 类别：2 条 simple、1 条 compositional；
- 每条查询：Top-20；
- 重复：1 次；
- pointwise 调用：60 次；
- listwise 调用：3 次；
- 正式计时前：使用同一 Qwen 做 1 次不计时单图 warm-up；
- relevance judgments：无。

warm-up 用于排除 Qwen 懒加载只计入 pointwise 的不公平情况。索引、逐调用 JSON
和模型权重均位于 Git 忽略的 `artifacts/` 或模型目录中。

## 4. 公平热态结果

| 指标 | Pointwise | Listwise |
|---|---:|---:|
| 查询数 | 3 | 3 |
| 模型调用数 | 60 | 3 |
| 平均每查询延迟 | 44.377 s | 9.143 s |
| 总计时延迟 | 133.131 s | 27.429 s |
| 硬失败率 | 0% | 0% |
| 部分回退率 | 不适用 | 33.3%（1/3） |
| 峰值 CUDA 显存 | 4.039 GiB | 4.092 GiB |

按总延迟计算，listwise 在这三条查询上约为 pointwise 的 **4.85×**。listwise
峰值显存仅高约 0.053 GiB，仍明显低于 8GB。

发生部分回退的 compositional 查询中，模型重复了 2 个 ID，并遗漏 3 个 ID；
系统去重后把遗漏项按原检索顺序补到末尾。该次不是硬失败，但报告中保留
`degraded=true`，不能与完整模型排序混为一谈。

## 5. 复现

```bash
conda run -n vlm-course python scripts/benchmark_listwise_top20.py \
  --config configs/benchmark_8gb.yaml \
  --split val --branches image \
  --top-k 20 --query-limit 3 --repeats 1 \
  --output artifacts/evaluation/m6_listwise_top20_warm_3queries_8gb.json
```

切换搜索服务到 listwise：

```yaml
retrieval:
  rerank_method: listwise
  rerank_count: 20
  result_count: 8
```

## 6. 结论边界

本实验能证明：

- RTX 4060 Laptop 8GB 可以运行 Top-20 listwise；
- 单张 contact sheet 方案没有 OOM；
- listwise 显著减少模型调用次数和本地延迟；
- 输出异常存在可测试、可审计的降级路径。

本实验不能证明 listwise 的检索质量优于 pointwise 或无重排。正式 A6 仍需要人工
relevance judgments，并在同一候选池上比较 MRR、nDCG、Recall 和失败/降级率。
