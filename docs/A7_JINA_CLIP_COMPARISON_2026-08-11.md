# A7 Chinese-CLIP 与 jina-clip-v2 资源对比

日期：2026-08-11

环境：RTX 4060 Laptop 8GB，Conda `vlm-course`，Torch 2.13.0 + CUDA 13.0
对照模型：Chinese-CLIP ViT-B/16 与 jina-clip-v2

## 1. 与技术提案的对应关系

技术提案在图像塔章节提出 Chinese-CLIP 与 jina-clip-v2 二选一，并在消融表 A7
明确要求 `Chinese-CLIP ViT-B/16 vs jina-clip-v2`。本次完成的是 A7 的编码器适配、
可切换建库链路和本机资源对照。

当前没有人工 query relevance judgments，因此本报告只回答“能否在 8GB 机器运行、
需要多少资源”，不回答“哪个模型检索质量更好”。正式质量消融仍须复用同一批人工
query/qrels，分别报告 Recall@K、MRR、mAP 与 nDCG@10。

## 2. 已实现内容

- 新增 `JinaClipV2Encoder`，图像与文本共用 jina-clip-v2 的跨模态空间；
- 支持 32、64、128、256、512、768、1024 七种 Matryoshka 截断维度；
- image-only、full 三路建库和服务加载均可通过配置切换编码器；
- 索引元数据保存 `encoder_type`、`encoder_options`、模型指纹和构建参数；
- 新增 64 图成对资源基准脚本和无标注条件下的明确质量边界；
- 对编码器输出增加 NaN/Inf 硬校验，非有限向量不会进入 FAISS；
- 兼容 Transformers 5 移除的 `clip_loss`，并确定性重建 Jina checkpoint 未保存的
  EVA RoPE buffer；
- Linux/Windows 环境加入 `timm`、`einops` 与 `xformers==0.0.35`。

模型权重和实验 JSON/FAISS 索引均保存在 Git 忽略目录，不进入仓库。当前本地
jina-clip-v2 模型卡声明 CC-BY-NC-4.0，课程演示可用，但后续商用必须重新检查许可。
`trust_remote_code=True` 所加载的 Jina 动态代码已在本次运行前人工审查。

## 3. NaN 故障与修复

最初 jina-clip-v2 可以加载，但图像向量全部为 NaN。逐层钩子定位结果是：模型权重、
512×512 预处理像素、patch embedding 和第 0 层 LayerNorm 均为有限值；NaN 第一次
出现在视觉塔第 0 层 RoPE。

原因是 Jina EVA 实现把 `freqs_cos/freqs_sin` 注册为非持久 buffer，它们不在
checkpoint 中；当前 Transformers 5 的低内存加载会留下未初始化 buffer。修复是在
模型加载后按官方公式和当前 vision config 确定性重建 RoPE，再移动到 CUDA。

同时保留输出端 `np.isfinite()` 校验。这样即使未来模型代码或 CUDA 内核再次产生
非有限值，建库也会直接失败并给出错误，而不是生成表面可查询、实际无效的 FAISS
索引。修复后同一组 8 图冒烟和 64 图正式资源测试均返回有效向量与非空 Top-5。

## 4. 实验设计

两种编码器严格使用 Val manifest 的前 64 张相同图片和五条固定中文查询：

1. 雨夜城市；
2. 户外自然风景；
3. 室内有人；
4. 暖色调建筑；
5. 道路上的汽车。

共同设置为 float16、512 维、内积检索和 L2 归一化。Chinese-CLIP batch size 为 4，
Jina batch size 为 1；Jina 使用 xFormers 0.0.35。建库时间包含模型冷加载、64 张图
编码和 FAISS 构建；查询延迟先做一次 warm-up，再取五条查询均值。CUDA 指标使用
PyTorch `max_memory_allocated()`，不是整机 `nvidia-smi` 占用。

## 5. 64 图实测结果

| 指标 | Chinese-CLIP 512 | jina-clip-v2 512 |
|---|---:|---:|
| 本地模型目录 | 1.403 GiB | 1.628 GiB |
| 图片数 | 64 | 64 |
| batch size | 4 | 1 |
| 输出维度 | 512 | 512 |
| 冷启动建库 | 6.060 s | 10.510 s |
| 建库吞吐 | 10.560 image/s | 6.089 image/s |
| CUDA 峰值分配 | 0.394 GiB | 2.577 GiB |
| 保存索引大小 | 129.560 KiB | 129.599 KiB |
| 平均热查询 | 2.251 ms | 47.701 ms |

资源层面的可复现结论是：

- 两个模型都能在 RTX 4060 Laptop 8GB 上完成 64 图建库和查询；
- 在本次 512 维、对应 batch 设置下，Chinese-CLIP 建库更快、显存更低、查询更快；
- 两者向量维度相同，所以 FAISS 主体大小几乎相同，几十字节差异来自元数据；
- jina-clip-v2 的 89 语言、512×512 输入和 Matryoshka 能力是模型设计特性，但本次
  无标注资源测试不能证明这些特性带来本数据集上的检索质量提升。

不同模型的余弦/内积分数不在同一标尺上，不能直接比较分数绝对值；五条查询的 Top-5
也不能替代人工相关性标注。

## 6. 复现方法

Conda 环境更新后，确认模型分别位于
`models/chinese-clip-vit-base-patch16` 和 `models/jina-clip-v2`。本机使用：

```bash
conda activate vlm-course
python -m pip install xformers==0.0.35
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/benchmark_image_encoders.py \
  --config configs/benchmark_8gb.yaml \
  --limit 64 \
  --output artifacts/a7_encoder_comparison_64.json
```

外层模型启用了 `local_files_only`，但 Jina 动态代码内部仍按 Hugging Face 模型名解析
文本子模型。在所需文件已缓存时使用离线环境变量，可以避免无效代理或网络波动影响
复现。

正式索引切换配置：

```yaml
retrieval:
  image_encoder_type: jina_clip_v2
  jina_clip_truncate_dim: 512
  jina_clip_local_files_only: true
  jina_clip_image_batch_size: 1
```

切回 `chinese_clip` 即恢复默认图像塔。不同编码器的索引不能混用；切换后必须重新
构建 image 索引。

## 7. 当前完成度与剩余工作

A7 已完成代码级适配、配置切换、索引元数据、8GB 本地真实联调和 64 图资源对比。
其“工程与资源可行性”已经完成；“检索质量消融”仍等待队友提供人工审核的 query 与
relevance judgments。拿到 qrels 后应固定同一 512 维设置重建两份 Val 索引，再运行
统一评测脚本，不能复用本报告的 Top-5 作为质量结论。
