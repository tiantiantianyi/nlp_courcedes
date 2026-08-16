# M3–M5 Qwen3.5 标注迁移与联调记录

日期：2026-08-16

## 1. 工作边界与归属

- M0–M2 及 Qwen3.5 canonical v1.3 标注由队友交付，张添翼未重新实现或修改。
- M3–M5 的参考实现来自队友交付包。
- 张添翼负责将其中的 Qwen3.5 adapter 与导入 CLI 选择性迁移到当前主仓库，补充回归测试，并继续完成 M3→M7 的接口联调与验收。
- 本次迁移不整体覆盖当前仓库，不引入队友交付包中的旧版 M6/M7 文件。

## 2. Canonical 数据边界

默认只读输入为：

```text
../M1_clean_annotations_v1.3/qwen3.5_9b_annotations.jsonl
```

已验收的上游事实：

| 项目 | 数量 |
|---|---:|
| Train manifest | 2000 |
| Val manifest | 369 |
| Qwen3.5 canonical 记录 | 2362 |
| 预期 Train 导入 | 1993 |
| 预期 Val 导入 | 369 |

缺失 Train 数字 ID 为 `48`、`649`、`764`、`899`、`1155`、`1217`、`1918`。
导入器不会合成这些记录，也不会自动回退到其他模型。

## 3. Adapter 映射

`adapt_annotation()` 识别含顶层 `annotation` 的 canonical v1.3 记录，并映射为项目统一的 `ImageAnnotation`：

- manifest 决定规范 `train-/val-` image ID、split、POSIX 相对路径与图片 SHA-256；
- `processed_sha256` 与 manifest 不一致时拒绝导入；
- `captions` 映射为摘要、检索种子与生成上下文；
- `entities` 映射为对象、精确数量、动作、颜色、材质和状态；
- `scene` 与 `capture_visual` 映射为场景、时间和天气等检索属性；
- `ocr`、`relations`、`event`、`subjective` 与 `uncertainties` 保留到统一字段；
- 英文枚举同时保留 code 并生成中文检索值，例如 `night → 夜晚`、`rain → 雨天`。

## 4. 导入接口

```python
import_annotations(
    source: Path,
    artifacts: Path,
    require_complete: bool = False,
) -> dict
```

输出位于 `artifacts/annotations/`，包括 Train/Val JSONL 与导入报告。源标注和原图保持只读，生成物不提交 Git。

运行方式：

```bash
conda run -n vlm-course python scripts/import_m1_qwen35.py \
  --config configs/default.yaml
```

## 5. 当前验收与后续门禁

本阶段的代码级门禁是 adapter 身份/哈希回归测试、Train/Val 导入分流测试和 `compileall`。真实 2362 条导入、三路索引、12×20 M5 导出以及 M6/M7 运行结果将在后续任务完成后追加记录；在产物生成前不声明这些实验已经完成。
