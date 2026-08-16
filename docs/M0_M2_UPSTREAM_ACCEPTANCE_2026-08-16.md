# M0–M2 上游输入验收

- 验收日期：2026-08-16
- 责任归属：队友提供，本文件不将 M0–M2 记为张添翼个人实现
- 验收边界：只读检查输入、哈希、数量与 M3 消费条件，不修改上游代码或原始数据

## Canonical 标注包

- Schema：`annotation_payload.canonical-v1.3.schema.json`
- `verification_status`：`passed`
- Manifest 图片总数：2369
- Qwen3.5-9B normalized annotations：2362
- Qwen3.5-9B 状态：2158 valid，204 valid_with_lossy_repairs，3 review_required，4 unrecoverable
- Qwen3.5-9B manifest coverage、workspace manifest hashes、processed hashes、source image identity 和 schema/semantic validation 均通过
- SHA256SUMS：schema、三份 JSONL 与 `verification.json` 全部校验为 OK

## 项目 Manifest 与预期覆盖

- Train manifest：2000
- Val manifest：369
- Qwen3.5 预期导入：Train 1993，Val 369
- 缺失 Train 数字 ID：48、649、764、899、1155、1217、1918
- M3 只能读取这些输入，不得重写原图、canonical JSONL 或上游 verification 文件

## 额外风险记录

- Qwen3-VL-8B-Instruct 上游 run 的 `source_image_hashes_exact=false`；该问题不影响本阶段选用的 Qwen3.5-9B canonical 输入，但不得据此宣称三模型均具有完全相同的源图字节身份。
- Pairwise 报告中有 83 个样本尚不适合直接三模型两两比较；这不阻止 Qwen3.5 单模型 canonical 导入，但会限制后续 A4 多模型对比结论。

## 基线验证

- Git 分支：`main`
- 实施前回归：`183 passed in 8.99s`
- 工作区：无未提交文件；本地 `main` 比 `origin/main` ahead 22
