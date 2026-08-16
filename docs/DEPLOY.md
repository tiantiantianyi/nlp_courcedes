# AskAlbum 本地与 GPU 容器部署

本文档部署现有 M3--M7 应用，不重新生成课程图片、模型或索引。Docker 镜像仅包含代码和 `pixi.lock` 锁定的 Linux 环境；数据、模型和索引通过只读卷提供，只有 `artifacts/generated/` 可写。

## 1. 前提与只读检查

所有命令均从仓库根目录执行。宿主机运行需要 NVIDIA 驱动以及 Conda 或 Pixi；容器运行还需要 Docker、Compose 和 NVIDIA Container Toolkit。Windows 推荐 WSL2 + Docker Desktop，并启用 WSL 集成与 GPU 支持。

以下检查只读，不写产物：

```bash
test -d ../Val
test -f artifacts/indexes/val/annotations.json
test -f artifacts/indexes/val/manifest.json
test -d models/chinese-clip-vit-base-patch16
test -d models/bge-small-zh-v1.5
```

启用 M6/M7 重排、问答或故事时再检查 Qwen；启用缺图生成时再检查 Stable Diffusion：

```bash
test -d Qwen--Qwen3-VL-2B-Instruct/snapshots/master
test -d stablediffusion
mkdir -p artifacts/generated
```

最后一条只创建生成目录。索引不存在时，先按 README 的 image-only 或 full 流程在宿主机建库；建库会写 `artifacts/indexes/`，不要在容器的只读索引卷中建库。

## 2. Conda 启动

首次创建环境会写 Conda 环境与包缓存，但不写项目数据：

```bash
conda env create -f environment.yml
```

已有环境的更新命令会修改该 Conda 环境：

```bash
conda env update -n vlm-course -f environment.yml --prune
```

应用只读模型和索引；仅缺图补全会写 `artifacts/generated/`：

```bash
conda activate vlm-course
python scripts/launch_app.py \
  --config configs/default.yaml --split val --host 127.0.0.1 --port 7860
```

访问 <http://127.0.0.1:7860/>。普通搜索默认不加载 Qwen3-VL。

## 3. Pixi 启动

首次安装会写 `.pixi/` 锁定环境，不写课程数据或模型：

```bash
pixi install --locked
pixi run python scripts/launch_app.py \
  --config configs/default.yaml --split val --host 127.0.0.1 --port 7860
```

Windows PowerShell 命令相同。建议把仓库、Val、模型和索引放在 WSL2 的 Linux 文件系统，避免跨文件系统读取大量图片。

## 4. Docker 与 Compose

### 4.1 配置专项检查

以下命令只解析 Dockerfile/Compose，不构建索引或写课程产物；Docker 可能读取镜像元数据和本地缓存：

```bash
docker build --check .
docker compose config --quiet
```

Docker/Markdown 是配置与文档，使用 Docker 解析器和 Compose 插值检查作为专项验收；Python 行为仍由完整 pytest 回归覆盖。

### 4.2 构建与启动

构建会写 Docker 镜像层和缓存；`.dockerignore` 排除图片、模型、索引、评测产物、生成图片、密钥与 Git 元数据：

```bash
docker build -t askalbum-vlm:local .
```

首次构建需联网下载 Pixi 基础镜像及锁文件依赖。模型必须提前在宿主机准备，构建不会下载 Hugging Face 或 ModelScope 权重。

固定挂载契约：

| 宿主机路径 | 容器路径 | 权限 |
|---|---|---|
| `../Val` | `/data/Val` | 只读 |
| `./artifacts/indexes` | `/app/artifacts/indexes` | 只读 |
| `./models` | `/app/models` | 只读 |
| `./Qwen--Qwen3-VL-2B-Instruct` | `/app/Qwen--Qwen3-VL-2B-Instruct` | 只读 |
| `./stablediffusion` | `/app/stablediffusion` | 只读 |
| `./artifacts/generated` | `/app/artifacts/generated` | 可写 |

启动会创建容器和网络；故事补图会写宿主机生成目录：

```bash
docker compose up --build
docker compose up --build -d
```

停止会删除容器和网络，但保留镜像、挂载文件和生成图片：

```bash
docker compose down
docker compose logs --tail=200 askalbum
```

日志命令本身只读。

## 5. 直接 docker run

下列命令创建临时容器，只有生成目录可写：

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v "$(pwd)/../Val:/data/Val:ro" \
  -v "$(pwd)/artifacts/indexes:/app/artifacts/indexes:ro" \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/Qwen--Qwen3-VL-2B-Instruct:/app/Qwen--Qwen3-VL-2B-Instruct:ro" \
  -v "$(pwd)/stablediffusion:/app/stablediffusion:ro" \
  -v "$(pwd)/artifacts/generated:/app/artifacts/generated" \
  askalbum-vlm:local
```

PowerShell 使用 `${PWD}` 替换 `$(pwd)`；Docker Desktop 需要允许共享这些目录。

## 6. RTX 4060 Laptop 8GB 设置

- 保持 `rerank_default: false`，普通检索不自动加载 Qwen3-VL。
- Qwen3-VL 与 Stable Diffusion 串行切换，不并发问答和缺图生成。
- A6 Top-20 listwise 保持 tile 192、5 列 contact sheet。
- image batch size 默认 8；显存碎片化时依次降至 4 或 1，再重启。
- jina-clip-v2 对照使用 batch size 1 和本地离线模型。

`nvidia-smi` 只读显示驱动、GPU 与空闲显存。

## 7. 离线模式与 API Key

模型完整后可在当前 shell 启用离线变量；第一次准备模型时不要启用：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Compose 继承这两个变量，未设置时默认 0。默认 rules 后端不需要密钥；API 后端只从环境变量读取：

```bash
export SILICONFLOW_API_KEY="替换为本人的密钥"
docker compose up
```

不要把密钥写入 YAML、Dockerfile、Compose、终端截图或 Git。

## 8. 健康检查与恢复

下列命令只读 HTTP 和 GPU 状态：

```bash
curl --fail --silent --show-error http://127.0.0.1:7860/ >/dev/null
nvidia-smi
```

常见故障与处理：

- `enabled retrieval indexes are missing`：val 索引不完整，重建对应分支；重建会写索引。
- 路径不存在：检查六个挂载目录；Val 应位于仓库同级。
- 容器无 GPU：运行 `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`；首次可能拉取诊断镜像，但不写项目。
- CUDA OOM：关闭视觉重排或降低批大小，再重启应用。
- 离线加载失败：确认模型完整；首次下载时取消两个 offline 变量。
- 端口占用：改为 `7861:7860`，或停止原 7860 进程。

恢复命令只改变容器状态，不删除挂载产物：

```bash
docker compose restart askalbum
docker compose down
docker compose up -d
```

不要使用 `docker compose down -v`，也不要删除宿主机挂载目录。
