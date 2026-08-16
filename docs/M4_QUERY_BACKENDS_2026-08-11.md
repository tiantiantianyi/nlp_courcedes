# M4 查询理解三后端实测报告

日期：2026-08-11  
环境：RTX 4060 Laptop 8GB，Conda vlm-course  
本地模型：Qwen3-VL-2B-Instruct

## 1. 实现结论

M4 现在支持 rules、local_qwen 与 openai_compatible 三种可配置后端：

- rules：始终可用的确定性解析；
- local_qwen：使用本地 Qwen 做 semantic_text 改写与软描述字段补充；
- openai_compatible：可接 SiliconFlow、OpenRouter 或其他兼容
  /chat/completions 的免费/低成本文本接口。

无论 LLM 是否成功，确定性规则都会先提取否定、数量、场景、时间、天气、颜色与
OCR。LLM 失败时不阻断检索，而是回退规则结果。

## 2. 真实本地 Qwen 联调

### 2.1 首次暴露的问题

第一轮真实运行中，Qwen 把 scene、mood 等数组字段输出为字符串，三条查询中两条
触发 schema 回退；另一次还把提示词示例中的“不要、排除、没有”错误写入
excluded_terms。

修复包括：

1. 对 LLM 的数组字段执行 string→list、null→空数组归一化；
2. 收紧 prompt，明确数组、计数和否定字段契约；
3. 硬过滤只采用确定性规则结果；
4. LLM 仅补充 actions、mood、style 等软字段和 semantic_text；
5. 记录 effective_backend 与 fallback_error，便于报告真实回退率。

### 2.2 修复后结果

同一模型、同一组三查询的最终结果：

| 查询类型 | 查询 | 有效后端 | 回退 | 耗时 |
|---|---|---|---:|---:|
| 否定复合 | 不要人物，寻找冷色调的雨夜城市 | llm | 否 | 10.085 s |
| 数量 | 至少三辆车的城市街景 | llm | 否 | 2.714 s |
| OCR | 找招牌写着“老王面馆”的照片 | llm | 否 | 2.680 s |

首条包含模型冷启动；两条热调用均值约 2.697 s。最终 LLM 成功率为 3/3，
规则回退率为 0/3。

关键硬槽位保持正确：

- 否定查询：excluded_terms=[人物]，time_of_day=[夜晚]，
  weather=[雨天]，colors=[冷色]；
- 数量查询：count_target=汽车，count_value=3，count_operator=gte；
- OCR 查询：ocr_terms=[老王面馆]。

这只是接口与结构正确性验证，不是查询理解准确率。正式准确率需要独立人工槽位真值。

## 3. 免费 API 模式

OpenAICompatibleTextClient 已实现：

- Bearer Key 只从指定环境变量读取；
- 标准 /chat/completions POST；
- model、messages、temperature=0 与 max_tokens；
- 字符串和多段文本 response content 兼容；
- 超时及 429/HTTP/网络/JSON 错误重试；
- 失败后由 QueryParser 自动回退规则。

单元测试使用可控 HTTP opener 验证了 URL、Authorization header、请求 JSON、
响应解析、瞬时网络失败重试以及缺 Key 报错。SearchService 测试同时证明 API 后端
不会加载本地 Qwen。

当前机器检查到以下 Key 均不存在：SILICONFLOW_API_KEY、OPENAI_API_KEY、
DASHSCOPE_API_KEY、DEEPSEEK_API_KEY、OPENROUTER_API_KEY。因此不能诚实声称已经
完成外部免费服务调用。

在无 Key 条件下运行 openai_compatible 后端，三条查询均在 0.005 秒内安全回退
到规则解析，并明确记录：

    RuntimeError: missing API key; set environment variable SILICONFLOW_API_KEY

补齐免费 Key 后执行：

    export SILICONFLOW_API_KEY=你的密钥
    python scripts/verify_m4_query_parser.py --backend openai_compatible \
      --query "雨夜城市街道，没有人物" \
      --output artifacts/evaluation/m4_api_smoke.json

配置默认值为：

    retrieval:
      query_parser_backend: openai_compatible
      query_parser_api:
        base_url: https://api.siliconflow.cn/v1
        model: Qwen/Qwen2.5-7B-Instruct
        api_key_env: SILICONFLOW_API_KEY
        timeout_seconds: 30
        max_retries: 2

模型名称应以申请 Key 时平台当前可用模型列表为准。

## 4. 与技术提案 M4 的符合度

| 提案要求 | 当前状态 |
|---|---|
| semantic_text 稠密检索文本 | 已实现规则清洗与可选 LLM 改写 |
| 场景、时间、颜色、数量、OCR hard filters | 已实现并由规则层保护 |
| simple/compositional/negative/count/ocr 路由 | 已实现 |
| 免费 LLM API | 客户端、配置、重试、回退和协议测试完成；真实外部调用待 Key |
| 否定查询 case study | 代码可运行；正式检索效果需要 relevance judgments |

工程上 M4 本地模式已完成；免费 API 模式完成到“提供 Key 即可运行”，外部真实请求
是当前唯一非代码阻塞。不能把三条联调结果写成查询理解精度或检索质量提升。

## 5. 复现命令

提交前完整回归、编译和补丁格式检查均通过：117 passed。

    conda activate vlm-course
    python scripts/verify_m4_query_parser.py --backend rules \
      --queries-file configs/m6_benchmark_queries.jsonl \
      --output artifacts/evaluation/m4_rules_12.json
    python scripts/verify_m4_query_parser.py --backend local_qwen \
      --query "雨夜城市街道，没有人物" \
      --output artifacts/evaluation/m4_local_qwen_smoke.json
    python scripts/verify_m4_query_parser.py --backend openai_compatible \
      --query "雨夜城市街道，没有人物" \
      --output artifacts/evaluation/m4_api_smoke.json

定向测试：

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
      tests/unit/test_query_parser.py \
      tests/unit/test_m4_structured.py \
      tests/unit/test_openai_compatible.py \
      tests/unit/test_service_m4_backends.py
