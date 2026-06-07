# GraphRAG-V1

GraphRAG-V1 is a sample-level GraphRAG question-answering system for Chinese criminal-law materials. It demonstrates how structured legal data, graph paths, hybrid retrieval, query rewriting, routing, and grounded LLM answering can work together in an end-to-end application.

GraphRAG-V1 是一个面向中文刑法材料的样本级 GraphRAG 问答系统。项目展示了如何把结构化法律数据、图谱路径、混合检索、问题改写、检索路由和基于依据的 LLM 回答整合成一个端到端应用。

## What It Does / 项目能力

- Parses and normalizes sample criminal-law articles and criminal case records.
- Builds law/case chunks, retrieval chunks, graph nodes, and graph edges.
- Retrieves evidence through lexical, vector, and graph-based retrievers.
- Merges candidates with a hybrid reranker and keeps citations traceable.
- Uses a Query Rewriter LoRA to rewrite multi-turn questions and extract structured intent.
- Uses a Router LoRA to decide whether retrieval is needed and which sources to retrieve.
- Uses LangGraph to orchestrate memory, rewriting, routing, retrieval, evidence checks, answering, fallback, logs, timing, and token usage.
- Generates grounded answers with citation IDs, graph paths, warnings, and safe fallback behavior.
- Provides a React frontend with login, conversation history, answer evidence, orchestration logs, timing, and token display.

- 解析并规范化刑法条文样本和刑事案例样本。
- 构建法条/案例 chunk、检索 chunk、图节点和图边。
- 通过词法检索、向量检索和图谱检索召回依据。
- 使用混合 rerank 合并候选，并保留可追溯引用。
- 使用 Query Rewriter LoRA 进行多轮问题改写和结构化意图抽取。
- 使用 Router LoRA 判断是否检索以及检索哪些数据源。
- 使用 LangGraph 编排记忆、改写、路由、检索、证据检查、回答、fallback、日志、耗时和 token 统计。
- 输出带 citation、graph paths、warnings 和安全 fallback 的 grounded answer。
- 提供 React 前端，支持登录、会话历史、回答依据、编排日志、耗时和 token 展示。

## Runtime Flow / 运行链路

```text
User question + history
-> FastAPI
-> LangGraph StateGraph
-> Structured memory
-> Query Rewriter LoRA
-> Deterministic rewrite/schema completion
-> Router LoRA + priority rules
-> Direct answer or retrieval
-> Law / Case / Graph / Vector retrievers
-> Hybrid merge and rerank
-> Evidence sufficiency check
-> DeepSeek/OpenAI-compatible grounded answer or safe fallback
-> Citations + graph paths + warnings + logs + metrics
-> Frontend
```

```text
用户问题 + 历史消息
-> FastAPI
-> LangGraph StateGraph
-> 结构化会话记忆
-> Query Rewriter LoRA
-> 规则补齐和 schema 修正
-> Router LoRA + 优先规则
-> 直接回答或进入检索
-> 法条 / 案例 / 图谱 / 向量检索
-> Hybrid merge and rerank
-> 证据充分性检查
-> DeepSeek/OpenAI-compatible grounded answer 或安全 fallback
-> 引用 + 图路径 + 提示 + 日志 + 指标
-> 前端展示
```

## Tech Stack / 技术栈

- Backend: FastAPI, Pydantic, SQLite
- Frontend: React, TypeScript, Vite
- Orchestration: LangGraph, LangChain runnable compatibility
- Retrieval: lexical retrieval, FAISS vector index, graph path retrieval, hybrid reranking
- LLM: DeepSeek/OpenAI-compatible chat completion API
- Local models: Qwen2.5-1.5B-Instruct with LoRA adapters for Rewriter and Router
- Evaluation: retrieval eval, answer eval, orchestration eval, adversarial eval, hidden eval assets

- 后端：FastAPI、Pydantic、SQLite
- 前端：React、TypeScript、Vite
- 编排：LangGraph，保留 LangChain Runnable 兼容路径
- 检索：词法检索、FAISS 向量索引、图路径检索、混合排序
- LLM：DeepSeek/OpenAI-compatible Chat Completion API
- 本地模型：Qwen2.5-1.5B-Instruct + Rewriter/Router LoRA
- 评测：检索评测、回答评测、编排评测、对抗评测、hidden eval 资产

## Repository Layout / 目录结构

```text
app/              FastAPI backend, auth, chat APIs, provider client, schemas
frontend/         React frontend
processing/       Data processing, retrieval, graph, LoRA training, eval scripts
processing_data/  Tracked sample data, graph outputs, vector index sample, eval reports
docs/             Architecture, schema, artifact, and archive documentation
tests/            Backend and orchestration tests
models/           Local base models, ignored by Git
raw/              Local raw source data, ignored by Git
```

```text
app/              FastAPI 后端、认证、问答接口、Provider client、schema
frontend/         React 前端
processing/       数据处理、检索、图谱、LoRA 训练和评测脚本
processing_data/  已提交的样本数据、图谱产物、向量索引样本和评测报告
docs/             架构、schema、产物清单和归档文档
tests/            后端和编排测试
models/           本地基础模型，Git 忽略
raw/              本地原始数据，Git 忽略
```

## Configuration / 配置

Copy `.env.example` to `.env` and fill in the provider key if provider-backed answering is needed.

如需调用 Provider 生成回答，复制 `.env.example` 为 `.env` 并填写 API Key。

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash
USE_LANGCHAIN_ORCHESTRATION=true
QUERY_REWRITER_MODE=lora
ROUTER_MODE=lora
QUERY_UNDERSTANDING_MODE=disabled
```

Large local artifacts are intentionally not committed:

- `.env`
- `raw/`
- `models/`
- `processing_data/query_rewriter_lora/`
- `processing_data/router_lora/`
- `processing_data/query_understanding_lora/`
- `processing_data/query_router_lora/`

这些大型或私有本地产物不会提交到 Git：

- `.env`
- `raw/`
- `models/`
- `processing_data/query_rewriter_lora/`
- `processing_data/router_lora/`
- `processing_data/query_understanding_lora/`
- `processing_data/query_router_lora/`

See `docs/artifact_inventory.md` for artifact notes.

产物说明见 `docs/artifact_inventory.md`。

## Run Locally / 本地运行

Install backend dependencies:

安装后端依赖：

```powershell
cd D:\GraphRAG-V1
pip install -e .
```

Start the backend:

启动后端：

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Install frontend dependencies:

安装前端依赖：

```powershell
cd D:\GraphRAG-V1\frontend
npm install
```

Start the frontend:

启动前端：

```powershell
npm run dev
```

Default local frontend URL:

默认前端地址：

```text
http://127.0.0.1:5173
```

Default demo login values are configured in `.env.example`:

默认演示登录配置见 `.env.example`：

```env
AUTH_BOOTSTRAP_USERNAME=admin
AUTH_BOOTSTRAP_PASSWORD=1
```

## Sample Questions / 示例问题

```text
盗窃罪一般怎么量刑？
吴必定盗窃罪案判了多久？
这个案子适用了哪些法条？
盗窃40000元一般会怎么处理？
入室盗窃和普通盗窃有什么区别？
刑法主要涉及到哪些违法行为？
```

## Evaluation / 评测

Lightweight checks:

轻量检查：

```powershell
pytest tests\test_langgraph_orchestration.py tests\test_query_rewriter.py -q
npm test --prefix frontend
python -m compileall app processing tests
```

Retrieval and orchestration checks:

检索和编排评测：

```powershell
python processing\search_eval_sample.py --fail-on-warn
python processing\vector_search_eval_sample.py --fail-on-warn
python processing\graph_retriever_eval_sample.py --fail-on-warn
python processing\hybrid_search_eval_sample.py --fail-on-warn
python processing\langchain_orchestration_eval_sample.py --fail-on-warn
```

## Current Scope / 当前边界

This is a sample-level engineering prototype. It is designed to validate data processing, graph-enhanced retrieval, query understanding, grounded generation, and traceable evaluation. It is not a production legal consultation system.

这是一个样本级工程原型，用于验证数据处理、图谱增强检索、查询理解、基于依据的生成和可追溯评测。它不是生产级法律咨询系统。

Known limitations:

- The committed dataset is sample-scale.
- Full raw data, base models, and LoRA checkpoints are local artifacts.
- Runtime code still imports some `processing/*_sample.py` modules directly.
- Rewriter and Router LoRA behavior still depends on deterministic correction and regression tests.
- Provider-backed generation requires a valid API key and network access.
- Larger-scale deployment would need stronger service boundaries, incremental indexing, monitoring, and legal-risk controls.

已知限制：

- 当前提交的是样本级数据。
- 完整 raw 数据、基础模型和 LoRA checkpoint 是本地产物。
- 运行时仍直接依赖部分 `processing/*_sample.py` 模块。
- Rewriter 和 Router LoRA 仍需要规则补齐和回归测试兜底。
- Provider 生成需要有效 API Key 和网络访问。
- 更大规模部署需要更清晰的 service 层、增量索引、监控和法律风险控制。

