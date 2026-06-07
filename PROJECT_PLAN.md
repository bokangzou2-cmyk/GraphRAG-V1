# GraphRAG-V1 Project Plan

Last synchronized: 2026-06-07

## Project Status

GraphRAG-V1 is a sample-level Chinese criminal-law GraphRAG system. The project is no longer only a small retrieval script set: it now has a working frontend, FastAPI backend, LangGraph orchestration runtime, Query Rewriter LoRA, Router LoRA, hybrid retrieval, provider-backed grounded answering, audit logs, and regression assets.

The system remains a high-quality prototype, not a production legal-advice system. The sample data, retrieval contracts, graph paths, citations, confidence gates, and evaluation reports are the source of truth for current development.

## Current Runtime Chain

The frontend answer path is:

```text
User question + history
-> FastAPI auth/session
-> /api/chat/orchestration
-> LangGraph StateGraph in processing/langchain_orchestration_sample.py
-> structured memory node
-> Query Rewriter LoRA
-> deterministic rewrite/schema completion
-> Router LoRA + priority rules
-> direct answer or conditional retrieval
-> law/case/graph/vector retriever nodes
-> hybrid merge/rerank
-> evidence sufficiency gate
-> DeepSeek/OpenAI-compatible grounded answer or safe fallback
-> orchestration_logs + metrics + token usage
-> frontend answer, citations, warnings, graph paths, timing panel
```

`/api/chat` is now a compatibility endpoint that delegates to the same orchestration implementation with provider generation. It should not be treated as an independent stable legacy answer path.

## Active Query Understanding Architecture

The old unified Query Understanding LoRA is not the active runtime direction.

Current active split:

- Query Rewriter LoRA: rewrites multi-turn questions and extracts case references, case names, crimes, amounts, and field intents.
- Deterministic completion: repairs Rewriter schema, fills missing structured fields, and prevents obvious over-rewrite drift.
- Router LoRA: decides whether retrieval is needed and which targets to retrieve.
- Scope/direct rules: handle non-law, unsafe, or broad direct-answer cases before retrieval.

Important runtime defaults are documented in `.env.example`:

```env
USE_LANGCHAIN_ORCHESTRATION=true
QUERY_REWRITER_MODE=lora
ROUTER_MODE=lora
QUERY_UNDERSTANDING_MODE=disabled
LLM_PROVIDER=deepseek
```

## Current Data And Retrieval Assets

Tracked sample assets under `processing_data/` include:

- law and case sample JSONL files;
- enriched cases and chunks;
- retrieval chunks;
- graph nodes/edges/snapshot files;
- FAISS vector sample index and metadata;
- lexical/vector/graph/hybrid/answer/orchestration/adversarial/hidden eval reports.

Ignored local artifacts are documented in `docs/artifact_inventory.md`:

- `raw/`
- `models/`
- `.env`
- local SQLite chat history;
- LoRA datasets, checkpoints, and adapter output directories.

## Retrieval Architecture

The retrieval layer is still self-owned Python code. LangGraph orchestrates it but does not replace it.

Active retrieval modules:

- `processing/search_sample.py`: lexical and exact-match baseline retrieval.
- `processing/vector_search_sample.py`: vector retrieval over `processing_data/vector_index_sample/`.
- `processing/graph_retriever_sample.py`: graph path candidate recall.
- `processing/hybrid_search_sample.py`: merge, deduplicate, rank, and package retrieval context.

The sample suffix is historical. These modules are currently runtime-critical and must not be deleted or bypassed without a replacement service layer.

## Answer Architecture

Current user-facing answer generation is controlled by the LangGraph runtime:

- direct answer for non-retrieval criminal-law general questions;
- grounded answer when retrieved context is sufficient;
- fallback answer when evidence is weak, missing, or provider output is invalid;
- internal provider errors are logged but not exposed raw to users.

Provider calls use the OpenAI-compatible client in `app/llm/minimax_client.py`. The filename is historical; current defaults target DeepSeek through `LLM_BASE_URL` and `LLM_MODEL`.

Legacy/diagnostic answer paths remain:

- `processing/answer_sample.py`: retrieval-only baseline and eval harness.
- `app/llm/chat_service.py`: shared helpers and legacy comparison support.
- `processing/chat_vs_orchestration_eval.py`: diagnostic comparison.

These should be preserved until service-layer convergence is complete.

## LangGraph Boundary

Use LangGraph for:

- explicit state-machine orchestration;
- structured memory and multi-turn flow;
- conditional retrieval branches;
- evidence sufficiency routing;
- audit-friendly logs, metrics, and node timing.

Do not use LangGraph/LangChain for:

- parsing raw files;
- mutating `raw/`;
- replacing self-owned retrieval scoring;
- hiding citation IDs, graph paths, source files, text hashes, confidence inputs, warnings, or fallback reasons;
- letting the LLM read anything outside selected retrieval context.

## Completed Milestones

- Sample data pipeline and schema reports.
- Law/case chunk generation.
- Graph node/edge construction.
- Vector index construction.
- Lexical/vector/graph/hybrid retrieval.
- Query Rewriter-only LoRA training assets and runtime integration.
- Router LoRA v2 processed-dataset training assets and runtime integration.
- LangGraph `StateGraph` orchestration runtime.
- Direct, grounded, and fallback answer branches.
- Evidence sufficiency gate.
- Frontend integration with answer evidence, orchestration logs, timing, and token usage.
- Repository archive and GitHub publication.

## Current Risks

- Runtime code still imports `processing/*_sample.py` modules directly.
- Rewriter and Router LoRA behavior still needs broader regression coverage.
- Amount role extraction and similar-case matching remain fragile.
- Provider JSON validity, citation repair, and fallback behavior remain high-risk user-facing areas.
- Current sample scale does not prove full-data behavior.
- Large generated artifacts are local-only and must remain documented.

## Next Milestone

The next milestone is architecture convergence, not adding more ad hoc rules:

1. Introduce an `app/services/` layer that wraps current processing modules without changing behavior.
2. Move runtime orchestration out of `processing/` after service wrappers are stable.
3. Define strong contracts for `RewriteResult`, `RouteDecision`, `RetrievalResult`, `MergedContext`, and `AnswerContract`.
4. Keep current evals green while moving code boundaries.
5. Add targeted regression cases for broad criminal-law questions, conservative rewriting, amount roles, multi-case comparison, and low-evidence fallback.
6. Keep provider errors user-safe and visible only through audit logs.

## Validation Commands

Run lightweight checks after runtime changes:

```powershell
pytest tests\test_langgraph_orchestration.py tests\test_query_rewriter.py -q
npm test --prefix frontend
python -m compileall app processing tests
```

Run broader sample checks when retrieval/data behavior changes:

```powershell
python processing\run_sample_pipeline.py
python processing\search_eval_sample.py --fail-on-warn
python processing\vector_search_eval_sample.py --fail-on-warn
python processing\graph_retriever_eval_sample.py --fail-on-warn
python processing\hybrid_search_eval_sample.py --fail-on-warn
python processing\langchain_orchestration_eval_sample.py --fail-on-warn
```

Do not run the sample pipeline in parallel with eval commands that read `processing_data/`.
