from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables import RunnableLambda

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # LangGraph is the target runtime, but keep old environments usable.
    END = None
    StateGraph = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.processing_bridge  # noqa: F401
from app.config import settings
from app.llm.chat_service import (
    answer_without_retrieval,
    citation_from_context,
    confidence_for,
    fallback_answer,
    latest_user_query,
    contextualized_retrieval_query,
    align_confidence_with_route,
    can_answer_low_confidence_generally,
    parse_llm_json,
    reconcile_provider_payload,
    refusal_answer,
    refusal_reason_for,
    should_use_llm,
    low_confidence_llm_answer,
)
from app.llm.minimax_client import MinimaxClient, chat_grounded_json
from app.llm.query_rewriter import (
    amount_constraints_for as memory_amount_constraints_for,
    case_ref_indices,
    extract_case_candidates,
    field_intents_for as memory_field_intents_for,
    infer_crimes_from_text,
    rewrite_query,
)
from app.llm.query_understanding import QueryUnderstandingResult, postprocess_payload, router_input_from_rewrite_payload, validate_payload
from app.llm.query_router import RouteDecision, route_query
from app.llm.scope_guard import is_out_of_scope_query, out_of_scope_response
from graph_retriever_sample import retrieve as graph_retrieve
from hybrid_search_sample import merge_retrieval_results, vector_hits
from search_sample import search as lexical_search


MOCK_LLM_MODES = {"grounded", "invalid_json", "missing_citation", "empty_answer", "no_answer"}
GENERATION_MODES = {"mock", "provider"}
RESPONSE_KEYS = {
    "answer",
    "answer_type",
    "citations",
    "graph_paths",
    "warnings",
    "confidence",
    "refusal_reason",
    "retrieval_context",
    "orchestration_logs",
    "provider_error",
    "metrics",
}


class OrchestrationState(TypedDict, total=False):
    messages: list[dict]
    messages_for_rewrite: list[dict]
    messages_for_answer: list[dict]
    top_k: int
    mock_llm_mode: str
    generation_mode: str
    query: str
    route_query: str
    retrieval_query: str
    memory: dict
    memory_compression: dict
    rewrite_result: object
    rewrite_payload: dict
    query_understanding: object
    orchestration_logs: list[dict]
    route: RouteDecision
    enabled_retrievers: list[str]
    retrieval_targets: list[str]
    _lexical_all_result: dict
    law_result: dict
    case_result: dict
    lexical_result: dict
    graph_result: dict
    vector_result: list[dict]
    vector_warnings: list[str]
    retrieval_result: dict
    merge_confidence: dict
    context: list[dict]
    warnings: list[str]
    confidence: dict
    graph_paths: list[dict]
    citations: list[dict]
    llm_messages: list[dict]
    raw_llm_output: str | None
    llm_provider_error: str
    provider_error_detail: str
    answer: str
    answer_type: str
    retrieval_context: list[dict]
    refusal_reason: str | None
    provider_error: dict
    evidence_sufficient: bool
    evidence_sufficiency: dict
    response: dict
    metrics: dict

ORCHESTRATION_GROUNDED_SYSTEM_PROMPT = """你是中文刑法 GraphRAG 问答助手。只能基于给定 retrieval_context 和 graph_paths 回答。
硬性要求：
1. 只能使用 retrieval_context 中出现的事实、法条、案名、裁判理由和量刑信息；不要补充外部知识。
2. citation_ids 必须来自 retrieval_context 中的 retrieval_id，且要覆盖回答中使用的关键法条、案例事实或裁判理由。
3. 如果回答涉及法条依据，优先引用 law_article；如果回答涉及案例事实、证据、辩解、裁判理由或量刑，必须引用对应 case_chunk。
4. 如果问题问 1979 年刑法、旧刑法或历史法条，但 retrieval_context 未明确提供 1979 年刑法文本，不得自动用现行刑法替代；应输出 no_answer，并加入 warning: historical_law_context_missing。
5. 如果检索上下文不足以支持结论，输出 no_answer 或加入 warning，不要编造。
6. 不要因为上下文中有多个候选片段就自由扩写；只回答能被引用片段直接支持的内容。
7. 回答金额类问题时必须区分金额角色：盗窃数额、罚金、退赔/追缴、违法所得不能互相替代；如 amount_diagnostics 显示角色不一致或距离较大，只能作为风险提示，不能作为主类案依据。
8. 面向用户回答时使用 retrieval_context.display_title 里的自然短案名，例如“王士东案”“刘方祥案”；完整判决书标题只作为引用资料，不要写成“第一个参考案例是《……一审刑事判决书》”这类首句。
9. 回答法定刑时必须逐字核对 retrieval_context 中的条文，不得把“三年以上七年以下”写成“七年以上”，不得把“以下/以上”方向写反。
10. 必须输出严格 JSON，不要 Markdown，不要解释 JSON 外文本。

JSON schema:
{
  "answer": "中文回答；如果证据不足则明确说明无法确认",
  "answer_type": "llm_grounded | no_answer",
  "citation_ids": ["retrieval_id"],
  "warnings": ["warning_code"],
  "refusal_reason": "insufficient_context | low_confidence | out_of_scope | null"
}"""


def display_title(title: str | None) -> str:
    text = str(title or "").strip()
    if not text:
        return ""
    prefix = re.split(r"、|，|,|犯|与|等|一审|二审|再审|刑事|民事|行政", text, maxsplit=1)[0].strip()
    if 2 <= len(prefix) <= 8 and not prefix.endswith("案"):
        return f"{prefix}案"
    return prefix or text


def add_log(state: dict, event: str, payload: dict | None = None) -> dict:
    state.setdefault("orchestration_logs", []).append({"event": event, "payload": payload or {}})
    return state


def add_node_metric(state: dict, node: str, duration_ms: int) -> None:
    metrics = state.setdefault("metrics", {"nodes": {}, "token_usage": {}, "total_duration_ms": None})
    node_metrics = metrics.setdefault("nodes", {})
    node_metrics[node] = {"duration_ms": duration_ms}


def timed_node(node_name: str, func):
    def wrapped(state: dict) -> dict:
        started = time.perf_counter()
        result = func(state)
        duration_ms = int((time.perf_counter() - started) * 1000)
        if isinstance(result, dict):
            add_node_metric(result, node_name, duration_ms)
            for item in reversed(result.get("orchestration_logs", [])):
                if item.get("event") == node_name:
                    item.setdefault("payload", {})["duration_ms"] = duration_ms
                    break
        return result

    return wrapped


def record_latest_event_duration(state: dict, event: str, duration_ms: int) -> None:
    add_node_metric(state, event, duration_ms)
    for item in reversed(state.get("orchestration_logs", [])):
        if item.get("event") == event:
            item.setdefault("payload", {})["duration_ms"] = duration_ms
            break


def compact_context(rows: list[dict], limit: int = 5) -> list[dict]:
    return [
        {
            "rank": row.get("rank"),
            "retrieval_id": row.get("retrieval_id"),
            "source_type": row.get("source_type"),
            "title": row.get("title"),
            "field": row.get("field"),
            "hybrid_score": row.get("hybrid_score"),
            "retrieval_sources": row.get("retrieval_sources", []),
            "ranking_adjustments": row.get("ranking_adjustments", {}),
            "graph_path_count": len(row.get("graph_paths", [])),
        }
        for row in rows[:limit]
    ]


def build_orchestration_llm_messages(query: str, context: list[dict], graph_paths: list[dict], warnings: list[str], history: list[dict]) -> list[dict]:
    user_history = [
        {"role": "user", "content": m.get("content", "")}
        for m in history
        if m.get("role") == "user" and m.get("content")
    ][-4:]
    evidence = {
        "query": query,
        "retrieval_context": [
            {
                "rank": item.get("rank"),
                "retrieval_id": item.get("retrieval_id"),
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "display_title": display_title(item.get("title")),
                "field": item.get("field"),
                "text_preview": item.get("text_preview"),
                "law_metadata": item.get("law_metadata"),
                "source_scores": item.get("source_scores"),
                "retrieval_sources": item.get("retrieval_sources"),
                "ranking_adjustments": item.get("ranking_adjustments"),
                "amount_diagnostics": item.get("amount_diagnostics"),
                "graph_path_count": len(item.get("graph_paths", [])),
            }
            for item in context
        ],
        "graph_paths": graph_paths,
        "warnings": warnings,
    }
    return [
        {"role": "system", "content": ORCHESTRATION_GROUNDED_SYSTEM_PROMPT},
        *user_history,
        {"role": "user", "content": "请基于以下检索证据回答最后一个问题：\n" + json.dumps(evidence, ensure_ascii=False, indent=2)},
    ]


def build_citations(context: list[dict]) -> list[dict]:
    citations = []
    seen = set()
    for item in context:
        key = item.get("retrieval_id")
        if key and key not in seen:
            seen.add(key)
            citations.append(citation_from_context(item))
    return citations


def enabled_retrievers(route: RouteDecision) -> list[str]:
    if not route.needs_retrieval:
        return []
    return ["lexical", "graph", "vector"]


def normalize_retrieval_targets(route: RouteDecision) -> list[str]:
    if not route.needs_retrieval:
        return []
    targets = set(route.retrieval_targets or [])
    normalized: list[str] = []
    if "law_articles" in targets:
        normalized.append("law")
    if "cases" in targets:
        normalized.append("case")
    if "graph" in targets or "cases" in targets:
        normalized.append("graph")
    normalized.append("vector")
    return normalized


def summarize_memory(messages: list[dict]) -> dict:
    latest = latest_user_query(messages)
    history_text = "\n".join(str(message.get("content") or "") for message in messages[-8:])
    cases = extract_case_candidates(messages)
    ref_indices = case_ref_indices(latest, len(cases))
    active_case = None
    compared_cases = []
    if ref_indices:
        compared_cases = [case for case in cases if int(case.get("index") or 0) in ref_indices]
        active_case = compared_cases[-1] if compared_cases else None
    elif cases:
        active_case = cases[0]
    crimes = infer_crimes_from_text(f"{history_text}\n{latest}")
    amounts = memory_amount_constraints_for(f"{history_text}\n{latest}")
    field_intents = memory_field_intents_for(latest, [])
    constraints = []
    if re.search(r"不要查案例|不用查案例|不查案例|别查案例|不要案例|不用案例|别用案例", latest):
        constraints.append("no_case_retrieval")
    if re.search(r"不要查法条|不用查法条|不查法条|别查法条|不要法条|不用法条", latest):
        constraints.append("no_law_retrieval")
    return {
        "case_refs": cases[:6],
        "active_case": active_case,
        "compared_cases": compared_cases[:4],
        "crime_mentions": crimes,
        "amount_constraints": amounts,
        "field_intents": field_intents,
        "user_constraints": constraints,
    }


def memory_case_line(case: dict) -> str:
    index = case.get("index")
    title = case.get("case_title")
    short_title = case.get("short_title")
    amount = case.get("amount")
    parts = []
    if index:
        parts.append(f"第{index}个参考案例")
    if title:
        parts.append(f"《{title}》")
    if short_title:
        parts.append(f"简称{short_title}")
    if amount:
        parts.append(f"金额约{amount}元")
    return "，".join(parts)


def memory_to_message_content(memory: dict) -> str:
    lines = ["结构化上下文记忆（不是检索证据，仅用于问题改写和指代消解）："]
    case_lines = [memory_case_line(case) for case in memory.get("case_refs", []) if isinstance(case, dict)]
    case_lines = [line for line in case_lines if line]
    if case_lines:
        lines.append("案例上下文：" + "；".join(case_lines))
    active_case = memory.get("active_case")
    if isinstance(active_case, dict) and active_case.get("case_title"):
        lines.append(f"当前焦点案件：《{active_case['case_title']}》")
    compared = [case.get("case_title") for case in memory.get("compared_cases", []) if isinstance(case, dict) and case.get("case_title")]
    if compared:
        lines.append("对比案件：" + "、".join(f"《{title}》" for title in compared))
    if memory.get("crime_mentions"):
        lines.append("已提及罪名：" + "、".join(map(str, memory["crime_mentions"])))
    if memory.get("amount_constraints"):
        amount_parts = [
            f"{item.get('raw_text') or item.get('value')}({item.get('role', 'unknown_amount')})"
            for item in memory["amount_constraints"]
            if isinstance(item, dict)
        ]
        if amount_parts:
            lines.append("已提及金额：" + "、".join(amount_parts))
    if memory.get("field_intents"):
        lines.append("字段关注：" + "、".join(map(str, memory["field_intents"])))
    if memory.get("user_constraints"):
        lines.append("用户限制：" + "、".join(map(str, memory["user_constraints"])))
    lines.append("没有出现的字段保持为空；不要补充未在对话中明确出现的案件、罪名、金额、事实或法条。")
    return "\n".join(lines)


def compress_messages_for_context(messages: list[dict], memory: dict) -> tuple[list[dict], dict]:
    threshold = max(0, settings.memory_compression_message_threshold)
    recent_limit = max(1, settings.memory_compression_recent_messages)
    if len(messages) <= threshold:
        return messages, {
            "enabled": False,
            "reason": "below_threshold",
            "message_count": len(messages),
            "threshold": threshold,
            "recent_message_count": len(messages),
            "dropped_message_count": 0,
        }
    recent = list(messages[-recent_limit:])
    memory_message = {"role": "assistant", "content": memory_to_message_content(memory)}
    if recent:
        compressed = [*recent[:-1], memory_message, recent[-1]]
    else:
        compressed = [memory_message]
    return compressed, {
        "enabled": True,
        "reason": "message_count_exceeded_threshold",
        "message_count": len(messages),
        "threshold": threshold,
        "recent_message_count": len(recent),
        "dropped_message_count": max(0, len(messages) - len(recent)),
        "memory_message_chars": len(memory_message["content"]),
    }


def init_state(inputs: dict) -> dict:
    messages = inputs.get("messages", [])
    top_k = int(inputs.get("top_k", 10))
    mock_llm_mode = inputs.get("mock_llm_mode", "grounded")
    generation_mode = inputs.get("generation_mode", "mock")
    if mock_llm_mode not in MOCK_LLM_MODES:
        raise ValueError(f"unknown mock_llm_mode: {mock_llm_mode}")
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    return add_log(
        {
            "messages": messages,
            "top_k": top_k,
            "mock_llm_mode": mock_llm_mode,
            "generation_mode": generation_mode,
            "query": latest_user_query(messages),
            "route_query": contextualized_retrieval_query(messages),
            "retrieval_query": contextualized_retrieval_query(messages),
            "orchestration_logs": [],
            "metrics": {"nodes": {}, "token_usage": {}, "started_perf": time.perf_counter(), "total_duration_ms": None},
        },
        "state_initialized",
        {"message_count": len(messages), "top_k": top_k, "mock_llm_mode": mock_llm_mode, "generation_mode": generation_mode},
    )


def memory_state(state: dict) -> dict:
    memory = summarize_memory(state["messages"])
    compressed_messages, compression = compress_messages_for_context(state["messages"], memory)
    state["memory"] = memory
    state["memory_compression"] = compression
    state["messages_for_rewrite"] = compressed_messages
    state["messages_for_answer"] = compressed_messages
    return add_log(
        state,
        "memory_summary",
        {
            "compression": compression,
            "case_ref_count": len(memory.get("case_refs", [])),
            "active_case": memory.get("active_case", {}),
            "compared_case_count": len(memory.get("compared_cases", [])),
            "crime_mentions": memory.get("crime_mentions", []),
            "amount_constraints": memory.get("amount_constraints", []),
            "field_intents": memory.get("field_intents", []),
            "user_constraints": memory.get("user_constraints", []),
        },
    )


def query_rewriter_state(state: dict) -> dict:
    rewrite = rewrite_query(state.get("messages_for_rewrite") or state["messages"])
    state["rewrite_result"] = rewrite
    state["rewrite_payload"] = dict(rewrite.payload)
    state["route_query"] = rewrite.standalone_query or state.get("route_query") or state["query"]
    state["retrieval_query"] = state["route_query"]
    if rewrite.metrics:
        state.setdefault("metrics", {}).setdefault("token_usage", {})["query_rewriter"] = rewrite.metrics
    return add_log(state, "query_rewriter_result", rewrite.as_log_payload())


def structured_fix_state(state: dict) -> dict:
    rewrite = state["rewrite_result"]
    payload = dict(state["rewrite_payload"])
    query = payload.get("standalone_query") or state.get("route_query") or state["query"]
    resolved = list(payload.get("resolved_case_refs") or [])
    route = route_query(router_input_from_rewrite_payload(payload, state["query"]))
    if len(resolved) >= 2 and re.search(r"差异|区别|对比|比较|分别|两者|二者|两个", str(query)):
        route = RouteDecision(
            "criminal_law_multi_case",
            True,
            ["cases", "graph"],
            True,
            False,
            False,
            max(route.confidence, float(payload.get("confidence") or 0.0), 0.82),
            "rules",
            "rewriter_multi_case_completion",
        )
    warnings = list(payload.get("warnings") or [])
    warnings.append(f"query_rewriter:{rewrite.source}")
    understanding_payload = postprocess_payload(
        validate_payload(
            {
                "standalone_query": str(query),
                "route_label": route.label,
                "needs_retrieval": route.needs_retrieval,
                "retrieval_targets": route.retrieval_targets,
                "case_required": route.case_required,
                "law_required": route.law_required,
                "case_name_mentions": list(payload.get("case_name_mentions") or []),
                "resolved_case_refs": resolved,
                "field_intents": list(payload.get("field_intents") or []),
                "crime_mentions": list(payload.get("crime_mentions") or []),
                "amount_constraints": list(payload.get("amount_constraints") or []),
                "confidence": max(route.confidence, float(payload.get("confidence") or 0.0)),
                "warnings": warnings,
            }
        ),
        state.get("messages_for_rewrite") or state["messages"],
    )
    query_understanding = QueryUnderstandingResult(
        understanding_payload,
        rewrite.source,
        f"{rewrite.reason}_then_route_query",
        rewrite.raw_output,
    )
    state["query_understanding"] = query_understanding
    state["route_query"] = query_understanding.standalone_query or state["route_query"]
    state["retrieval_query"] = query_understanding.standalone_query or state["retrieval_query"]
    if state["query"] and state["query"] not in state["retrieval_query"]:
        state["retrieval_query"] = f"{state['retrieval_query']} {state['query']}".strip()
    add_log(
        state,
        "structured_fix",
        {
            "resolved_case_refs": understanding_payload.get("resolved_case_refs", []),
            "amount_constraints": understanding_payload.get("amount_constraints", []),
            "field_intents": understanding_payload.get("field_intents", []),
            "warnings": understanding_payload.get("warnings", []),
        },
    )
    return add_log(
        state,
        "start",
        {
            "query": state["query"],
            "route_query": state["route_query"],
            "retrieval_query": state["retrieval_query"],
            "query_understanding": query_understanding.as_log_payload(),
            "top_k": state["top_k"],
            "mock_llm_mode": state["mock_llm_mode"],
            "generation_mode": state["generation_mode"],
            "use_langchain_orchestration": settings.use_langchain_orchestration,
        },
    )


def start_state(inputs: dict) -> dict:
    started = time.perf_counter()
    state = init_state(inputs)
    record_latest_event_duration(state, "state_initialized", int((time.perf_counter() - started) * 1000))
    started = time.perf_counter()
    state = memory_state(state)
    record_latest_event_duration(state, "memory_summary", int((time.perf_counter() - started) * 1000))
    started = time.perf_counter()
    state = query_rewriter_state(state)
    record_latest_event_duration(state, "query_rewriter_result", int((time.perf_counter() - started) * 1000))
    started = time.perf_counter()
    state = structured_fix_state(state)
    record_latest_event_duration(state, "structured_fix", int((time.perf_counter() - started) * 1000))
    return state


def router_state(state: dict) -> dict:
    query = state["route_query"]
    if not query:
        route = RouteDecision("low_quality_or_incomplete", False, ["none"], False, False, False, 0.0, "rules", "empty_query")
    elif state.get("query_understanding"):
        route = state["query_understanding"].route_decision()
    else:
        route = route_query(query)
    state["route"] = route
    state["enabled_retrievers"] = enabled_retrievers(route)
    state["retrieval_targets"] = normalize_retrieval_targets(route)
    return add_log(
        state,
        "router_result",
        {
            **route.as_dict(),
            "enabled_retrievers": state["enabled_retrievers"],
            "normalized_retrieval_targets": state["retrieval_targets"],
            "source_types": route.source_types,
        },
    )


def lexical_all_result(state: dict) -> dict:
    if "_lexical_all_result" not in state:
        state["_lexical_all_result"] = lexical_search(state["retrieval_query"], top_k=24, expansion_k=12, context_k=max(state["top_k"], 12))
    return state["_lexical_all_result"]


def filter_lexical_result(result: dict, source_type: str) -> dict:
    filtered = dict(result)
    filtered["direct_hits"] = [hit for hit in result.get("direct_hits", []) if hit.get("source_type") == source_type]
    filtered["final_context"] = [hit for hit in result.get("final_context", []) if hit.get("source_type") == source_type]
    if source_type != "case_chunk":
        filtered["graph_expansions"] = []
    return filtered


def law_retriever_state(state: dict) -> dict:
    if "law" not in state.get("retrieval_targets", []):
        state["law_result"] = {"direct_hits": [], "final_context": []}
        return add_log(state, "law_retriever", {"enabled": False, "law_count": 0})
    result = filter_lexical_result(lexical_all_result(state), "law_article")
    state["law_result"] = result
    return add_log(state, "law_retriever", {"enabled": True, "law_count": len(result.get("direct_hits", []))})


def case_retriever_state(state: dict) -> dict:
    if "case" not in state.get("retrieval_targets", []):
        state["case_result"] = {"direct_hits": [], "final_context": []}
        return add_log(state, "case_retriever", {"enabled": False, "case_count": 0})
    result = filter_lexical_result(lexical_all_result(state), "case_chunk")
    state["case_result"] = result
    return add_log(state, "case_retriever", {"enabled": True, "case_count": len(result.get("direct_hits", []))})


def expand_law_from_case_hits(state: dict) -> dict:
    if "law" not in state.get("retrieval_targets", []):
        return {"direct_hits": [], "final_context": []}
    if state.get("law_result", {}).get("direct_hits"):
        return {"direct_hits": [], "final_context": []}
    terms: list[str] = []
    for hit in state.get("case_result", {}).get("direct_hits", [])[:8]:
        metadata = hit.get("case_metadata") or {}
        for key in ("crimes", "applied_articles", "cited_articles"):
            for value in metadata.get(key) or []:
                text = str(value or "").strip()
                if text and text not in terms:
                    terms.append(text)
    if not terms:
        return {"direct_hits": [], "final_context": []}
    result = lexical_search(" ".join(terms), top_k=12, expansion_k=4, context_k=12)
    return filter_lexical_result(result, "law_article")


def lexical_retriever_state(state: dict) -> dict:
    law_expansion = expand_law_from_case_hits(state)
    if law_expansion.get("direct_hits"):
        existing = {hit.get("retrieval_id") for hit in state.get("law_result", {}).get("direct_hits", [])}
        state.setdefault("law_result", {}).setdefault("direct_hits", [])
        state.setdefault("law_result", {}).setdefault("final_context", [])
        for hit in law_expansion.get("direct_hits", []):
            if hit.get("retrieval_id") not in existing:
                state["law_result"]["direct_hits"].append(hit)
                existing.add(hit.get("retrieval_id"))
        state["law_result"]["final_context"].extend(law_expansion.get("final_context", []))
        add_log(state, "case_law_expansion", {"enabled": True, "law_count": len(law_expansion.get("direct_hits", []))})
    else:
        add_log(state, "case_law_expansion", {"enabled": False, "law_count": 0})
    direct_hits = [
        *state.get("law_result", {}).get("direct_hits", []),
        *state.get("case_result", {}).get("direct_hits", []),
    ]
    final_context = [
        *state.get("law_result", {}).get("final_context", []),
        *state.get("case_result", {}).get("final_context", []),
    ]
    state["lexical_result"] = {"direct_hits": direct_hits, "final_context": final_context}
    return add_log(
        state,
        "lexical_retriever",
        {
            "enabled": bool(direct_hits),
            "lexical_count": len(direct_hits),
            "law_count": len(state.get("law_result", {}).get("direct_hits", [])),
            "case_count": len(state.get("case_result", {}).get("direct_hits", [])),
        },
    )


def graph_retriever_state(state: dict) -> dict:
    if "graph" not in state.get("retrieval_targets", []):
        state["graph_result"] = {"candidates": [], "warnings": []}
        return add_log(state, "graph_retriever", {"enabled": False, "graph_count": 0, "warnings": []})
    result = graph_retrieve(state["retrieval_query"], top_k=30)
    state["graph_result"] = result
    return add_log(
        state,
        "graph_retriever",
        {"enabled": True, "graph_count": len(result.get("candidates", [])), "warnings": result.get("warnings", [])},
    )


def vector_retriever_state(state: dict) -> dict:
    if "vector" not in state.get("retrieval_targets", []):
        state["vector_result"] = []
        state["vector_warnings"] = []
        return add_log(state, "vector_retriever", {"enabled": False, "vector_count": 0, "warnings": []})
    vectors, warnings = vector_hits(state["retrieval_query"], 12)
    state["vector_result"] = vectors
    state["vector_warnings"] = warnings
    return add_log(state, "vector_retriever", {"enabled": True, "vector_count": len(vectors), "warnings": warnings})


def merge_rerank_state(state: dict) -> dict:
    route: RouteDecision = state["route"]
    if not route.needs_retrieval:
        state["retrieval_result"] = {"warnings": ["retrieval_skipped"], "selected_context": [], "search_summary": {}}
        return add_log(
            state,
            "hybrid_merge_rerank",
            {
                "enabled": False,
                "router_result": route.as_dict(),
                "enabled_retrievers": state["enabled_retrievers"],
                "lexical_count": 0,
                "graph_count": 0,
                "vector_count": 0,
                "final_context_count": 0,
                "confidence": {"level": "high", "score": route.confidence, "router": route.as_dict()},
                "warnings": ["retrieval_skipped"],
            },
        )
    result = merge_retrieval_results(
        state["retrieval_query"],
        top_k=state["top_k"],
        lexical=state["lexical_result"],
        graph=state["graph_result"],
        vectors=state["vector_result"],
        warnings=state["vector_warnings"],
        source_types=route.source_types,
    )
    state["retrieval_result"] = result
    confidence = confidence_for(result)
    confidence["router"] = route.as_dict()
    confidence = align_confidence_with_route(confidence, route)
    state["merge_confidence"] = confidence
    summary = result.get("search_summary", {})
    return add_log(
        state,
        "hybrid_merge_rerank",
        {
            "router_result": route.as_dict(),
            "enabled_retrievers": state["enabled_retrievers"],
            "lexical_count": summary.get("lexical_hits", 0),
            "graph_count": summary.get("graph_candidates", 0),
            "vector_count": summary.get("vector_hits", 0),
            "final_context_count": len(result.get("selected_context", [])),
            "confidence": confidence,
            "warnings": result.get("warnings", []),
            "selected_context": compact_context(result.get("selected_context", [])),
        },
    )


def evidence_sufficiency_state(state: dict) -> dict:
    route: RouteDecision = state["route"]
    confidence = state.get("merge_confidence") or {"level": "high" if not route.needs_retrieval else "none", "score": route.confidence}
    context_count = len(state.get("retrieval_result", {}).get("selected_context", []))
    sufficient = not route.needs_retrieval or (context_count > 0 and confidence.get("level") not in {"none", "low"})
    state["evidence_sufficient"] = sufficient
    state["evidence_sufficiency"] = {
        "sufficient": sufficient,
        "context_count": context_count,
        "confidence_level": confidence.get("level"),
        "route_label": route.label,
        "needs_retrieval": route.needs_retrieval,
    }
    return add_log(state, "evidence_sufficiency", state["evidence_sufficiency"])


def prepare_state(state: dict) -> dict:
    route: RouteDecision = state["route"]
    if not route.needs_retrieval:
        confidence = {"level": "high", "score": route.confidence, "query_type": route.label, "router": route.as_dict()}
        state.update({"context": [], "warnings": ["retrieval_skipped"], "confidence": confidence, "graph_paths": [], "citations": [], "llm_messages": []})
        return add_log(
            state,
            "answer_contract_prepared",
            {"confidence": confidence, "citation_count": 0, "graph_path_count": 0, "warnings": state["warnings"], "citation_ids": [], "llm_message_role_counts": {}},
        )

    result = state["retrieval_result"]
    context = result.get("selected_context", [])
    warnings = list(result.get("warnings", [])) + [f"router:{route.label}"]
    confidence = state.get("merge_confidence") or confidence_for(result)
    confidence["router"] = route.as_dict()
    confidence = align_confidence_with_route(confidence, route)
    graph_paths = [path for item in context for path in item.get("graph_paths", [])]
    citations = build_citations(context)
    answer_history = state.get("messages_for_answer") or state["messages"]
    llm_messages = build_orchestration_llm_messages(state["query"], context, graph_paths, warnings, answer_history) if state["query"] else []
    role_counts: dict[str, int] = {}
    for message in llm_messages:
        role = str(message.get("role") or "")
        role_counts[role] = role_counts.get(role, 0) + 1
    state.update({"context": context, "warnings": warnings, "confidence": confidence, "graph_paths": graph_paths, "citations": citations, "llm_messages": llm_messages})
    return add_log(
        state,
        "answer_contract_prepared",
        {
            "confidence": confidence,
            "citation_count": len(citations),
            "graph_path_count": len(graph_paths),
            "warnings": warnings,
            "citation_ids": [citation.get("retrieval_id") for citation in citations[:10]],
            "llm_message_role_counts": role_counts,
        },
    )


def mock_llm_output(state: dict) -> str:
    mode = state["mock_llm_mode"]
    citation_ids = [item["retrieval_id"] for item in state["citations"] if item.get("retrieval_id")]
    if mode == "invalid_json":
        return "不是 JSON"
    if mode == "missing_citation":
        payload = {"answer": "这是一个带无效引用的模拟回答。", "answer_type": "llm_grounded", "citation_ids": ["missing-retrieval-id"], "warnings": [], "refusal_reason": None}
    elif mode == "empty_answer":
        payload = {"answer": "", "answer_type": "llm_grounded", "citation_ids": citation_ids[:1], "warnings": [], "refusal_reason": None}
    elif mode == "no_answer":
        payload = {"answer": "检索上下文不足，无法可靠回答。", "answer_type": "no_answer", "citation_ids": [], "warnings": ["insufficient_context"], "refusal_reason": "insufficient_context"}
    else:
        titles = [item.get("title") for item in state["context"][:2] if item.get("title")]
        payload = {"answer": "根据检索上下文，" + ("、".join(titles) if titles else "相关材料") + "可以支持该回答。", "answer_type": "llm_grounded", "citation_ids": citation_ids[:5], "warnings": [], "refusal_reason": None}
    return json.dumps(payload, ensure_ascii=False)


def generate_state(state: dict) -> dict:
    route: RouteDecision = state["route"]
    if not state["query"]:
        state["raw_llm_output"] = None
        return add_log(state, "llm_skipped", {"reason": "empty_query"})
    if not route.needs_retrieval:
        state["raw_llm_output"] = None
        return add_log(state, "llm_skipped", {"reason": "direct_answer_path"})
    confidence = state["confidence"]
    warnings = state["warnings"]
    if confidence.get("level") in {"none", "low"}:
        state["raw_llm_output"] = None
        return add_log(state, "llm_skipped", {"reason": f"confidence_{confidence.get('level')}"})
    if not should_use_llm(state["context"], state["citations"], confidence, warnings):
        state["raw_llm_output"] = None
        return add_log(state, "llm_skipped", {"reason": "confidence_gate"})
    if state["generation_mode"] == "provider":
        client = MinimaxClient()
        if not client.enabled():
            state["raw_llm_output"] = None
            return add_log(state, "llm_skipped", {"reason": "provider_not_configured"})
        try:
            state["raw_llm_output"] = asyncio.run(chat_grounded_json(client, state["llm_messages"]))
            if client.last_usage:
                state.setdefault("metrics", {}).setdefault("token_usage", {})["provider_llm"] = client.last_usage
        except Exception as exc:
            state["raw_llm_output"] = None
            state["llm_provider_error"] = type(exc).__name__
            state["provider_error_detail"] = str(exc)[:500]
            return add_log(state, "provider_llm_failed", {"error_type": type(exc).__name__, "message": str(exc)[:240]})
        return add_log(state, "provider_llm_output", {"provider": settings.llm_provider, "raw_preview": state["raw_llm_output"][:240]})
    state["raw_llm_output"] = mock_llm_output(state)
    return add_log(state, "mock_llm_output", {"mode": state["mock_llm_mode"], "raw_preview": state["raw_llm_output"][:240]})


async def direct_answer_for_state(state: dict) -> dict:
    return await answer_without_retrieval(state["query"], state.get("messages_for_answer") or state["messages"], state["route"])


def finalize_state(state: dict) -> dict:
    route: RouteDecision = state["route"]
    if not state.get("query"):
        response = {"answer": "请输入刑法相关问题。", "answer_type": "no_answer", "citations": [], "graph_paths": [], "warnings": ["empty_query", "insufficient_context"], "confidence": {"level": "none", "score": 0.0}, "refusal_reason": "insufficient_context", "retrieval_context": [], "orchestration_logs": state.get("orchestration_logs", [])}
    elif route.label != "criminal_law_general_direct" and is_out_of_scope_query(state["query"]):
        response = out_of_scope_response()
        response["orchestration_logs"] = state.get("orchestration_logs", [])
    elif not route.needs_retrieval:
        response = asyncio.run(direct_answer_for_state(state))
        response["orchestration_logs"] = state.get("orchestration_logs", [])
    else:
        context = state.get("context", [])
        warnings = state.get("warnings", [])
        confidence = state.get("confidence", {"level": "none", "score": 0.0})
        citations = state.get("citations", [])
        graph_paths = state.get("graph_paths", [])
        if confidence.get("level") == "none":
            warnings.append("insufficient_context")
            answer = "没有检索到足够的刑法案例或法条上下文，无法可靠回答。"
            answer_type = "no_answer"
            refusal_reason = "insufficient_context"
        elif confidence.get("level") == "low":
            warnings.append("low_confidence")
            if can_answer_low_confidence_generally(confidence) and settings.answer_use_llm and MinimaxClient().enabled():
                answer, answer_type, refusal_reason = asyncio.run(
                    low_confidence_llm_answer(state["query"], state.get("messages_for_answer") or state.get("messages", []), warnings)
                )
            else:
                answer = refusal_answer("low_confidence")
                answer_type = "no_answer"
                refusal_reason = "low_confidence"
            citations = []
            graph_paths = []
            context = []
        elif state.get("raw_llm_output"):
            try:
                allowed_ids = {item["retrieval_id"] for item in citations if item.get("retrieval_id")}
                parsed = parse_llm_json(state["raw_llm_output"], allowed_ids)
                warnings.extend(parsed["warnings"])
                answer, answer_type, citations, refusal_reason, misrefused = reconcile_provider_payload(
                    parsed,
                    citations,
                    confidence,
                    warnings,
                    correct_misrefusal=state.get("generation_mode") == "provider",
                )
                if misrefused:
                    context = []
                    graph_paths = []
            except Exception as exc:
                warnings.append("llm_call_failed")
                if isinstance(exc, (ValueError, json.JSONDecodeError)):
                    warnings.append("llm_invalid_json")
                state["llm_provider_error"] = type(exc).__name__
                state["provider_error_detail"] = str(exc)[:500]
                answer = fallback_answer(state["query"], context, warnings, confidence, reason="llm_call_failed")
                answer_type = "retrieval_fallback"
                refusal_reason = refusal_reason_for(confidence, warnings)
        elif state.get("llm_provider_error"):
            warnings.append("llm_call_failed")
            answer = fallback_answer(state["query"], context, warnings, confidence, reason="llm_call_failed")
            answer_type = "retrieval_fallback"
            refusal_reason = refusal_reason_for(confidence, warnings)
        else:
            warnings.append("llm_skipped_by_confidence_gate")
            answer = fallback_answer(state["query"], context, warnings, confidence)
            answer_type = "retrieval_fallback"
            refusal_reason = refusal_reason_for(confidence, warnings)
        if answer_type == "no_answer":
            answer = refusal_answer(refusal_reason, answer)
            citations = []
            graph_paths = []
            context = []
        response = {"answer": answer, "answer_type": answer_type, "citations": citations[:10], "graph_paths": graph_paths[:20], "warnings": sorted(set(warnings)), "confidence": confidence, "refusal_reason": refusal_reason, "retrieval_context": context, "orchestration_logs": state.get("orchestration_logs", [])}
        if state.get("llm_provider_error"):
            response["provider_error"] = {
                "error_type": state.get("llm_provider_error"),
                "message": state.get("provider_error_detail", ""),
            }

    add_log(
        state,
        "final_response",
        {
            "answer_type": response["answer_type"],
            "warning_count": len(response["warnings"]),
            "citation_count": len(response["citations"]),
            "refusal_reason": response["refusal_reason"],
        },
    )
    response["orchestration_logs"] = state.get("orchestration_logs", [])
    metrics = state.get("metrics") or {}
    if metrics:
        started = metrics.get("started_perf")
        if isinstance(started, (int, float)):
            metrics["total_duration_ms"] = int((time.perf_counter() - started) * 1000)
        metrics.pop("started_perf", None)
        response["metrics"] = metrics
    return response


def route_after_router(state: dict) -> str:
    route: RouteDecision = state["route"]
    if not state.get("query"):
        return "direct_answer"
    if route.label != "criminal_law_general_direct" and is_out_of_scope_query(state["query"]):
        return "direct_answer"
    if not route.needs_retrieval:
        return "direct_answer"
    return "law_retriever"


def route_after_evidence(state: dict) -> str:
    return "grounded_answer" if state.get("evidence_sufficient") else "fallback_answer"


def direct_answer_state(state: dict) -> dict:
    route: RouteDecision = state["route"]
    if not state.get("query"):
        response = {
            "answer": "请输入刑法相关问题。",
            "answer_type": "no_answer",
            "citations": [],
            "graph_paths": [],
            "warnings": ["empty_query", "insufficient_context"],
            "confidence": {"level": "none", "score": 0.0},
            "refusal_reason": "insufficient_context",
            "retrieval_context": [],
            "orchestration_logs": state.get("orchestration_logs", []),
        }
    elif route.label != "criminal_law_general_direct" and is_out_of_scope_query(state["query"]):
        response = out_of_scope_response()
        response["orchestration_logs"] = state.get("orchestration_logs", [])
    else:
        response = asyncio.run(direct_answer_for_state(state))
        response["orchestration_logs"] = state.get("orchestration_logs", [])
        response_metrics = response.get("metrics") or {}
        token_usage = response_metrics.get("token_usage") if isinstance(response_metrics, dict) else None
        if isinstance(token_usage, dict):
            state.setdefault("metrics", {}).setdefault("token_usage", {}).update(token_usage)
    state["response"] = response
    return add_log(
        state,
        "direct_answer",
        {"answer_type": response["answer_type"], "refusal_reason": response.get("refusal_reason"), "warning_count": len(response.get("warnings", []))},
    )


def retrieval_response_from_state(state: dict) -> dict:
    context = state.get("context", [])
    warnings = state.get("warnings", [])
    confidence = state.get("confidence", {"level": "none", "score": 0.0})
    citations = state.get("citations", [])
    graph_paths = state.get("graph_paths", [])
    if confidence.get("level") == "none":
        warnings.append("insufficient_context")
        answer = "没有检索到足够的刑法案例或法条上下文，无法可靠回答。"
        answer_type = "no_answer"
        refusal_reason = "insufficient_context"
    elif confidence.get("level") == "low":
        warnings.append("low_confidence")
        if can_answer_low_confidence_generally(confidence) and settings.answer_use_llm and MinimaxClient().enabled():
            answer, answer_type, refusal_reason = asyncio.run(
                low_confidence_llm_answer(state["query"], state.get("messages_for_answer") or state.get("messages", []), warnings)
            )
        else:
            answer = refusal_answer("low_confidence")
            answer_type = "no_answer"
            refusal_reason = "low_confidence"
        citations = []
        graph_paths = []
        context = []
    elif state.get("raw_llm_output"):
        try:
            allowed_ids = {item["retrieval_id"] for item in citations if item.get("retrieval_id")}
            parsed = parse_llm_json(state["raw_llm_output"], allowed_ids)
            warnings.extend(parsed["warnings"])
            answer, answer_type, citations, refusal_reason, misrefused = reconcile_provider_payload(
                parsed,
                citations,
                confidence,
                warnings,
                correct_misrefusal=state.get("generation_mode") == "provider",
            )
            if misrefused:
                context = []
                graph_paths = []
        except Exception as exc:
            warnings.append("llm_call_failed")
            if isinstance(exc, (ValueError, json.JSONDecodeError)):
                warnings.append("llm_invalid_json")
            state["llm_provider_error"] = type(exc).__name__
            state["provider_error_detail"] = str(exc)[:500]
            answer = fallback_answer(state["query"], context, warnings, confidence, reason="llm_call_failed")
            answer_type = "retrieval_fallback"
            refusal_reason = refusal_reason_for(confidence, warnings)
    elif state.get("llm_provider_error"):
        warnings.append("llm_call_failed")
        answer = fallback_answer(state["query"], context, warnings, confidence, reason="llm_call_failed")
        answer_type = "retrieval_fallback"
        refusal_reason = refusal_reason_for(confidence, warnings)
    else:
        warnings.append("llm_skipped_by_confidence_gate")
        answer = fallback_answer(state["query"], context, warnings, confidence)
        answer_type = "retrieval_fallback"
        refusal_reason = refusal_reason_for(confidence, warnings)
    if answer_type == "no_answer":
        answer = refusal_answer(refusal_reason, answer)
        citations = []
        graph_paths = []
        context = []
    response = {
        "answer": answer,
        "answer_type": answer_type,
        "citations": citations[:10],
        "graph_paths": graph_paths[:20],
        "warnings": sorted(set(warnings)),
        "confidence": confidence,
        "refusal_reason": refusal_reason,
        "retrieval_context": context,
        "orchestration_logs": state.get("orchestration_logs", []),
    }
    if state.get("llm_provider_error"):
        response["provider_error"] = {
            "error_type": state.get("llm_provider_error"),
            "message": state.get("provider_error_detail", ""),
        }
    return response


def grounded_answer_state(state: dict) -> dict:
    started = time.perf_counter()
    state = prepare_state(state)
    record_latest_event_duration(state, "answer_contract_prepared", int((time.perf_counter() - started) * 1000))
    started = time.perf_counter()
    state = generate_state(state)
    duration_ms = int((time.perf_counter() - started) * 1000)
    for event in ("provider_llm_output", "provider_llm_failed", "mock_llm_output", "llm_skipped"):
        if any(item.get("event") == event for item in state.get("orchestration_logs", [])):
            record_latest_event_duration(state, event, duration_ms)
            break
    response = retrieval_response_from_state(state)
    state["response"] = response
    return add_log(
        state,
        "grounded_answer",
        {"answer_type": response["answer_type"], "citation_count": len(response["citations"]), "warning_count": len(response["warnings"])},
    )


def fallback_answer_state(state: dict) -> dict:
    started = time.perf_counter()
    state = prepare_state(state)
    record_latest_event_duration(state, "answer_contract_prepared", int((time.perf_counter() - started) * 1000))
    state["raw_llm_output"] = None
    response = retrieval_response_from_state(state)
    state["response"] = response
    return add_log(
        state,
        "fallback_answer",
        {"answer_type": response["answer_type"], "refusal_reason": response.get("refusal_reason"), "warning_count": len(response["warnings"])},
    )


def final_response_state(state: dict) -> dict:
    response = state["response"]
    add_log(
        state,
        "final_response",
        {
            "answer_type": response["answer_type"],
            "warning_count": len(response["warnings"]),
            "citation_count": len(response["citations"]),
            "refusal_reason": response["refusal_reason"],
        },
    )
    response["orchestration_logs"] = state.get("orchestration_logs", [])
    metrics = state.get("metrics") or {}
    if metrics:
        started = metrics.get("started_perf")
        if isinstance(started, (int, float)):
            metrics["total_duration_ms"] = int((time.perf_counter() - started) * 1000)
        metrics.pop("started_perf", None)
        response["metrics"] = metrics
    return response


RouterRunnable = RunnableLambda(router_state)
LawRetrieverRunnable = RunnableLambda(timed_node("law_retriever", law_retriever_state))
CaseRetrieverRunnable = RunnableLambda(timed_node("case_retriever", case_retriever_state))
LexicalRetrieverRunnable = RunnableLambda(timed_node("lexical_retriever", lexical_retriever_state))
GraphRetrieverRunnable = RunnableLambda(timed_node("graph_retriever", graph_retriever_state))
VectorRetrieverRunnable = RunnableLambda(timed_node("vector_retriever", vector_retriever_state))
HybridMergeRerankRunnable = RunnableLambda(timed_node("hybrid_merge_rerank", merge_rerank_state))
EvidenceSufficiencyRunnable = RunnableLambda(timed_node("evidence_sufficiency", evidence_sufficiency_state))
AnswerRunnable = RunnableLambda(prepare_state) | RunnableLambda(generate_state) | RunnableLambda(finalize_state)


def build_chain():
    graph_chain = build_state_graph()
    if graph_chain is not None:
        return graph_chain
    return build_runnable_chain()


def build_runnable_chain():
    return (
        RunnableLambda(start_state)
        | RouterRunnable
        | LawRetrieverRunnable
        | CaseRetrieverRunnable
        | LexicalRetrieverRunnable
        | GraphRetrieverRunnable
        | VectorRetrieverRunnable
        | HybridMergeRerankRunnable
        | EvidenceSufficiencyRunnable
        | AnswerRunnable
    )


def build_state_graph():
    if StateGraph is None or END is None:
        return None

    graph = StateGraph(OrchestrationState)
    graph.add_node("start", timed_node("start", start_state))
    graph.add_node("router", timed_node("router_result", router_state))
    graph.add_node("direct_answer", timed_node("direct_answer", direct_answer_state))
    graph.add_node("law_retriever", timed_node("law_retriever", law_retriever_state))
    graph.add_node("case_retriever", timed_node("case_retriever", case_retriever_state))
    graph.add_node("lexical_retriever", timed_node("lexical_retriever", lexical_retriever_state))
    graph.add_node("graph_retriever", timed_node("graph_retriever", graph_retriever_state))
    graph.add_node("vector_retriever", timed_node("vector_retriever", vector_retriever_state))
    graph.add_node("hybrid_merge_rerank", timed_node("hybrid_merge_rerank", merge_rerank_state))
    graph.add_node("evidence_sufficiency", timed_node("evidence_sufficiency", evidence_sufficiency_state))
    graph.add_node("grounded_answer", timed_node("grounded_answer", grounded_answer_state))
    graph.add_node("fallback_answer", timed_node("fallback_answer", fallback_answer_state))
    graph.add_node("final_response", timed_node("final_response", final_response_state))

    graph.set_entry_point("start")
    graph.add_edge("start", "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "direct_answer": "direct_answer",
            "law_retriever": "law_retriever",
        },
    )
    graph.add_edge("law_retriever", "case_retriever")
    graph.add_edge("case_retriever", "lexical_retriever")
    graph.add_edge("lexical_retriever", "graph_retriever")
    graph.add_edge("graph_retriever", "vector_retriever")
    graph.add_edge("vector_retriever", "hybrid_merge_rerank")
    graph.add_edge("hybrid_merge_rerank", "evidence_sufficiency")
    graph.add_conditional_edges(
        "evidence_sufficiency",
        route_after_evidence,
        {
            "grounded_answer": "grounded_answer",
            "fallback_answer": "fallback_answer",
        },
    )
    graph.add_edge("direct_answer", "final_response")
    graph.add_edge("grounded_answer", "final_response")
    graph.add_edge("fallback_answer", "final_response")
    graph.add_edge("final_response", END)
    return graph.compile()


def run_orchestration(messages: list[dict], top_k: int = 10, mock_llm_mode: str = "grounded") -> dict:
    return response_from_chain_output(build_chain().invoke({"messages": messages, "top_k": top_k, "mock_llm_mode": mock_llm_mode}))


def run_orchestration_with_mode(messages: list[dict], top_k: int = 10, mock_llm_mode: str = "grounded", generation_mode: str = "provider") -> dict:
    return response_from_chain_output(build_chain().invoke({"messages": messages, "top_k": top_k, "mock_llm_mode": mock_llm_mode, "generation_mode": generation_mode}))


def response_from_chain_output(output: dict) -> dict:
    if "answer_type" not in output:
        return output
    return {key: output[key] for key in RESPONSE_KEYS if key in output}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangChain orchestration over the self-owned sample pipeline.")
    parser.add_argument("query", nargs="?", default="吴必定案判了多久")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mock-llm-mode", choices=sorted(MOCK_LLM_MODES), default="grounded")
    parser.add_argument("--generation-mode", choices=sorted(GENERATION_MODES), default="provider")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_orchestration_with_mode([{"role": "user", "content": args.query}], top_k=args.top_k, mock_llm_mode=args.mock_llm_mode, generation_mode=args.generation_mode)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"query: {args.query}")
    print(f"answer_type: {result['answer_type']}")
    print(f"confidence: {result['confidence']['level']} {result['confidence']['score']}")
    print(f"warnings: {', '.join(result['warnings']) if result['warnings'] else '-'}")
    print(f"citations: {len(result['citations'])}")
    print(result["answer"])


if __name__ == "__main__":
    main()
