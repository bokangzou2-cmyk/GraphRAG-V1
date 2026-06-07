from __future__ import annotations

import json
import re
from typing import Literal

import app.processing_bridge  # noqa: F401
from app.config import settings
from app.llm.minimax_client import MinimaxClient, chat_grounded_json
from app.llm.prompting import build_llm_messages
from app.llm.query_understanding import understand_query
from app.llm.query_router import RouteDecision, route_query
from app.llm.scope_guard import is_out_of_scope_query, out_of_scope_response
from hybrid_search_sample import search as hybrid_search
from pydantic import BaseModel, Field, ValidationError, field_validator


class LLMAnswerPayload(BaseModel):
    answer: str = Field(min_length=1)
    answer_type: Literal["llm_grounded", "no_answer"] = "llm_grounded"
    citation_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("llm_answer_empty")
        return value


def latest_user_query(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content", "").strip():
            return message["content"].strip()
    return ""


CRIME_ALIAS_PATTERNS = [
    ("过失致人死亡罪", re.compile(r"过失杀人|过失致人死亡|过失(?:让|使|导致)人死亡")),
    ("故意杀人罪", re.compile(r"故意杀人|杀人")),
    ("盗窃罪", re.compile(r"盗窃|偷东西|偷了|偷窃")),
    ("诈骗罪", re.compile(r"诈骗|骗取|骗了")),
    ("抢劫罪", re.compile(r"抢劫|持刀抢")),
    ("故意伤害罪", re.compile(r"故意伤害|打伤|打成轻伤|打成重伤")),
]

QUERY_EQUIVALENTS = {
    "入室盗窃": "入户盗窃",
    "入室偷窃": "入户盗窃",
    "入室偷东西": "入户盗窃",
    "过失杀人": "过失致人死亡罪",
    "过失让人死亡": "过失致人死亡罪",
    "过失使人死亡": "过失致人死亡罪",
    "过失导致人死亡": "过失致人死亡罪",
}


def normalize_query_terms(query: str) -> str:
    normalized = query.strip()
    expansions = [target for source, target in QUERY_EQUIVALENTS.items() if source in normalized and target not in normalized]
    if expansions:
        normalized = f"{normalized} {' '.join(expansions)}"
    return normalized


def infer_crimes_from_text(text: str) -> list[str]:
    crimes: list[str] = []
    for crime, pattern in CRIME_ALIAS_PATTERNS:
        if pattern.search(text):
            crimes.append(crime)
    for explicit in re.findall(r"[\u4e00-\u9fff]{2,16}罪", text):
        if explicit not in crimes:
            crimes.append(explicit)
    return crimes


def infer_crime_from_text(text: str) -> str:
    crimes = infer_crimes_from_text(text)
    return crimes[0] if crimes else ""


def contextualized_retrieval_query(messages: list[dict]) -> str:
    """Build a retrieval query that keeps short follow-ups tied to the prior criminal-law topic."""
    latest = normalize_query_terms(latest_user_query(messages))
    if not latest:
        return ""
    crime = infer_crime_from_text(latest)
    if not crime:
        for message in reversed(messages[:-1]):
            if message.get("role") not in {"user", "assistant"}:
                continue
            crime = infer_crime_from_text(str(message.get("content") or ""))
            if crime:
                break
    if crime and crime not in latest:
        latest = f"{crime} {latest}"
    if re.search(r"类似|参考|类案|相似|有没有.*案例|案件参考", latest):
        previous_user = next(
            (
                normalize_query_terms(str(message.get("content") or ""))
                for message in reversed(messages[:-1])
                if message.get("role") == "user" and message.get("content")
            ),
            "",
        )
        if previous_user:
            return f"{latest} {previous_user}"
    return latest


def contextualized_route_query(messages: list[dict]) -> str:
    """Keep route classification focused on the latest turn while preserving the topic crime."""
    latest = normalize_query_terms(latest_user_query(messages))
    if not latest:
        return ""
    crime = infer_crime_from_text(latest)
    if not crime:
        for message in reversed(messages[:-1]):
            if message.get("role") not in {"user", "assistant"}:
                continue
            crime = infer_crime_from_text(str(message.get("content") or ""))
            if crime:
                break
    if crime and crime not in latest:
        return f"{crime} {latest}"
    return latest


def citation_from_context(item: dict) -> dict:
    return {
        "retrieval_id": item.get("retrieval_id"),
        "chunk_id": item.get("chunk_id"),
        "title": item.get("title"),
        "source_type": item.get("source_type"),
        "field": item.get("field"),
        "case_id": item.get("case_id"),
        "source_file": item.get("source_file"),
        "text_hash": item.get("text_hash"),
        "article_no": item.get("article_no") or (item.get("law_metadata") or {}).get("article_no"),
        "law_version": item.get("law_version") or (item.get("law_metadata") or {}).get("law_version"),
        "hybrid_score": item.get("hybrid_score"),
        "retrieval_sources": item.get("retrieval_sources", []),
        "evidence_preview": item.get("text_preview"),
        "amount_diagnostics": item.get("amount_diagnostics"),
    }


def title_matches_case(case_name: str, row: dict) -> bool:
    title = row.get("title") or ""
    return bool(case_name and (case_name in title or (len(case_name) >= 3 and case_name[:3] in title)))


def derive_query_type(profile: dict) -> str:
    if profile.get("law_intent"):
        return "law_lookup"
    intents = profile.get("intents") or []
    if intents:
        return str(intents[0])
    if profile.get("case_name"):
        return "case_lookup"
    return "general_criminal"


def confidence_evidence_checks(rows: list[dict], profile: dict) -> list[dict]:
    top_rows = rows[:8]
    preferred_fields = set(profile.get("preferred_fields") or [])
    case_name = str(profile.get("case_name") or "")
    law_intent = bool(profile.get("law_intent"))
    checks: list[dict] = []

    if case_name:
        case_match = any(title_matches_case(case_name, row) for row in top_rows) or any(
            title_matches_case(case_name, {"title": path.get("case_title")})
            for row in top_rows
            for path in row.get("graph_paths", [])
        )
        checks.append(
            {
                "name": "case_name_match",
                "required": True,
                "passed": case_match,
                "expected": case_name,
            }
        )

    if law_intent:
        law_article = any(row.get("source_type") == "law_article" for row in top_rows[:5])
        law_graph = any(
            path.get("edge_type") in {"CASE_APPLIES_ARTICLE", "CASE_CITES_ARTICLE"}
            for row in top_rows
            for path in row.get("graph_paths", [])
        )
        checks.append(
            {
                "name": "law_evidence",
                "required": True,
                "passed": law_article or law_graph,
                "expected": "law_article_or_article_graph_path",
            }
        )

    if preferred_fields and case_name and not law_intent:
        field_match = any(row.get("field") in preferred_fields and title_matches_case(case_name, row) for row in top_rows)
        checks.append(
            {
                "name": "intent_field_match",
                "required": True,
                "passed": field_match,
                "expected": sorted(preferred_fields),
            }
        )

    return checks


def confidence_for(result: dict) -> dict:
    rows = result.get("selected_context", [])
    profile = (result.get("search_summary") or {}).get("query_profile", {})
    query_type = derive_query_type(profile)
    if not rows:
        return {"level": "none", "score": 0.0, "reason": "no_context", "query_type": query_type, "evidence_checks": []}
    top = float(rows[0].get("hybrid_score", 0.0))
    graph_supported = sum(1 for row in rows[:5] if row.get("graph_paths"))
    source_diversity = len({source for row in rows[:5] for source in row.get("retrieval_sources", [])})
    citation_ready = sum(
        1
        for row in rows[:5]
        if row.get("retrieval_id") and row.get("source_file") and row.get("text_hash")
    )
    vector_supported = sum(1 for row in rows[:5] if "vector" in row.get("retrieval_sources", []))
    score = min(
        1.0,
        (top / 420.0)
        + graph_supported * 0.07
        + source_diversity * 0.05
        + citation_ready * 0.03
        + vector_supported * 0.03,
    )
    if citation_ready == 0:
        score = min(score, 0.34)
    evidence_checks = confidence_evidence_checks(rows, profile)
    failed_required = [check["name"] for check in evidence_checks if check.get("required") and not check.get("passed")]
    if failed_required:
        score = min(score, 0.32)
    if score >= 0.65:
        level = "high"
    elif score >= 0.35:
        level = "medium"
    else:
        level = "low"
    reason = f"missing_required_evidence:{','.join(failed_required)}" if failed_required else None
    return {
        "level": level,
        "score": round(score, 3),
        "reason": reason,
        "query_type": query_type,
        "top_hybrid_score": round(top, 3),
        "graph_supported_contexts": graph_supported,
        "source_diversity": source_diversity,
        "citation_ready_contexts": citation_ready,
        "vector_supported_contexts": vector_supported,
        "evidence_checks": evidence_checks,
    }


def refusal_reason_for(confidence: dict, warnings: list[str]) -> str | None:
    if confidence.get("level") == "none":
        return "insufficient_context"
    if confidence.get("level") == "low":
        return "low_confidence"
    if "out_of_scope" in warnings:
        return "out_of_scope"
    if "insufficient_context" in warnings:
        return "insufficient_context"
    return None


def should_use_llm(context: list[dict], citations: list[dict], confidence: dict, warnings: list[str]) -> bool:
    hard_warnings = {"insufficient_context", "out_of_scope"}
    valid_context_count = sum(1 for item in context if item.get("retrieval_id") and item.get("source_file"))
    if hard_warnings & set(warnings):
        return False
    if confidence.get("level") not in {"medium", "high"} or not citations:
        return False
    router = confidence.get("router") or {}
    if router.get("label") == "criminal_law_law_only" and any(item.get("source_type") == "law_article" for item in context):
        return valid_context_count >= 1
    if confidence.get("query_type") == "law_lookup" and any(item.get("source_type") == "law_article" for item in context):
        return valid_context_count >= 1
    return valid_context_count >= 2


def can_answer_low_confidence_generally(confidence: dict) -> bool:
    reason = str(confidence.get("reason") or "")
    if "case_name_match" in reason or "intent_field_match" in reason:
        return False
    return confidence.get("query_type") in {"general_criminal", "law_lookup"}


def align_confidence_with_route(confidence: dict, route: RouteDecision) -> dict:
    """Avoid carrying law-only evidence requirements into case-only follow-up retrieval."""
    if route.law_required:
        return confidence
    checks = list(confidence.get("evidence_checks") or [])
    changed = False
    for check in checks:
        if check.get("name") == "law_evidence":
            check["required"] = False
            changed = True
    if not changed:
        return confidence
    remaining_failed = [check["name"] for check in checks if check.get("required") and not check.get("passed")]
    confidence["evidence_checks"] = checks
    confidence["reason"] = f"missing_required_evidence:{','.join(remaining_failed)}" if remaining_failed else None
    if not remaining_failed and confidence.get("level") == "low":
        score = max(float(confidence.get("score") or 0.0), 0.65)
        confidence["score"] = round(min(score, 1.0), 3)
        confidence["level"] = "high" if confidence["score"] >= 0.65 else "medium"
    return confidence


def parse_llm_json(text: str, allowed_ids: set[str]) -> dict:
    payload = text.strip()
    if not payload.startswith("{"):
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not match:
            raise ValueError("llm_response_not_json")
        payload = match.group(0)
    raw_data = json.loads(payload)
    if not isinstance(raw_data, dict):
        raise ValueError("llm_response_not_object")
    try:
        data = LLMAnswerPayload.model_validate(raw_data)
    except ValidationError as exc:
        raise ValueError("llm_payload_validation_failed") from exc
    citation_ids = [item for item in data.citation_ids if item in allowed_ids]
    if data.answer_type == "llm_grounded" and not citation_ids:
        raise ValueError("llm_missing_valid_citations")
    if data.answer_type == "no_answer" and data.refusal_reason is None:
        data.refusal_reason = "insufficient_context"
    if data.answer_type == "llm_grounded" and data.refusal_reason is not None:
        raise ValueError("llm_grounded_with_refusal_reason")
    return {
        "answer": data.answer,
        "answer_type": data.answer_type,
        "citation_ids": citation_ids,
        "warnings": data.warnings,
        "refusal_reason": data.refusal_reason,
    }


def reconcile_provider_payload(
    parsed: dict,
    citations: list[dict],
    confidence: dict,
    warnings: list[str],
    correct_misrefusal: bool = True,
) -> tuple[str, str, list[dict], str | None, bool]:
    """Normalize provider output against retrieval confidence and citation contract."""
    answer = parsed["answer"]
    answer_type = parsed["answer_type"]
    refusal_reason = parsed["refusal_reason"] or refusal_reason_for(confidence, warnings)
    if correct_misrefusal and answer_type == "no_answer" and confidence.get("level") == "high" and citations:
        warnings.append("provider_misrefused")
        return refusal_answer(refusal_reason), "no_answer", [], refusal_reason or "insufficient_context", True
    if answer_type == "no_answer":
        return refusal_answer(refusal_reason, answer), answer_type, [], refusal_reason or "insufficient_context", False

    used_ids = set(parsed.get("citation_ids") or [])
    selected = [item for item in citations if item.get("retrieval_id") in used_ids]
    min_required = min(2, len(citations))
    if len(selected) < min_required:
        supplement_ids = {item.get("retrieval_id") for item in selected}
        for citation in citations:
            score = citation.get("hybrid_score")
            if isinstance(score, (int, float)) and score <= 0:
                continue
            if citation.get("retrieval_id") not in supplement_ids:
                selected.append(citation)
                supplement_ids.add(citation.get("retrieval_id"))
            if len(selected) >= min_required:
                break
        if len(selected) < min_required:
            min_required = len(selected)
        else:
            warnings.append("provider_citation_count_decreased")
    return answer, answer_type, selected, refusal_reason, False


def refusal_answer(reason: str | None = None, provider_answer: str | None = None) -> str:
    if reason == "out_of_scope":
        return "当前问题超出本系统的刑法样本知识库范围，无法解答。"
    if provider_answer and len(provider_answer.strip()) <= 80 and re.search(r"无法|不足|不能", provider_answer):
        return provider_answer.strip()
    return "当前问题无法根据现有资料可靠解答。"


def fallback_answer(query: str, context: list[dict], warnings: list[str], confidence: dict, reason: str = "llm_not_configured") -> str:
    if not context:
        return "没有检索到足够的刑法案例或法条上下文，无法可靠回答。"
    parts = []
    for item in context[:3]:
        title = item.get("title") or "未命名来源"
        field = item.get("field") or "unknown"
        preview = item.get("text_preview") or ""
        amount = item.get("amount_diagnostics") or {}
        amount_note = ""
        if amount:
            amount_note = (
                f"；金额匹配：{amount.get('matched_amount_raw_text')}"
                f" / {amount.get('matched_amount_role')}"
                f" / 距离{amount.get('amount_distance')}"
            )
        parts.append(f"{title}（{field}{amount_note}）：{preview}")
    prefix = "以下先基于已检索到的资料整理参考。"
    if reason == "llm_call_failed":
        prefix = "以下先基于已检索到的资料整理参考。"
    if confidence.get("level") == "low":
        warnings.append("low_confidence")
        return "当前问题无法根据现有资料可靠解答。"
    return prefix + "\n" + "\n".join(parts)


def general_legal_messages(query: str, history: list[dict]) -> list[dict]:
    recent_history = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in history
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ][-6:]
    return [
        {
            "role": "system",
            "content": (
                "你是中文刑法问答助手。检索资料置信度不足时，只能给一般法律知识解释。"
                "不要编造案例、不要声称已检索到资料、不要给确定个案结论；"
                "如果问题要求规避责任、编造案例或明显超出刑法范围，只回答当前问题无法解答。"
            ),
        },
        *recent_history,
        {"role": "user", "content": f"请回答当前问题：{query}"},
    ]


async def low_confidence_llm_answer(query: str, messages: list[dict], warnings: list[str]) -> tuple[str, str, str | None]:
    try:
        answer = await MinimaxClient().chat(general_legal_messages(query, messages), temperature=0.2, max_tokens=900)
        return answer.strip() or refusal_answer("low_confidence"), "llm_direct", None
    except Exception:
        warnings.append("llm_call_failed")
        return refusal_answer("low_confidence"), "no_answer", "low_confidence"


def direct_llm_messages(query: str, history: list[dict]) -> list[dict]:
    recent_history = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in history
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ][-8:]
    is_criminal_general = bool(re.search(r"刑法|刑事|犯罪|刑罚|犯罪构成", query))
    system_content = (
        "你是一个普通中文问答助手。用户问题不需要刑法知识库检索时，直接自然回答。"
        "不要编造实时信息；如果问题依赖当前位置、实时天气、新闻或价格，要说明需要用户提供信息或联网查询。"
    )
    if is_criminal_general:
        system_content = (
            "你是中文刑法常识问答助手。当前问题不依赖本项目样本法条库或案例库，直接用一般刑法知识回答。"
            "不要编造具体案例、案号或样本库内容；涉及法条数量、施行时间、修法变化等可能变化的信息时，说明应以最新官方文本为准。"
            "如果用户问具体罪名量刑、具体案件、证据、辩护、裁判理由或类案，应提示需要结合具体事实和资料。"
        )
    return [
        {
            "role": "system",
            "content": system_content,
        },
        *recent_history,
    ]


async def answer_without_retrieval(query: str, messages: list[dict], route: RouteDecision) -> dict:
    warnings = ["retrieval_skipped", f"router:{route.label}"]
    confidence = {
        "level": "high",
        "score": route.confidence,
        "query_type": route.label,
        "router": route.as_dict(),
    }
    client = MinimaxClient()
    if settings.answer_use_llm and client.enabled() and route.label not in {"low_quality_or_incomplete", "non_criminal_legal"}:
        try:
            answer = await client.chat(direct_llm_messages(query, messages), temperature=0.3, max_tokens=900)
            return {
                "answer": answer,
                "answer_type": "llm_direct",
                "citations": [],
                "graph_paths": [],
                "warnings": warnings,
                "confidence": confidence,
                "refusal_reason": None,
                "retrieval_context": [],
                "metrics": {"token_usage": {"provider_llm": client.last_usage}} if client.last_usage else {},
            }
        except Exception as exc:
            warnings.append("llm_call_failed")
            return {
                "answer": "当前生成服务暂时不可用，请稍后再试。",
                "answer_type": "no_answer",
                "citations": [],
                "graph_paths": [],
                "warnings": sorted(set(warnings)),
                "confidence": confidence,
                "refusal_reason": "llm_call_failed",
                "retrieval_context": [],
            }
    answer = "该问题不需要刑法知识库检索；当前未启用可直接回答的 LLM。"
    refusal_reason = "llm_not_configured"
    if route.label == "low_quality_or_incomplete":
        answer = "问题不完整或噪声较多，请补充更明确的问题。"
    elif route.label == "non_criminal_legal":
        answer = "该问题不属于当前刑法样本知识库范围，无法解答。"
        refusal_reason = "out_of_scope"
    return {
        "answer": answer,
        "answer_type": "no_answer",
        "citations": [],
        "graph_paths": [],
        "warnings": sorted(set(warnings + ["llm_not_configured"])),
        "confidence": confidence,
        "refusal_reason": refusal_reason,
        "retrieval_context": [],
    }


async def answer_chat(messages: list[dict], top_k: int = 10) -> dict:
    query = latest_user_query(messages)
    if not query:
        return {
            "answer": "请输入刑法相关问题。",
            "answer_type": "no_answer",
            "citations": [],
            "graph_paths": [],
            "warnings": ["empty_query", "insufficient_context"],
            "confidence": {"level": "none", "score": 0.0},
            "refusal_reason": "insufficient_context",
            "retrieval_context": [],
        }
    query_understanding = understand_query(messages)
    retrieval_query = query_understanding.standalone_query or contextualized_retrieval_query(messages)
    route = query_understanding.route_decision()
    if route.label != "criminal_law_general_direct" and is_out_of_scope_query(query):
        return out_of_scope_response()
    if not route.needs_retrieval:
        return await answer_without_retrieval(query, messages, route)
    result = hybrid_search(retrieval_query, top_k=top_k, source_types=route.source_types)
    context = result.get("selected_context", [])
    warnings = list(result.get("warnings", [])) + [f"router:{route.label}", f"query_understanding:{query_understanding.source}"]
    warnings.extend(query_understanding.payload.get("warnings", []))
    confidence = confidence_for(result)
    confidence["router"] = route.as_dict()
    confidence = align_confidence_with_route(confidence, route)
    graph_paths = [path for item in context for path in item.get("graph_paths", [])]
    citations = []
    seen = set()
    for item in context:
        key = item.get("retrieval_id")
        if key and key not in seen:
            seen.add(key)
            citations.append(citation_from_context(item))

    if confidence["level"] == "none":
        warnings.append("insufficient_context")
        answer = "没有检索到足够的刑法案例或法条上下文，无法可靠回答。"
        answer_type = "no_answer"
        refusal_reason = "insufficient_context"
    elif confidence["level"] == "low":
        warnings.append("low_confidence")
        if can_answer_low_confidence_generally(confidence) and settings.answer_use_llm and MinimaxClient().enabled():
            answer, answer_type, refusal_reason = await low_confidence_llm_answer(query, messages, warnings)
            citations = []
            graph_paths = []
            context = []
        else:
            answer = refusal_answer("low_confidence")
            answer_type = "no_answer"
            refusal_reason = "low_confidence"
    elif settings.answer_use_llm and MinimaxClient().enabled() and should_use_llm(context, citations, confidence, warnings):
        try:
            raw_answer = await chat_grounded_json(MinimaxClient(), build_llm_messages(query, context, graph_paths, warnings, messages))
            parsed = parse_llm_json(raw_answer, {item["retrieval_id"] for item in citations if item.get("retrieval_id")})
            warnings.extend(parsed["warnings"])
            answer, answer_type, citations, refusal_reason, misrefused = reconcile_provider_payload(parsed, citations, confidence, warnings)
            if misrefused:
                context = []
                graph_paths = []
        except Exception as exc:
            warnings.append("llm_call_failed")
            if isinstance(exc, (ValueError, json.JSONDecodeError)):
                warnings.append("llm_invalid_json")
            answer = fallback_answer(query, context, warnings, confidence, reason="llm_call_failed")
            answer_type = "retrieval_fallback"
            refusal_reason = refusal_reason_for(confidence, warnings)
    elif settings.answer_use_llm and MinimaxClient().enabled():
        warnings.append("llm_skipped_by_confidence_gate")
        answer = fallback_answer(query, context, warnings, confidence)
        answer_type = "retrieval_fallback"
        refusal_reason = refusal_reason_for(confidence, warnings)
    else:
        warnings.append("llm_not_configured")
        answer = fallback_answer(query, context, warnings, confidence)
        answer_type = "retrieval_fallback"
        refusal_reason = refusal_reason_for(confidence, warnings)

    return {
        "answer": answer,
        "answer_type": answer_type,
        "citations": [] if answer_type == "no_answer" else citations[:10],
        "graph_paths": [] if answer_type == "no_answer" else graph_paths[:20],
        "warnings": sorted(set(warnings)),
        "confidence": confidence,
        "refusal_reason": refusal_reason,
        "retrieval_context": [] if answer_type == "no_answer" else context,
    }
