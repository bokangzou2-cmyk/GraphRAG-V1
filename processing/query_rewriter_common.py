from __future__ import annotations

import json
import re
from typing import Any


REWRITER_SYSTEM_PROMPT = (
    "你是刑法 GraphRAG 的 Query Rewriter，只负责在必要时把用户问题改写成独立问题，"
    "并抽取案名、罪名、金额、字段意图和历史案例指代。"
    "如果用户问题本身已经清楚，standalone_query 必须尽量保持原文。"
    "不要补充用户没有明确说出的罪名、案名、金额、事实、法条或检索关键词。"
    "不要判断是否需要检索，不要输出 route_label、needs_retrieval、retrieval_targets、case_required 或 law_required。"
    "必须只输出一个 JSON 对象，不要 Markdown，不要解释。"
)

REWRITER_SCHEMA_EXAMPLE = {
    "standalone_query": "王士东、吉得才盗窃罪一审刑事判决书 案情 法院认定事实 量刑",
    "resolved_case_refs": [
        {
            "ref": "第一个参考案例",
            "index": 1,
            "case_title": "王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书",
            "short_title": "王士东案",
            "source": "assistant_history",
        }
    ],
    "case_name_mentions": ["王士东案"],
    "crime_mentions": ["盗窃罪"],
    "amount_constraints": [{"value": 40000, "role": "theft_amount_total_or_single", "raw_text": "40000元"}],
    "field_intents": ["court_found_facts_text", "reasoning_text", "judgment_text"],
    "rewrite_required": True,
    "confidence": 0.9,
    "warnings": [],
}

REQUIRED_REWRITER_FIELDS = {
    "standalone_query",
    "resolved_case_refs",
    "case_name_mentions",
    "crime_mentions",
    "amount_constraints",
    "field_intents",
    "rewrite_required",
    "confidence",
    "warnings",
}

FORBIDDEN_REWRITER_FIELDS = {
    "route_label",
    "needs_retrieval",
    "retrieval_targets",
    "case_required",
    "law_required",
}

VALID_FIELD_INTENTS = {
    "court_found_facts_text",
    "evidence_text",
    "defense_text",
    "reasoning_text",
    "judgment_text",
    "legal_basis_text",
}


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def query_rewriter_prompt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_history = "\n".join(
        f"{message.get('role')}: {str(message.get('content') or '')[:900]}"
        for message in messages[-8:]
    )
    return [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "请按下面 schema 输出 JSON，字段不能缺失。\n"
                f"schema 示例：{json.dumps(REWRITER_SCHEMA_EXAMPLE, ensure_ascii=False)}\n\n"
                "输出约束：\n"
                "- 只做必要的问题改写和实体/字段抽取。\n"
                "- 原问题已经清楚时，standalone_query 保持原文或只做极小规范化。\n"
                "- 不要把泛化刑法常识问题改成具体罪名、案例、构成要件或量刑检索。\n"
                "- 不要补充用户没有明确说出的罪名、案名、金额、事实、法条或检索关键词。\n"
                "- 不要判断是否检索。\n"
                "- 多轮问题必须消解“第一个案例/第二个案例/这个案子/刚才那个/前面两个”等指代。\n"
                "- 只有用户明确表达具体犯罪行为或罪名时，才把生活化罪名归一到规范罪名，例如入室盗窃->盗窃罪，过失杀人->过失致人死亡罪。\n"
                "- 对“刑法主要涉及哪些违法行为/刑法管什么/犯罪一般有哪些类型”等泛化常识问题，保持原问题，crime_mentions 为空。\n\n"
                f"最近对话：\n{compact_history}"
            ),
        },
    ]


def build_rewriter_messages(source_messages: list[dict[str, Any]], answer: dict[str, Any]) -> list[dict[str, str]]:
    return query_rewriter_prompt(source_messages) + [{"role": "assistant", "content": compact_json(answer)}]


def extract_json(text: str) -> dict[str, Any]:
    payload = str(text or "").strip()
    if not payload.startswith("{"):
        match = re.search(r"\{.*\}", payload, flags=re.S)
        if not match:
            raise ValueError("rewriter_output_not_json")
        payload = match.group(0)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("rewriter_output_not_object")
    return data


def normalize_rewriter_answer(answer: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(answer)
    for field in ["resolved_case_refs", "case_name_mentions", "crime_mentions", "amount_constraints", "field_intents", "warnings"]:
        if not isinstance(normalized.get(field), list):
            normalized[field] = []
    normalized["field_intents"] = [
        str(item)
        for item in normalized.get("field_intents", [])
        if str(item) in VALID_FIELD_INTENTS
    ]
    normalized["standalone_query"] = re.sub(r"\s+", " ", str(normalized.get("standalone_query") or "")).strip()
    normalized["rewrite_required"] = bool(normalized.get("rewrite_required"))
    try:
        normalized["confidence"] = max(0.0, min(1.0, float(normalized.get("confidence", 0.85))))
    except (TypeError, ValueError):
        normalized["confidence"] = 0.85
    return normalized


def validate_rewriter_answer(answer: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_REWRITER_FIELDS - set(answer))
    if missing:
        errors.extend(f"missing:{field}" for field in missing)
    forbidden = sorted(FORBIDDEN_REWRITER_FIELDS & set(answer))
    if forbidden:
        errors.extend(f"forbidden:{field}" for field in forbidden)
    if not str(answer.get("standalone_query") or "").strip():
        errors.append("empty_standalone_query")
    for field in ["resolved_case_refs", "case_name_mentions", "crime_mentions", "amount_constraints", "field_intents", "warnings"]:
        if not isinstance(answer.get(field), list):
            errors.append(f"not_list:{field}")
    return errors
