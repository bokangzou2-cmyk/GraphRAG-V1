from __future__ import annotations

import json
import re


SYSTEM_PROMPT = """你是中文刑法 GraphRAG 问答助手。只能基于给定 retrieval_context 回答。
要求：
1. 不要编造未在上下文出现的法条、案例、事实或结论。
2. 回答要简洁，说明依据来自哪些案例/法条。
3. 如果上下文不足，要明确说无法确认。
4. 保留法条版本差异，例如 1979 年刑法不能当作现行刑法。
5. 只能引用 retrieval_context 中存在的 retrieval_id。
6. 回答金额类问题时必须区分金额角色：盗窃数额、罚金、退赔/追缴、违法所得不能互相替代；如 amount_diagnostics 显示角色不一致或距离较大，只能作为风险提示，不能作为主类案依据。
7. 面向用户回答时使用自然短案名，例如“王士东案”“刘方祥案”；完整判决书标题只作为引用资料，不要写成“第一个参考案例是《……一审刑事判决书》”这类首句。
8. 回答法定刑时必须逐字核对 retrieval_context 中的条文，不得把“三年以上七年以下”写成“七年以上”，不得把“以下/以上”方向写反。
9. 必须输出严格 JSON，不要输出 Markdown 或额外解释。

JSON schema:
{
  "answer": "中文回答；如果证据不足则明确拒答",
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


def build_llm_messages(query: str, context: list[dict], graph_paths: list[dict], warnings: list[str], history: list[dict]) -> list[dict]:
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
                "graph_path_count": len(item.get("graph_paths", [])),
                "amount_diagnostics": item.get("amount_diagnostics"),
            }
            for item in context
        ],
        "graph_paths": graph_paths,
        "warnings": warnings,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *user_history,
        {"role": "user", "content": "请基于以下检索证据回答最后一个问题：\n" + json.dumps(evidence, ensure_ascii=False, indent=2)},
    ]
