from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万零〇两0-9]+条")
HISTORICAL_RE = re.compile(r"1979|一九七九|历史|旧刑法")


def _context_text(response: dict) -> str:
    parts = []
    for row in response.get("retrieval_context", []) or []:
        parts.append(str(row.get("title") or ""))
        parts.append(str(row.get("field") or ""))
        parts.append(str(row.get("text_preview") or ""))
        parts.append(json.dumps(row.get("law_metadata") or {}, ensure_ascii=False))
    return "\n".join(parts)


def _citation_ids(response: dict) -> set[str]:
    return {str(item.get("retrieval_id")) for item in response.get("citations", []) if item.get("retrieval_id")}


def _context_ids(response: dict) -> set[str]:
    return {str(item.get("retrieval_id")) for item in response.get("retrieval_context", []) if item.get("retrieval_id")}


def validate_grounding(query: str, response: dict, baseline: dict | None = None) -> dict:
    answer = str(response.get("answer") or "")
    warnings = set(response.get("warnings") or [])
    answer_type = response.get("answer_type")
    context = response.get("retrieval_context", []) or []
    context_text = _context_text(response)
    citation_ids = _citation_ids(response)
    context_ids = _context_ids(response)
    graph_paths = response.get("graph_paths", []) or []
    issues: list[dict] = []

    if answer_type == "llm_grounded" and not citation_ids:
        issues.append({"severity": "high", "type": "grounded_answer_without_citations"})
    if answer_type == "llm_grounded" and not context:
        issues.append({"severity": "high", "type": "grounded_answer_without_context"})
    invalid_citations = sorted(citation_ids - context_ids)
    if invalid_citations:
        issues.append({"severity": "high", "type": "citation_outside_context", "ids": invalid_citations[:10]})

    law_context_ids = {str(row.get("retrieval_id")) for row in context if row.get("source_type") == "law_article" and row.get("retrieval_id")}
    cited_law_ids = citation_ids & law_context_ids
    if law_context_ids and answer_type == "llm_grounded" and not cited_law_ids and re.search(r"法条|依据|规定|第.+条|刑法", query):
        issues.append({"severity": "medium", "type": "law_context_not_cited", "law_context_count": len(law_context_ids)})

    answer_articles = set(ARTICLE_RE.findall(answer))
    context_articles = set(ARTICLE_RE.findall(context_text))
    unsupported_articles = sorted(answer_articles - context_articles)
    if unsupported_articles:
        issues.append({"severity": "high", "type": "unsupported_article_claims", "articles": unsupported_articles})

    if "llm_invalid_json" in warnings or "llm_call_failed" in warnings:
        issues.append({"severity": "medium", "type": "provider_generation_failure", "warnings": sorted(warnings)})

    historical_query = bool(HISTORICAL_RE.search(query))
    if historical_query:
        context_has_1979 = "1979" in context_text or "一九七九" in context_text
        answer_mentions_current = bool(re.search(r"现行|1997|修正|第264条|第234条|第263条", answer))
        answer_refuses = response.get("answer_type") == "no_answer" or "无法确认" in answer or "未提供" in answer
        if answer_mentions_current and not context_has_1979 and not answer_refuses:
            issues.append({"severity": "high", "type": "historical_law_mismatch"})
        elif not context_has_1979 and response.get("answer_type") == "llm_grounded":
            issues.append({"severity": "medium", "type": "historical_law_context_missing"})

    if baseline is not None:
        baseline_graph_count = len(baseline.get("graph_paths", []) or [])
        if baseline_graph_count and not graph_paths:
            issues.append({"severity": "high", "type": "graph_paths_lost", "baseline_graph_paths": baseline_graph_count})
        baseline_citations = len(baseline.get("citations", []) or [])
        if len(citation_ids) < baseline_citations:
            issues.append({"severity": "medium", "type": "citation_count_decreased", "baseline": baseline_citations, "actual": len(citation_ids)})

    score = 1.0
    for issue in issues:
        score -= 0.35 if issue["severity"] == "high" else 0.18
    score = max(0.0, round(score, 3))
    citation_coverage_score = 1.0
    if law_context_ids and re.search(r"法条|依据|规定|第.+条|刑法", query):
        citation_coverage_score = min(1.0, round(len(cited_law_ids) / max(1, min(len(law_context_ids), 3)), 3))
    hallucination_risk_score = round(1.0 - score, 3)
    refusal_quality_score = 1.0
    if answer_type == "no_answer":
        refusal_quality_score = 1.0 if ("无法" in answer or "不足" in answer or "未提供" in answer) else 0.45
    return {
        "grounded_score": score,
        "citation_coverage_score": citation_coverage_score,
        "hallucination_risk_score": hallucination_risk_score,
        "refusal_quality_score": refusal_quality_score,
        "issues": issues,
        "summary": {
            "answer_type": answer_type,
            "context_count": len(context),
            "citation_count": len(citation_ids),
            "graph_path_count": len(graph_paths),
            "historical_query": historical_query,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate grounding risk for one response object.")
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    response = json.loads(args.response.read_text(encoding="utf-8-sig"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig")) if args.baseline else None
    print(json.dumps(validate_grounding(args.query, response, baseline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
