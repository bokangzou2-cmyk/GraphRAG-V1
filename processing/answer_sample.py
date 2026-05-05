from __future__ import annotations

import argparse
import json
import re
from typing import Any

from common import normalize_text
from search_sample import search


def is_sentence_query(query: str) -> bool:
    return bool(re.search(r"判|刑|罚金|处罚|多久|有期徒刑|拘役|缓刑|死刑|无期徒刑", query))


def is_applied_article_query(query: str) -> bool:
    return bool(re.search(r"适用|适用了|依据|裁判依据|法条", query) and re.search(r"案|案件", query))


def is_law_article_query(query: str) -> bool:
    return bool(re.search(r"法条|刑法|第.+条|规定", query))


def is_fact_query(query: str) -> bool:
    return bool(re.search(r"事实|经过|查明|发生了什么|案情", query))


def is_evidence_query(query: str) -> bool:
    return bool(re.search(r"证据|证人|证言|供述|鉴定|证明", query))


def is_reasoning_query(query: str) -> bool:
    return bool(re.search(r"为什么|理由|认定|构成|采纳|不予采纳", query))


def is_amount_query(query: str) -> bool:
    return bool(re.search(r"多少|金额|数额|立方米|元|价值|蓄积量", query))


def citation_from_hit(hit: dict) -> dict:
    return {
        "retrieval_id": hit.get("retrieval_id"),
        "chunk_id": hit.get("chunk_id"),
        "title": hit.get("title"),
        "source_type": hit.get("source_type"),
        "field": hit.get("field"),
        "case_id": hit.get("case_id"),
        "source_file": hit.get("source_file"),
    }


def graph_path_from_expansion(expansion: dict, case_hit: dict | None = None) -> dict:
    law_chunk = expansion.get("law_chunk") or {}
    return {
        "case_id": expansion.get("case_id"),
        "case_title": case_hit.get("title") if case_hit else "",
        "edge_type": expansion.get("edge_type"),
        "article": expansion.get("article"),
        "article_no": expansion.get("article_no"),
        "law_name": expansion.get("law_name"),
        "law_version": expansion.get("law_version"),
        "resolution": expansion.get("resolution"),
        "law_retrieval_id": law_chunk.get("retrieval_id"),
        "law_chunk_id": law_chunk.get("chunk_id"),
        "law_title": law_chunk.get("title"),
    }


def first_case_hit(result: dict, case_id: str | None = None) -> dict | None:
    for hit in result.get("direct_hits", []):
        if hit.get("source_type") != "case_chunk":
            continue
        if case_id and hit.get("case_id") != case_id:
            continue
        return hit
    return None


def unique_citations(citations: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    for citation in citations:
        key = citation.get("retrieval_id") or citation.get("chunk_id")
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(citation)
    return rows


def clean_preview(text: str) -> str:
    text = normalize_text(text)
    text = re.split(r"如不服本判决|本判决为终审判决", text)[0].strip()
    return text.rstrip("。；;") + ("。" if text and not text.endswith("。") else "")


def answer_field_query(result: dict, answer_type: str, fields: list[str], label: str) -> dict:
    warnings = []
    hits = [
        hit for hit in result.get("direct_hits", [])
        if hit.get("source_type") == "case_chunk" and hit.get("field") in fields
    ]
    case_name_match = re.search(r"([\u4e00-\u9fff]{2,8})案", result.get("query", ""))
    if case_name_match:
        case_name = case_name_match.group(1)
        hits.sort(key=lambda hit: 0 if case_name in (hit.get("title") or "") else 1)
    if not hits:
        return {
            "answer_type": answer_type,
            "answer": f"没有在检索结果中找到可引用的{label}内容。",
            "citations": [],
            "graph_paths": [],
            "warnings": ["insufficient_context"],
        }

    hit = hits[0]
    title = hit.get("title") or "该案"
    answer_text = clean_preview(hit.get("text_preview", ""))
    if not answer_text:
        warnings.append("empty_preview")
        answer_text = f"检索命中了{title}的{label}字段，但摘要为空。"
    return {
        "answer_type": answer_type,
        "answer": f"{title}的{label}内容为：{answer_text}",
        "citations": [citation_from_hit(hit)],
        "graph_paths": [],
        "warnings": warnings,
    }


def answer_amount_query(result: dict) -> dict:
    preferred_fields = ["court_found_facts_text", "accusation_text", "reasoning_text", "judgment_text"]
    hits = [
        hit for hit in result.get("direct_hits", [])
        if hit.get("source_type") == "case_chunk" and hit.get("field") in preferred_fields
    ]
    amount_pattern = re.compile(r"[^。；;，,]{0,40}(?:人民币)?\d+(?:\.\d+)?(?:万余)?(?:元|万元|立方米|株|亩)[^。；;，,]{0,80}")
    for hit in hits:
        preview = normalize_text(hit.get("text_preview", ""))
        snippets = [normalize_text(match.group(0)) for match in amount_pattern.finditer(preview)]
        if snippets:
            title = hit.get("title") or "该案"
            return {
                "answer_type": "amount",
                "answer": f"{title}中与数额相关的内容包括：" + "；".join(snippets[:3]) + "。",
                "citations": [citation_from_hit(hit)],
                "graph_paths": [],
                "warnings": [],
            }
    return answer_field_query(result, "amount", preferred_fields, "数额")


def answer_sentence_query(result: dict) -> dict:
    warnings = []
    citations = []
    judgment_hit = next(
        (hit for hit in result.get("direct_hits", []) if hit.get("source_type") == "case_chunk" and hit.get("field") == "judgment_text"),
        None,
    )
    case_hit = judgment_hit or first_case_hit(result)
    if not case_hit:
        return {
            "answer_type": "sentence",
            "answer": "没有在检索结果中找到可引用的判决结果。",
            "citations": [],
            "graph_paths": [],
            "warnings": ["insufficient_context"],
        }

    metadata = case_hit.get("case_metadata") or {}
    sentencing = [normalize_text(item) for item in metadata.get("sentencing", []) if normalize_text(item)]
    judgment_preview = judgment_hit.get("text_preview", "") if judgment_hit else case_hit.get("text_preview", "")
    for extra in re.findall(r"缓刑[一二三四五六七八九十\d]+(?:年|个月|月)(?:[一二三四五六七八九十\d]+(?:个月|月))?", judgment_preview):
        if extra and all(extra not in item for item in sentencing):
            sentencing.append(extra)
    title = case_hit.get("title") or "该案"
    if sentencing:
        answer = f"{title}的量刑结果包括：" + "；".join(sentencing) + "。"
    else:
        answer = clean_preview(case_hit.get("text_preview", ""))
        warnings.append("sentencing_metadata_missing")

    if judgment_hit:
        citations.append(citation_from_hit(judgment_hit))
    else:
        citations.append(citation_from_hit(case_hit))
        warnings.append("judgment_chunk_missing")

    return {
        "answer_type": "sentence",
        "answer": answer,
        "citations": unique_citations(citations),
        "graph_paths": [],
        "warnings": warnings,
    }


def answer_applied_articles_query(result: dict) -> dict:
    warnings = []
    citations = []
    graph_paths = []
    applied_edges = [
        item for item in result.get("graph_expansions", [])
        if item.get("edge_type") == "CASE_APPLIES_ARTICLE"
    ]
    if not applied_edges:
        case_hit = first_case_hit(result)
        if case_hit:
            citations.append(citation_from_hit(case_hit))
        return {
            "answer_type": "applied_articles",
            "answer": "没有在图扩展结果中找到可引用的适用法条。",
            "citations": unique_citations(citations),
            "graph_paths": [],
            "warnings": ["insufficient_context"],
        }

    answer_parts = []
    for edge in applied_edges:
        case_hit = first_case_hit(result, edge.get("case_id"))
        if case_hit:
            citations.append(citation_from_hit(case_hit))
        law_chunk = edge.get("law_chunk")
        if law_chunk:
            citations.append(citation_from_hit(law_chunk))
        if edge.get("resolution") == "no_current_law_chunk":
            warnings.append("no_current_law_chunk")
        graph_paths.append(graph_path_from_expansion(edge, case_hit))
        law_version = edge.get("law_version") or "unknown"
        law_name = edge.get("law_name") or "未知法律"
        resolution = "已补现行刑法条文" if edge.get("resolution") == "law_chunk_found" else "未补现行刑法条文"
        answer_parts.append(f"{edge.get('article')}（{law_name}，law_version={law_version}，{resolution}）")

    case_title = first_case_hit(result, applied_edges[0].get("case_id"))
    prefix = f"{case_title.get('title')}适用的法条包括：" if case_title else "检索命中的案例适用法条包括："
    return {
        "answer_type": "applied_articles",
        "answer": prefix + "；".join(answer_parts) + "。",
        "citations": unique_citations(citations),
        "graph_paths": graph_paths,
        "warnings": sorted(set(warnings)),
    }


def answer_law_article_query(result: dict) -> dict:
    warnings = []
    law_hits = [hit for hit in result.get("direct_hits", []) if hit.get("source_type") == "law_article"]
    crime_tokens = re.findall(r"[\u4e00-\u9fff]{2,12}罪", result.get("query", ""))
    exact_crime_hits = [
        hit for hit in law_hits
        if (hit.get("law_metadata") or {}).get("title") in crime_tokens
    ]
    if exact_crime_hits:
        law_hits = exact_crime_hits
    if not law_hits:
        return {
            "answer_type": "law_articles",
            "answer": "没有在检索结果中找到可引用的刑法条文。",
            "citations": [],
            "graph_paths": [],
            "warnings": ["insufficient_context"],
        }

    answer_parts = []
    citations = []
    for hit in law_hits[:3]:
        metadata = hit.get("law_metadata") or {}
        article_no = metadata.get("article_no") or ""
        title = metadata.get("title") or hit.get("title") or ""
        answer_parts.append(f"第{article_no}条 {title}：{clean_preview(hit.get('text_preview', ''))}")
        citations.append(citation_from_hit(hit))

    return {
        "answer_type": "law_articles",
        "answer": "相关刑法条文包括：" + " ".join(answer_parts),
        "citations": unique_citations(citations),
        "graph_paths": [],
        "warnings": warnings,
    }


def answer_fallback(result: dict) -> dict:
    hit = next((item for item in result.get("final_context", []) if item.get("retrieval_id")), None)
    if not hit:
        return {
            "answer_type": "fallback",
            "answer": "没有找到可引用的上下文，无法回答。",
            "citations": [],
            "graph_paths": [],
            "warnings": ["insufficient_context"],
        }
    return {
        "answer_type": "fallback",
        "answer": clean_preview(hit.get("text_preview", "")),
        "citations": [citation_from_hit(hit)],
        "graph_paths": [],
        "warnings": ["fallback_answer"],
    }


def answer(query: str, top_k: int = 8, expansion_k: int = 12, context_k: int = 12) -> dict:
    result = search(query, top_k=top_k, expansion_k=expansion_k, context_k=context_k)
    if is_applied_article_query(query):
        answer_result = answer_applied_articles_query(result)
    elif is_law_article_query(query):
        answer_result = answer_law_article_query(result)
    elif is_sentence_query(query):
        answer_result = answer_sentence_query(result)
    elif is_amount_query(query):
        answer_result = answer_amount_query(result)
    elif is_fact_query(query):
        answer_result = answer_field_query(result, "fact", ["court_found_facts_text", "accusation_text"], "事实")
    elif is_evidence_query(query):
        answer_result = answer_field_query(
            result,
            "evidence",
            [
                "evidence_text",
                "expert_opinion_texts",
                "documentary_evidence_texts",
                "inspection_records_texts",
                "witness_testimony_texts",
                "defendant_confession_texts",
                "victim_statement_texts",
            ],
            "证据",
        )
    elif is_reasoning_query(query):
        answer_result = answer_field_query(result, "reasoning", ["reasoning_text"], "裁判理由")
    else:
        answer_result = answer_fallback(result)

    return {
        "query": query,
        **answer_result,
        "search_summary": {
            "direct_hit_count": len(result.get("direct_hits", [])),
            "graph_expansion_count": len(result.get("graph_expansions", [])),
            "final_context_count": len(result.get("final_context", [])),
            "top_direct_hits": [
                {
                    "source_type": hit.get("source_type"),
                    "field": hit.get("field"),
                    "title": hit.get("title"),
                    "retrieval_id": hit.get("retrieval_id"),
                    "score": hit.get("score"),
                }
                for hit in result.get("direct_hits", [])[:5]
            ],
        },
    }


def print_text(result: dict) -> None:
    print(f"query: {result['query']}")
    print(f"answer_type: {result['answer_type']}")
    print(f"answer: {result['answer']}")
    if result.get("warnings"):
        print(f"warnings: {', '.join(result['warnings'])}")
    print("\nCitations:")
    for idx, citation in enumerate(result.get("citations", []), 1):
        print(f"{idx}. {citation['retrieval_id']} | {citation['source_type']} {citation['field']} | {citation['title']}")
    if result.get("graph_paths"):
        print("\nGraph paths:")
        for idx, path in enumerate(result["graph_paths"], 1):
            print(
                f"{idx}. Case:{path['case_id']} --{path['edge_type']}--> "
                f"{path['article']} ({path['law_version']}, {path['resolution']})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question using sample search context only.")
    parser.add_argument("query", nargs="?", default="吴必定案判了多久")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--expansion-k", type=int, default=12)
    parser.add_argument("--context-k", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = answer(args.query, top_k=args.top_k, expansion_k=args.expansion_k, context_k=args.context_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)


if __name__ == "__main__":
    main()
