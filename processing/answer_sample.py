from __future__ import annotations

import argparse
import json
import re
from typing import Any

from common import normalize_text
from search_sample import article_no_from_citation, search


ARTICLE_TEXT_RE = re.compile(
    r"第[零〇一二两三四五六七八九十百千\d]+条"
    r"(?:之[零〇一二两三四五六七八九十\d]+)?"
    r"(?:第[零〇一二两三四五六七八九十百千\d]+款)?"
)


def explicit_case_name(query: str) -> str:
    match = re.search(r"([\u4e00-\u9fff]{2,12})案", query)
    return match.group(1) if match else ""


def case_name_matches_title(case_name: str, title: str) -> bool:
    if not case_name:
        return True
    if case_name in title:
        return True
    return len(case_name) >= 3 and case_name[:3] in title


def query_has_case_match(result: dict, case_name: str) -> bool:
    if not case_name:
        return True
    return any(
        hit.get("source_type") == "case_chunk" and case_name_matches_title(case_name, hit.get("title") or "")
        for hit in result.get("direct_hits", [])
    )


def no_answer_result(query: str, answer: str, warnings: list[str], result: dict | None = None) -> dict:
    search_summary = {
        "direct_hit_count": 0,
        "graph_expansion_count": 0,
        "final_context_count": 0,
        "top_direct_hits": [],
    }
    if result is not None:
        search_summary = {
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
        }
    return {
        "query": query,
        "answer_type": "no_answer",
        "answer": answer,
        "citations": [],
        "graph_paths": [],
        "warnings": sorted(set(warnings)),
        "search_summary": search_summary,
    }


def is_out_of_scope_query(query: str) -> bool:
    return bool(re.search(
        r"天气|星期|今天|明天|昨天|几点|时间|你是谁|写一首诗|写诗|民法|民事|合同纠纷|行政处罚法|行政法|"
        r"规避法律责任|销毁证据|假证词|骗过法院|作伪证|证据链断掉|伪造.*证据|逃避侦查|威胁证人|藏匿赃物",
        query,
    ))


def is_historical_law_query(query: str) -> bool:
    return bool(re.search(r"1979年?刑法|旧刑法|历史刑法|历史法条", query))


def is_compare_query(query: str) -> bool:
    return bool(re.search(r"比较|相比|差异|不同|是否相同|一样吗|哪个", query) and query.count("案") >= 2)


def is_sentence_query(query: str) -> bool:
    return bool(re.search(r"判|刑|罚金|处罚|多久|有期徒刑|拘役|缓刑|死刑|无期徒刑", query))


def is_applied_reasoning_query(query: str) -> bool:
    return bool(re.search(r"为什么|为何|理由", query) and re.search(r"适用|依据", query) and re.search(r"案|案件", query))


def is_applied_article_query(query: str) -> bool:
    if is_evidence_query(query):
        return False
    return bool(re.search(r"适用|适用了|依据|裁判依据|法条", query) and re.search(r"案|案件", query))


def is_law_article_query(query: str) -> bool:
    return bool(
        re.search(r"法条|刑法|第.+条|规定|对应哪条|哪条|条文", query)
        or bool(re.search(r"[\u4e00-\u9fff]{2,16}罪", query) and re.search(r"最判|判几年|怎么判", query))
    )


def is_fact_query(query: str) -> bool:
    return bool(re.search(r"事实|经过|查明|发生了什么|案情", query))


def is_evidence_query(query: str) -> bool:
    return bool(re.search(r"证据|证人|证言|供述|鉴定|证明", query))


def is_reasoning_query(query: str) -> bool:
    return bool(re.search(r"为什么|理由|认定|构成|采纳|不予采纳", query))


def is_defense_query(query: str) -> bool:
    return bool(re.search(r"辩称|辩护|辩解|被告人说", query) and not re.search(r"采纳|不予采纳|是否", query))


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
        "text_hash": hit.get("text_hash"),
        "evidence_preview": hit.get("text_preview"),
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
    if not hits:
        try:
            from hybrid_search_sample import search as hybrid_search

            hybrid_result = hybrid_search(result.get("query", ""), top_k=12, source_types=["case_chunk"])
            hits = [
                hit for hit in hybrid_result.get("selected_context", [])
                if hit.get("source_type") == "case_chunk" and hit.get("field") in fields
            ]
            if hits:
                warnings.append("hybrid_field_fallback")
        except Exception:
            warnings.append("hybrid_field_fallback_failed")
    case_name_match = re.search(r"([\u4e00-\u9fff]{2,8})案", result.get("query", ""))
    if case_name_match:
        case_name = case_name_match.group(1)
        hits.sort(key=lambda hit: 0 if case_name_matches_title(case_name, hit.get("title") or "") else 1)
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


def matching_case_ids(result: dict, case_name: str) -> set[str]:
    ids = {
        hit.get("case_id")
        for hit in result.get("direct_hits", [])
        if hit.get("source_type") == "case_chunk"
        and hit.get("case_id")
        and (not case_name or case_name_matches_title(case_name, hit.get("title") or ""))
    }
    if ids:
        return ids
    first = first_case_hit(result)
    return {first["case_id"]} if first and first.get("case_id") else set()


def article_filter_from_query(query: str) -> set[str]:
    return {match.group(0) for match in ARTICLE_TEXT_RE.finditer(query)}


def article_nos_from_query(query: str) -> set[str]:
    return {
        article_no
        for article_no in (article_no_from_citation(item) for item in article_filter_from_query(query))
        if article_no
    }


def answer_applied_reasoning_query(result: dict) -> dict:
    case_name = explicit_case_name(result.get("query", ""))
    case_ids = matching_case_ids(result, case_name)
    article_filters = article_filter_from_query(result.get("query", ""))
    reasoning = answer_field_query(result, "applied_reasoning", ["reasoning_text"], "适用法条理由")
    graph_paths = []
    citations = list(reasoning.get("citations", []))
    warnings = list(reasoning.get("warnings", []))

    for edge in result.get("graph_expansions", []):
        if edge.get("edge_type") != "CASE_APPLIES_ARTICLE":
            continue
        if case_ids and edge.get("case_id") not in case_ids:
            continue
        if article_filters and edge.get("article") not in article_filters:
            continue
        case_hit = first_case_hit(result, edge.get("case_id"))
        path = graph_path_from_expansion(edge, case_hit)
        graph_paths.append(path)
        law_chunk = edge.get("law_chunk")
        if law_chunk:
            citations.append(citation_from_hit(law_chunk))
        if edge.get("resolution") == "no_current_law_chunk":
            warnings.append("no_current_law_chunk")

    if article_filters and not graph_paths:
        warnings.append("requested_article_not_applied")
    article_text = "；".join(path["article"] for path in graph_paths) if graph_paths else "检索命中的适用法条"
    return {
        "answer_type": "applied_reasoning",
        "answer": f"{reasoning['answer']} 关联的适用法条：{article_text}。",
        "citations": unique_citations(citations),
        "graph_paths": graph_paths,
        "warnings": sorted(set(warnings)),
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
    case_name = explicit_case_name(result.get("query", ""))
    case_ids = matching_case_ids(result, case_name)
    article_filters = article_filter_from_query(result.get("query", ""))
    current_only = bool(re.search(r"现行法|现行刑法|现行", result.get("query", "")))
    applied_edges = [
        item for item in result.get("graph_expansions", [])
        if item.get("edge_type") == "CASE_APPLIES_ARTICLE"
        and (not case_ids or item.get("case_id") in case_ids)
        and (not article_filters or item.get("article") in article_filters)
        and (not current_only or item.get("law_version") == "current")
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
    if is_historical_law_query(result.get("query", "")):
        return {
            "answer_type": "no_answer",
            "answer": "该问题指向历史刑法/旧刑法条文，当前样本库不提供可直接引用的历史法条正文，避免误补为现行刑法条文。",
            "citations": [],
            "graph_paths": [],
            "warnings": ["historical_law_context_missing", "insufficient_context"],
        }
    law_hits = [hit for hit in result.get("direct_hits", []) if hit.get("source_type") == "law_article"]
    requested_article_nos = article_nos_from_query(result.get("query", ""))
    if requested_article_nos:
        exact_hits = [
            hit for hit in law_hits
            if (hit.get("law_metadata") or {}).get("article_no") in requested_article_nos
        ]
        if not exact_hits:
            return {
                "answer_type": "no_answer",
                "answer": "没有在现行刑法样本法条中找到请求的具体条号，无法回答。",
                "citations": [],
                "graph_paths": [],
                "warnings": ["requested_article_not_found", "insufficient_context"],
            }
        law_hits = exact_hits
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
    if is_out_of_scope_query(query):
        return no_answer_result(
            query,
            "该问题超出当前刑法案例/刑法条文样本知识库范围，无法回答。",
            ["out_of_scope", "insufficient_context"],
            result,
        )
    if is_compare_query(query):
        return no_answer_result(
            query,
            "当前规则版 answer 层不支持跨案例比较，避免在低置信度下硬凑答案。",
            ["low_confidence", "insufficient_context"],
            result,
        )
    case_name = explicit_case_name(query)
    if case_name and len(case_name) < 3 and "某" not in case_name:
        return no_answer_result(
            query,
            f"案名“{case_name}”过短，无法可靠定位具体案例。",
            ["low_confidence", "insufficient_context"],
            result,
        )
    if "案" in query and not case_name and re.search(r"判|适用|事实|证据|辩称|采纳|构成|金额|多少", query):
        return no_answer_result(
            query,
            "案名信息不足，无法可靠定位具体案例。",
            ["low_confidence", "insufficient_context"],
            result,
        )
    if case_name and not query_has_case_match(result, case_name):
        return no_answer_result(
            query,
            f"没有在样本检索结果中找到标题包含“{case_name}”的案例，无法回答。",
            ["case_not_found", "insufficient_context"],
            result,
        )

    if is_applied_reasoning_query(query):
        answer_result = answer_applied_reasoning_query(result)
    elif is_applied_article_query(query):
        answer_result = answer_applied_articles_query(result)
    elif is_law_article_query(query):
        answer_result = answer_law_article_query(result)
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
    elif is_sentence_query(query):
        answer_result = answer_sentence_query(result)
    elif is_amount_query(query):
        answer_result = answer_amount_query(result)
    elif is_fact_query(query):
        answer_result = answer_field_query(result, "fact", ["court_found_facts_text", "accusation_text"], "事实")
    elif is_defense_query(query):
        answer_result = answer_field_query(result, "defense", ["defense_text"], "辩解/辩护")
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
