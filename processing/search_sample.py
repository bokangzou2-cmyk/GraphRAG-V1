from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from common import OUT_DIR, chinese_to_int, iter_jsonl, normalize_text


RETRIEVAL_PATH = OUT_DIR / "retrieval_chunks_sample.jsonl"
GRAPH_PATH = OUT_DIR / "graph_sample.jsonl"

FIELD_WEIGHTS = {
    "judgment_text": 1.45,
    "reasoning_text": 1.35,
    "court_found_facts_text": 1.25,
    "accusation_text": 1.1,
    "defense_text": 1.15,
    "witness_testimony_texts": 1.2,
    "defendant_confession_texts": 1.2,
    "expert_opinion_texts": 1.15,
    "evidence_text": 1.05,
    "law_article": 1.35,
}
INTENT_FIELD_BOOSTS = [
    (re.compile(r"判|刑|罚金|处罚|多久|有期徒刑|缓刑|剥夺政治权利"), {"judgment_text": 2.4}),
    (re.compile(r"法条|刑法|第.+条|适用|依据|规定|对应哪条|哪条|条文"), {"law_article": 2.2, "reasoning_text": 1.5, "judgment_text": 1.2}),
    (re.compile(r"证人|证言|谁说|证明"), {"witness_testimony_texts": 2.3, "evidence_text": 1.3}),
    (re.compile(r"供述|辩解|被告人说"), {"defendant_confession_texts": 2.2, "defense_text": 1.4}),
    (re.compile(r"自首|坦白|认罪|悔罪|谅解|退赔|赔偿"), {"reasoning_text": 1.9, "defense_text": 1.4, "judgment_text": 1.2}),
    (re.compile(r"为什么|理由|认定|采纳|不予采纳|构成"), {"reasoning_text": 2.2, "court_found_facts_text": 1.4}),
]
STOP_TOKENS = {
    "这个", "那个", "哪些", "什么", "怎么", "如何", "是否", "以及", "相关", "一下",
    "案件", "案例", "法院", "被告人", "判决", "刑法",
}
ARTICLE_RE = re.compile(
    r"第?(?P<num>[零〇一二两三四五六七八九十百千\d]+)条"
    r"(?P<suffix>之[零〇一二两三四五六七八九十\d]+)?"
    r"(?P<rest>第[零〇一二两三四五六七八九十百千\d]+款)?"
)


@dataclass
class SearchIndex:
    retrieval_rows: list[dict]
    law_by_article_no: dict[str, dict]
    graph_by_case: dict[str, list[dict]]


def load_index() -> SearchIndex:
    rows = list(iter_jsonl(RETRIEVAL_PATH))
    law_by_article_no = {}
    for row in rows:
        if row.get("source_type") != "law_article":
            continue
        metadata = row.get("law_metadata") or {}
        article_no = normalize_text(metadata.get("article_no"))
        if article_no:
            law_by_article_no[article_no] = row

    graph_by_case: dict[str, list[dict]] = defaultdict(list)
    for edge in iter_jsonl(GRAPH_PATH):
        source = normalize_text(edge.get("source"))
        if source.startswith("Case:"):
            graph_by_case[source.removeprefix("Case:")].append(edge)
    return SearchIndex(rows, law_by_article_no, dict(graph_by_case))


def article_no_from_citation(citation: str) -> str:
    match = ARTICLE_RE.search(citation)
    if not match:
        return ""
    num = chinese_to_int(match.group("num"))
    if num is None:
        return ""
    suffix = ""
    if match.group("suffix"):
        suffix_no = chinese_to_int(match.group("suffix").removeprefix("之"))
        suffix = f".{suffix_no}" if suffix_no is not None else ""
    return f"{num}{suffix}"


def query_tokens(query: str) -> list[str]:
    query = normalize_text(query)
    for source, target in {
        "故意伤人罪": "故意伤害罪",
        "危险架驶罪": "危险驾驶罪",
        "抢却罪": "抢劫罪",
        "诈骗人罪": "诈骗罪",
        "偷盗罪": "盗窃罪",
    }.items():
        query = query.replace(source, target)
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,12}罪|第[零〇一二两三四五六七八九十百千\d]+条(?:之[零〇一二两三四五六七八九十\d]+)?(?:第[零〇一二两三四五六七八九十百千\d]+款)?|[\u4e00-\u9fff]{2,8}|[A-Za-z0-9.]+", query))
    for token in list(tokens):
        if token.endswith("案") and len(token) > 2:
            tokens.add(token[:-1])
    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", query):
        for size in (2, 3, 4):
            for idx in range(0, max(len(seq) - size + 1, 0)):
                tokens.add(seq[idx:idx + size])
    return sorted(token for token in tokens if token not in STOP_TOKENS)


def field_boost(query: str, field: str) -> float:
    boost = FIELD_WEIGHTS.get(field, 1.0)
    for pattern, boosts in INTENT_FIELD_BOOSTS:
        if pattern.search(query):
            boost *= boosts.get(field, 1.0)
    return boost


def metadata_text(row: dict) -> str:
    if row.get("source_type") == "law_article":
        return " ".join(str(v) for v in (row.get("law_metadata") or {}).values())
    metadata = row.get("case_metadata") or {}
    parts: list[str] = []
    for value in metadata.values():
        if isinstance(value, list):
            parts.append(json.dumps(value, ensure_ascii=False))
        else:
            parts.append(str(value))
    return " ".join(parts)


def score_row(query: str, tokens: list[str], row: dict) -> tuple[float, list[str]]:
    text = row.get("text", "")
    title = row.get("title", "")
    field = row.get("field", "")
    source_type = row.get("source_type", "")
    meta = metadata_text(row)
    haystack = f"{title} {field} {text} {meta}"
    score = 0.0
    reasons = []

    for token in tokens:
        text_hits = min(text.count(token), 3)
        title_hits = min(title.count(token), 2)
        meta_hits = min(meta.count(token), 2)
        if text_hits:
            score += text_hits * max(len(token), 2)
            reasons.append(f"text:{token}")
        if title_hits:
            score += title_hits * max(len(token), 2) * 2.5
            reasons.append(f"title:{token}")
        if meta_hits:
            score += meta_hits * max(len(token), 2) * 1.6
            reasons.append(f"metadata:{token}")

    for citation in ARTICLE_RE.finditer(query):
        article_no = article_no_from_citation(citation.group(0))
        if article_no and row.get("source_type") == "law_article" and (row.get("law_metadata") or {}).get("article_no") == article_no:
            score += 500
            reasons.append(f"article_no:{article_no}")
        elif article_no and row.get("source_type") == "law_article":
            score *= 0.35
            reasons.append(f"article_no_mismatch:{article_no}")

    explicit_case_query = bool(re.search(r"[\u4e00-\u9fff]{2,8}案", query))
    if source_type == "law_article" and score > 0 and re.search(r"法条|刑法|第.+条|规定|对应哪条|哪条|条文", query):
        score += 50
        reasons.append("law_intent")
        if explicit_case_query:
            score *= 0.35
            reasons.append("explicit_case_query_law_penalty")
        else:
            score += 120
            reasons.append("general_law_query_boost")
    if source_type == "law_article" and re.search(r"[\u4e00-\u9fff]{2,12}罪", query) and any(token.endswith("罪") and token in row.get("title", "") for token in tokens):
        score += 35
        reasons.append("law_title_crime")
        if any(token.endswith("罪") and token == row.get("title", "") for token in tokens):
            score += 220
            reasons.append("exact_law_title_crime")
    if source_type == "case_chunk" and score > 0 and re.search(r"案|案件|判|证据|法院|被告人", query):
        score += 3
        reasons.append("case_intent")
    if source_type == "case_chunk" and explicit_case_query and any(token in title and len(token) >= 3 for token in tokens):
        score += 180
        reasons.append("explicit_case_title_match")
    if source_type == "case_chunk" and any(re.fullmatch(r"[\u4e00-\u9fff]某", token) and token in title for token in tokens):
        score += 120
        reasons.append("masked_name_title_match")

    if score:
        score *= field_boost(query, field)
        if row.get("quality_flags"):
            score *= 0.92
            reasons.append("quality_flag_penalty")
    return round(score, 4), sorted(set(reasons))


def direct_search(index: SearchIndex, query: str, top_k: int) -> list[dict]:
    tokens = query_tokens(query)
    scored = []
    for row in index.retrieval_rows:
        score, reasons = score_row(query, tokens, row)
        if score <= 0:
            continue
        scored.append((score, row, reasons))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [format_hit(row, score, reasons) for score, row, reasons in scored[:top_k]]


def format_hit(row: dict, score: float, reasons: list[str]) -> dict:
    return {
        "score": score,
        "source_type": row.get("source_type"),
        "retrieval_id": row.get("retrieval_id"),
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "field": row.get("field"),
        "case_id": row.get("case_id"),
        "source_file": row.get("source_file"),
        "text_hash": row.get("text_hash"),
        "reasons": reasons[:12],
        "text_preview": row.get("text", "")[:260],
        "case_metadata": row.get("case_metadata"),
        "law_metadata": row.get("law_metadata"),
    }


def current_law_article_allowed(edge: dict) -> bool:
    return edge.get("type") == "CASE_APPLIES_ARTICLE" and edge.get("law_version") == "current"


def graph_expansions(index: SearchIndex, direct_hits: list[dict], max_items: int) -> list[dict]:
    expansions = []
    seen = set()
    case_hits = [hit for hit in direct_hits if hit.get("source_type") == "case_chunk" and hit.get("case_id")]
    for hit in case_hits:
        case_id = hit["case_id"]
        edges = sorted(
            index.graph_by_case.get(case_id, []),
            key=lambda edge: 0 if edge.get("type") == "CASE_APPLIES_ARTICLE" else 1,
        )
        for edge in edges:
            if edge.get("type") not in {"CASE_APPLIES_ARTICLE", "CASE_CITES_ARTICLE"}:
                continue
            article = normalize_text(edge.get("article") or edge.get("target", "").removeprefix("Article:"))
            article_no = normalize_text(edge.get("article_no")) or article_no_from_citation(article)
            law_row = index.law_by_article_no.get(article_no) if current_law_article_allowed(edge) else None
            key = (case_id, edge.get("type"), article, law_row.get("chunk_id") if law_row else "")
            if key in seen:
                continue
            seen.add(key)
            expansions.append({
                "case_id": case_id,
                "edge_type": edge.get("type"),
                "article": article,
                "article_no": article_no,
                "law_name": edge.get("law_name", ""),
                "law_version": edge.get("law_version", "unknown"),
                "edge_evidence_text": edge.get("evidence_text"),
                "resolution": "law_chunk_found" if law_row else "no_current_law_chunk",
                "law_chunk": format_hit(law_row, 0.0, ["graph_expansion"]) if law_row else None,
            })
            if len(expansions) >= max_items:
                return expansions
    return expansions


def build_final_context(direct_hits: list[dict], expansions: list[dict], max_items: int) -> list[dict]:
    rows = []
    seen = set()
    for hit in direct_hits:
        key = hit.get("retrieval_id") or hit.get("chunk_id")
        if key in seen:
            continue
        seen.add(key)
        rows.append({"context_source": "direct", **hit})
        if len(rows) >= max_items:
            return rows
    for expansion in expansions:
        law_chunk = expansion.get("law_chunk")
        if not law_chunk:
            continue
        key = law_chunk.get("retrieval_id") or law_chunk.get("chunk_id")
        if key in seen:
            continue
        seen.add(key)
        rows.append({"context_source": "graph_expansion", **law_chunk, "graph_edge": {
            "case_id": expansion.get("case_id"),
            "edge_type": expansion.get("edge_type"),
            "article": expansion.get("article"),
            "law_name": expansion.get("law_name"),
            "law_version": expansion.get("law_version"),
        }})
        if len(rows) >= max_items:
            break
    return rows


def search(query: str, top_k: int = 10, expansion_k: int = 8, context_k: int = 12) -> dict:
    index = load_index()
    direct_hits = direct_search(index, query, top_k)
    expansions = graph_expansions(index, direct_hits, expansion_k)
    final_context = build_final_context(direct_hits, expansions, context_k)
    return {
        "query": query,
        "tokens": query_tokens(query),
        "direct_hits": direct_hits,
        "graph_expansions": expansions,
        "final_context": final_context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search sample GraphRAG retrieval chunks.")
    parser.add_argument("query", nargs="?", default="吴必定案判了多久")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expansion-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    args = parser.parse_args()

    result = search(args.query, args.top_k, args.expansion_k, args.context_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"query: {result['query']}")
    print(f"tokens: {', '.join(result['tokens'])}")
    print("\nDirect hits:")
    for idx, hit in enumerate(result["direct_hits"], 1):
        print(f"{idx}. score={hit['score']} {hit['source_type']} {hit['field']} {hit['title']}")
        print(f"   id={hit['retrieval_id']}")
        print(f"   reasons={', '.join(hit['reasons'])}")
        print(f"   text={hit['text_preview']}")

    print("\nGraph expansions:")
    for idx, item in enumerate(result["graph_expansions"], 1):
        law_chunk = item.get("law_chunk") or {}
        print(f"{idx}. {item['edge_type']} {item['article']} ({item.get('law_version')}) -> {law_chunk.get('title', item.get('resolution'))}")

    print("\nFinal context:")
    for idx, item in enumerate(result["final_context"], 1):
        print(f"{idx}. [{item['context_source']}] {item['source_type']} {item['field']} {item['title']}")
        print(f"   id={item['retrieval_id']}")


if __name__ == "__main__":
    main()
