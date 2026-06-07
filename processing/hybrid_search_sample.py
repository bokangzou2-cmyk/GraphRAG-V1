from __future__ import annotations

import argparse
import json
import re

from amount_utils import amount_distance, amount_query_profile
from common import OUT_DIR, iter_jsonl
from graph_retriever_sample import retrieve as graph_retrieve
from search_sample import search as lexical_search


INTENT_FIELD_GROUPS = [
    ("defense", re.compile(r"辩称|辩护|辩解|被告人说"), {"defense_text"}),
    ("evidence", re.compile(r"证据|证人|证言|供述|鉴定|证明"), {"evidence_text", "witness_testimony_texts", "defendant_confession_texts", "expert_opinion_texts", "inspection_records_texts"}),
    ("reasoning", re.compile(r"为什么|为何|理由|认定|构成|采纳|不予采纳"), {"reasoning_text"}),
    ("fact", re.compile(r"事实|经过|查明|案情|发生了什么"), {"court_found_facts_text", "accusation_text"}),
    ("sentence", re.compile(r"判|刑|罚金|处罚|多久|缓刑|有期徒刑|拘役"), {"judgment_text"}),
]
GENERIC_LAW_TITLES = {"自首", "从犯", "主犯", "缓刑", "共同犯罪的概念", "适用条件", "有期徒刑的期限", "罚金的缴纳", "罚金数额的裁量"}
GENERIC_CASE_WORDS = {"案件", "案例", "类案", "本案", "此案", "该案", "类似案件", "相似案例", "没有类似"}
GENERIC_CASE_NAME_RE = re.compile(r"(盗窃|诈骗|抢劫|伤害|犯罪|刑法|法条|金额|数额|罚金|案例|类案|案件|投案|报案|案发|案情)")
AMOUNTS_PATH = OUT_DIR / "case_amounts_sample.jsonl"


def extract_case_name(query: str) -> str:
    matches = re.findall(r"([\u4e00-\u9fff]{2,12})案", query)
    for value in matches:
        value = re.sub(r"^(?:查询|检索|搜索|查找|看看|了解|分析|关于|有关)", "", value).strip()
        if not value:
            continue
        candidate = f"{value}案"
        if candidate in GENERIC_CASE_WORDS or value in GENERIC_CASE_WORDS:
            continue
        if value.endswith(("案件", "案例", "类案", "本案", "此案", "该案")):
            continue
        if value.endswith(("投", "报", "案发", "案情")):
            continue
        if GENERIC_CASE_NAME_RE.search(value):
            continue
        if re.search(r"类似|相似|参考|没有|哪些|什么", value):
            continue
        return value
    return ""


def query_amount_numbers(query: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", query)


def load_case_amounts() -> dict[str, list[dict]]:
    if not AMOUNTS_PATH.exists():
        return {}
    rows: dict[str, list[dict]] = {}
    for amount in iter_jsonl(AMOUNTS_PATH):
        rows.setdefault(amount.get("case_id", ""), []).append(amount)
    return rows


def normalize_query_terms(query: str) -> str:
    for source, target in {
        "故意伤人罪": "故意伤害罪",
        "危险架驶罪": "危险驾驶罪",
        "抢却罪": "抢劫罪",
        "诈骗人罪": "诈骗罪",
        "偷盗罪": "盗窃罪",
        "过失杀人": "过失致人死亡罪",
        "过失让人死亡": "过失致人死亡罪",
        "过失使人死亡": "过失致人死亡罪",
        "过失导致人死亡": "过失致人死亡罪",
    }.items():
        query = query.replace(source, target)
    return query


def query_profile(query: str) -> dict:
    query = normalize_query_terms(query)
    preferred_fields: set[str] = set()
    intents = []
    for name, pattern, fields in INTENT_FIELD_GROUPS:
        if pattern.search(query):
            intents.append(name)
            preferred_fields.update(fields)
    article_mentions = re.findall(r"第[零〇一二两三四五六七八九十百千\d]+条(?:之[零〇一二两三四五六七八九十\d]+)?", query)
    crime_mentions = set(re.findall(r"[\u4e00-\u9fff]{2,16}罪", query))
    inferred_crimes = {
        "盗窃罪": r"盗窃|偷|小偷|入户|入室|扒窃",
        "诈骗罪": r"诈骗|骗取|被骗",
        "抢劫罪": r"抢劫|持刀抢",
        "过失致人死亡罪": r"过失杀人|过失致人死亡|过失(?:让|使|导致)人死亡",
        "故意杀人罪": r"故意杀人|杀人",
        "故意伤害罪": r"故意伤害|伤人",
    }
    for crime, pattern in inferred_crimes.items():
        if re.search(pattern, query):
            crime_mentions.add(crime)
    amount_profile = amount_query_profile(query)
    return {
        "intents": intents,
        "preferred_fields": preferred_fields,
        "law_intent": bool(
            re.search(r"法条|刑法|第几条|第.+条|规定|适用|依据|对应哪条|哪条|条文", query)
            or bool((crime_mentions or re.search(r"[\u4e00-\u9fff]{2,16}罪", query)) and re.search(r"最判|判几年|怎么判", query))
        ),
        "case_name": extract_case_name(query),
        "article_mentions": article_mentions,
        "crime_mentions": sorted(crime_mentions),
        "compare_intent": bool(re.search(r"比较|差异|区别|是否相同|多个案例|多案", query)),
        "historical_intent": bool(re.search(r"1979年|旧刑法|历史法", query)),
        "amount_mentions": re.findall(r"\d+(?:\.\d+)?\s*(?:元|万元)?", query),
        **amount_profile,
    }


def case_name_matches(case_name: str, title: str) -> bool:
    return not case_name or case_name in title or (len(case_name) >= 3 and case_name[:3] in title)


def crime_matches_text(crimes: set[str], text: str) -> bool:
    return any(crime in text or crime.removesuffix("罪") in text for crime in crimes)


def vector_hits(query: str, top_k: int) -> tuple[list[dict], list[str]]:
    try:
        from vector_search_sample import search as vector_search

        return vector_search(query, top_k=top_k).get("hits", []), []
    except FileNotFoundError:
        return [], ["vector_index_missing"]
    except Exception:
        return [], ["vector_search_failed"]


def normalize_lexical_hit(hit: dict) -> dict:
    return {
        "retrieval_id": hit.get("retrieval_id"),
        "chunk_id": hit.get("chunk_id"),
        "source_type": hit.get("source_type"),
        "case_id": hit.get("case_id"),
        "title": hit.get("title"),
        "field": hit.get("field"),
        "source_file": hit.get("source_file"),
        "text_hash": hit.get("text_hash"),
        "text_preview": hit.get("text_preview", ""),
        "case_metadata": hit.get("case_metadata"),
        "law_metadata": hit.get("law_metadata"),
        "quality_flags": [],
    }


def normalize_graph_candidate(candidate: dict) -> dict:
    row = candidate["chunk"]
    return {
        "retrieval_id": row.get("retrieval_id"),
        "chunk_id": row.get("chunk_id"),
        "source_type": row.get("source_type"),
        "case_id": row.get("case_id"),
        "title": row.get("title"),
        "field": row.get("field"),
        "source_file": row.get("source_file"),
        "text_hash": row.get("text_hash"),
        "text_preview": row.get("text", "")[:360],
        "case_metadata": row.get("case_metadata"),
        "law_metadata": row.get("law_metadata"),
        "quality_flags": row.get("quality_flags", []),
        "candidate_type": candidate.get("candidate_type"),
        "path_score": candidate.get("path_score"),
    }


def merge_candidate(candidates: dict, row: dict, source: str, score: float, graph_paths: list[dict] | None = None) -> None:
    key = row.get("retrieval_id")
    if not key:
        return
    current = candidates.setdefault(key, {
        **row,
        "hybrid_score": 0.0,
        "source_scores": {},
        "retrieval_sources": [],
        "graph_paths": [],
    })
    current["hybrid_score"] += score
    for field in (
        "chunk_id",
        "source_type",
        "case_id",
        "title",
        "field",
        "source_file",
        "text_hash",
        "text_preview",
        "case_metadata",
        "law_metadata",
        "quality_flags",
        "article_no",
        "law_name",
        "law_version",
        "candidate_type",
        "path_score",
    ):
        if not current.get(field) and row.get(field):
            current[field] = row[field]
    current["source_scores"][source] = max(float(score), current["source_scores"].get(source, 0.0))
    if source not in current["retrieval_sources"]:
        current["retrieval_sources"].append(source)
    for path in graph_paths or []:
        if path not in current["graph_paths"]:
            current["graph_paths"].append(path)


def best_amount_match(case_amounts: list[dict], profile: dict) -> dict | None:
    values = profile.get("amount_query_values") or []
    if not values or not case_amounts:
        return None
    preferred_roles = set(profile.get("preferred_amount_roles") or [])
    best: dict | None = None
    for amount in case_amounts:
        if amount.get("value") is None:
            continue
        distance = min(amount_distance(value, float(amount["value"])) for value in values)
        role_match = amount.get("role") in preferred_roles if preferred_roles else True
        weak_role = bool(preferred_roles and not role_match)
        candidate = {
            "amount_query_value": values[0],
            "amount_query_role": profile.get("amount_query_role"),
            "matched_amount_id": amount.get("amount_id"),
            "matched_amount_value": amount.get("value"),
            "matched_amount_role": amount.get("role"),
            "matched_amount_raw_text": amount.get("raw_text"),
            "matched_amount_context": amount.get("context"),
            "amount_distance": int(distance) if float(distance).is_integer() else round(distance, 3),
            "amount_role_match": role_match,
            "amount_role_weak_match": weak_role,
            "amount_role_mismatch_warning": None if role_match else "用户问的是盗窃/涉案金额，但该金额不是对应金额角色，不能作为主类案匹配依据。",
        }
        ranking = (0 if role_match else 1, distance)
        current = (
            0 if best and best["amount_role_match"] else 1,
            float(best["amount_distance"]) if best else float("inf"),
        )
        if best is None or ranking < current:
            best = candidate
    return best


def apply_intent_rerank(candidates: dict[str, dict], profile: dict, amounts_by_case: dict[str, list[dict]] | None = None) -> None:
    preferred_fields = profile["preferred_fields"]
    law_intent = profile["law_intent"]
    case_name = profile["case_name"]
    amounts_by_case = amounts_by_case or {}
    for item in candidates.values():
        adjustments: dict[str, float] = {}
        field = item.get("field")
        title = item.get("title") or ""
        source_type = item.get("source_type")
        sources = set(item.get("retrieval_sources", []))
        law_metadata = item.get("law_metadata") or {}
        case_metadata = item.get("case_metadata") or {}
        article_no = item.get("article_no") or law_metadata.get("article_no")
        graph_only = sources == {"graph"}
        preview = item.get("text_preview") or ""
        case_amount_text = " ".join(str(value) for value in case_metadata.get("amounts") or [])
        crime_mentions = set(profile.get("crime_mentions") or [])
        case_crime_label_text = f"{title} {' '.join(str(value) for value in case_metadata.get('crimes') or [])}"
        case_crime_match = bool(crime_mentions and crime_matches_text(crime_mentions, case_crime_label_text))

        if case_name and source_type == "case_chunk":
            if case_name_matches(case_name, title):
                adjustments["case_name_match"] = 720.0
            else:
                adjustments["case_name_mismatch"] = -900.0

        if preferred_fields:
            if field in preferred_fields:
                base = 180.0 if "graph" in sources and "vector" in sources else 130.0
                adjustments["intent_field_match"] = base
            elif source_type == "case_chunk":
                adjustments["intent_field_mismatch"] = -55.0

        if law_intent:
            if source_type == "law_article":
                adjustments["law_intent_article"] = 900.0
                if case_name:
                    adjustments["case_law_intent_article"] = 1400.0
            elif field in {"witness_testimony_texts", "defendant_confession_texts", "documentary_evidence_texts", "evidence_text"}:
                adjustments["law_intent_evidence_penalty"] = -70.0

        if source_type == "law_article" and law_metadata.get("law_version") == "current":
            adjustments["current_law"] = adjustments.get("current_law", 0.0) + 10.0
        if source_type == "law_article" and profile.get("article_mentions"):
            from search_sample import article_no_from_citation

            requested_articles = {article_no_from_citation(value) for value in profile["article_mentions"]}
            if article_no in requested_articles:
                adjustments["article_exact_match"] = 1200.0
        if source_type == "law_article" and profile.get("crime_mentions"):
            if law_intent and (title in set(profile["crime_mentions"]) or any(str(mention).endswith(title) for mention in profile["crime_mentions"])):
                adjustments["crime_title_exact_match"] = 850.0
            elif preferred_fields and (title in set(profile["crime_mentions"]) or any(str(mention).endswith(title) for mention in profile["crime_mentions"])):
                adjustments["supporting_crime_law_article"] = 1250.0
            elif graph_only and title in GENERIC_LAW_TITLES:
                adjustments["generic_graph_law_penalty"] = -780.0
        if source_type == "law_article" and preferred_fields and not law_intent:
            adjustments["case_intent_law_article_penalty"] = -650.0
        if source_type == "case_chunk" and profile.get("crime_mentions"):
            if any(crime in title for crime in crime_mentions):
                adjustments["case_crime_title_match"] = 360.0
            elif crime_matches_text(crime_mentions, title):
                adjustments["case_crime_title_match"] = 320.0
            elif any(crime in preview for crime in crime_mentions):
                adjustments["case_crime_text_match"] = 180.0
            elif sources == {"graph"}:
                adjustments["graph_only_crime_mismatch_penalty"] = -900.0
            else:
                adjustments["case_crime_mismatch_penalty"] = -360.0
        if source_type == "case_chunk" and re.search(r"入户盗窃|入室盗窃", preview + case_amount_text):
            adjustments["case_burglary_mode_match"] = 420.0
        if source_type == "case_chunk" and profile.get("amount_query_values"):
            amount_match = best_amount_match(amounts_by_case.get(item.get("case_id") or "", []), profile)
            if amount_match:
                item["amount_diagnostics"] = amount_match
                distance = float(amount_match["amount_distance"])
                role_match = bool(amount_match["amount_role_match"])
                if profile.get("preferred_amount_roles") and "盗窃罪" in crime_mentions and not case_crime_match:
                    adjustments["amount_crime_mismatch_penalty"] = -1700.0
                    role_match = False
                if role_match:
                    if distance == 0:
                        adjustments["amount_role_exact_match"] = 1600.0
                    elif distance <= max(2500.0, float(amount_match["amount_query_value"]) * 0.12):
                        adjustments["amount_role_near_match"] = 1100.0
                    elif distance > max(10000.0, float(amount_match["amount_query_value"]) * 0.5):
                        adjustments["amount_role_far_distance_penalty"] = -360.0
                    else:
                        adjustments["amount_role_far_match"] = max(80.0, 420.0 - min(distance / 100.0, 340.0))
                else:
                    adjustments["amount_role_mismatch_penalty"] = -520.0
            else:
                adjustments["case_amount_missing_penalty"] = -120.0
        if source_type == "law_article" and graph_only and not profile.get("article_mentions") and title in GENERIC_LAW_TITLES:
            adjustments["weak_graph_only_law_penalty"] = adjustments.get("weak_graph_only_law_penalty", 0.0) - 420.0
        if profile.get("historical_intent") and source_type == "law_article" and law_metadata.get("law_version") == "current":
            adjustments["historical_current_law_penalty"] = -4200.0
        if profile.get("compare_intent") and source_type == "case_chunk":
            adjustments["multi_case_candidate"] = 60.0

        item["ranking_adjustments"] = adjustments
        item["hybrid_score"] += sum(adjustments.values())


def annotate_ranking(rows: list[dict]) -> None:
    for idx, row in enumerate(rows):
        source_scores = row.get("source_scores", {})
        row["winning_source"] = max(source_scores, key=source_scores.get) if source_scores else None
        next_score = rows[idx + 1]["hybrid_score"] if idx + 1 < len(rows) else None
        row["score_margin_to_next"] = round(row["hybrid_score"] - next_score, 6) if next_score is not None else None


def apply_diversity_rerank(candidates: dict[str, dict]) -> None:
    rows = sorted(candidates.values(), key=lambda item: item["hybrid_score"], reverse=True)
    seen_case_fields: dict[tuple[str, str], int] = {}
    for item in rows:
        adjustments = item.setdefault("ranking_adjustments", {})
        delta = 0.0
        if item.get("quality_flags"):
            adjustments["quality_flag_penalty"] = adjustments.get("quality_flag_penalty", 0.0) - 40.0
            delta -= 40.0
        if item.get("source_type") != "case_chunk":
            item["hybrid_score"] += delta
            continue
        case_id = item.get("case_id") or ""
        field = item.get("field") or ""
        if not case_id or not field:
            item["hybrid_score"] += delta
            continue
        key = (case_id, field)
        repeat_count = seen_case_fields.get(key, 0)
        seen_case_fields[key] = repeat_count + 1
        if repeat_count:
            penalty = 140.0 if field.endswith("_texts") else 85.0
            adjustments["diversity_repeat_field_penalty"] = adjustments.get("diversity_repeat_field_penalty", 0.0) - penalty * repeat_count
            delta -= penalty * repeat_count
        item["hybrid_score"] += delta


def add_case_law_metadata_paths(rows: list[dict]) -> None:
    for row in rows:
        if row.get("source_type") != "case_chunk":
            continue
        metadata = row.get("case_metadata") or {}
        applied_laws = metadata.get("applied_laws") or []
        articles = []
        for item in applied_laws:
            if isinstance(item, dict):
                article = str(item.get("article") or "").strip()
                if article:
                    articles.append({"article": article, "law_name": str(item.get("law_name") or "").strip()})
        for article in metadata.get("applied_articles") or []:
            text = str(article or "").strip()
            if text and not any(item["article"] == text for item in articles):
                articles.append({"article": text, "law_name": ""})
        if not articles:
            continue
        paths = row.setdefault("graph_paths", [])
        for item in articles[:6]:
            path = {
                "edge_type": "CASE_APPLIES_ARTICLE",
                "case_title": row.get("title"),
                "article": item["article"],
                "law_name": item["law_name"],
                "source": "case_metadata",
            }
            if path not in paths:
                paths.append(path)


def merge_retrieval_results(
    query: str,
    top_k: int = 12,
    lexical: dict | None = None,
    graph: dict | None = None,
    vectors: list[dict] | None = None,
    warnings: list[str] | None = None,
    source_types: list[str] | None = None,
) -> dict:
    warnings = list(warnings or [])
    candidates: dict[str, dict] = {}
    profile = query_profile(query)
    amounts_by_case = load_case_amounts()
    lexical = lexical or {"direct_hits": []}
    graph = graph or {"candidates": [], "warnings": []}
    vectors = vectors or []

    for rank, hit in enumerate(lexical.get("direct_hits", []), 1):
        merge_candidate(candidates, normalize_lexical_hit(hit), "lexical", max(0.0, float(hit.get("score", 0.0))) / max(rank, 1))

    warnings.extend(graph.get("warnings", []))
    for item in graph.get("candidates", []):
        row = normalize_graph_candidate(item)
        multiplier = 1.45
        if row.get("source_type") == "law_article" and item.get("candidate_type") == "case_to_article":
            multiplier = 1.2
        merge_candidate(candidates, row, "graph", float(item.get("score", 0.0)) * multiplier, item.get("graph_paths", []))

    for hit in vectors:
        merge_candidate(candidates, hit, "vector", float(hit.get("score", 0.0)) * 120.0)

    for item in candidates.values():
        sources = set(item.get("retrieval_sources", []))
        if {"vector", "graph"} <= sources:
            item["hybrid_score"] += 45.0
        if {"lexical", "vector"} <= sources:
            item["hybrid_score"] += 25.0
        if item.get("graph_paths"):
            item["hybrid_score"] += min(35.0, len(item["graph_paths"]) * 8.0)
        law_metadata = item.get("law_metadata") or {}
        if item.get("source_type") == "law_article" and law_metadata.get("law_version") == "current":
            item["hybrid_score"] += 10.0

    apply_intent_rerank(candidates, profile, amounts_by_case)
    apply_diversity_rerank(candidates)
    rows = sorted(candidates.values(), key=lambda item: item["hybrid_score"], reverse=True)
    if profile.get("law_intent"):
        add_case_law_metadata_paths(rows)
    if source_types is not None:
        allowed_source_types = set(source_types)
        rows = [row for row in rows if row.get("source_type") in allowed_source_types]
        if allowed_source_types == {"law_article"} and any(float(row.get("hybrid_score") or 0.0) > 0 for row in rows):
            rows = [row for row in rows if float(row.get("hybrid_score") or 0.0) > 0]
    annotate_ranking(rows)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["hybrid_score"] = round(row["hybrid_score"], 6)
    selected_context = rows[:top_k]
    if (
        source_types is not None
        and {"law_article", "case_chunk"} <= set(source_types)
        and profile.get("law_intent")
        and selected_context
        and not any(row.get("source_type") == "law_article" for row in selected_context)
    ):
        law_row = next((row for row in rows[top_k:] if row.get("source_type") == "law_article"), None)
        if law_row:
            selected_context = [*selected_context[:-1], law_row]
            for rank, row in enumerate(selected_context, 1):
                row["rank"] = rank
    if not rows:
        warnings.append("insufficient_context")
    if rows and source_types is not None and set(source_types) == {"law_article"}:
        warnings = [warning for warning in warnings if warning != "no_current_law_chunk"]
    return {
        "query": query,
        "warnings": sorted(set(warnings)),
        "selected_context": selected_context,
        "search_summary": {
            "lexical_hits": len(lexical.get("direct_hits", [])),
            "graph_candidates": len(graph.get("candidates", [])),
            "vector_hits": len(vectors),
            "query_profile": {
                "intents": profile["intents"],
                "preferred_fields": sorted(profile["preferred_fields"]),
                "law_intent": profile["law_intent"],
                "case_name": profile["case_name"],
                "crime_mentions": profile["crime_mentions"],
                "article_mentions": profile["article_mentions"],
                "compare_intent": profile["compare_intent"],
                "historical_intent": profile["historical_intent"],
                "amount_mentions": re.findall(r"\d+(?:\.\d+)?\s*(?:元|万元)?", query),
            },
        },
    }


def search(
    query: str,
    top_k: int = 12,
    lexical_k: int = 12,
    vector_k: int = 12,
    graph_k: int = 30,
    source_types: list[str] | None = None,
) -> dict:
    warnings: list[str] = []
    query = normalize_query_terms(query)
    lexical = lexical_search(query, top_k=lexical_k, expansion_k=graph_k, context_k=max(top_k, 12))
    graph = graph_retrieve(query, top_k=graph_k)
    vectors, vector_warnings = vector_hits(query, vector_k)
    warnings.extend(vector_warnings)
    return merge_retrieval_results(
        query,
        top_k=top_k,
        lexical=lexical,
        graph=graph,
        vectors=vectors,
        warnings=warnings,
        source_types=source_types,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid sample search over lexical, graph, and vector candidates.")
    parser.add_argument("query", nargs="?", default="吴必定案判了多久")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = search(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"query: {result['query']}")
    print(f"warnings: {', '.join(result['warnings']) if result['warnings'] else '-'}")
    for item in result["selected_context"]:
        print(f"{item['rank']}. score={item['hybrid_score']} {item['source_type']} {item['field']} {item['title']}")
        print(f"   sources={','.join(item['retrieval_sources'])} id={item['retrieval_id']}")


if __name__ == "__main__":
    main()
