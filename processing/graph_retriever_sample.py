from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from amount_utils import amount_distance, amount_query_profile
from common import OUT_DIR, iter_jsonl
from search_sample import ARTICLE_RE, article_no_from_citation, query_tokens


RETRIEVAL_PATH = OUT_DIR / "retrieval_chunks_sample.jsonl"
GRAPH_PATH = OUT_DIR / "graph_sample.jsonl"

CASE_FIELDS_BY_INTENT = [
    ("defense", re.compile(r"辩称|辩护|辩解|被告人说"), {"defense_text"}),
    ("evidence", re.compile(r"证据|证人|证言|供述|鉴定|证明"), {"evidence_text", "witness_testimony_texts", "defendant_confession_texts", "expert_opinion_texts", "inspection_records_texts"}),
    ("reasoning", re.compile(r"为什么|为何|理由|认定|构成|采纳|不予采纳|法院认为"), {"reasoning_text"}),
    ("fact", re.compile(r"事实|经过|查明|案情|发生了什么|法院认定"), {"court_found_facts_text", "accusation_text"}),
    ("sentence", re.compile(r"判|刑|罚金|处罚|多久|缓刑|有期徒刑|拘役"), {"judgment_text"}),
]


def load_rows() -> tuple[list[dict], list[dict]]:
    return list(iter_jsonl(RETRIEVAL_PATH)), list(iter_jsonl(GRAPH_PATH))


def wanted_fields(query: str) -> set[str]:
    fields = set()
    for _, pattern, values in CASE_FIELDS_BY_INTENT:
        if pattern.search(query):
            fields.update(values)
    return fields


def article_nos_from_query(query: str) -> set[str]:
    return {
        article_no
        for article_no in (article_no_from_citation(match.group(0)) for match in ARTICLE_RE.finditer(query))
        if article_no
    }


def crime_tokens_from_query(query: str) -> set[str]:
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,16}罪", query))
    tokens.update(token for token in query_tokens(query) if token.endswith("罪"))
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
            tokens.add(crime)
    return tokens


def case_name_from_query(query: str) -> str:
    match = re.search(r"([\u4e00-\u9fff]{2,12})案", query)
    return match.group(1) if match else ""


def title_matches_case(case_name: str, title: str) -> bool:
    return bool(case_name and (case_name in title or (len(case_name) >= 3 and case_name[:3] in title)))


def crime_matches_text(crimes: set[str], text: str) -> bool:
    return any(crime in text or crime.removesuffix("罪") in text for crime in crimes)


def law_intent(query: str) -> bool:
    return bool(re.search(r"法条|刑法|第几条|规定|依据|适用|条文", query))


def case_intent(query: str) -> bool:
    return bool(re.search(r"案例|案|判决|判了|事实|证据|辩护|理由|认定|量刑", query))


def path(path_type: str, score: float, **payload: object) -> dict:
    return {"path_type": path_type, "path_score": round(score, 4), **payload}


def add_candidate(candidates: dict[str, dict], row: dict, score: float, candidate_type: str, graph_path: dict) -> None:
    key = row.get("retrieval_id")
    if not key:
        return
    current = candidates.setdefault(
        key,
        {
            "retrieval_id": key,
            "chunk": row,
            "score": 0.0,
            "candidate_type": candidate_type,
            "graph_paths": [],
            "path_scores": [],
        },
    )
    current["score"] += score
    current["path_scores"].append(score)
    if graph_path not in current["graph_paths"]:
        current["graph_paths"].append(graph_path)
    if score > max(current.get("path_scores", [0.0])):
        current["candidate_type"] = candidate_type


def build_indexes(rows: list[dict], edges: list[dict]) -> dict:
    rows_by_id = {row.get("retrieval_id"): row for row in rows if row.get("retrieval_id")}
    law_by_article_no = {
        (row.get("law_metadata") or {}).get("article_no") or row.get("article_no"): row
        for row in rows
        if row.get("source_type") == "law_article" and ((row.get("law_metadata") or {}).get("article_no") or row.get("article_no"))
    }
    case_chunks: dict[str, list[dict]] = defaultdict(list)
    article_cases: dict[str, list[dict]] = defaultdict(list)
    applies_by_case: dict[str, list[dict]] = defaultdict(list)
    cites_by_case: dict[str, list[dict]] = defaultdict(list)
    case_crimes: dict[str, set[str]] = defaultdict(set)
    case_titles: dict[str, str] = {}
    case_amounts: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source_type") == "case_chunk" and row.get("case_id"):
            case_chunks[row["case_id"]].append(row)
            case_titles[row["case_id"]] = row.get("title", "")
    for edge in edges:
        source_case = edge.get("source", "").removeprefix("Case:")
        if edge.get("type") == "CASE_OF_CRIME":
            case_crimes[source_case].add(edge.get("target", "").removeprefix("Crime:"))
        elif edge.get("type") == "CASE_APPLIES_ARTICLE":
            applies_by_case[source_case].append(edge)
            if edge.get("article_no"):
                article_cases[edge["article_no"]].append(edge)
        elif edge.get("type") == "CASE_CITES_ARTICLE":
            cites_by_case[source_case].append(edge)
            if edge.get("article_no"):
                article_cases[edge["article_no"]].append(edge)
        elif edge.get("type") == "CASE_HAS_AMOUNT":
            case_amounts[source_case].append(edge)
    return {
        "rows_by_id": rows_by_id,
        "law_by_article_no": law_by_article_no,
        "case_chunks": case_chunks,
        "article_cases": article_cases,
        "applies_by_case": applies_by_case,
        "cites_by_case": cites_by_case,
        "case_crimes": case_crimes,
        "case_titles": case_titles,
        "case_amounts": case_amounts,
    }


def chunk_score_for_query(query: str, row: dict, fields: set[str], case_name: str, crimes: set[str]) -> float:
    score = 0.0
    title = row.get("title", "")
    haystack = f"{title} {row.get('text', '')} {json.dumps(row.get('case_metadata') or {}, ensure_ascii=False)}"
    for token in query_tokens(query):
        if len(token) < 2:
            continue
        if token in title:
            score += len(token) * 8
        elif token in haystack:
            score += len(token) * 1.5
    if case_name and title_matches_case(case_name, title):
        score += 180.0
    if fields and row.get("field") in fields:
        score += 90.0
    metadata_crimes = set((row.get("case_metadata") or {}).get("crimes") or [])
    if crimes & metadata_crimes:
        score += 130.0 + 15.0 * len(crimes & metadata_crimes)
    return score


def retrieve(query: str, top_k: int = 12) -> dict:
    rows, edges = load_rows()
    idx = build_indexes(rows, edges)
    fields = wanted_fields(query)
    crimes = crime_tokens_from_query(query)
    for known_crimes in idx["case_crimes"].values():
        crimes.update(crime for crime in known_crimes if crime and crime in query)
    requested_articles = article_nos_from_query(query)
    case_name = case_name_from_query(query)
    wants_law = law_intent(query)
    wants_case = case_intent(query)
    historical_query = bool(re.search(r"1979年|旧刑法|历史法", query))
    candidates: dict[str, dict] = {}
    warnings: list[str] = []
    amount_profile = amount_query_profile(query)
    amount_values = amount_profile.get("amount_query_values") or []
    preferred_amount_roles = set(amount_profile.get("preferred_amount_roles") or [])

    if historical_query:
        warnings.append("historical_law_context_missing")

    for article_no in requested_articles:
        law_row = idx["law_by_article_no"].get(article_no)
        if law_row and not historical_query:
            score = 1200.0 if wants_law else 620.0
            add_candidate(
                candidates,
                law_row,
                score,
                "article_law_chunk",
                path("query_to_article", score, article_no=article_no, retrieval_id=law_row.get("retrieval_id"), resolution="law_chunk_found"),
            )
        elif historical_query:
            warnings.append("no_current_law_chunk")

        if wants_case or historical_query:
            for edge in idx["article_cases"].get(article_no, [])[:30]:
                case_id = edge.get("source", "").removeprefix("Case:")
                if historical_query and edge.get("law_version") == "current":
                    continue
                for chunk in idx["case_chunks"].get(case_id, []):
                    if fields and chunk.get("field") not in fields:
                        continue
                    score = 520.0 + (120.0 if edge.get("type") == "CASE_APPLIES_ARTICLE" else 40.0)
                    if historical_query:
                        score += 260.0
                    if chunk.get("field") in fields:
                        score += 90.0
                    add_candidate(
                        candidates,
                        chunk,
                        score,
                        "article_to_related_case_chunk",
                        path(
                            "article_to_case_to_chunk",
                            score,
                            article_no=article_no,
                            edge_type=edge.get("type"),
                            case_id=case_id,
                            case_title=chunk.get("title"),
                            field=chunk.get("field"),
                            law_version=edge.get("law_version"),
                            retrieval_id=chunk.get("retrieval_id"),
                        ),
                    )

    if crimes:
        for case_id, case_crimes in idx["case_crimes"].items():
            overlap = crimes & case_crimes
            if not overlap:
                continue
            for chunk in idx["case_chunks"].get(case_id, []):
                if wants_law and chunk.get("source_type") == "case_chunk":
                    continue
                if fields and chunk.get("field") not in fields:
                    continue
                score = 460.0 + len(overlap) * 70.0
                if chunk.get("field") in fields:
                    score += 100.0
                add_candidate(
                    candidates,
                    chunk,
                    score,
                    "crime_to_case_chunk",
                    path("crime_to_case_to_chunk", score, crime=sorted(overlap)[0], case_id=case_id, case_title=chunk.get("title"), field=chunk.get("field"), retrieval_id=chunk.get("retrieval_id")),
                )
            for edge in idx["applies_by_case"].get(case_id, []):
                if edge.get("law_version") != "current":
                    warnings.append("no_current_law_chunk")
                    continue
                law_row = idx["law_by_article_no"].get(edge.get("article_no"))
                if not law_row:
                    continue
                score = (880.0 if wants_law else 45.0) + len(overlap) * (45.0 if wants_law else 5.0)
                if wants_law and law_row.get("title") in crimes:
                    score += 260.0
                add_candidate(
                    candidates,
                    law_row,
                    score,
                    "crime_to_applied_article",
                    path(
                        "crime_to_case_to_article",
                        score,
                        crime=sorted(overlap)[0],
                        case_id=case_id,
                        edge_type="CASE_APPLIES_ARTICLE",
                        article=edge.get("article"),
                        article_no=edge.get("article_no"),
                        law_name=edge.get("law_name"),
                        law_version=edge.get("law_version"),
                        retrieval_id=law_row.get("retrieval_id"),
                        resolution="law_chunk_found",
                    ),
                )

    if amount_values:
        for case_id, amounts in idx["case_amounts"].items():
            case_crime_label_text = f"{idx['case_titles'].get(case_id, '')} {' '.join(idx['case_crimes'].get(case_id, set()))}"
            if crimes and preferred_amount_roles and not crime_matches_text(crimes, case_crime_label_text):
                continue
            best_amount: dict | None = None
            best_distance: float | None = None
            best_role_match = False
            for amount in amounts:
                if amount.get("value") is None:
                    continue
                distance = min(amount_distance(value, float(amount["value"])) for value in amount_values)
                role_match = amount.get("role") in preferred_amount_roles if preferred_amount_roles else True
                ranking_tuple = (0 if role_match else 1, distance)
                if best_amount is None or ranking_tuple < (0 if best_role_match else 1, best_distance if best_distance is not None else float("inf")):
                    best_amount = amount
                    best_distance = distance
                    best_role_match = role_match
            if not best_amount:
                continue
            query_value = float(amount_values[0])
            far_threshold = max(10000.0, query_value * 0.5)
            if best_role_match and best_distance is not None and best_distance > far_threshold:
                continue
            base = 680.0 if best_role_match else 80.0
            closeness = max(0.0, 420.0 - min(float(best_distance or 0.0), 420.0))
            for chunk in idx["case_chunks"].get(case_id, []):
                if fields and chunk.get("field") not in fields:
                    continue
                score = base + closeness
                if chunk.get("field") in fields:
                    score += 80.0
                graph_path = path(
                    "amount_to_case_to_chunk",
                    score,
                    case_id=case_id,
                    case_title=chunk.get("title"),
                    field=chunk.get("field"),
                    retrieval_id=chunk.get("retrieval_id"),
                    amount_id=best_amount.get("amount_id"),
                    amount_value=best_amount.get("value"),
                    amount_role=best_amount.get("role"),
                    amount_raw_text=best_amount.get("raw_text"),
                    amount_query_value=amount_values[0],
                    amount_query_role=amount_profile.get("amount_query_role"),
                    amount_distance=best_distance,
                    amount_role_match=best_role_match,
                    amount_role_mismatch_warning=None if best_role_match else "用户问的金额角色与该金额角色不一致，不能作为主类案匹配依据。",
                )
                add_candidate(candidates, chunk, score, "amount_to_case_chunk", graph_path)

    if case_name:
        for case_id, chunks in idx["case_chunks"].items():
            if not chunks or not title_matches_case(case_name, chunks[0].get("title", "")):
                continue
            for chunk in chunks:
                if fields and chunk.get("field") not in fields:
                    continue
                score = 260.0 if wants_law and not fields else 760.0
                score += 160.0 if chunk.get("field") in fields else 0.0
                add_candidate(
                    candidates,
                    chunk,
                    score,
                    "case_to_chunk",
                    path("case_to_chunk", score, case_id=case_id, case_title=chunk.get("title"), edge_type="CASE_HAS_CHUNK", field=chunk.get("field"), retrieval_id=chunk.get("retrieval_id")),
                )
            for edge in [*idx["applies_by_case"].get(case_id, []), *idx["cites_by_case"].get(case_id, [])]:
                if edge.get("law_version") != "current":
                    warnings.append("no_current_law_chunk")
                    continue
                law_row = idx["law_by_article_no"].get(edge.get("article_no"))
                if not law_row:
                    continue
                score = 620.0 if wants_law else 120.0
                add_candidate(
                    candidates,
                    law_row,
                    score,
                    "case_to_article",
                    path(
                        "case_to_article",
                        score,
                        case_id=case_id,
                        case_title=chunks[0].get("title", ""),
                        edge_type=edge.get("type"),
                        article=edge.get("article"),
                        article_no=edge.get("article_no"),
                        law_name=edge.get("law_name"),
                        law_version=edge.get("law_version"),
                        retrieval_id=law_row.get("retrieval_id"),
                        resolution="law_chunk_found",
                    ),
                )

    # Text-backed graph recall fallback for direct field/case matching.
    for row in rows:
        if row.get("source_type") != "case_chunk":
            continue
        score = chunk_score_for_query(query, row, fields, case_name, crimes)
        if score <= 0:
            continue
        if fields and row.get("field") not in fields and not case_name:
            continue
        add_candidate(
            candidates,
            row,
            min(score, 620.0),
            "metadata_to_case_chunk",
            path("metadata_to_case_chunk", min(score, 620.0), case_id=row.get("case_id"), case_title=row.get("title"), field=row.get("field"), retrieval_id=row.get("retrieval_id")),
        )

    ordered = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    for item in ordered:
        item["score"] = round(float(item["score"]), 4)
        item["path_score"] = round(max(item.get("path_scores") or [0.0]), 4)
    return {
        "query": query,
        "candidates": ordered[:top_k],
        "warnings": sorted(set(warnings)),
        "search_summary": {
            "case_name": case_name,
            "crime_tokens": sorted(crimes),
            "article_nos": sorted(requested_articles),
            "wanted_fields": sorted(fields),
            "law_intent": wants_law,
            "case_intent": wants_case,
            "historical_query": historical_query,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-based sample chunk recall.")
    parser.add_argument("query", nargs="?", default="吴必定案为什么适用第十二条第一款")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = retrieve(args.query, args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"query: {result['query']}")
    if result["warnings"]:
        print(f"warnings: {', '.join(result['warnings'])}")
    for idx, item in enumerate(result["candidates"], 1):
        chunk = item["chunk"]
        print(f"{idx}. score={item['score']} type={item['candidate_type']} {chunk.get('source_type')} {chunk.get('field')} {chunk.get('title')}")
        print(f"   id={item['retrieval_id']}")


if __name__ == "__main__":
    main()
