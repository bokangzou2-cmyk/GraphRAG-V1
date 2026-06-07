from __future__ import annotations

from common import OUT_DIR, iter_jsonl, write_jsonl


INPUT = OUT_DIR / "cases_enriched_sample.jsonl"
CASE_CHUNKS = OUT_DIR / "case_chunks_sample.jsonl"
LAW_CHUNKS = OUT_DIR / "law_chunks_sample.jsonl"
AMOUNTS = OUT_DIR / "case_amounts_sample.jsonl"
OUTPUT = OUT_DIR / "graph_sample.jsonl"
NODES_OUTPUT = OUT_DIR / "graph_nodes_sample.jsonl"
EDGES_OUTPUT = OUT_DIR / "graph_edges_sample.jsonl"


def article_no_from_article(article: str) -> str:
    from common import chinese_to_int
    import re

    match = re.search(
        r"第(?P<num>[零〇一二两三四五六七八九十百千\d]+)条"
        r"(?P<suffix>之[零〇一二两三四五六七八九十\d]+)?",
        article,
    )
    if not match:
        return ""
    number = chinese_to_int(match.group("num"))
    if number is None:
        return ""
    suffix = ""
    if match.group("suffix"):
        suffix_no = chinese_to_int(match.group("suffix").removeprefix("之"))
        suffix = f".{suffix_no}" if suffix_no is not None else ""
    return f"{number}{suffix}"


def normalized_citation(article: str) -> str:
    return article.strip()


def law_version(law_name: str) -> str:
    if "1979年" in law_name:
        return "1979"
    if "解释" in law_name or "最高人民法院" in law_name or "最高人民检察院" in law_name:
        return "judicial_interpretation"
    if law_name == "中华人民共和国刑法":
        return "current"
    if law_name:
        return "other"
    return "unknown"


def law_family(law_name: str) -> str:
    if "刑法" in law_name:
        return "criminal_law"
    if "解释" in law_name or "最高人民法院" in law_name or "最高人民检察院" in law_name:
        return "judicial_interpretation"
    if law_name:
        return "other_law"
    return "unknown"


def current_law_article_nos() -> set[str]:
    return {
        chunk.get("article_no", "")
        for chunk in iter_jsonl(LAW_CHUNKS)
        if chunk.get("article_no")
    }


def applied_law_lookup(case: dict) -> dict[str, dict]:
    rows = {}
    for item in case.get("applied_laws", []):
        article = item.get("article", "")
        law_name = item.get("law_name", "")
        if article:
            rows[article] = {
                "law_name": law_name,
                "law_version": law_version(law_name),
                "law_family": law_family(law_name),
                "article_no": article_no_from_article(article),
            }
    return rows


def case_chunk_lookup() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for chunk in iter_jsonl(CASE_CHUNKS):
        rows.setdefault(chunk.get("case_id", ""), []).append(chunk)
    return rows


def source_chunk_for_apply(case_id: str, evidence_text: str, chunks_by_case: dict[str, list[dict]]) -> dict:
    chunks = chunks_by_case.get(case_id, [])
    for preferred_field in ("reasoning_text", "judgment_text", "accusation_text"):
        for chunk in chunks:
            if chunk.get("field") != preferred_field:
                continue
            text = chunk.get("text", "")
            if evidence_text and (evidence_text in text or text in evidence_text):
                return chunk
    for chunk in chunks:
        if chunk.get("field") == "reasoning_text":
            return chunk
    return {}


def infer_cited_law_info(article: str, applied_lookup: dict[str, dict], legal_basis_text: str, catalog_article_nos: set[str]) -> dict:
    if article in applied_lookup:
        return applied_lookup[article] | {"resolution_status": "matched_applied_law"}
    article_no = article_no_from_article(article)
    if not article_no:
        return {"law_name": "", "law_version": "unknown", "law_family": "unknown", "article_no": "", "resolution_status": "unparsed_article"}
    if "1979年" in legal_basis_text and article in legal_basis_text:
        return {"law_name": "1979年中华人民共和国刑法", "law_version": "1979", "law_family": "criminal_law", "article_no": article_no, "resolution_status": "historical_context"}
    if article_no in catalog_article_nos:
        return {"law_name": "中华人民共和国刑法", "law_version": "current", "law_family": "criminal_law", "article_no": article_no, "resolution_status": "inferred_current_catalog"}
    return {"law_name": "", "law_version": "unknown", "law_family": "unknown", "article_no": article_no, "resolution_status": "catalog_miss"}


def build_edges() -> list[dict]:
    rows: list[dict] = []
    chunks_by_case = case_chunk_lookup()
    catalog_article_nos = current_law_article_nos()
    for case in iter_jsonl(INPUT):
        case_id = case["case_id"]
        applied_lookup = applied_law_lookup(case)
        for article in case.get("cited_articles", []):
            law_info = infer_cited_law_info(article, applied_lookup, case.get("legal_basis_text", ""), catalog_article_nos)
            rows.append({
                "type": "CASE_CITES_ARTICLE",
                "source": f"Case:{case_id}",
                "target": f"Article:{law_info.get('law_version', 'unknown')}:{law_info.get('article_no') or article}",
                "raw_citation": article,
                "normalized_citation": normalized_citation(article),
                "article": article,
                "article_no": law_info.get("article_no", ""),
                "law_name": law_info.get("law_name", ""),
                "law_version": law_info.get("law_version", "unknown"),
                "law_family": law_info.get("law_family", "unknown"),
                "citation_role": "cited",
                "resolution_status": law_info.get("resolution_status", ""),
                "evidence_text": article,
                "confidence": 0.7 if law_info.get("law_version") != "current" else 0.6,
                "extraction_method": "regex_citation",
            })
        for article in case.get("applied_articles", []):
            law_info = applied_lookup.get(article, {})
            evidence_text = case.get("legal_basis_text", article)
            source_chunk = source_chunk_for_apply(case_id, evidence_text, chunks_by_case)
            article_no = law_info.get("article_no") or article_no_from_article(article)
            version = law_info.get("law_version", "unknown")
            rows.append({
                "type": "CASE_APPLIES_ARTICLE",
                "source": f"Case:{case_id}",
                "target": f"Article:{version}:{article_no or article}",
                "raw_citation": article,
                "normalized_citation": normalized_citation(article),
                "article": article,
                "article_no": article_no,
                "law_name": law_info.get("law_name", ""),
                "law_version": version,
                "law_family": law_info.get("law_family", "unknown"),
                "citation_role": "applied",
                "resolution_status": "law_name_parsed" if law_info.get("law_name") else "law_name_missing",
                "source_chunk_id": source_chunk.get("chunk_id", ""),
                "source_retrieval_id": f"case:{source_chunk.get('chunk_id')}" if source_chunk else "",
                "source_field": source_chunk.get("field", ""),
                "source_text_hash": source_chunk.get("text_hash", ""),
                "provenance_status": "chunk_grounded" if source_chunk else "case_level_only",
                "evidence_text": evidence_text,
                "confidence": 0.85,
                "extraction_method": "regex_legal_basis",
            })
        for crime in case.get("crimes", []):
            rows.append({
                "type": "CASE_OF_CRIME",
                "source": f"Case:{case_id}",
                "target": f"Crime:{crime}",
                "evidence_text": crime,
                "confidence": 0.7,
                "extraction_method": "regex_crime",
            })
    if AMOUNTS.exists():
        for amount in iter_jsonl(AMOUNTS):
            amount_id = amount.get("amount_id", "")
            case_id = amount.get("case_id", "")
            if not amount_id or not case_id:
                continue
            rows.append({
                "type": "CASE_HAS_AMOUNT",
                "source": f"Case:{case_id}",
                "target": f"Amount:{amount_id}",
                "case_id": case_id,
                "amount_id": amount_id,
                "value": amount.get("value"),
                "unit": amount.get("unit", "CNY"),
                "role": amount.get("role", "unknown_amount"),
                "raw_text": amount.get("raw_text", ""),
                "field": amount.get("field", ""),
                "context": amount.get("context", ""),
                "source_file": amount.get("source_file", ""),
                "text_hash": amount.get("text_hash", ""),
                "evidence_text": amount.get("context", ""),
                "confidence": amount.get("confidence", 0.0),
                "extraction_method": amount.get("extraction_method", "regex_context_role_v1"),
            })
    for chunk in iter_jsonl(CASE_CHUNKS):
        rows.append({
            "type": "CASE_HAS_CHUNK",
            "source": f"Case:{chunk.get('case_id')}",
            "target": f"Chunk:{chunk.get('chunk_id')}",
            "chunk_id": chunk.get("chunk_id"),
            "retrieval_id": f"case:{chunk.get('chunk_id')}",
            "field": chunk.get("field"),
            "title": chunk.get("title"),
            "evidence_text": chunk.get("text", "")[:240],
            "confidence": 1.0,
            "extraction_method": "chunk_metadata",
        })
    for chunk in iter_jsonl(LAW_CHUNKS):
        article_no = chunk.get("article_no", "")
        rows.append({
            "type": "ARTICLE_HAS_CHUNK",
            "source": f"Article:{article_no}",
            "target": f"Chunk:{chunk.get('chunk_id')}",
            "article": f"第{article_no}条",
            "article_no": article_no,
            "chunk_id": chunk.get("chunk_id"),
            "retrieval_id": f"law:{chunk.get('chunk_id')}",
            "field": "law_article",
            "title": chunk.get("title"),
            "law_name": (chunk.get("law_metadata") or {}).get("law_name", "中华人民共和国刑法"),
            "law_version": "current",
            "evidence_text": chunk.get("text", "")[:240],
            "confidence": 1.0,
            "extraction_method": "chunk_metadata",
        })
    return rows


def build_nodes(edges: list[dict]) -> list[dict]:
    nodes: dict[str, dict] = {}
    for edge in edges:
        for endpoint in ("source", "target"):
            node_id = edge.get(endpoint)
            if not node_id:
                continue
            if node_id.startswith("Case:"):
                node_type = "case"
            elif node_id.startswith("Article:"):
                node_type = "article"
            elif node_id.startswith("Crime:"):
                node_type = "crime"
            elif node_id.startswith("Chunk:"):
                node_type = "chunk"
            elif node_id.startswith("Amount:"):
                node_type = "amount"
            else:
                node_type = "unknown"
            nodes.setdefault(node_id, {"node_id": node_id, "node_type": node_type, "source_file": "", "text_hash": ""})
        if edge.get("type") == "CASE_HAS_CHUNK":
            nodes[edge["target"]].update({
                "retrieval_id": edge.get("retrieval_id"),
                "field": edge.get("field"),
                "title": edge.get("title"),
            })
        elif edge.get("type") == "ARTICLE_HAS_CHUNK":
            nodes[edge["source"]].update({
                "article_no": edge.get("article_no"),
                "law_name": edge.get("law_name"),
                "law_version": edge.get("law_version"),
                "title": edge.get("title"),
            })
        elif edge.get("type") in {"CASE_APPLIES_ARTICLE", "CASE_CITES_ARTICLE"}:
            nodes[edge["target"]].update({
                "article_no": edge.get("article_no"),
                "law_name": edge.get("law_name"),
                "law_version": edge.get("law_version"),
                "law_family": edge.get("law_family"),
            })
        elif edge.get("type") == "CASE_HAS_AMOUNT":
            nodes[edge["target"]].update({
                "amount_id": edge.get("amount_id"),
                "value": edge.get("value"),
                "unit": edge.get("unit"),
                "role": edge.get("role"),
                "raw_text": edge.get("raw_text"),
                "field": edge.get("field"),
                "source_file": edge.get("source_file"),
                "text_hash": edge.get("text_hash"),
            })
    return sorted(nodes.values(), key=lambda row: row["node_id"])


def main() -> None:
    rows = build_edges()
    nodes = build_nodes(rows)
    edge_count = write_jsonl(OUTPUT, rows)
    write_jsonl(EDGES_OUTPUT, rows)
    node_count = write_jsonl(NODES_OUTPUT, nodes)
    print(f"wrote {edge_count} graph edges -> {OUTPUT}")
    print(f"wrote {edge_count} graph edges -> {EDGES_OUTPUT}")
    print(f"wrote {node_count} graph nodes -> {NODES_OUTPUT}")


if __name__ == "__main__":
    main()
