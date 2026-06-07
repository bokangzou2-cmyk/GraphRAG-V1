from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import OUT_DIR, iter_jsonl


OUTPUT = OUT_DIR / "data_quality_report_sample.json"

LAW_PATH = OUT_DIR / "law_articles_sample.jsonl"
CASES_PATH = OUT_DIR / "cases_sample.jsonl"
ENRICHED_PATH = OUT_DIR / "cases_enriched_sample.jsonl"
CHUNKS_PATH = OUT_DIR / "case_chunks_sample.jsonl"
LAW_CHUNKS_PATH = OUT_DIR / "law_chunks_sample.jsonl"
RETRIEVAL_CHUNKS_PATH = OUT_DIR / "retrieval_chunks_sample.jsonl"
GRAPH_PATH = OUT_DIR / "graph_sample.jsonl"
GRAPH_NODES_PATH = OUT_DIR / "graph_nodes_sample.jsonl"
GRAPH_EDGES_PATH = OUT_DIR / "graph_edges_sample.jsonl"

REQUIRED_FILES = [LAW_PATH, CASES_PATH, ENRICHED_PATH, CHUNKS_PATH, LAW_CHUNKS_PATH, RETRIEVAL_CHUNKS_PATH, GRAPH_PATH, GRAPH_NODES_PATH, GRAPH_EDGES_PATH]
REQUIRED_FIELDS = {
    "law_articles_sample.jsonl": ["article_no", "title", "content"],
    "cases_sample.jsonl": ["case_id", "title", "fact_text", "reasoning_text", "judgment_text", "source_file"],
    "cases_enriched_sample.jsonl": [
        "case_id",
        "title",
        "source_file",
        "crimes",
        "cited_articles",
        "legal_basis_text",
        "applied_articles",
        "applied_laws",
        "sentencing",
    ],
    "case_chunks_sample.jsonl": [
        "chunk_id",
        "case_id",
        "title",
        "field",
        "list_index",
        "chunk_index",
        "text",
        "text_length",
        "char_start",
        "char_end",
        "text_hash",
        "source_file",
        "extraction_method",
        "quality_flags",
        "case_metadata",
    ],
    "law_chunks_sample.jsonl": [
        "chunk_id",
        "source_type",
        "article_no",
        "title",
        "text",
        "text_length",
        "text_hash",
        "source_file",
        "extraction_method",
        "quality_flags",
        "law_metadata",
    ],
    "retrieval_chunks_sample.jsonl": [
        "retrieval_id",
        "source_type",
        "chunk_id",
        "text",
        "text_length",
        "text_hash",
        "title",
        "source_file",
        "field",
        "case_id",
        "case_metadata",
        "law_metadata",
        "extraction_method",
        "quality_flags",
    ],
    "graph_sample.jsonl": ["type", "source", "target", "evidence_text", "confidence", "extraction_method"],
    "graph_nodes_sample.jsonl": ["node_id", "node_type"],
    "graph_edges_sample.jsonl": ["type", "source", "target", "evidence_text", "confidence", "extraction_method"],
}
CHUNK_FIELDS = {
    "accusation_text",
    "defense_text",
    "court_found_facts_text",
    "evidence_text",
    "witness_testimony_texts",
    "victim_statement_texts",
    "defendant_confession_texts",
    "expert_opinion_texts",
    "documentary_evidence_texts",
    "inspection_records_texts",
    "audio_video_evidence_texts",
    "reasoning_text",
    "judgment_text",
}
LIST_CHUNK_FIELDS = {
    "witness_testimony_texts",
    "victim_statement_texts",
    "defendant_confession_texts",
    "expert_opinion_texts",
    "documentary_evidence_texts",
    "inspection_records_texts",
    "audio_video_evidence_texts",
}
GRAPH_TYPES = {"CASE_CITES_ARTICLE", "CASE_APPLIES_ARTICLE", "CASE_OF_CRIME", "CASE_HAS_CHUNK", "ARTICLE_HAS_CHUNK", "CASE_HAS_AMOUNT"}
GRAPH_NODE_TYPES = {"case", "article", "crime", "chunk", "amount", "unknown"}
RETRIEVAL_TYPES = {"case_chunk", "law_article"}


def read_jsonl_checked(path: Path, errors: list[dict]) -> list[dict]:
    if not path.exists():
        errors.append({"code": "missing_file", "file": path.name, "message": f"missing {path}"})
        return []
    rows = []
    try:
        for row in iter_jsonl(path):
            rows.append(row)
    except Exception as exc:
        errors.append({"code": "invalid_jsonl", "file": path.name, "message": str(exc)})
    return rows


def missing_required(rows: list[dict], filename: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        for field in REQUIRED_FIELDS[filename]:
            if field not in row:
                counts[field] += 1
    return dict(sorted(counts.items()))


def blank_required(rows: list[dict], fields: list[str]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        for field in fields:
            if not row.get(field):
                counts[field] += 1
    return dict(sorted(counts.items()))


def duplicate_count(values: list[Any]) -> int:
    return sum(1 for count in Counter(values).values() if count > 1)


def validate_chunks(chunks: list[dict]) -> dict:
    field_counts = Counter(row.get("field") for row in chunks)
    quality_flags = Counter(flag for row in chunks for flag in row.get("quality_flags", []))
    duplicate_chunk_ids = duplicate_count([row.get("chunk_id") for row in chunks])
    invalid_fields = sorted({row.get("field") for row in chunks if row.get("field") not in CHUNK_FIELDS})
    list_index_errors = 0
    scalar_index_errors = 0
    for row in chunks:
        field = row.get("field")
        if field in LIST_CHUNK_FIELDS and row.get("list_index") is None:
            list_index_errors += 1
        if field not in LIST_CHUNK_FIELDS and row.get("list_index") is not None:
            scalar_index_errors += 1
    lengths = [row.get("text_length", 0) for row in chunks]
    over_max = [row for row in chunks if row.get("text_length", 0) > 1200]
    empty_text = sum(1 for row in chunks if not row.get("text"))
    bad_text_length = sum(1 for row in chunks if row.get("text_length") != len(row.get("text", "")))
    return {
        "count": len(chunks),
        "duplicate_chunk_id": duplicate_chunk_ids,
        "empty_text": empty_text,
        "bad_text_length": bad_text_length,
        "invalid_fields": invalid_fields,
        "list_index_errors": list_index_errors,
        "scalar_index_errors": scalar_index_errors,
        "over_max_chars": len(over_max),
        "max_text_length": max(lengths) if lengths else 0,
        "avg_text_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "field_counts": dict(sorted(field_counts.items())),
        "quality_flags": dict(sorted(quality_flags.items())),
    }


def validate_law_chunks(chunks: list[dict]) -> dict:
    lengths = [row.get("text_length", 0) for row in chunks]
    return {
        "count": len(chunks),
        "duplicate_chunk_id": duplicate_count([row.get("chunk_id") for row in chunks]),
        "empty_text": sum(1 for row in chunks if not row.get("text")),
        "bad_text_length": sum(1 for row in chunks if row.get("text_length") != len(row.get("text", ""))),
        "duplicate_article_no": duplicate_count([row.get("article_no") for row in chunks]),
        "max_text_length": max(lengths) if lengths else 0,
        "avg_text_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
    }


def validate_retrieval_chunks(rows: list[dict]) -> dict:
    lengths = [row.get("text_length", 0) for row in rows]
    source_type_counts = Counter(row.get("source_type") for row in rows)
    invalid_source_types = sorted({row.get("source_type") for row in rows if row.get("source_type") not in RETRIEVAL_TYPES})
    case_rows = [row for row in rows if row.get("source_type") == "case_chunk"]
    law_rows = [row for row in rows if row.get("source_type") == "law_article"]
    return {
        "count": len(rows),
        "duplicate_retrieval_id": duplicate_count([row.get("retrieval_id") for row in rows]),
        "empty_text": sum(1 for row in rows if not row.get("text")),
        "bad_text_length": sum(1 for row in rows if row.get("text_length") != len(row.get("text", ""))),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "invalid_source_types": invalid_source_types,
        "case_rows_without_case_metadata": sum(1 for row in case_rows if not row.get("case_metadata")),
        "law_rows_without_law_metadata": sum(1 for row in law_rows if not row.get("law_metadata")),
        "max_text_length": max(lengths) if lengths else 0,
        "avg_text_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
    }


def validate_enriched(cases: list[dict]) -> dict:
    applied_not_in_basis = []
    suspicious_crimes = []
    for row in cases:
        legal_basis = row.get("legal_basis_text", "")
        for article in row.get("applied_articles", []):
            if article not in legal_basis:
                applied_not_in_basis.append({
                    "case_id": row.get("case_id"),
                    "title": row.get("title"),
                    "article": article,
                })
        for crime in row.get("crimes", []):
            if crime.startswith("罪") or any(noise in crime for noise in ("罪事实", "罪情节", "数罪", "悔罪", "同种罪")):
                suspicious_crimes.append({
                    "case_id": row.get("case_id"),
                    "title": row.get("title"),
                    "crime": crime,
                })
    return {
        "count": len(cases),
        "duplicate_case_source": duplicate_count([(row.get("case_id"), row.get("source_file")) for row in cases]),
        "blank_key_fields": blank_required(cases, ["case_id", "title", "source_file", "crimes"]),
        "metadata_coverage": {
            "with_crimes": sum(1 for row in cases if row.get("crimes")),
            "with_cited_articles": sum(1 for row in cases if row.get("cited_articles")),
            "with_applied_articles": sum(1 for row in cases if row.get("applied_articles")),
            "with_sentencing": sum(1 for row in cases if row.get("sentencing")),
            "with_amounts": sum(1 for row in cases if row.get("amount_texts")),
            "with_court_name": sum(1 for row in cases if row.get("court_name")),
            "with_judgment_date_text": sum(1 for row in cases if row.get("judgment_date_text")),
        },
        "applied_articles_not_in_legal_basis": applied_not_in_basis[:20],
        "suspicious_crimes": suspicious_crimes[:20],
    }


def validate_graph(graph_rows: list[dict]) -> dict:
    type_counts = Counter(row.get("type") for row in graph_rows)
    law_version_counts = Counter(row.get("law_version", "") for row in graph_rows if row.get("type") in {"CASE_CITES_ARTICLE", "CASE_APPLIES_ARTICLE"})
    invalid_types = sorted({row.get("type") for row in graph_rows if row.get("type") not in GRAPH_TYPES})
    blank_edges = sum(1 for row in graph_rows if not row.get("source") or not row.get("target"))
    historical_law_edges = sum(1 for row in graph_rows if row.get("law_version") == "1979")
    current_law_edges = sum(1 for row in graph_rows if row.get("law_version") == "current")
    unknown_law_edges = sum(1 for row in graph_rows if row.get("type") in {"CASE_CITES_ARTICLE", "CASE_APPLIES_ARTICLE"} and row.get("law_version") == "unknown")
    unresolved_applied_law_edges = sum(
        1 for row in graph_rows
        if row.get("type") == "CASE_APPLIES_ARTICLE" and row.get("law_version") in {"unknown", ""}
    )
    non_current_applied_law_edges = sum(
        1 for row in graph_rows
        if row.get("type") == "CASE_APPLIES_ARTICLE" and row.get("law_version") not in {"current", "unknown", ""}
    )
    chunk_grounded_applies = sum(1 for row in graph_rows if row.get("type") == "CASE_APPLIES_ARTICLE" and row.get("provenance_status") == "chunk_grounded")
    case_level_only_applies = sum(1 for row in graph_rows if row.get("type") == "CASE_APPLIES_ARTICLE" and row.get("provenance_status") == "case_level_only")
    return {
        "count": len(graph_rows),
        "type_counts": dict(sorted(type_counts.items())),
        "law_version_counts": dict(sorted(law_version_counts.items())),
        "historical_law_edges": historical_law_edges,
        "current_law_edges": current_law_edges,
        "unknown_law_edges": unknown_law_edges,
        "unresolved_applied_law_edges": unresolved_applied_law_edges,
        "non_current_applied_law_edges": non_current_applied_law_edges,
        "chunk_grounded_applies": chunk_grounded_applies,
        "case_level_only_applies": case_level_only_applies,
        "invalid_types": invalid_types,
        "blank_edges": blank_edges,
    }


def validate_graph_nodes(nodes: list[dict]) -> dict:
    return {
        "count": len(nodes),
        "duplicate_node_id": duplicate_count([row.get("node_id") for row in nodes]),
        "invalid_node_types": sorted({row.get("node_type") for row in nodes if row.get("node_type") not in GRAPH_NODE_TYPES}),
        "blank_nodes": sum(1 for row in nodes if not row.get("node_id") or not row.get("node_type")),
    }


def build_report() -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []
    rows_by_file = {path.name: read_jsonl_checked(path, errors) for path in REQUIRED_FILES}

    for filename, rows in rows_by_file.items():
        missing = missing_required(rows, filename)
        if missing:
            errors.append({"code": "missing_required_fields", "file": filename, "fields": missing})

    laws = rows_by_file[LAW_PATH.name]
    cases = rows_by_file[CASES_PATH.name]
    enriched = rows_by_file[ENRICHED_PATH.name]
    chunks = rows_by_file[CHUNKS_PATH.name]
    law_chunks = rows_by_file[LAW_CHUNKS_PATH.name]
    retrieval_chunks = rows_by_file[RETRIEVAL_CHUNKS_PATH.name]
    graph_rows = rows_by_file[GRAPH_PATH.name]
    graph_nodes = rows_by_file[GRAPH_NODES_PATH.name]

    chunk_report = validate_chunks(chunks)
    law_chunk_report = validate_law_chunks(law_chunks)
    retrieval_report = validate_retrieval_chunks(retrieval_chunks)
    enriched_report = validate_enriched(enriched)
    graph_report = validate_graph(graph_rows)
    graph_node_report = validate_graph_nodes(graph_nodes)

    if chunk_report["duplicate_chunk_id"]:
        errors.append({"code": "duplicate_chunk_id", "count": chunk_report["duplicate_chunk_id"]})
    if chunk_report["over_max_chars"]:
        errors.append({"code": "chunk_over_max_chars", "count": chunk_report["over_max_chars"]})
    if chunk_report["empty_text"]:
        errors.append({"code": "empty_chunk_text", "count": chunk_report["empty_text"]})
    if chunk_report["quality_flags"]:
        warnings.append({"code": "chunk_quality_flags_present", "flags": chunk_report["quality_flags"]})
    if enriched_report["applied_articles_not_in_legal_basis"]:
        errors.append({
            "code": "applied_articles_not_in_legal_basis",
            "examples": enriched_report["applied_articles_not_in_legal_basis"],
        })
    if enriched_report["suspicious_crimes"]:
        warnings.append({"code": "suspicious_crimes", "examples": enriched_report["suspicious_crimes"]})
    if law_chunk_report["duplicate_chunk_id"] or law_chunk_report["empty_text"]:
        errors.append({"code": "invalid_law_chunks", "summary": law_chunk_report})
    if retrieval_report["duplicate_retrieval_id"] or retrieval_report["empty_text"] or retrieval_report["invalid_source_types"]:
        errors.append({"code": "invalid_retrieval_chunks", "summary": retrieval_report})
    if graph_report["invalid_types"]:
        errors.append({"code": "invalid_graph_types", "types": graph_report["invalid_types"]})
    if graph_node_report["duplicate_node_id"] or graph_node_report["invalid_node_types"] or graph_node_report["blank_nodes"]:
        errors.append({"code": "invalid_graph_nodes", "summary": graph_node_report})

    return {
        "status": "PASS" if not errors else "FAIL",
        "files": {name: {"count": len(rows)} for name, rows in rows_by_file.items()},
        "laws": {
            "count": len(laws),
            "duplicate_article_no": duplicate_count([row.get("article_no") for row in laws]),
            "blank_key_fields": blank_required(laws, ["article_no", "title", "content"]),
        },
        "cases": {
            "count": len(cases),
            "duplicate_case_source": duplicate_count([(row.get("case_id"), row.get("source_file")) for row in cases]),
            "blank_key_fields": blank_required(cases, ["case_id", "title", "fact_text", "source_file"]),
        },
        "enriched_cases": enriched_report,
        "chunks": chunk_report,
        "law_chunks": law_chunk_report,
        "retrieval_chunks": retrieval_report,
        "graph": graph_report,
        "graph_nodes": graph_node_report,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    report = build_report()
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote data quality report -> {OUTPUT}")
    print(f"status: {report['status']}")


if __name__ == "__main__":
    main()
