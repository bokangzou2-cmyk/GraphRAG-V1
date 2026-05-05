from __future__ import annotations

import json
import random
from collections import Counter

from common import OUT_DIR, iter_jsonl


LAW_PATH = OUT_DIR / "law_articles_sample.jsonl"
CASE_PATH = OUT_DIR / "cases_enriched_sample.jsonl"
GRAPH_PATH = OUT_DIR / "graph_sample.jsonl"
CHUNK_PATH = OUT_DIR / "case_chunks_sample.jsonl"
LAW_CHUNK_PATH = OUT_DIR / "law_chunks_sample.jsonl"
RETRIEVAL_CHUNK_PATH = OUT_DIR / "retrieval_chunks_sample.jsonl"
QUALITY_PATH = OUT_DIR / "data_quality_report_sample.json"
OUTPUT = OUT_DIR / "sample_report.txt"


def pct(part: int, total: int) -> str:
    return "0.0%" if total == 0 else f"{part / total * 100:.1f}%"


def main() -> None:
    laws = list(iter_jsonl(LAW_PATH))
    cases = list(iter_jsonl(CASE_PATH))
    graph_rows = list(iter_jsonl(GRAPH_PATH))
    chunks = list(iter_jsonl(CHUNK_PATH)) if CHUNK_PATH.exists() else []
    law_chunks = list(iter_jsonl(LAW_CHUNK_PATH)) if LAW_CHUNK_PATH.exists() else []
    retrieval_chunks = list(iter_jsonl(RETRIEVAL_CHUNK_PATH)) if RETRIEVAL_CHUNK_PATH.exists() else []
    with_crime = sum(1 for row in cases if row.get("crimes"))
    with_article = sum(1 for row in cases if row.get("cited_articles"))
    with_applied_article = sum(1 for row in cases if row.get("applied_articles"))
    sample = random.Random(7).sample(cases, min(3, len(cases)))
    chunk_sample = random.Random(11).sample(chunks, min(3, len(chunks)))
    first_case = cases[0] if cases else {}
    chunk_lengths = [row.get("text_length", 0) for row in chunks]
    chunk_field_counts = Counter(row.get("field", "") for row in chunks)
    chunk_quality_flags = Counter(flag for row in chunks for flag in row.get("quality_flags", []))
    overlong_chunks = [row for row in chunks if row.get("text_length", 0) > 1200]
    quality_report = json.loads(QUALITY_PATH.read_text(encoding="utf-8")) if QUALITY_PATH.exists() else {}

    lines = [
        "Sample pipeline report",
        f"法条数量: {len(laws)}",
        f"案例数量: {len(cases)}",
        f"图边数量: {len(graph_rows)}",
        f"chunk 总数: {len(chunks)}",
        f"law chunk 总数: {len(law_chunks)}",
        f"retrieval chunk 总数: {len(retrieval_chunks)}",
        f"有 crime 的比例: {with_crime}/{len(cases)} ({pct(with_crime, len(cases))})",
        f"有 article 的比例: {with_article}/{len(cases)} ({pct(with_article, len(cases))})",
        f"有 applied_article 的比例: {with_applied_article}/{len(cases)} ({pct(with_applied_article, len(cases))})",
        "",
        "质量校验:",
        f"status: {quality_report.get('status', 'MISSING')}",
        f"errors: {len(quality_report.get('errors', []))}",
        f"warnings: {len(quality_report.get('warnings', []))}",
        f"chunk duplicate_chunk_id: {quality_report.get('chunks', {}).get('duplicate_chunk_id', '')}",
        f"chunk over_max_chars: {quality_report.get('chunks', {}).get('over_max_chars', '')}",
        f"chunk quality_flags: {json.dumps(quality_report.get('chunks', {}).get('quality_flags', {}), ensure_ascii=False)}",
        f"law_chunks: {json.dumps(quality_report.get('law_chunks', {}), ensure_ascii=False)}",
        f"retrieval_chunks: {json.dumps(quality_report.get('retrieval_chunks', {}), ensure_ascii=False)}",
        f"graph law_version_counts: {json.dumps(quality_report.get('graph', {}).get('law_version_counts', {}), ensure_ascii=False)}",
        f"graph historical_law_edges: {quality_report.get('graph', {}).get('historical_law_edges', '')}",
        f"graph unresolved_applied_law_edges: {quality_report.get('graph', {}).get('unresolved_applied_law_edges', '')}",
        "",
        "首条案例结构化抽查:",
        f"title: {first_case.get('title', '')}",
        f"legal_basis_text: {first_case.get('legal_basis_text', '')}",
        f"cited_articles: {json.dumps(first_case.get('cited_articles', []), ensure_ascii=False)}",
        f"applied_articles: {json.dumps(first_case.get('applied_articles', []), ensure_ascii=False)}",
        f"applied_laws: {json.dumps(first_case.get('applied_laws', []), ensure_ascii=False)}",
        f"court_name: {first_case.get('court_name', '')}",
        f"prosecution_org: {first_case.get('prosecution_org', '')}",
        f"case_location_texts: {json.dumps(first_case.get('case_location_texts', []), ensure_ascii=False)}",
        f"punishment_location_texts: {json.dumps(first_case.get('punishment_location_texts', []), ensure_ascii=False)}",
        f"crime_time_texts: {json.dumps(first_case.get('crime_time_texts', []), ensure_ascii=False)}",
        f"judgment_date_text: {first_case.get('judgment_date_text', '')}",
        f"defendant_attitude_texts: {json.dumps(first_case.get('defendant_attitude_texts', []), ensure_ascii=False)}",
        f"amount_texts: {json.dumps(first_case.get('amount_texts', []), ensure_ascii=False)}",
        f"punishments: {json.dumps(first_case.get('punishments', []), ensure_ascii=False)}",
        f"accusation_text_len: {len(first_case.get('accusation_text', ''))}",
        f"court_found_facts_text_len: {len(first_case.get('court_found_facts_text', ''))}",
        f"defense_text_len: {len(first_case.get('defense_text', ''))}",
        f"evidence_text_len: {len(first_case.get('evidence_text', ''))}",
        f"witness_testimony_count: {len(first_case.get('witness_testimony_texts', []))}",
        f"defendant_confession_count: {len(first_case.get('defendant_confession_texts', []))}",
        f"expert_opinion_count: {len(first_case.get('expert_opinion_texts', []))}",
        "",
        "chunk 统计:",
        f"chunk 总数: {len(chunks)}",
        f"每个字段的 chunk 数量: {json.dumps(dict(sorted(chunk_field_counts.items())), ensure_ascii=False)}",
        f"平均 chunk 长度: {sum(chunk_lengths) / len(chunk_lengths):.1f}" if chunk_lengths else "平均 chunk 长度: 0.0",
        f"最大 chunk 长度: {max(chunk_lengths) if chunk_lengths else 0}",
        f"是否存在超过 1200 字的 chunk: {'是' if overlong_chunks else '否'}",
        f"chunk quality_flags 统计: {json.dumps(dict(sorted(chunk_quality_flags.items())), ensure_ascii=False)}",
        "",
        "随机案例 3 条:",
    ]
    for idx, row in enumerate(sample, 1):
        lines.extend([
            f"{idx}. {row.get('title')}",
            f"   case_id: {row.get('case_id')}",
            f"   crimes: {json.dumps(row.get('crimes', []), ensure_ascii=False)}",
            f"   cited_articles: {json.dumps(row.get('cited_articles', []), ensure_ascii=False)}",
            f"   applied_articles: {json.dumps(row.get('applied_articles', []), ensure_ascii=False)}",
            f"   sentencing: {json.dumps(row.get('sentencing', []), ensure_ascii=False)}",
            f"   punishments: {json.dumps(row.get('punishments', []), ensure_ascii=False)[:600]}",
            f"   amount_texts: {json.dumps(row.get('amount_texts', []), ensure_ascii=False)[:600]}",
        ])

    lines.extend(["", "随机 chunk 3 条:"])
    for idx, row in enumerate(chunk_sample, 1):
        lines.extend([
            f"{idx}. {row.get('chunk_id')}",
            f"   title: {row.get('title')}",
            f"   field: {row.get('field')}",
            f"   list_index: {row.get('list_index')}",
            f"   chunk_index: {row.get('chunk_index')}",
            f"   text_length: {row.get('text_length')}",
            f"   quality_flags: {json.dumps(row.get('quality_flags', []), ensure_ascii=False)}",
            f"   text_preview: {row.get('text', '')[:240]}",
        ])

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report -> {OUTPUT}")


if __name__ == "__main__":
    main()
