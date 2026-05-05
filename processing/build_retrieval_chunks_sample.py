from __future__ import annotations

from common import OUT_DIR, iter_jsonl, write_jsonl


CASE_INPUT = OUT_DIR / "case_chunks_sample.jsonl"
LAW_INPUT = OUT_DIR / "law_chunks_sample.jsonl"
OUTPUT = OUT_DIR / "retrieval_chunks_sample.jsonl"


def normalize_case_chunk(row: dict) -> dict:
    return {
        "retrieval_id": f"case:{row['chunk_id']}",
        "source_type": "case_chunk",
        "chunk_id": row["chunk_id"],
        "text": row.get("text", ""),
        "text_length": row.get("text_length", 0),
        "text_hash": row.get("text_hash", ""),
        "title": row.get("title", ""),
        "source_file": row.get("source_file", ""),
        "field": row.get("field", ""),
        "case_id": row.get("case_id", ""),
        "case_metadata": row.get("case_metadata", {}),
        "law_metadata": None,
        "extraction_method": row.get("extraction_method", ""),
        "quality_flags": row.get("quality_flags", []),
    }


def normalize_law_chunk(row: dict) -> dict:
    return {
        "retrieval_id": f"law:{row['chunk_id']}",
        "source_type": "law_article",
        "chunk_id": row["chunk_id"],
        "text": row.get("text", ""),
        "text_length": row.get("text_length", 0),
        "text_hash": row.get("text_hash", ""),
        "title": row.get("title", ""),
        "source_file": row.get("source_file", ""),
        "field": "law_article",
        "case_id": None,
        "case_metadata": None,
        "law_metadata": row.get("law_metadata", {}),
        "extraction_method": row.get("extraction_method", ""),
        "quality_flags": row.get("quality_flags", []),
    }


def build_rows() -> list[dict]:
    rows = [normalize_case_chunk(row) for row in iter_jsonl(CASE_INPUT)]
    rows.extend(normalize_law_chunk(row) for row in iter_jsonl(LAW_INPUT))
    return rows


def main() -> None:
    rows = build_rows()
    count = write_jsonl(OUTPUT, rows)
    print(f"wrote {count} retrieval chunks -> {OUTPUT}")


if __name__ == "__main__":
    main()
