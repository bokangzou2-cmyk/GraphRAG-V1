from __future__ import annotations

import hashlib

from common import OUT_DIR, iter_jsonl, normalize_text, write_jsonl


INPUT = OUT_DIR / "law_articles_sample.jsonl"
OUTPUT = OUT_DIR / "law_chunks_sample.jsonl"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_chunks() -> list[dict]:
    rows = []
    for article in iter_jsonl(INPUT):
        article_no = normalize_text(article.get("article_no"))
        title = normalize_text(article.get("title"))
        content = normalize_text(article.get("content"))
        text = normalize_text(f"第{article_no}条 {title} {content}" if title else f"第{article_no}条 {content}")
        if not article_no or not text:
            continue
        rows.append({
            "chunk_id": f"law:criminal_law:{article_no}",
            "source_type": "law_article",
            "article_no": article_no,
            "title": title,
            "text": text,
            "text_length": len(text),
            "text_hash": text_hash(text),
            "source_file": "刑法.txt",
            "extraction_method": "law_article_chunk",
            "quality_flags": [],
            "law_metadata": {
                "law_name": "中华人民共和国刑法",
                "article_no": article_no,
                "title": title,
                "law_version": "current",
            },
        })
    return rows


def main() -> None:
    rows = build_chunks()
    count = write_jsonl(OUTPUT, rows)
    print(f"wrote {count} law chunks -> {OUTPUT}")


if __name__ == "__main__":
    main()
