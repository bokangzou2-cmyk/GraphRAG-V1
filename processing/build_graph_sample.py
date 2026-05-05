from __future__ import annotations

from common import OUT_DIR, iter_jsonl, write_jsonl


INPUT = OUT_DIR / "cases_enriched_sample.jsonl"
OUTPUT = OUT_DIR / "graph_sample.jsonl"


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


def law_version(law_name: str) -> str:
    if "1979年" in law_name:
        return "1979"
    if law_name == "中华人民共和国刑法":
        return "current"
    if law_name:
        return "other"
    return "unknown"


def applied_law_lookup(case: dict) -> dict[str, dict]:
    rows = {}
    for item in case.get("applied_laws", []):
        article = item.get("article", "")
        law_name = item.get("law_name", "")
        if article:
            rows[article] = {
                "law_name": law_name,
                "law_version": law_version(law_name),
                "article_no": article_no_from_article(article),
            }
    return rows


def build_edges() -> list[dict]:
    rows: list[dict] = []
    for case in iter_jsonl(INPUT):
        case_id = case["case_id"]
        applied_lookup = applied_law_lookup(case)
        for article in case.get("cited_articles", []):
            rows.append({
                "type": "CASE_CITES_ARTICLE",
                "source": f"Case:{case_id}",
                "target": f"Article:{article}",
                "article": article,
                "article_no": article_no_from_article(article),
                "law_name": "",
                "law_version": "unknown",
                "evidence_text": article,
                "confidence": 0.7,
                "extraction_method": "regex_citation",
            })
        for article in case.get("applied_articles", []):
            law_info = applied_lookup.get(article, {})
            rows.append({
                "type": "CASE_APPLIES_ARTICLE",
                "source": f"Case:{case_id}",
                "target": f"Article:{article}",
                "article": article,
                "article_no": law_info.get("article_no") or article_no_from_article(article),
                "law_name": law_info.get("law_name", ""),
                "law_version": law_info.get("law_version", "unknown"),
                "evidence_text": case.get("legal_basis_text", article),
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
    return rows


def main() -> None:
    rows = build_edges()
    count = write_jsonl(OUTPUT, rows)
    print(f"wrote {count} graph edges -> {OUTPUT}")


if __name__ == "__main__":
    main()
