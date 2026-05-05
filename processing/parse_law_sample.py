from __future__ import annotations

import re
from pathlib import Path

from common import RAW_DIR, OUT_DIR, chinese_to_int, normalize_text, write_jsonl


INPUT = RAW_DIR / "刑法.txt"
OUTPUT = OUT_DIR / "law_articles_sample.jsonl"


ARTICLE_RE = re.compile(
    r"(?m)^\s*第(?P<num>[零〇一二两三四五六七八九十百千\d]+)条"
    r"(?P<suffix>之[零〇一二两三四五六七八九十\d]+)?\s+【"
)
TITLE_RE = re.compile(r"^【(?P<title>[^】]+)】")


def parse_articles(path: Path = INPUT) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if "条文" in text:
        text = text.split("条文", 1)[1]

    matches = list(ARTICLE_RE.finditer(text))
    rows: list[dict] = []
    for idx, match in enumerate(matches):
        start = match.end() - 1
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = normalize_text(text[start:end])
        if len(body) < 10:
            continue

        title = ""
        title_match = TITLE_RE.match(body)
        if title_match:
            title = title_match.group("title").strip()
            body = normalize_text(body[title_match.end():])
        body = strip_trailing_headings(body)

        article_no = chinese_to_int(match.group("num"))
        if article_no is None:
            continue
        suffix = normalize_suffix(match.group("suffix"))
        rows.append({
            "article_no": f"{article_no}{suffix}",
            "title": title,
            "content": body,
        })
    return rows


def normalize_suffix(raw_suffix: str | None) -> str:
    if not raw_suffix:
        return ""
    suffix_no = chinese_to_int(raw_suffix.removeprefix("之"))
    if suffix_no is None:
        return raw_suffix
    return f".{suffix_no}"


def strip_trailing_headings(text: str) -> str:
    heading_re = re.compile(
        r"\s*第[零〇一二两三四五六七八九十百千]+[编章节]\s+"
        r"[\u4e00-\u9fff、（）()]+(?:\s+[\u4e00-\u9fff、（）()]+)*$"
    )
    previous = None
    while previous != text:
        previous = text
        text = heading_re.sub("", text).strip()
    return text


def main() -> None:
    rows = parse_articles()
    count = write_jsonl(OUTPUT, rows)
    print(f"wrote {count} law articles -> {OUTPUT}")


if __name__ == "__main__":
    main()
