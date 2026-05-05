from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "processing_data"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


CN_NUM = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def chinese_to_int(text: str) -> int | None:
    text = str(text).strip()
    if text.isdigit():
        return int(text)
    total = 0
    current = 0
    for ch in text:
        if ch in CN_NUM:
            current = CN_NUM[ch]
        elif ch in CN_UNIT:
            unit = CN_UNIT[ch]
            total += (current or 1) * unit
            current = 0
        else:
            return None
    return total + current
