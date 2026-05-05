from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

from common import OUT_DIR, iter_jsonl, normalize_text, write_jsonl


INPUT = OUT_DIR / "cases_enriched_sample.jsonl"
OUTPUT = OUT_DIR / "case_chunks_sample.jsonl"

CHUNK_FIELDS = [
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
]
LIST_FIELDS = {
    "witness_testimony_texts",
    "victim_statement_texts",
    "defendant_confession_texts",
    "expert_opinion_texts",
    "documentary_evidence_texts",
    "inspection_records_texts",
    "audio_video_evidence_texts",
}
TARGET_MIN = 700
TARGET_MAX = 900
TARGET_CHARS = 850
MAX_CHARS = 1200
OVERLAP_CHARS = 120

SENTENCE_RE = re.compile(r"[^。；！？!?\n]+[。；！？!?\n]?")
SOFT_SPLIT_RE = re.compile(r"[^，、,；;：:\n]+[，、,；;：:\n]?")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def id_part(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        sentence = normalize_text(match.group(0))
        if sentence:
            spans.append((match.start(), match.end(), sentence))
    return spans


def split_oversized_sentence(start: int, sentence: str) -> list[tuple[int, int, str]]:
    parts = []
    cursor = 0
    current = []
    current_start = 0
    current_len = 0
    soft_parts = [(m.start(), m.end(), normalize_text(m.group(0))) for m in SOFT_SPLIT_RE.finditer(sentence)]
    if not soft_parts:
        soft_parts = [(0, len(sentence), sentence)]

    for part_start, part_end, part in soft_parts:
        if not part:
            continue
        if current and current_len + len(part) > MAX_CHARS:
            text = normalize_text("".join(current))
            parts.append((start + current_start, start + cursor, text))
            current = []
            current_start = part_start
            current_len = 0
        if not current:
            current_start = part_start
        current.append(part)
        current_len += len(part)
        cursor = part_end

    if current:
        text = normalize_text("".join(current))
        parts.append((start + current_start, start + cursor, text))
    return parts


def normalized_items(value: object, field: str) -> Iterator[tuple[int | None, str]]:
    if field in LIST_FIELDS:
        if not isinstance(value, list):
            return
        for idx, item in enumerate(value):
            text = normalize_text(item)
            if text:
                yield idx, text
    else:
        text = normalize_text(value)
        if text:
            yield None, text


def chunk_text(text: str) -> list[dict]:
    spans = []
    for start, end, sentence in sentence_spans(text):
        if len(sentence) > MAX_CHARS:
            spans.extend(split_oversized_sentence(start, sentence))
        else:
            spans.append((start, end, sentence))

    chunks = []
    index = 0
    while index < len(spans):
        chunk_spans = []
        chunk_len = 0
        quality_flags = []
        while index < len(spans):
            start, end, sentence = spans[index]
            next_len = chunk_len + len(sentence)
            if chunk_spans and next_len > TARGET_CHARS and chunk_len >= TARGET_MIN:
                break
            if chunk_spans and next_len > MAX_CHARS:
                break
            chunk_spans.append((start, end, sentence))
            chunk_len = next_len
            if len(sentence) > TARGET_MAX:
                quality_flags.append("long_sentence")
            index += 1

        if not chunk_spans:
            start, end, sentence = spans[index]
            chunk_spans.append((start, end, sentence))
            if len(sentence) > MAX_CHARS:
                quality_flags.append("oversize_chunk")
            index += 1

        chunk_text_value = normalize_text("".join(part[2] for part in chunk_spans))
        if len(chunk_text_value) <= 50:
            quality_flags.append("short_chunk")
        if re.search(r"(?:\d+[、．.]|[（(][一二三四五六七八九十]+[）)])\s*$", chunk_text_value):
            quality_flags.append("dangling_enumeration")
        chunks.append({
            "text": chunk_text_value,
            "char_start": chunk_spans[0][0],
            "char_end": chunk_spans[-1][1],
            "quality_flags": sorted(set(quality_flags)),
        })

        if index >= len(spans):
            break

        overlap_len = 0
        overlap_count = 0
        for _, _, sentence in reversed(chunk_spans):
            overlap_len += len(sentence)
            overlap_count += 1
            if overlap_len >= OVERLAP_CHARS:
                break
        index = max(index - overlap_count, 0)
        if overlap_count >= len(chunk_spans):
            index += 1

    return chunks


def case_metadata(row: dict) -> dict:
    return {
        "crimes": row.get("crimes", []),
        "cited_articles": row.get("cited_articles", []),
        "applied_articles": row.get("applied_articles", []),
        "applied_laws": row.get("applied_laws", []),
        "sentencing": row.get("sentencing", []),
        "amounts": row.get("amount_texts", []),
        "court_name": row.get("court_name", ""),
        "judgment_date_text": row.get("judgment_date_text", ""),
    }


def build_chunks() -> list[dict]:
    rows = []
    for case in iter_jsonl(INPUT):
        metadata = case_metadata(case)
        case_id = case.get("case_id", "")
        source_file = case.get("source_file", "")
        source_id = id_part(source_file)
        for field in CHUNK_FIELDS:
            for list_index, item_text in normalized_items(case.get(field), field):
                for chunk_index, chunk in enumerate(chunk_text(item_text)):
                    text = chunk["text"]
                    if not text:
                        continue
                    rows.append({
                        "chunk_id": f"{case_id}:{source_id}:{field}:{list_index if list_index is not None else 'scalar'}:{chunk_index}",
                        "case_id": case_id,
                        "title": case.get("title", ""),
                        "field": field,
                        "list_index": list_index,
                        "chunk_index": chunk_index,
                        "text": text,
                        "text_length": len(text),
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                        "text_hash": text_hash(text),
                        "source_file": source_file,
                        "extraction_method": "field_sentence_chunk",
                        "quality_flags": chunk["quality_flags"],
                        "case_metadata": metadata,
                    })
    return rows


def main() -> None:
    rows = build_chunks()
    count = write_jsonl(OUTPUT, rows)
    print(f"wrote {count} case chunks -> {OUTPUT}")


if __name__ == "__main__":
    main()
