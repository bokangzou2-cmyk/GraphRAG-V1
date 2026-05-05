from __future__ import annotations

from pathlib import Path

from common import RAW_DIR, OUT_DIR, normalize_text, read_json, write_jsonl


CASE_ROOT = RAW_DIR / "data" / "data" / "类案检索" / "train" / "train_set" / "candidates"
OUTPUT = OUT_DIR / "cases_sample.jsonl"
LIMIT = 150

# 这些规则只做轻量分段：保留 fact_text 原文，同时增加可检索的粗粒度事实/证据字段。
COURT_FOUND_RE = r"(?:经审理查明|另查明|本院查明|本院认定)"
EVIDENCE_RE = (
    r"(?:上述事实[^。；]{0,60}(?:证据|证明|证实)"
    r"|为证实上述指控[^。；]{0,40}"
    r"|公诉机关(?:当庭)?(?:宣读|提交|出示|提供)[^。；]{0,60}(?:证据|证明)"
    r"|下列证据[：:]"
    r"|认定本案事实的综合证据如下[：:])"
)
DEFENSE_RE = r"(?:被告人[^。；]{0,30}(?:辩称|辩解|提出)|辩护人[^。；]{0,30}(?:提出|认为|辩称|发表))"
EVIDENCE_ITEM_RE = r"(?:证人|被害人|被告人|鉴定|检验|辨认|指认|勘验|检查|搜查|书证|户籍|到案|抓获|视听资料|电子数据|监控|光盘)"

EVIDENCE_TYPE_PATTERNS = {
    "witness_testimony_texts": r"证人[^。；]{0,30}(?:证言|证实|证明)",
    "victim_statement_texts": r"被害人[^。；]{0,30}(?:陈述|证实|证明)",
    "defendant_confession_texts": r"被告人[^。；]{0,30}(?:供述|供述与辩解|辩解)",
    "expert_opinion_texts": r"(?:鉴定意见|鉴定书|检验报告|价格认定|法医学尸体检验)",
    "documentary_evidence_texts": r"(?:书证|户籍|到案经过|抓获经过|立案|受案|证明)",
    "inspection_records_texts": r"(?:辨认|指认|勘验|检查|搜查|扣押|提取)[^。；]{0,20}(?:笔录|清单|照片)?",
    "audio_video_evidence_texts": r"(?:视听资料|电子数据|监控录像|录像|光盘)",
}
TESTIMONIAL_EVIDENCE_KEYS = {
    "witness_testimony_texts",
    "victim_statement_texts",
    "defendant_confession_texts",
}


def iter_case_files(limit: int = LIMIT) -> list[Path]:
    return sorted(CASE_ROOT.rglob("*.json"), key=lambda p: str(p))[:limit]


def first_match_start(text: str, pattern: str) -> int | None:
    import re

    match = re.search(pattern, text)
    return None if match is None else match.start()


def slice_text(text: str, start: int | None, end: int | None = None) -> str:
    if start is None:
        return ""
    return normalize_text(text[start:end])


def split_fact_sections(fact_text: str) -> dict:
    court_start = first_match_start(fact_text, COURT_FOUND_RE)
    evidence_start = first_match_start(fact_text, EVIDENCE_RE)
    defense_start = first_match_start(fact_text, DEFENSE_RE)

    accusation_end_candidates = [pos for pos in (defense_start, court_start, evidence_start) if pos is not None]
    accusation_end = min(accusation_end_candidates) if accusation_end_candidates else None
    defense_end_candidates = [pos for pos in (court_start, evidence_start) if pos is not None and defense_start is not None and pos > defense_start]
    court_end = evidence_start if evidence_start is not None else None

    evidence_section_text = slice_text(fact_text, evidence_start)
    evidence_sections = split_evidence_types(evidence_section_text)

    sections = {
        "accusation_text": normalize_text(fact_text[:accusation_end]) if accusation_end else fact_text,
        "defense_text": slice_text(fact_text, defense_start, min(defense_end_candidates) if defense_end_candidates else court_start),
        "court_found_facts_text": slice_text(fact_text, court_start, court_end),
        "evidence_text": evidence_sections.pop("objective_evidence_text"),
    }
    sections.update(evidence_sections)
    return sections


def split_evidence_types(evidence_text: str) -> dict[str, list[str]]:
    import re

    result = {key: [] for key in EVIDENCE_TYPE_PATTERNS}
    objective_chunks: list[str] = []
    if not evidence_text:
        result["objective_evidence_text"] = ""
        return result

    evidence_text = re.split(r"(?:上述|以上)证据[，,]", evidence_text, maxsplit=1)[0]
    anchors = list(re.finditer(EVIDENCE_ITEM_RE, evidence_text))
    chunks: list[str] = []
    for idx, anchor in enumerate(anchors):
        start = anchor.start()
        end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(evidence_text)
        chunk = clean_evidence_chunk(evidence_text[start:end])
        if len(chunk) >= 15:
            chunks.append(chunk)

    for chunk in chunks:
        matched_key = ""
        for key, pattern in EVIDENCE_TYPE_PATTERNS.items():
            if re.search(pattern, chunk):
                result[key].append(chunk)
                matched_key = key
                break
        if matched_key not in TESTIMONIAL_EVIDENCE_KEYS:
            objective_chunks.append(chunk)
    result["objective_evidence_text"] = normalize_text(" ".join(objective_chunks))
    return result


def clean_evidence_chunk(text: str) -> str:
    # 证据段末尾常跟“上述证据，庭审中经举证、质证...”，这不是具体证据材料。
    text = normalize_text(text).split("上述证据", 1)[0]
    return text.strip(" ，。；")


def parse_cases(limit: int = LIMIT) -> list[dict]:
    rows: list[dict] = []
    for path in iter_case_files(limit):
        data = read_json(path)
        case_id = normalize_text(data.get("ajId")) or normalize_text(data.get("writId")) or path.stem
        fact_text = normalize_text(data.get("ajjbqk"))
        sections = split_fact_sections(fact_text)
        rows.append({
            "case_id": case_id,
            "title": normalize_text(data.get("writName") or data.get("ajName")),
            "fact_text": fact_text,
            **sections,
            "reasoning_text": normalize_text(data.get("cpfxgc")),
            "judgment_text": normalize_text(data.get("pjjg")),
            "source_file": str(path.relative_to(RAW_DIR)).replace("\\", "/"),
        })
    return rows


def main() -> None:
    rows = parse_cases()
    count = write_jsonl(OUTPUT, rows)
    print(f"wrote {count} cases -> {OUTPUT}")


if __name__ == "__main__":
    main()
