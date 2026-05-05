from __future__ import annotations

import re

from common import RAW_DIR, OUT_DIR, iter_jsonl, normalize_text, read_json, write_jsonl


INPUT = OUT_DIR / "cases_sample.jsonl"
OUTPUT = OUT_DIR / "cases_enriched_sample.jsonl"

CRIME_PATTERNS = (
    re.compile(r"犯(?P<crime>[\u4e00-\u9fff、]{2,24}?罪)"),
    re.compile(r"构成(?P<crime>[\u4e00-\u9fff、]{2,24}?罪)"),
    re.compile(r"以(?P<crime>[\u4e00-\u9fff、]{2,24}?罪)追究刑事责任"),
)
CITATION_DETAIL_RE = re.compile(
    r"(?P<citation>第[零〇一二两三四五六七八九十百千\d]+条"
    r"(?:之[零〇一二两三四五六七八九十\d]+)?"
    r"(?:第[零〇一二两三四五六七八九十百千\d]+款)?"
    r"(?:第[零〇一二两三四五六七八九十百千\d]+项)?)"
)
LAW_AND_CITATION_RE = re.compile(
    r"(?P<law>(?:\d{4}年)?《[^》]{2,60}》)?"
    r"(?P<citation>第[零〇一二两三四五六七八九十百千\d]+条"
    r"(?:之[零〇一二两三四五六七八九十\d]+)?"
    r"(?:第[零〇一二两三四五六七八九十百千\d]+款)?"
    r"(?:第[零〇一二两三四五六七八九十百千\d]+项)?)"
)
SENTENCE_RE = re.compile(r"(有期徒刑[^，。；]*|拘役[^，。；]*|管制[^，。；]*|无期徒刑|死刑[^，。；]*|罚金(?:人民币)?[^，。；]*)")
PUNISHMENT_RE = re.compile(
    r"(?:判处|决定执行|宣告缓刑|并处|单处|免予刑事处罚|从轻处罚|减轻处罚|追缴|退赔|赔偿|发还|没收)"
    r"|有期徒刑|拘役|管制|无期徒刑|死刑|缓刑|罚金|剥夺政治权利|没收(?:个人)?财产"
)
CORE_PUNISHMENT_RE = re.compile(r"判处|决定执行|并处|单处|免予刑事处罚|追缴|退赔|赔偿|发还|没收|犯")
AMOUNT_RE = re.compile(
    r"(?:人民币)?(?:\d+(?:,\d{3})*(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+)"
    r"(?:余)?(?:元|万元|亿元)"
)
ATTITUDE_RE = re.compile(
    r"自首|坦白|如实供述|认罪|悔罪|积极赔偿|赔偿[^。；]{0,40}(?:损失|被害人|谅解)"
    r"|取得[^。；]{0,30}谅解|退赃|退赔|主动投案|到案后[^。；]{0,80}供述"
)
TIME_RE = re.compile(
    r"(?:\d{4}年|[一二三四五六七八九十〇零]{2,4}年)[^。；，]{0,40}"
    r"(?:月|日|时|期间|以来|左右|许)"
)
JUDGMENT_DATE_RE = re.compile(
    r"(?:二[〇零一二三四五六七八九十]{3}|\d{4})年"
    r"[一二三四五六七八九十〇零\d]{1,3}月"
    r"[一二三四五六七八九十〇零\d]{1,3}日"
)
LOCATION_RE = re.compile(
    r"[\u4e00-\u9fff]{2,18}(?:省|自治区|市|县|区|旗|镇|乡|村|街道|路|公司|人民法院|人民检察院)"
)
NOISE = ("犯罪", "认罪", "悔罪", "数罪", "罪名", "罪行", "本罪", "同种罪", "罪事实", "罪情节")


def unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = normalize_text(value).strip("，。；:： ")
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def extract_crimes(text: str) -> list[str]:
    crimes: list[str] = []
    for pattern in CRIME_PATTERNS:
        for match in pattern.finditer(text):
            crime = match.group("crime")
            if not crime.startswith("罪") and not any(noise in crime for noise in NOISE):
                crimes.append(crime)
    if not crimes:
        crimes.extend(re.findall(r"[\u4e00-\u9fff]{2,20}罪", text[:1200]))
    return unique(crimes)[:5]


def extract_legal_basis(reasoning_text: str) -> str:
    """提取裁判依据段，区别于全文中一般性提到的法条。"""
    if not reasoning_text:
        return ""
    matches = list(re.finditer(r"(?:依照|根据)[^。；]{0,700}?(?:判决如下|裁定如下)", reasoning_text))
    if matches:
        return normalize_text(matches[-1].group(0))

    start = max(reasoning_text.rfind("依照"), reasoning_text.rfind("根据"))
    if start == -1:
        return ""
    tail = reasoning_text[start:]
    end_match = re.search(r"(?:判决如下|裁定如下|。)", tail)
    return normalize_text(tail[:end_match.end()] if end_match else tail)


def extract_citations(text: str) -> list[str]:
    return unique([match.group("citation") for match in CITATION_DETAIL_RE.finditer(text)])[:30]


def extract_applied_laws(legal_basis_text: str) -> list[dict]:
    """从裁判依据段抽取适用法律；无法律名时仍保留 article。"""
    rows: list[dict] = []
    current_law = ""
    seen = set()
    for match in LAW_AND_CITATION_RE.finditer(legal_basis_text):
        law_name = normalize_law_name(match.group("law"))
        if law_name:
            current_law = law_name
        citation = match.group("citation")
        key = (current_law, citation)
        if citation and key not in seen:
            seen.add(key)
            rows.append({"law_name": current_law, "article": citation})
    return rows


def normalize_law_name(value: str | None) -> str:
    value = normalize_text(value)
    return value.replace("《", "").replace("》", "")


def split_sentences(text: str) -> list[str]:
    return unique(re.findall(r"[^。；!?！？]+[。；!?！？]?", normalize_text(text)))


def source_qw(row: dict) -> str:
    source_file = normalize_text(row.get("source_file"))
    if not source_file:
        return ""
    path = RAW_DIR / source_file
    if not path.exists():
        return ""
    return normalize_text(read_json(path).get("qw"))


def extract_court_name(qw: str) -> str:
    match = re.search(r"([\u4e00-\u9fff]{2,80}人民法院)", qw[:300])
    return normalize_text(match.group(1)) if match else ""


def extract_prosecution_org(text: str) -> str:
    match = re.search(r"公诉机关([\u4e00-\u9fff]{2,80}人民检察院)", text[:800])
    if match:
        return normalize_text(match.group(1))
    match = re.search(r"([\u4e00-\u9fff]{2,80}人民检察院)指控", text[:800])
    return normalize_text(match.group(1)) if match else ""


def extract_judgment_date_text(qw: str) -> str:
    matches = list(JUDGMENT_DATE_RE.finditer(qw[-500:]))
    return normalize_text(matches[-1].group(0)) if matches else ""


def extract_texts_by_pattern(text: str, pattern: re.Pattern, limit: int = 30) -> list[str]:
    rows = [sentence for sentence in split_sentences(text) if pattern.search(sentence)]
    return unique(rows)[:limit]


def extract_texts_by_pattern_from_fields(fields: list[str], pattern: re.Pattern, limit: int = 30) -> list[str]:
    rows = []
    for field in fields:
        rows.extend(sentence for sentence in split_sentences(field) if pattern.search(sentence))
    return unique(rows)[:limit]


def extract_crime_time_texts(row: dict) -> list[str]:
    text = " ".join([row.get("accusation_text", ""), row.get("court_found_facts_text", "")])
    rows = []
    for sentence in split_sentences(text):
        if TIME_RE.search(sentence) and not re.search(r"抓获|归案|拘留|逮捕|羁押|判决执行|刑期", sentence):
            rows.append(sentence)
    return unique(rows)[:20]


def extract_case_location_texts(row: dict) -> list[str]:
    text = " ".join([row.get("accusation_text", ""), row.get("court_found_facts_text", "")])
    rows = []
    for sentence in split_sentences(text):
        if LOCATION_RE.search(sentence) and not re.search(r"人民法院|人民检察院|判决|起诉|公诉|抓获|归案|公安局|鉴定中心", sentence):
            rows.append(sentence)
    return unique(rows)[:20]


def extract_defendants(text: str) -> list[str]:
    names = []
    for match in re.finditer(r"(?:被告人|被告单位|上诉人|原审被告人)([^，。；（(]{1,40})", text):
        value = re.split(r"(?:犯|判处|决定|系|其|退|违法|赃款|非法|，|、?等人)", match.group(1))[0]
        names.extend(re.split(r"[、和及]", value))
    return unique([name.strip() for name in names if name.strip()])[:20]


def extract_punishments(judgment_text: str) -> list[dict]:
    text = re.split(r"如不服本判决|本判决为终审判决", normalize_text(judgment_text), maxsplit=1)[0]
    rows = []
    seen = set()
    last_defendants = [""]
    for sentence in split_sentences(text):
        if not PUNISHMENT_RE.search(sentence) or not CORE_PUNISHMENT_RE.search(sentence):
            continue
        defendants = extract_defendants(sentence)
        if not defendants:
            defendants = last_defendants
        else:
            last_defendants = defendants
        for defendant in defendants:
            key = (defendant, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"defendant": defendant, "punishment_text": sentence})
    return rows[:80]


def enrich(row: dict) -> dict:
    text = " ".join([row.get("fact_text", ""), row.get("reasoning_text", ""), row.get("judgment_text", "")])
    qw = source_qw(row)
    court_name = extract_court_name(qw)
    prosecution_org = extract_prosecution_org(" ".join([qw, row.get("fact_text", "")]))
    legal_basis_text = extract_legal_basis(row.get("reasoning_text", ""))
    applied_laws = extract_applied_laws(legal_basis_text)
    row = dict(row)
    row["court_name"] = court_name
    row["prosecution_org"] = prosecution_org
    row["case_location_texts"] = extract_case_location_texts(row)
    row["punishment_location_texts"] = unique([court_name, prosecution_org])
    row["crime_time_texts"] = extract_crime_time_texts(row)
    row["judgment_date_text"] = extract_judgment_date_text(qw)
    row["defendant_attitude_texts"] = extract_texts_by_pattern_from_fields(
        [row.get("reasoning_text", ""), row.get("defense_text", ""), row.get("fact_text", "")],
        ATTITUDE_RE,
    )
    row["amount_texts"] = extract_texts_by_pattern_from_fields(
        [row.get("fact_text", ""), row.get("reasoning_text", ""), row.get("judgment_text", "")],
        AMOUNT_RE,
        limit=50,
    )
    row["punishments"] = extract_punishments(row.get("judgment_text", ""))
    row["crimes"] = extract_crimes(text)
    row["legal_basis_text"] = legal_basis_text
    row["cited_articles"] = extract_citations(text)
    row["applied_articles"] = unique([item["article"] for item in applied_laws])
    row["applied_laws"] = applied_laws
    row["applied_law_names"] = unique([item["law_name"] for item in applied_laws if item["law_name"]])
    row["sentencing"] = unique([m.group(0) for m in SENTENCE_RE.finditer(row.get("judgment_text", "") or text)])[:10]
    return row


def main() -> None:
    rows = [enrich(row) for row in iter_jsonl(INPUT)]
    count = write_jsonl(OUTPUT, rows)
    print(f"wrote {count} enriched cases -> {OUTPUT}")


if __name__ == "__main__":
    main()
