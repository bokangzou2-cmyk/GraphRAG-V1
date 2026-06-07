from __future__ import annotations

import hashlib
import re
from typing import Iterable

from common import chinese_to_int, normalize_text


AMOUNT_ROLES = {
    "theft_amount_total",
    "theft_amount_single",
    "fine_amount",
    "restitution_amount",
    "compensation_amount",
    "illegal_gain_amount",
    "victim_loss_amount",
    "unknown_amount",
}

THEFT_AMOUNT_ROLES = {"theft_amount_total", "theft_amount_single", "victim_loss_amount"}
FINE_AMOUNT_ROLES = {"fine_amount"}
RESTITUTION_AMOUNT_ROLES = {"restitution_amount"}

THEFT_CONTEXT_RE = re.compile(r"盗窃|盗得|窃得|被盗|偷走|偷得|入户盗窃|入室盗窃|扒窃|赃物")
THEFT_TOTAL_RE = re.compile(r"合计价值|共计|总计|涉案金额|鉴定价格|盗窃.*金额|盗窃.*价值|窃得.*合计|被盗.*共计|价值共计")
THEFT_SINGLE_RE = re.compile(r"盗得|窃得|被盗|偷走|偷得|入户盗窃|入室盗窃|价值人民币|现金人民币")

AMOUNT_RE = re.compile(
    r"(?P<raw>(?:人民币)?(?P<num>\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)(?P<more>余|多)?(?P<unit>万元|万余元|元))"
)


def parse_amount_value(num_text: str, unit: str) -> float | None:
    text = normalize_text(num_text)
    if not text:
        return None
    multiplier = 1.0
    if "万" in unit or text.endswith("万"):
        multiplier = 10000.0
        text = text.removesuffix("万")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text) * multiplier
    value = chinese_to_int(text)
    return float(value) * multiplier if value is not None else None


def classify_amount_role(context: str) -> tuple[str, float]:
    text = normalize_text(context)
    if re.search(r"罚金|单处罚金|并处罚金|处罚金|罚款", text):
        return "fine_amount", 0.92
    if re.search(r"退赔|退赃|退还|退出赃款|追缴|责令.*退|发还", text):
        return "restitution_amount", 0.88
    if re.search(r"赔偿|附带民事|经济损失", text):
        return "compensation_amount", 0.84
    theft_context = bool(THEFT_CONTEXT_RE.search(text))
    if theft_context and re.search(r"损失|被害人损失|造成.*损失", text):
        return "victim_loss_amount", 0.78
    if theft_context and THEFT_TOTAL_RE.search(text):
        return "theft_amount_total", 0.86
    if theft_context and THEFT_SINGLE_RE.search(text):
        return "theft_amount_single", 0.76
    if re.search(r"违法所得|非法获利|获利|赃款赃物", text):
        return "illegal_gain_amount", 0.78
    if re.search(r"损失|被害人损失|造成.*损失", text):
        return "victim_loss_amount", 0.58
    return "unknown_amount", 0.45


def amount_id_for(case_id: str, field: str, start: int, raw_text: str, context: str) -> str:
    digest = hashlib.sha1(f"{case_id}|{field}|{start}|{raw_text}|{context}".encode("utf-8")).hexdigest()[:12]
    return f"amount:{case_id}:{digest}"


def iter_text_fields(case_row: dict) -> Iterable[tuple[str, str]]:
    text_fields = [
        "fact_text",
        "accusation_text",
        "defense_text",
        "court_found_facts_text",
        "evidence_text",
        "reasoning_text",
        "judgment_text",
        "legal_basis_text",
    ]
    list_fields = [
        "witness_testimony_texts",
        "victim_statement_texts",
        "defendant_confession_texts",
        "expert_opinion_texts",
        "documentary_evidence_texts",
        "inspection_records_texts",
        "amount_texts",
        "sentencing",
    ]
    for field in text_fields:
        value = case_row.get(field)
        if value:
            yield field, str(value)
    for field in list_fields:
        for index, value in enumerate(case_row.get(field) or []):
            if value:
                yield f"{field}:{index}", str(value)


def extract_amounts_from_case(case_row: dict) -> list[dict]:
    case_id = case_row.get("case_id", "")
    source_file = case_row.get("source_file", "")
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for field, text in iter_text_fields(case_row):
        normalized = normalize_text(text)
        for match in AMOUNT_RE.finditer(normalized):
            value = parse_amount_value(match.group("num"), match.group("unit"))
            if value is None:
                continue
            start, end = match.span("raw")
            context = normalized[max(0, start - 80): min(len(normalized), end + 100)]
            role_context = normalized[max(0, start - 50): min(len(normalized), end + 24)]
            role, confidence = classify_amount_role(role_context)
            raw_text = match.group("raw")
            key = (field, raw_text, context)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "amount_id": amount_id_for(case_id, field, start, raw_text, context),
                "case_id": case_id,
                "value": int(value) if value.is_integer() else value,
                "unit": "CNY",
                "raw_text": raw_text,
                "role": role,
                "field": field.split(":", 1)[0],
                "field_instance": field,
                "context": context,
                "confidence": confidence,
                "source_file": source_file,
                "text_hash": hashlib.sha1(context.encode("utf-8")).hexdigest()[:16],
                "extraction_method": "regex_context_role_v1",
            })
    return rows


def amount_query_profile(query: str) -> dict:
    query = normalize_text(query)
    values = []
    for match in AMOUNT_RE.finditer(query):
        value = parse_amount_value(match.group("num"), match.group("unit"))
        if value is not None:
            values.append(int(value) if value.is_integer() else value)
    role = "unknown_amount"
    preferred_roles: set[str] = set()
    if re.search(r"罚金|罚款|判处罚金", query):
        role = "fine_amount"
        preferred_roles = set(FINE_AMOUNT_ROLES)
    elif re.search(r"退赔|退赃|退还|追缴", query):
        role = "restitution_amount"
        preferred_roles = set(RESTITUTION_AMOUNT_ROLES)
    elif re.search(r"盗窃|偷|入户|入室|涉案金额|总金额|数额|金额|价值", query):
        role = "theft_amount_total_or_single"
        preferred_roles = set(THEFT_AMOUNT_ROLES)
    return {
        "amount_query_values": values,
        "amount_query_value": values[0] if values else None,
        "amount_query_role": role,
        "preferred_amount_roles": sorted(preferred_roles),
    }


def amount_distance(query_value: float, matched_value: float) -> float:
    return abs(float(query_value) - float(matched_value))
