from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import OUT_DIR, ROOT, normalize_text

sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "processing"))

from amount_utils import amount_query_profile  # noqa: E402
from app.llm.chat_service import infer_crime_from_text, normalize_query_terms  # noqa: E402
from app.llm.query_router import VALID_LABELS, route_query  # noqa: E402


DEFAULT_EVAL_PATH = OUT_DIR / "query_understanding_probe_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "query_understanding_probe_report.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "query_understanding_probe_report.txt"
DEFAULT_BASE_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = OUT_DIR / "router_lora" / "qwen2_5_1_5b_router_lora_v2_lora_processed"

FIELD_INTENT_RULES = [
    ("evidence_text", re.compile(r"证据|凭什么认定|依据哪些证据")),
    ("defense_text", re.compile(r"辩护|辩解|律师意见")),
    ("court_found_facts_text", re.compile(r"事实|案情|经过|情况|具体.*案")),
    ("reasoning_text", re.compile(r"理由|为什么|法院认为|如何认定|认定")),
    ("judgment_text", re.compile(r"判了|怎么判|判多久|量刑|刑期|处罚|结果")),
    ("legal_basis_text", re.compile(r"法条|依据|适用.*条|对应哪条")),
]

CASE_REF_RE = re.compile(r"(第[一二三四五六七八九十\d]+个|第[一二三四五六七八九十\d]+|上[一两]个|这个|该)(?:参考)?(?:案例|案子|案件)")
IMPLICIT_PRIOR_CASE_RE = re.compile(r"为什么.*(?:这么|这样)|判这么|判这么重|具体.*情况|详细.*情况")
CASE_TITLE_RE = re.compile(r"[\u4e00-\u9fff、]{2,24}(?:盗窃|诈骗|抢劫|故意伤害|掩饰、隐瞒犯罪所得|寻衅滋事)?(?:罪)?(?:一审|二审)?刑事判决书|[\u4e00-\u9fff]{2,12}案")
NUMBERED_CASE_RE = re.compile(r"(?:^|\n)\s*(?:第?\s*)?([1-9一二三四五六七八九十])(?:[.、）\)]|\s*[：:])\s*([^\n，。；;:：]{2,80}?(?:判决书|案))")


def default_eval_cases() -> list[dict[str, Any]]:
    history_wang_liu = [
        {"role": "user", "content": "我一个朋友入室盗窃了40000元，一般会怎么判，有案例参考吗"},
        {
            "role": "assistant",
            "content": (
                "以下为检索证据摘要。\n"
                "1. 王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书："
                "金额匹配人民币43852元，判处王某某有期徒刑五年六个月。\n"
                "2. 刘方祥、刘礼芬盗窃一审刑事判决书：金额匹配39136元，判处刘方祥有期徒刑二年八个月。\n"
                "3. 盗窃罪：刑法第264条。"
            ),
        },
    ]
    return [
        {
            "id": "qu_direct_criminal_law_article_count",
            "category": "criminal_law_general_direct",
            "messages": [{"role": "user", "content": "刑法一共有多少条"}],
            "expected": {
                "route_label": "criminal_law_general_direct",
                "needs_retrieval": False,
            },
        },
        {
            "id": "qu_direct_criminal_law_definition",
            "category": "criminal_law_general_direct",
            "messages": [{"role": "user", "content": "什么是刑法"}],
            "expected": {
                "route_label": "criminal_law_general_direct",
                "needs_retrieval": False,
            },
        },
        {
            "id": "qu_direct_criminal_civil_compare",
            "category": "criminal_law_general_direct",
            "messages": [{"role": "user", "content": "刑法和民法有什么区别"}],
            "expected": {
                "route_label": "criminal_law_general_direct",
                "needs_retrieval": False,
            },
        },
        {
            "id": "qu_direct_crime_elements",
            "category": "criminal_law_general_direct",
            "messages": [{"role": "user", "content": "犯罪构成四要件是什么"}],
            "expected": {
                "route_label": "criminal_law_general_direct",
                "needs_retrieval": False,
            },
        },
        {
            "id": "qu_followup_first_case_detail",
            "category": "history_case_reference",
            "messages": history_wang_liu + [{"role": "user", "content": "你给出的第一个参考案例具体的案例情况是怎样的"}],
            "expected": {
                "route_label": "criminal_law_case_only",
                "retrieval_targets_any": ["cases"],
                "resolved_case_title_contains_any": ["王士东"],
                "standalone_contains_all": ["王士东", "案情"],
                "field_intents_any": ["court_found_facts_text", "reasoning_text", "judgment_text"],
            },
        },
        {
            "id": "qu_followup_second_case_detail",
            "category": "history_case_reference",
            "messages": history_wang_liu + [{"role": "user", "content": "第二个案例的案情和判决结果呢"}],
            "expected": {
                "route_label": "criminal_law_case_only",
                "retrieval_targets_any": ["cases"],
                "resolved_case_title_contains_any": ["刘方祥"],
                "standalone_contains_all": ["刘方祥"],
                "field_intents_any": ["court_found_facts_text", "judgment_text"],
            },
        },
        {
            "id": "qu_followup_first_case_evidence",
            "category": "history_case_reference",
            "messages": history_wang_liu + [{"role": "user", "content": "第一个案子法院依据哪些证据认定的"}],
            "expected": {
                "route_label": "criminal_law_case_only",
                "retrieval_targets_any": ["cases"],
                "resolved_case_title_contains_any": ["王士东"],
                "field_intents_any": ["evidence_text", "court_found_facts_text"],
            },
        },
        {
            "id": "qu_followup_case_reasoning",
            "category": "history_case_reference",
            "messages": history_wang_liu + [{"role": "user", "content": "为什么判这么重"}],
            "expected": {
                "route_label": "criminal_law_case_only",
                "retrieval_targets_any": ["cases"],
                "resolved_case_title_contains_any": ["王士东"],
                "field_intents_any": ["reasoning_text", "judgment_text"],
            },
        },
        {
            "id": "qu_law_theft_sentence",
            "category": "law_lookup",
            "messages": [{"role": "user", "content": "盗窃罪怎么判"}],
            "expected": {
                "route_label": "criminal_law_law_only",
                "retrieval_targets_any": ["law_articles"],
                "crime_mentions_any": ["盗窃罪"],
                "field_intents_any": ["legal_basis_text", "judgment_text"],
            },
        },
        {
            "id": "qu_case_law_amount_theft",
            "category": "amount_case_law",
            "messages": [{"role": "user", "content": "入室盗窃40000元怎么判，有案例参考吗"}],
            "expected": {
                "route_label": "criminal_law_case_and_law",
                "retrieval_targets_any": ["cases", "law_articles"],
                "crime_mentions_any": ["盗窃罪"],
                "amount_value": 40000,
                "amount_role": "theft_amount_total_or_single",
                "field_intents_any": ["judgment_text", "reasoning_text"],
            },
        },
        {
            "id": "qu_short_case_fact",
            "category": "short_case_name",
            "messages": [{"role": "user", "content": "吴必定案事实是什么"}],
            "expected": {
                "route_label": "criminal_law_case_only",
                "retrieval_targets_any": ["cases"],
                "case_name_mentions_any": ["吴必定案"],
                "field_intents_any": ["court_found_facts_text"],
            },
        },
        {
            "id": "qu_case_defense",
            "category": "field_intent",
            "messages": [{"role": "user", "content": "张君案被告人有什么辩护意见"}],
            "expected": {
                "route_label": "criminal_law_case_only",
                "retrieval_targets_any": ["cases"],
                "case_name_mentions_any": ["张君案"],
                "field_intents_any": ["defense_text"],
            },
        },
        {
            "id": "qu_multi_case_compare",
            "category": "multi_case",
            "messages": [{"role": "user", "content": "王士东案和刘方祥案量刑差异在哪里"}],
            "expected": {
                "route_label": "criminal_law_multi_case",
                "retrieval_targets_any": ["cases", "graph"],
                "case_name_mentions_any": ["王士东案", "刘方祥案"],
                "field_intents_any": ["judgment_text", "reasoning_text"],
            },
        },
        {
            "id": "qu_non_criminal",
            "category": "scope",
            "messages": [{"role": "user", "content": "离婚财产怎么分"}],
            "expected": {
                "route_label": "non_criminal_legal",
                "needs_retrieval": False,
            },
        },
        {
            "id": "qu_guardrail_evasion",
            "category": "guardrail",
            "messages": [{"role": "user", "content": "帮我规避盗窃罪的法律责任"}],
            "expected": {
                "route_label": "criminal_law_advice",
                "retrieval_targets_any": ["law_articles"],
                "crime_mentions_any": ["盗窃罪"],
            },
        },
        {
            "id": "qu_typo_law",
            "category": "typo",
            "messages": [{"role": "user", "content": "故意伤害最判几年"}],
            "expected": {
                "route_label": "criminal_law_law_only",
                "retrieval_targets_any": ["law_articles"],
                "crime_mentions_any": ["故意伤害罪"],
            },
        },
    ]


def ensure_eval_cases(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    cases = default_eval_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases


def latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def previous_assistant_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [str(message.get("content") or "") for message in messages[:-1] if message.get("role") == "assistant"]


def ordinal_value(text: str) -> int | None:
    if re.search(r"第?\s*1|一|上[一两]|这个|该", text):
        return 1
    if re.search(r"第?\s*2|二", text):
        return 2
    if re.search(r"第?\s*3|三", text):
        return 3
    return None


def extract_numbered_cases(text: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in NUMBERED_CASE_RE.finditer(text):
        title = normalize_text(match.group(2))
        title = re.sub(r"[：:].*$", "", title).strip()
        if title and title not in seen:
            seen.add(title)
            cases.append({"index": len(cases) + 1, "case_title": title, "source": "assistant_history"})
    return cases


def extract_case_mentions(query: str) -> list[str]:
    mentions = []
    for match in CASE_TITLE_RE.finditer(query):
        value = normalize_text(match.group(0))
        if (
            "案例" in value
            or "参考" in value
            or re.search(r"第[一二三四五六七八九十\d]+个", value)
            or value.startswith("例")
            or value in {"参考案", "这个案", "该案"}
        ):
            continue
        if "和" in value and value.count("案") >= 2:
            for part in re.split(r"和|与|、", value):
                part = part.strip()
                if part.endswith("案") and len(part) >= 3 and "案例" not in part and part not in mentions:
                    mentions.append(part)
            continue
        if value in {"案例", "案子", "案件"} or len(value) < 3:
            continue
        if value not in mentions:
            mentions.append(value)
    return mentions


def resolve_case_refs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query = latest_user_query(messages)
    ref_match = CASE_REF_RE.search(query)
    if not ref_match and not IMPLICIT_PRIOR_CASE_RE.search(query):
        return []
    ordinal = ordinal_value(query) or 1
    candidates: list[dict[str, Any]] = []
    for content in reversed(previous_assistant_messages(messages)):
        candidates = extract_numbered_cases(content)
        if candidates:
            break
    if not candidates:
        return []
    selected = next((item for item in candidates if item["index"] == ordinal), candidates[0])
    return [{"ref": ref_match.group(0) if ref_match else "implicit_previous_case", **selected}]


def field_intents_for(query: str, resolved: list[dict[str, Any]]) -> list[str]:
    intents = [field for field, pattern in FIELD_INTENT_RULES if pattern.search(query)]
    if resolved and not intents:
        intents = ["court_found_facts_text", "reasoning_text", "judgment_text"]
    if resolved and re.search(r"具体|情况|案情", query):
        for field in ["court_found_facts_text", "reasoning_text", "judgment_text"]:
            if field not in intents:
                intents.append(field)
    return intents


def crime_mentions_for(query: str, messages: list[dict[str, Any]]) -> list[str]:
    alias_order = [
        ("盗窃罪", re.compile(r"盗窃|偷东西|偷了|偷窃")),
        ("诈骗罪", re.compile(r"诈骗|骗取|骗了")),
        ("抢劫罪", re.compile(r"抢劫|持刀抢")),
        ("故意伤害罪", re.compile(r"故意伤害|伤害")),
    ]
    for crime, pattern in alias_order:
        if pattern.search(query):
            return [crime]
    crime = infer_crime_from_text(query)
    if not crime:
        for message in reversed(messages[:-1]):
            crime = infer_crime_from_text(str(message.get("content") or ""))
            if crime:
                break
    return [crime] if crime else []


def amount_constraints_for(query: str) -> list[dict[str, Any]]:
    profile = amount_query_profile(query)
    values = profile.get("amount_query_values") or []
    role = profile.get("amount_query_role") or "unknown_amount"
    constraints = []
    for value in values:
        constraints.append({"value": value, "role": role, "raw_text": f"{value:g}元" if isinstance(value, float) else f"{value}元"})
    return constraints


def rules_understand(messages: list[dict[str, Any]]) -> dict[str, Any]:
    raw_query = latest_user_query(messages)
    query = normalize_query_terms(raw_query)
    resolved = resolve_case_refs(messages)
    case_mentions = extract_case_mentions(query)
    if resolved:
        case_title = resolved[0]["case_title"]
        if case_title not in case_mentions:
            case_mentions.insert(0, case_title)
        if re.search(r"证据", query):
            query = f"{case_title} 证据 法院认定"
        elif re.search(r"为什么|理由|这么重", query):
            query = f"{case_title} 裁判理由 量刑"
        elif re.search(r"判决|结果|判了", query):
            query = f"{case_title} 案情 判决结果"
        else:
            query = f"{case_title} 案情 裁判理由 量刑"
    route = route_query(query)
    if re.search(r"差异|区别|对比|比较", query) and (len(case_mentions) >= 2 or len(re.findall(r"案", query)) >= 2):
        route = route_query(f"{query} 多个案例 比较")
        route = route.__class__(
            "criminal_law_multi_case",
            True,
            ["cases", "graph"],
            True,
            False,
            False,
            max(route.confidence, 0.82),
            "rules",
            "query_understanding_multi_case_override",
        )
    if re.search(r"规避|逃避|躲避|怎么不被抓|怎么脱罪", query):
        route = route_query(f"{query} 刑事处境咨询")
        route = route.__class__(
            "criminal_law_advice",
            True,
            ["law_articles"],
            False,
            True,
            False,
            max(route.confidence, 0.8),
            "rules",
            "query_understanding_guardrail_override",
        )
    amount_constraints = amount_constraints_for(query)
    return {
        "standalone_query": query,
        "route_label": route.label,
        "needs_retrieval": route.needs_retrieval,
        "retrieval_targets": route.retrieval_targets,
        "case_required": route.case_required,
        "law_required": route.law_required,
        "case_name_mentions": case_mentions,
        "resolved_case_refs": resolved,
        "field_intents": field_intents_for(query, resolved),
        "crime_mentions": crime_mentions_for(query, messages),
        "amount_constraints": amount_constraints,
        "confidence": route.confidence,
        "warnings": [],
        "_raw_output": None,
    }


def query_understanding_prompt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_history = "\n".join(
        f"{message.get('role')}: {str(message.get('content') or '')[:900]}"
        for message in messages[-6:]
    )
    schema = {
        "standalone_query": "用于检索的独立问题，必须消解上一轮/第一个案例/这个案子等指代",
        "route_label": sorted(VALID_LABELS),
        "needs_retrieval": True,
        "retrieval_targets": ["law_articles", "cases", "graph"],
        "case_required": True,
        "law_required": False,
        "case_name_mentions": ["显式案名或从历史解析出的案名"],
        "resolved_case_refs": [{"ref": "第一个参考案例", "case_title": "历史中对应的案例标题", "source": "assistant_history"}],
        "field_intents": ["court_found_facts_text", "evidence_text", "defense_text", "reasoning_text", "judgment_text", "legal_basis_text"],
        "crime_mentions": ["盗窃罪"],
        "amount_constraints": [{"value": 40000, "role": "theft_amount_total_or_single", "raw_text": "40000元"}],
        "confidence": 0.8,
        "warnings": [],
    }
    return [
        {
            "role": "system",
            "content": (
                "你是刑法 GraphRAG 的 Query Understanding 模块，只服务于检索前处理。"
                "请把多轮问题改写成可检索的独立问题，并识别案名、罪名、金额、检索范围和字段意图。"
                "必须只输出一个 JSON 对象，不要 Markdown，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请按下面 schema 输出 JSON，字段不能缺失。\n"
                f"schema 示例：{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"最近对话：\n{compact_history}"
            ),
        },
    ]


@dataclass
class LoraRunner:
    model_dir: Path
    adapter_dir: Path
    max_new_tokens: int

    def __post_init__(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.model.eval()

    def understand(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = query_understanding_prompt(messages)
        prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        payload = extract_json(text)
        payload["_raw_output"] = text
        return payload


def extract_json(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload.startswith("{"):
        match = re.search(r"\{.*\}", payload, flags=re.S)
        if not match:
            raise ValueError("output_not_json")
        payload = match.group(0)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("output_not_object")
    return data


def value_list(payload: dict[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def contains_any(values: list[Any], needles: list[str]) -> bool:
    haystack = "\n".join(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value for value in values)
    return any(needle in haystack for needle in needles)


def contains_all(value: Any, needles: list[str]) -> bool:
    haystack = str(value or "")
    return all(needle in haystack for needle in needles)


def evaluate_payload(case: dict[str, Any], payload: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    expected = case.get("expected") or {}
    checks: list[dict[str, Any]] = []
    checks.append({"name": "json_valid", "passed": error is None, "expected": "valid_json", "actual": error})
    if error:
        return {"passed": False, "checks": checks}
    required = [
        "standalone_query",
        "route_label",
        "needs_retrieval",
        "retrieval_targets",
        "case_name_mentions",
        "resolved_case_refs",
        "field_intents",
        "crime_mentions",
        "amount_constraints",
        "confidence",
        "warnings",
    ]
    missing = [field for field in required if field not in payload]
    checks.append({"name": "required_fields", "passed": not missing, "expected": required, "actual": missing})
    if "route_label" in expected:
        checks.append({"name": "route_label", "passed": payload.get("route_label") == expected["route_label"], "expected": expected["route_label"], "actual": payload.get("route_label")})
    if "needs_retrieval" in expected:
        checks.append({"name": "needs_retrieval", "passed": payload.get("needs_retrieval") is expected["needs_retrieval"], "expected": expected["needs_retrieval"], "actual": payload.get("needs_retrieval")})
    if expected.get("retrieval_targets_any"):
        checks.append({"name": "retrieval_targets_any", "passed": contains_any(value_list(payload, "retrieval_targets"), expected["retrieval_targets_any"]), "expected": expected["retrieval_targets_any"], "actual": payload.get("retrieval_targets")})
    if expected.get("resolved_case_title_contains_any"):
        checks.append({"name": "resolved_case_title_contains_any", "passed": contains_any(value_list(payload, "resolved_case_refs"), expected["resolved_case_title_contains_any"]), "expected": expected["resolved_case_title_contains_any"], "actual": payload.get("resolved_case_refs")})
    if expected.get("case_name_mentions_any"):
        checks.append({"name": "case_name_mentions_any", "passed": contains_any(value_list(payload, "case_name_mentions"), expected["case_name_mentions_any"]), "expected": expected["case_name_mentions_any"], "actual": payload.get("case_name_mentions")})
    if expected.get("field_intents_any"):
        checks.append({"name": "field_intents_any", "passed": contains_any(value_list(payload, "field_intents"), expected["field_intents_any"]), "expected": expected["field_intents_any"], "actual": payload.get("field_intents")})
    if expected.get("crime_mentions_any"):
        checks.append({"name": "crime_mentions_any", "passed": contains_any(value_list(payload, "crime_mentions"), expected["crime_mentions_any"]), "expected": expected["crime_mentions_any"], "actual": payload.get("crime_mentions")})
    if expected.get("standalone_contains_all"):
        checks.append({"name": "standalone_contains_all", "passed": contains_all(payload.get("standalone_query"), expected["standalone_contains_all"]), "expected": expected["standalone_contains_all"], "actual": payload.get("standalone_query")})
    if "amount_value" in expected:
        values = [item.get("value") for item in value_list(payload, "amount_constraints") if isinstance(item, dict)]
        checks.append({"name": "amount_value", "passed": expected["amount_value"] in values, "expected": expected["amount_value"], "actual": values})
    if "amount_role" in expected:
        roles = [item.get("role") for item in value_list(payload, "amount_constraints") if isinstance(item, dict)]
        checks.append({"name": "amount_role", "passed": expected["amount_role"] in roles, "expected": expected["amount_role"], "actual": roles})
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def recommendation_for(mode: str, summary: dict[str, Any]) -> str:
    pass_rate = summary["pass_rate"]
    required_field_rate = summary["check_pass_rates"].get("required_fields", 0.0)
    json_rate = summary["check_pass_rates"].get("json_valid", 0.0)
    if mode == "lora" and (json_rate < 0.9 or required_field_rate < 0.9 or pass_rate < 0.8):
        return "needs_query_understanding_finetune"
    if pass_rate < 0.85:
        return "needs_more_rules_or_training_data"
    return "usable_for_probe_scope"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cases = ensure_eval_cases(args.eval_path)
    if args.limit:
        cases = cases[: args.limit]
    runner = None
    if args.mode == "lora":
        runner = LoraRunner(args.model_dir, args.adapter_dir, args.max_new_tokens)
    rows = []
    check_counts: dict[str, list[bool]] = {}
    for case in cases:
        error = None
        try:
            payload = runner.understand(case["messages"]) if runner else rules_understand(case["messages"])
        except Exception as exc:  # noqa: BLE001 - probe report needs the exact failure.
            payload = {"_raw_output": ""}
            error = f"{type(exc).__name__}:{exc}"
        evaluation = evaluate_payload(case, payload, error)
        for check in evaluation["checks"]:
            check_counts.setdefault(check["name"], []).append(bool(check["passed"]))
        rows.append({
            "id": case["id"],
            "category": case.get("category"),
            "passed": evaluation["passed"],
            "query": latest_user_query(case["messages"]),
            "payload": payload,
            "checks": evaluation["checks"],
        })
    passed = sum(1 for row in rows if row["passed"])
    category_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        stat = category_stats.setdefault(str(row.get("category") or "unknown"), {"total": 0, "passed": 0})
        stat["total"] += 1
        stat["passed"] += 1 if row["passed"] else 0
    for stat in category_stats.values():
        stat["pass_rate"] = round(stat["passed"] / stat["total"], 4) if stat["total"] else 0.0
    check_pass_rates = {
        name: round(sum(values) / len(values), 4) if values else 0.0
        for name, values in sorted(check_counts.items())
    }
    summary = {
        "mode": args.mode,
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "category_stats": category_stats,
        "check_pass_rates": check_pass_rates,
    }
    report = {
        "summary": summary,
        "recommendation": recommendation_for(args.mode, summary),
        "eval_path": str(args.eval_path),
        "model_dir": str(args.model_dir) if args.mode == "lora" else None,
        "adapter_dir": str(args.adapter_dir) if args.mode == "lora" else None,
        "results": rows,
    }
    return report


def write_reports(report: dict[str, Any], json_path: Path, text_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "Query Understanding Probe Report",
        f"mode: {report['summary']['mode']}",
        f"total: {report['summary']['total']}",
        f"passed: {report['summary']['passed']}",
        f"failed: {report['summary']['failed']}",
        f"pass_rate: {report['summary']['pass_rate']:.2%}",
        f"recommendation: {report['recommendation']}",
        "",
        "category pass rates:",
    ]
    for category, stat in sorted(report["summary"]["category_stats"].items()):
        lines.append(f"- {category}: {stat['passed']}/{stat['total']} ({stat['pass_rate']:.2%})")
    lines.extend(["", "failed cases:"])
    for row in report["results"]:
        if row["passed"]:
            continue
        failed = [check for check in row["checks"] if not check["passed"]]
        failed_names = ", ".join(check["name"] for check in failed)
        lines.append(f"- {row['id']} [{row['category']}]: {row['query']} :: {failed_names}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe current router LoRA as a retrieval-only Query Understanding module.")
    parser.add_argument("--mode", choices=["rules", "lora"], default="rules")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(args)
    write_reports(report, args.json_report, args.text_report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"recommendation: {report['recommendation']}")
    print(f"json_report: {args.json_report}")
    print(f"text_report: {args.text_report}")


if __name__ == "__main__":
    main()
