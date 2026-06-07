from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Literal

import app.processing_bridge  # noqa: F401
import torch
from amount_utils import amount_query_profile

from app.config import settings
from app.llm.query_rewriter import QueryRewriteResult, rewrite_query
from app.llm.query_router import RouteDecision, VALID_LABELS, decision_from_payload, route_query


FIELD_INTENT_RULES = [
    ("evidence_text", re.compile(r"证据|凭什么认定|依据哪些证据")),
    ("defense_text", re.compile(r"辩护|辩解|律师意见")),
    ("court_found_facts_text", re.compile(r"事实|案情|经过|情况|具体.*案")),
    ("reasoning_text", re.compile(r"理由|为什么|法院认为|如何认定|认定|判这么重")),
    ("judgment_text", re.compile(r"判了|怎么判|判多久|量刑|刑期|处罚|结果")),
    ("legal_basis_text", re.compile(r"法条|依据|适用.*条|对应哪条")),
]
CRIME_ALIAS_PATTERNS = [
    ("过失致人死亡罪", re.compile(r"过失杀人|过失致人死亡|过失(?:让|使|导致)人死亡")),
    ("故意杀人罪", re.compile(r"故意杀人|杀人")),
    ("盗窃罪", re.compile(r"盗窃|偷东西|偷了|偷窃|入户|入室")),
    ("诈骗罪", re.compile(r"诈骗|骗取|骗了")),
    ("抢劫罪", re.compile(r"抢劫|持刀抢")),
    ("故意伤害罪", re.compile(r"故意伤害|故意伤人|伤害")),
]
QUERY_EQUIVALENTS = {
    "入室盗窃": "入户盗窃",
    "入室偷窃": "入户盗窃",
    "入室偷东西": "入户盗窃",
    "过失杀人": "过失致人死亡罪",
    "过失让人死亡": "过失致人死亡罪",
    "过失使人死亡": "过失致人死亡罪",
    "过失导致人死亡": "过失致人死亡罪",
}
CASE_REF_RE = re.compile(r"(第[一二三四五六七八九十\d]+个|第[一二三四五六七八九十\d]+|上[一两]个|这个|该)(?:参考)?(?:案例|案子|案件)")
IMPLICIT_PRIOR_CASE_RE = re.compile(r"为什么.*(?:这么|这样)|判这么|判这么重|具体.*情况|详细.*情况")
CASE_TITLE_RE = re.compile(r"[\u4e00-\u9fff、]{2,24}(?:盗窃|诈骗|抢劫|故意伤害|掩饰、隐瞒犯罪所得|寻衅滋事)?(?:罪)?(?:一审|二审)?刑事判决书|[\u4e00-\u9fff]{2,12}案")
NUMBERED_CASE_RE = re.compile(r"(?:^|\n)\s*(?:第?\s*)?([1-9一二三四五六七八九十])(?:[.、）\)]|\s*[：:])\s*([^\n，。；;:：]{2,80}?(?:判决书|案))")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def normalize_query_terms(query: str) -> str:
    normalized = query.strip()
    expansions = [target for source, target in QUERY_EQUIVALENTS.items() if source in normalized and target not in normalized]
    if expansions:
        normalized = f"{normalized} {' '.join(expansions)}"
    return normalized


def infer_crimes_from_text(text: str) -> list[str]:
    crimes: list[str] = []
    for crime, pattern in CRIME_ALIAS_PATTERNS:
        if pattern.search(text):
            crimes.append(crime)
    for explicit in re.findall(r"[\u4e00-\u9fff]{2,16}罪", text):
        if explicit not in crimes:
            crimes.append(explicit)
    return crimes


def infer_crime_from_text(text: str) -> str:
    crimes = infer_crimes_from_text(text)
    return crimes[0] if crimes else ""


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
        if "案例" in value or "参考" in value or re.search(r"第[一二三四五六七八九十\d]+个", value):
            continue
        if value in {"案例", "案子", "案件", "参考案", "这个案", "该案"} or len(value) < 3:
            continue
        if "和" in value and value.count("案") >= 2:
            for part in re.split(r"和|与|、", value):
                part = part.strip()
                if part.endswith("案") and len(part) >= 3 and "案例" not in part and part not in mentions:
                    mentions.append(part)
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
    crimes = infer_crimes_from_text(query)
    if not crimes:
        for message in reversed(messages[:-1]):
            crimes = infer_crimes_from_text(str(message.get("content") or ""))
            if crimes:
                break
    return crimes


def amount_constraints_for(query: str) -> list[dict[str, Any]]:
    profile = amount_query_profile(query)
    values = profile.get("amount_query_values") or []
    role = profile.get("amount_query_role") or "unknown_amount"
    return [{"value": value, "role": role, "raw_text": f"{value:g}元" if isinstance(value, float) else f"{value}元"} for value in values]


def rules_understand(messages: list[dict[str, Any]], reason: str = "rules_query_understanding") -> dict[str, Any]:
    query = normalize_query_terms(latest_user_query(messages))
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
        route = RouteDecision("criminal_law_multi_case", True, ["cases", "graph"], True, False, False, max(route.confidence, 0.82), "rules", reason)
    if re.search(r"规避|逃避|躲避|怎么不被抓|怎么脱罪", query):
        route = RouteDecision("criminal_law_advice", True, ["law_articles"], False, True, False, max(route.confidence, 0.8), "rules", reason)
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
        "amount_constraints": amount_constraints_for(query),
        "confidence": route.confidence,
        "warnings": [],
        "_raw_output": None,
        "_source": "rules",
        "_reason": reason,
    }


def is_general_criminal_direct_query(query: str) -> bool:
    route = route_query(query)
    return route.label == "criminal_law_general_direct"


def query_understanding_prompt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_history = "\n".join(f"{message.get('role')}: {str(message.get('content') or '')[:900]}" for message in messages[-6:])
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
            "content": "请按下面 schema 输出 JSON，字段不能缺失。\n"
            f"schema 示例：{json.dumps(schema, ensure_ascii=False)}\n\n最近对话：\n{compact_history}",
        },
    ]


def extract_json(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload.startswith("{"):
        match = re.search(r"\{.*\}", payload, flags=re.S)
        if not match:
            raise ValueError("query_understanding_output_not_json")
        payload = match.group(0)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("query_understanding_output_not_object")
    return data


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "standalone_query",
        "route_label",
        "needs_retrieval",
        "retrieval_targets",
        "case_required",
        "law_required",
        "case_name_mentions",
        "resolved_case_refs",
        "field_intents",
        "crime_mentions",
        "amount_constraints",
        "confidence",
        "warnings",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"query_understanding_missing_fields:{','.join(missing)}")
    if payload["route_label"] not in VALID_LABELS:
        raise ValueError(f"query_understanding_unknown_label:{payload['route_label']}")
    payload["standalone_query"] = normalize_query_terms(str(payload.get("standalone_query") or ""))
    for field in ["retrieval_targets", "case_name_mentions", "resolved_case_refs", "field_intents", "crime_mentions", "amount_constraints", "warnings"]:
        if not isinstance(payload.get(field), list):
            payload[field] = []
    payload["needs_retrieval"] = bool(payload.get("needs_retrieval"))
    payload["case_required"] = bool(payload.get("case_required"))
    payload["law_required"] = bool(payload.get("law_required"))
    try:
        payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.8))))
    except (TypeError, ValueError):
        payload["confidence"] = 0.8
    return payload


def previous_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages[:-1]):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return normalize_query_terms(str(message["content"]).strip())
    return ""


def postprocess_payload(payload: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    latest = normalize_query_terms(latest_user_query(messages))
    previous_user = previous_user_query(messages)
    standalone = str(payload.get("standalone_query") or latest)

    if re.search(r"类似|参考|类案|相似|有没有.*案例|案件参考", latest) and previous_user and not CASE_REF_RE.search(latest):
        standalone = f"{latest} {previous_user}"
        payload["route_label"] = "criminal_law_case_only"
        payload["needs_retrieval"] = True
        payload["retrieval_targets"] = ["cases"]
        payload["case_required"] = True
        payload["law_required"] = False

    if latest and "退赔" not in latest and "退赃" not in latest and "追缴" not in latest:
        standalone = re.sub(r"，?重点提取[^。]*(?:退赃退赔|退赔|退赃|认罪认罚)[^。]*。?", "", standalone)
        standalone = standalone.replace("退赃退赔、", "").replace("退赃退赔", "")

    standalone = normalize_query_terms(standalone)
    if latest and latest not in standalone and re.search(r"\d+(?:\.\d+)?\s*(?:元|万元)", latest):
        standalone = f"{standalone} {latest}"
    payload["standalone_query"] = standalone

    local_amounts = amount_constraints_for(standalone)
    if local_amounts:
        payload["amount_constraints"] = local_amounts
    local_crimes = crime_mentions_for(standalone, messages)
    if local_crimes:
        payload["crime_mentions"] = local_crimes
        for crime in reversed(local_crimes):
            if crime not in standalone:
                standalone = f"{crime} {standalone}"
        payload["standalone_query"] = standalone
        if payload.get("route_label") == "criminal_law_law_only":
            payload["field_intents"] = [
                intent
                for intent in payload.get("field_intents", [])
                if intent != "judgment_text"
            ] or ["legal_basis_text"]
    if is_general_criminal_direct_query(latest):
        direct_route = route_query(latest)
        payload["standalone_query"] = latest
        payload["route_label"] = direct_route.label
        payload["needs_retrieval"] = False
        payload["retrieval_targets"] = ["none"]
        payload["case_required"] = False
        payload["law_required"] = False
        payload["case_name_mentions"] = []
        payload["resolved_case_refs"] = []
        payload["field_intents"] = []
        payload["crime_mentions"] = []
        payload["amount_constraints"] = []
        payload["confidence"] = max(float(payload.get("confidence") or 0.0), direct_route.confidence)
        payload.setdefault("warnings", [])
        if "direct_general_legal_answer" not in payload["warnings"]:
            payload["warnings"].append("direct_general_legal_answer")
    return validate_payload(payload)


@dataclass(frozen=True)
class QueryUnderstandingResult:
    payload: dict[str, Any]
    source: Literal["lora", "rules", "fallback", "disabled"]
    reason: str
    raw_output: str | None = None

    @property
    def standalone_query(self) -> str:
        return str(self.payload.get("standalone_query") or "")

    def route_decision(self) -> RouteDecision:
        route = decision_from_payload(
            {
                "label": self.payload["route_label"],
                "needs_retrieval": self.payload["needs_retrieval"],
                "retrieval_targets": self.payload.get("retrieval_targets") or ["none"],
                "case_required": self.payload["case_required"],
                "law_required": self.payload["law_required"],
                "clarify_required": self.payload["route_label"] == "unclear_need_clarification",
                "confidence": self.payload.get("confidence", 0.8),
                "reason": self.reason,
            },
            source="lora" if self.source == "lora" else ("disabled" if self.source == "disabled" else "rules"),
            raw_output=self.raw_output,
        )
        return route

    def as_log_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "reason": self.reason,
            "standalone_query": self.standalone_query,
            "route_label": self.payload.get("route_label"),
            "retrieval_targets": self.payload.get("retrieval_targets", []),
            "case_name_mentions": self.payload.get("case_name_mentions", []),
            "resolved_case_refs": self.payload.get("resolved_case_refs", []),
            "field_intents": self.payload.get("field_intents", []),
            "crime_mentions": self.payload.get("crime_mentions", []),
            "amount_constraints": self.payload.get("amount_constraints", []),
            "warnings": self.payload.get("warnings", []),
        }


class LocalQueryUnderstandingLora:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def enabled(self) -> bool:
        return (
            settings.query_understanding_mode == "lora"
            and settings.query_understanding_adapter_dir.exists()
            and settings.query_understanding_base_model.exists()
        )

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("query_understanding_lora_dependencies_missing") from exc
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            if not torch.cuda.is_available():
                dtype = torch.float32
            tokenizer = AutoTokenizer.from_pretrained(settings.query_understanding_adapter_dir, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.query_understanding_base_model,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model = PeftModel.from_pretrained(base_model, settings.query_understanding_adapter_dir)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True

    def understand(self, messages: list[dict[str, Any]]) -> QueryUnderstandingResult:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        prompt = tokenizer.apply_chat_template(query_understanding_prompt(messages), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=settings.query_understanding_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        payload = postprocess_payload(validate_payload(extract_json(raw_output)), messages)
        payload["_source"] = "lora"
        payload["_reason"] = "query_understanding_lora"
        return QueryUnderstandingResult(payload, "lora", "query_understanding_lora", raw_output)


local_query_understanding_lora = LocalQueryUnderstandingLora()


def router_input_from_rewrite_payload(payload: dict[str, Any], raw_question: str = "") -> str:
    features = {
        "case_ref_count": len(payload.get("resolved_case_refs") or []),
        "case_name_mentions": payload.get("case_name_mentions") or [],
        "crime_mentions": payload.get("crime_mentions") or [],
        "amount_constraints": [
            {
                "value": item.get("value"),
                "role": item.get("role"),
                "raw_text": item.get("raw_text"),
            }
            for item in payload.get("amount_constraints") or []
            if isinstance(item, dict)
        ],
        "field_intents": payload.get("field_intents") or [],
    }
    parts = [f"改写问题：{payload.get('standalone_query') or ''}"]
    if raw_question:
        parts.append(f"原始用户问题：{raw_question}")
    parts.append(f"Rewriter结构化线索：{json.dumps(features, ensure_ascii=False, separators=(',', ':'))}")
    return "\n".join(parts)


def payload_from_rewrite(rewrite: QueryRewriteResult, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload = dict(rewrite.payload)
    query = payload.get("standalone_query") or latest_user_query(messages)
    router_input = router_input_from_rewrite_payload(payload, latest_user_query(messages))
    route = route_query(router_input)
    resolved = list(payload.get("resolved_case_refs") or [])
    if len(resolved) >= 2 and re.search(r"差异|区别|对比|比较|分别|两者|二者|两个", str(query)):
        route = RouteDecision(
            "criminal_law_multi_case",
            True,
            ["cases", "graph"],
            True,
            False,
            False,
            max(route.confidence, float(payload.get("confidence") or 0.0), 0.82),
            "rules",
            "rewriter_multi_case_completion",
        )
    warnings = list(payload.get("warnings") or [])
    warnings.append(f"query_rewriter:{rewrite.source}")
    return validate_payload(
        {
            "standalone_query": str(query),
            "route_label": route.label,
            "needs_retrieval": route.needs_retrieval,
            "retrieval_targets": route.retrieval_targets,
            "case_required": route.case_required,
            "law_required": route.law_required,
            "case_name_mentions": list(payload.get("case_name_mentions") or []),
            "resolved_case_refs": resolved,
            "field_intents": list(payload.get("field_intents") or []),
            "crime_mentions": list(payload.get("crime_mentions") or []),
            "amount_constraints": list(payload.get("amount_constraints") or []),
            "confidence": max(route.confidence, float(payload.get("confidence") or 0.0)),
            "warnings": warnings,
        }
    )


def understand_query_with_rewriter(messages: list[dict[str, Any]]) -> QueryUnderstandingResult:
    rewrite = rewrite_query(messages)
    payload = postprocess_payload(payload_from_rewrite(rewrite, messages), messages)
    reason = f"{rewrite.reason}_then_route_query"
    return QueryUnderstandingResult(payload, rewrite.source, reason, rewrite.raw_output)


def understand_query(messages: list[dict[str, Any]]) -> QueryUnderstandingResult:
    if settings.query_rewriter_mode in {"lora", "rules"}:
        return understand_query_with_rewriter(messages)
    if settings.query_understanding_mode == "disabled":
        payload = postprocess_payload(validate_payload(rules_understand(messages, reason="query_understanding_disabled")), messages)
        return QueryUnderstandingResult(payload, "disabled", "query_understanding_disabled")
    if settings.query_understanding_mode == "lora":
        try:
            if local_query_understanding_lora.enabled():
                return local_query_understanding_lora.understand(messages)
            payload = postprocess_payload(validate_payload(rules_understand(messages, reason="query_understanding_lora_unavailable")), messages)
            payload["warnings"].append("query_understanding_lora_unavailable")
            return QueryUnderstandingResult(payload, "fallback", "query_understanding_lora_unavailable")
        except Exception as exc:
            payload = postprocess_payload(validate_payload(rules_understand(messages, reason=f"query_understanding_lora_failed:{type(exc).__name__}")), messages)
            payload["warnings"].append(f"query_understanding_lora_failed:{type(exc).__name__}")
            return QueryUnderstandingResult(payload, "fallback", f"query_understanding_lora_failed:{type(exc).__name__}")
    payload = postprocess_payload(validate_payload(rules_understand(messages)), messages)
    return QueryUnderstandingResult(payload, "rules", "rules_query_understanding")
