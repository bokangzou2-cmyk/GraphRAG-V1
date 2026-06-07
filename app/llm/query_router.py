from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Literal

import torch

from app.config import settings


RouterLabel = Literal[
    "general_no_legal_retrieval",
    "criminal_law_general_direct",
    "non_criminal_legal",
    "criminal_law_law_only",
    "criminal_law_case_only",
    "criminal_law_case_and_law",
    "criminal_law_multi_case",
    "criminal_law_advice",
    "unclear_need_clarification",
    "low_quality_or_incomplete",
]

VALID_LABELS = {
    "general_no_legal_retrieval",
    "criminal_law_general_direct",
    "non_criminal_legal",
    "criminal_law_law_only",
    "criminal_law_case_only",
    "criminal_law_case_and_law",
    "criminal_law_multi_case",
    "criminal_law_advice",
    "unclear_need_clarification",
    "low_quality_or_incomplete",
}

SYSTEM_PROMPT = (
    "你是一个刑法问答系统的问题路由分类器。"
    "只根据用户问题判断是否需要法律检索以及需要检索什么。"
    "必须只输出一个 JSON 对象，不要解释，不要输出 Markdown。"
)

LABEL_DESCRIPTIONS = {
    "general_no_legal_retrieval": "普通非法律问题，不触发法律检索。",
    "criminal_law_general_direct": "刑法/刑事法律常识、法典元信息或概念百科问题，不依赖本项目样本库，直接由 LLM 回答。",
    "non_criminal_legal": "非刑法法律问题，本刑法系统不触发检索。",
    "criminal_law_law_only": "刑法概念、构成要件、量刑规则、程序规则等，只需检索法条或规范。",
    "criminal_law_case_only": "明确要求案例、判例、类案或裁判观点，主要检索案例。",
    "criminal_law_case_and_law": "刑法问题同时需要法条依据和案例辅助。",
    "criminal_law_multi_case": "要求多个案例、类案比较、趋势归纳或案例群分析。",
    "criminal_law_advice": "具体个人刑事处境咨询，优先检索法条，必要时由后续链路再决定是否扩展案例。",
    "unclear_need_clarification": "问题语义不清或缺少关键信息，需要澄清。",
    "low_quality_or_incomplete": "问题残缺、噪声过多或不可判断。",
}

CRIMINAL_RE = re.compile(
    r"刑法|刑事|犯罪|罪名|构成要件|量刑|判刑|有期徒刑|拘役|缓刑|死刑|罚金|盗窃|诈骗|抢劫|故意伤害|"
    r"强奸|放火|贩毒|走私|帮信|寻衅滋事|取保候审|逮捕|公诉|检察院|公安机关|自首|立功"
)
CASE_RE = re.compile(r"案例|判例|类案|裁判|法院认为|判决书|案情|同类案件|案件通常|哪些案件|网红|某案")
CASE_NAME_RE = re.compile(r"[\u4e00-\u9fff]{2,12}案")
NEGATIVE_CASE_RE = re.compile(r"不要查案例|不用查案例|不查案例|别查案例|不要案例|不用案例|别用案例|不要具体案子|不用具体案子|别展开案例")
MULTI_CASE_RE = re.compile(r"多个|类案|同类|比较|汇总|趋势|通常|哪些案件|案例群")
LAW_RE = re.compile(r"法条|规定|依据|标准|构成要件|第.+条|怎么判|如何处罚|怎么处罚|判多久")
NON_CRIMINAL_LEGAL_RE = re.compile(r"合同|离婚|继承|劳动|工伤|仲裁|行政|民事|房产|租赁|公司法|税|专利|商标|著作权")
LOW_QUALITY_RE = re.compile(r"^\W*$|不知道.*什么|这个是啥东西吗")
GENERAL_CRIMINAL_DIRECT_RE = re.compile(
    r"刑法(?:一共|总共)?(?:有)?(?:多少|几)(?:条|个条文)|"
    r"什么是刑法|刑法是什么|刑法(?:和|与).{0,8}(?:民法|行政法|治安管理处罚法).{0,8}(?:区别|不同)|"
    r"刑法(?:总则|分则)(?:是什么|包括什么|什么意思|区别)|"
    r"刑法(?:主要)?(?:管|规制|调整|处罚|惩罚|打击)(?:什么|啥|哪些|哪类|哪几类)$|"
    r"刑法(?:主要)?(?:管|规制|调整|处罚|惩罚|打击)(?:什么|啥|哪些|哪类|哪几类)(?:，?简单说|，?通俗说|，?概括一下)?|"
    r"刑法(?:主要)?(?:涉及|管|规制|调整|处罚|惩罚|打击)(?:哪些|什么|哪类|哪几类)?.{0,8}(?:行为|违法行为|犯罪|犯罪行为|事情)|"
    r"刑法(?:主要)?(?:处罚|惩罚|打击)(?:什么|哪些|哪类|哪几类)|"
    r"(?:哪些|什么|哪类|哪几类).{0,8}(?:行为|违法行为|犯罪行为|犯罪)(?:属于|算|构成).{0,4}(?:刑事犯罪|犯罪)|"
    r"(?:犯罪|刑事犯罪)(?:主要)?(?:分为|包括|包含|有哪些|有哪几类)|"
    r"刑罚(?:种类|有哪些|是什么)|犯罪构成(?:四要件|三阶层|是什么|有哪些)|"
    r"刑法(?:什么时候|哪年)(?:施行|生效|颁布)|"
    r"刑法的(?:基本原则|任务|适用范围)(?:是什么|有哪些)"
)
SPECIFIC_RETRIEVAL_RE = re.compile(
    r"第.+条|法条|对应哪条|适用|依据|案例|判例|类案|判决书|案情|证据|辩护|法院认为|裁判理由|"
    r"怎么判|判几年|判多久|量刑|处罚|缓刑|自首|立功|数额|金额|入户|入室"
)
GENERAL_DIRECT_BLOCK_RE = re.compile(
    r"第.+条|法条|对应哪条|适用|依据|案例|判例|类案|判决书|案情|证据|辩护|法院认为|裁判理由|"
    r"怎么判|判几年|判多久|量刑|缓刑|自首|立功|数额|金额|入户|入室|盗窃|诈骗|抢劫|故意伤害|故意杀人|过失致人死亡"
)
RAW_USER_QUESTION_RE = re.compile(r"(?:^|\n)原始用户问题：([^\n]+)")


@dataclass(frozen=True)
class RouteDecision:
    label: RouterLabel
    needs_retrieval: bool
    retrieval_targets: list[str]
    case_required: bool
    law_required: bool
    clarify_required: bool
    confidence: float
    source: Literal["lora", "rules", "disabled", "fallback"]
    reason: str = ""
    raw_output: str | None = None

    @property
    def source_types(self) -> list[str] | None:
        if not self.needs_retrieval:
            return []
        targets = set(self.retrieval_targets)
        types: list[str] = []
        if "law_articles" in targets:
            types.append("law_article")
        if "cases" in targets or "graph" in targets:
            types.append("case_chunk")
        return types or None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "needs_retrieval": self.needs_retrieval,
            "retrieval_targets": self.retrieval_targets,
            "case_required": self.case_required,
            "law_required": self.law_required,
            "clarify_required": self.clarify_required,
            "confidence": self.confidence,
            "source": self.source,
            "reason": self.reason,
        }


def build_user_prompt(question: str) -> str:
    labels = "\n".join(f"- {label}: {description}" for label, description in LABEL_DESCRIPTIONS.items())
    return (
        "请对下面的用户问题或 Rewriter 结构化输入做路由分类。\n\n"
        f"可选标签：\n{labels}\n\n"
        "输出 JSON 字段固定为："
        "label, needs_retrieval, retrieval_targets, case_required, law_required, clarify_required。\n\n"
        f"Router 输入：{question}"
    )


def parse_router_json(text: str) -> dict:
    payload = text.strip()
    if not payload.startswith("{"):
        match = re.search(r"\{.*\}", payload, flags=re.S)
        if not match:
            raise ValueError("router_output_not_json")
        payload = match.group(0)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("router_output_not_object")
    label = data.get("label")
    if label not in VALID_LABELS:
        raise ValueError(f"router_unknown_label:{label}")
    return data


def decision_from_payload(payload: dict, *, source: RouteDecision.__annotations__["source"], raw_output: str | None = None) -> RouteDecision:
    label = payload["label"]
    return RouteDecision(
        label=label,
        needs_retrieval=bool(payload.get("needs_retrieval")),
        retrieval_targets=list(payload.get("retrieval_targets") or ["none"]),
        case_required=bool(payload.get("case_required")),
        law_required=bool(payload.get("law_required")),
        clarify_required=bool(payload.get("clarify_required")),
        confidence=float(payload.get("confidence", 0.9)),
        source=source,
        reason=str(payload.get("reason") or ""),
        raw_output=raw_output,
    )


def extract_original_question(router_input: str) -> str:
    match = RAW_USER_QUESTION_RE.search(router_input)
    if not match:
        return router_input
    return match.group(1).strip() or router_input


def rule_route(question: str, *, source: RouteDecision.__annotations__["source"] = "rules", reason: str = "rule_router") -> RouteDecision:
    query = question.strip()
    if not query or LOW_QUALITY_RE.search(query):
        return RouteDecision("low_quality_or_incomplete", False, ["none"], False, False, False, 0.75, source, reason)
    has_negative_case_constraint = bool(NEGATIVE_CASE_RE.search(query))
    if GENERAL_CRIMINAL_DIRECT_RE.search(query) and (not GENERAL_DIRECT_BLOCK_RE.search(query) or has_negative_case_constraint):
        return RouteDecision("criminal_law_general_direct", False, ["none"], False, False, False, 0.84, source, reason)
    has_case_name = bool(CASE_NAME_RE.search(query))
    has_positive_case_signal = bool(CASE_RE.search(query) or has_case_name) and not has_negative_case_constraint
    if CRIMINAL_RE.search(query) or has_case_name:
        if MULTI_CASE_RE.search(query) and not has_negative_case_constraint:
            return RouteDecision("criminal_law_multi_case", True, ["cases", "graph"], True, False, False, 0.82, source, reason)
        if has_positive_case_signal and LAW_RE.search(query):
            return RouteDecision("criminal_law_case_and_law", True, ["law_articles", "cases"], True, True, False, 0.82, source, reason)
        if has_positive_case_signal:
            return RouteDecision("criminal_law_case_only", True, ["cases"], True, False, False, 0.82, source, reason)
        return RouteDecision("criminal_law_law_only", True, ["law_articles"], False, True, False, 0.86, source, reason)
    if NON_CRIMINAL_LEGAL_RE.search(query):
        return RouteDecision("non_criminal_legal", False, ["none"], False, False, False, 0.78, source, reason)
    return RouteDecision("general_no_legal_retrieval", False, ["none"], False, False, False, 0.8, source, reason)


class LocalLoraRouter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def enabled(self) -> bool:
        return settings.router_mode == "lora" and settings.router_adapter_dir.exists() and settings.router_base_model.exists()

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
                raise RuntimeError("router_lora_dependencies_missing") from exc
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            if not torch.cuda.is_available():
                dtype = torch.float32
            tokenizer = AutoTokenizer.from_pretrained(settings.router_adapter_dir, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.router_base_model,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model = PeftModel.from_pretrained(base_model, settings.router_adapter_dir)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True

    def route(self, question: str) -> RouteDecision:
        self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
        ]
        tokenizer = self._tokenizer
        model = self._model
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=settings.router_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        payload = parse_router_json(text)
        return decision_from_payload(payload, source="lora", raw_output=text)


local_lora_router = LocalLoraRouter()


def route_query(question: str) -> RouteDecision:
    if settings.router_mode == "disabled":
        return RouteDecision("criminal_law_case_and_law", True, ["law_articles", "cases"], True, True, False, 0.0, "disabled", "router_disabled")
    original_question = extract_original_question(question)
    priority_rule = rule_route(original_question, source="rules", reason="priority_general_criminal_direct")
    if priority_rule.label == "criminal_law_general_direct":
        return priority_rule
    if priority_rule.label in {
        "criminal_law_law_only",
        "criminal_law_case_only",
        "criminal_law_case_and_law",
        "criminal_law_multi_case",
        "criminal_law_advice",
    }:
        return RouteDecision(
            priority_rule.label,
            priority_rule.needs_retrieval,
            priority_rule.retrieval_targets,
            priority_rule.case_required,
            priority_rule.law_required,
            priority_rule.clarify_required,
            priority_rule.confidence,
            priority_rule.source,
            "priority_explicit_retrieval_intent",
        )
    if settings.router_mode == "lora":
        try:
            if local_lora_router.enabled():
                return local_lora_router.route(question)
        except Exception as exc:
            return rule_route(question, source="fallback", reason=f"lora_router_failed:{type(exc).__name__}")
        return rule_route(question, source="fallback", reason="lora_router_unavailable")
    return rule_route(question)
