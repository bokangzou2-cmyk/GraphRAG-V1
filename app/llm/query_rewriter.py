from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Literal

import app.processing_bridge  # noqa: F401
import torch
from amount_utils import amount_query_profile
from query_rewriter_common import (
    extract_json,
    normalize_rewriter_answer,
    query_rewriter_prompt,
    validate_rewriter_answer,
)

from app.config import settings


FIELD_INTENT_RULES = [
    ("evidence_text", re.compile(r"证据|凭什么认定|依据哪些证据")),
    ("defense_text", re.compile(r"辩护|辩解|律师意见")),
    ("court_found_facts_text", re.compile(r"事实|案情|经过|情况|具体.*案")),
    ("reasoning_text", re.compile(r"理由|为什么|法院认为|如何认定|认定|判这么重|差异|区别|对比|比较")),
    ("judgment_text", re.compile(r"判了|怎么判|判多久|量刑|刑期|处罚|结果|差异|区别|对比|比较")),
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
KNOWN_CRIME_NAMES = [
    "盗窃罪",
    "诈骗罪",
    "抢劫罪",
    "故意伤害罪",
    "故意杀人罪",
    "过失致人死亡罪",
    "掩饰、隐瞒犯罪所得罪",
    "掩饰、隐瞒犯罪所得、犯罪所得收益罪",
    "犯罪所得收益罪",
    "寻衅滋事罪",
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
CASE_ORDINAL_RE = re.compile(r"(第[一二三四五六七八九十\d]+个|第[一二三四五六七八九十\d]+|上[一两]个|这个|该)(?:参考)?(?:案例|案子|案件)")
MULTI_CASE_REF_RE = re.compile(r"前面两个|上述两个|这两个|两个(?:参考)?(?:案例|案子|案件)|两[个起]案|二者|两者|分别|差异|区别|对比|比较")
ASSISTANT_CASE_RE = re.compile(
    r"(第[一二三四五六七八九十\d]+个)?(?:参考)?(?:案例|案子|案件)?(?:是|为)?《([^》]{2,120})》"
    r"(?:，?简称([^，。；;\s]{2,30}))?"
    r"(?:，?金额约?([0-9]+(?:\.[0-9]+)?)\s*元)?"
)
NUMBERED_CASE_RE = re.compile(r"(?:^|\n)\s*(?:第?\s*)?([1-9一二三四五六七八九十])(?:[.、）\)]|\s*[：:])\s*([^\n，。；;:：]{2,100}?(?:判决书|案))")
CASE_TITLE_RE = re.compile(r"[\u4e00-\u9fff、]{2,32}(?:盗窃|诈骗|抢劫|故意伤害|掩饰、隐瞒犯罪所得|寻衅滋事)?(?:罪)?(?:一审|二审)?刑事判决书|[\u4e00-\u9fff]{2,12}案")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def previous_assistant_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [str(message.get("content") or "") for message in messages[:-1] if message.get("role") == "assistant"]


def normalize_query_terms(query: str) -> str:
    normalized = normalize_text(query)
    expansions = [target for source, target in QUERY_EQUIVALENTS.items() if source in normalized and target not in normalized]
    if expansions:
        normalized = f"{normalized} {' '.join(expansions)}"
    return normalized


def chinese_ordinal_to_int(text: str | None) -> int | None:
    if not text:
        return None
    if re.search(r"\d+", text):
        return int(re.search(r"\d+", text).group(0))  # type: ignore[union-attr]
    values = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    for key, value in values.items():
        if key in text:
            return value
    return None


def case_ref_indices(query: str, candidate_count: int) -> list[int]:
    if candidate_count <= 0:
        return []
    if MULTI_CASE_REF_RE.search(query):
        return list(range(1, min(candidate_count, 2) + 1))
    match = CASE_ORDINAL_RE.search(query)
    if match:
        index = chinese_ordinal_to_int(match.group(1)) or 1
        return [index] if index <= candidate_count else []
    if re.search(r"为什么.*(?:这么|这样)|判这么|具体.*情况|详细.*情况", query):
        return [1]
    return []


def extract_case_candidates_from_text(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in ASSISTANT_CASE_RE.finditer(text):
        title = normalize_text(match.group(2))
        if not title or title in seen:
            continue
        ordinal = chinese_ordinal_to_int(match.group(1)) or len(candidates) + 1
        item: dict[str, Any] = {"index": ordinal, "case_title": title, "source": "assistant_history"}
        short_title = normalize_text(match.group(3) or "")
        if short_title:
            item["short_title"] = short_title
        amount = match.group(4)
        if amount:
            item["amount"] = int(float(amount))
        seen.add(title)
        candidates.append(item)

    for match in NUMBERED_CASE_RE.finditer(text):
        title = normalize_text(match.group(2))
        title = re.sub(r"[：:].*$", "", title).strip()
        if not title or title in seen:
            continue
        ordinal = chinese_ordinal_to_int(match.group(1)) or len(candidates) + 1
        seen.add(title)
        candidates.append({"index": ordinal, "case_title": title, "source": "assistant_history"})

    candidates.sort(key=lambda item: int(item.get("index") or 999))
    return candidates


def extract_case_candidates(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for content in reversed(previous_assistant_messages(messages)):
        candidates = extract_case_candidates_from_text(content)
        if candidates:
            return candidates
    return []


def extract_case_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for match in CASE_TITLE_RE.finditer(text):
        value = normalize_text(match.group(0))
        if value in {"案例", "案子", "案件", "参考案", "这个案", "该案"} or len(value) < 3:
            continue
        if value not in mentions:
            mentions.append(value)
    return mentions


def infer_crimes_from_text(text: str) -> list[str]:
    crimes: list[str] = []
    for crime, pattern in CRIME_ALIAS_PATTERNS:
        if pattern.search(text) and crime not in crimes:
            crimes.append(crime)
    for crime in KNOWN_CRIME_NAMES:
        if crime in text and crime not in crimes and not any(crime in existing and crime != existing for existing in crimes):
            crimes.append(crime)
    return crimes


def amount_constraints_for(text: str) -> list[dict[str, Any]]:
    profile = amount_query_profile(text)
    values = profile.get("amount_query_values") or []
    role = profile.get("amount_query_role") or "unknown_amount"
    return [{"value": value, "role": role, "raw_text": f"{value:g}元" if isinstance(value, float) else f"{value}元"} for value in values]


def field_intents_for(text: str, resolved: list[dict[str, Any]]) -> list[str]:
    intents = [field for field, pattern in FIELD_INTENT_RULES if pattern.search(text)]
    if resolved and not intents:
        intents = ["court_found_facts_text", "reasoning_text", "judgment_text"]
    return intents


def append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def ensure_rewriter_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "standalone_query",
        "resolved_case_refs",
        "case_name_mentions",
        "crime_mentions",
        "amount_constraints",
        "field_intents",
        "rewrite_required",
        "confidence",
        "warnings",
    }
    cleaned = {key: value for key, value in payload.items() if key in allowed}
    cleaned.setdefault("standalone_query", "")
    cleaned.setdefault("resolved_case_refs", [])
    cleaned.setdefault("case_name_mentions", [])
    cleaned.setdefault("crime_mentions", [])
    cleaned.setdefault("amount_constraints", [])
    cleaned.setdefault("field_intents", [])
    cleaned.setdefault("rewrite_required", False)
    cleaned.setdefault("confidence", 0.85)
    cleaned.setdefault("warnings", [])
    return normalize_rewriter_answer(cleaned)


def rule_rewrite(messages: list[dict[str, Any]], reason: str = "rules_query_rewriter") -> dict[str, Any]:
    latest = normalize_query_terms(latest_user_query(messages))
    candidates = extract_case_candidates(messages)
    indices = case_ref_indices(latest, len(candidates))
    resolved = [
        {"ref": latest, **candidate}
        for candidate in candidates
        if int(candidate.get("index") or 0) in indices
    ]
    standalone = latest
    if resolved:
        titles = [str(item["case_title"]) for item in resolved]
        intent_terms = " ".join(field_intents_for(latest, resolved))
        if len(titles) >= 2:
            standalone = f"{' 与 '.join(titles)} {latest} {intent_terms}"
        else:
            standalone = f"{titles[0]} {latest} {intent_terms}"
    payload = {
        "standalone_query": normalize_query_terms(standalone),
        "resolved_case_refs": resolved,
        "case_name_mentions": extract_case_mentions(standalone),
        "crime_mentions": infer_crimes_from_text(f"{latest} {standalone}"),
        "amount_constraints": amount_constraints_for(latest) or amount_constraints_for(standalone),
        "field_intents": field_intents_for(latest, resolved),
        "rewrite_required": bool(resolved or standalone != latest),
        "confidence": 0.75,
        "warnings": [reason],
    }
    return ensure_rewriter_payload_shape(payload)


def complete_rewriter_payload(payload: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    latest = normalize_query_terms(latest_user_query(messages))
    completed = ensure_rewriter_payload_shape(payload)
    warnings = completed["warnings"]
    standalone = normalize_query_terms(str(completed.get("standalone_query") or latest))
    candidates = extract_case_candidates(messages)
    indices = case_ref_indices(latest, len(candidates))

    if indices:
        existing_by_index = {
            int(item.get("index") or 0): item
            for item in completed.get("resolved_case_refs", [])
            if isinstance(item, dict) and item.get("index")
        }
        for candidate in candidates:
            index = int(candidate.get("index") or 0)
            if index not in indices or index in existing_by_index:
                continue
            completed["resolved_case_refs"].append({"ref": latest, **candidate})
            append_unique(warnings, "rewriter_rule_completed_case_ref")

        target_titles = [
            str(item.get("case_title") or "")
            for item in sorted(completed["resolved_case_refs"], key=lambda item: int(item.get("index") or 999))
            if int(item.get("index") or 0) in indices and item.get("case_title")
        ]
        for item in completed["resolved_case_refs"]:
            title = str(item.get("case_title") or "")
            short_title = str(item.get("short_title") or "")
            if title:
                append_unique(completed["case_name_mentions"], title)
            if short_title:
                append_unique(completed["case_name_mentions"], short_title)
        missing_titles = [title for title in target_titles if title and title not in standalone]
        if missing_titles or (len(target_titles) >= 2 and not re.search(r"差异|区别|对比|比较", standalone)):
            standalone = normalize_text(f"{' 与 '.join(target_titles)} {latest} {standalone}")
            append_unique(warnings, "rewriter_rule_completed_standalone_cases")

    local_amounts = amount_constraints_for(latest) or amount_constraints_for(standalone)
    if local_amounts:
        completed["amount_constraints"] = local_amounts

    crimes = infer_crimes_from_text(f"{latest} {standalone}")
    if crimes:
        completed["crime_mentions"] = [item for item in completed["crime_mentions"] if item in KNOWN_CRIME_NAMES]
        for crime in crimes:
            append_unique(completed["crime_mentions"], crime)
        for crime in reversed(crimes):
            if crime not in standalone:
                standalone = f"{crime} {standalone}"

    inferred_intents = field_intents_for(f"{latest} {standalone}", completed["resolved_case_refs"])
    for intent in inferred_intents:
        append_unique(completed["field_intents"], intent)

    completed["standalone_query"] = normalize_query_terms(standalone)
    completed["rewrite_required"] = bool(completed["rewrite_required"] or completed["standalone_query"] != latest)
    return ensure_rewriter_payload_shape(completed)


@dataclass(frozen=True)
class QueryRewriteResult:
    payload: dict[str, Any]
    source: Literal["lora", "rules", "fallback", "disabled"]
    reason: str
    raw_output: str | None = None
    metrics: dict[str, Any] | None = None

    @property
    def standalone_query(self) -> str:
        return str(self.payload.get("standalone_query") or "")

    def as_log_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "reason": self.reason,
            "standalone_query": self.standalone_query,
            "resolved_case_refs": self.payload.get("resolved_case_refs", []),
            "case_name_mentions": self.payload.get("case_name_mentions", []),
            "field_intents": self.payload.get("field_intents", []),
            "crime_mentions": self.payload.get("crime_mentions", []),
            "amount_constraints": self.payload.get("amount_constraints", []),
            "warnings": self.payload.get("warnings", []),
        }


class LocalQueryRewriterLora:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def enabled(self) -> bool:
        return (
            settings.query_rewriter_mode == "lora"
            and settings.query_rewriter_adapter_dir.exists()
            and settings.query_rewriter_base_model.exists()
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
                raise RuntimeError("query_rewriter_lora_dependencies_missing") from exc
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            if not torch.cuda.is_available():
                dtype = torch.float32
            tokenizer = AutoTokenizer.from_pretrained(settings.query_rewriter_adapter_dir, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.query_rewriter_base_model,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model = PeftModel.from_pretrained(base_model, settings.query_rewriter_adapter_dir)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True

    def rewrite(self, messages: list[dict[str, Any]]) -> QueryRewriteResult:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        prompt = tokenizer.apply_chat_template(query_rewriter_prompt(messages), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=settings.query_rewriter_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        payload = complete_rewriter_payload(extract_json(raw_output), messages)
        errors = validate_rewriter_answer(payload)
        if errors:
            raise ValueError(f"query_rewriter_invalid_payload:{','.join(errors)}")
        metrics = {
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "completion_tokens": int(generated.shape[-1]),
            "total_tokens": int(inputs["input_ids"].shape[-1] + generated.shape[-1]),
        }
        return QueryRewriteResult(payload, "lora", "query_rewriter_lora", raw_output, metrics)


local_query_rewriter_lora = LocalQueryRewriterLora()


def rewrite_query(messages: list[dict[str, Any]]) -> QueryRewriteResult:
    if settings.query_rewriter_mode == "disabled":
        payload = complete_rewriter_payload(rule_rewrite(messages, reason="query_rewriter_disabled"), messages)
        return QueryRewriteResult(payload, "disabled", "query_rewriter_disabled")
    if settings.query_rewriter_mode == "lora":
        try:
            if local_query_rewriter_lora.enabled():
                return local_query_rewriter_lora.rewrite(messages)
            payload = complete_rewriter_payload(rule_rewrite(messages, reason="query_rewriter_lora_unavailable"), messages)
            append_unique(payload["warnings"], "query_rewriter_lora_unavailable")
            return QueryRewriteResult(payload, "fallback", "query_rewriter_lora_unavailable")
        except Exception as exc:
            payload = complete_rewriter_payload(rule_rewrite(messages, reason=f"query_rewriter_lora_failed:{type(exc).__name__}"), messages)
            append_unique(payload["warnings"], f"query_rewriter_lora_failed:{type(exc).__name__}")
            return QueryRewriteResult(payload, "fallback", f"query_rewriter_lora_failed:{type(exc).__name__}")
    payload = complete_rewriter_payload(rule_rewrite(messages), messages)
    return QueryRewriteResult(payload, "rules", "rules_query_rewriter")
