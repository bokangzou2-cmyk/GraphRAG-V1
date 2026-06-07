from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common import OUT_DIR, ROOT, iter_jsonl

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm.query_rewriter import complete_rewriter_payload, rewrite_query, rule_rewrite  # noqa: E402
from app.llm.query_router import LABEL_DESCRIPTIONS, VALID_LABELS  # noqa: E402


DEFAULT_OLD_ROUTER_DATASET = OUT_DIR / "router_lora" / "dataset"
DEFAULT_SUBAGENT_SEED = OUT_DIR / "query_router_lora" / "subagent_router_seed_proposal.json"
DEFAULT_OUTPUT_DIR = OUT_DIR / "router_lora" / "dataset_v2_rewriter_processed"
SCHEMA_VERSION = "router-lora-rewriter-processed-v2"
REWRITE_MODE = "rules"

SYSTEM_PROMPT = (
    "你是刑法 GraphRAG 的 Router，只负责判断是否需要检索以及检索什么。"
    "输入已经由 Query Rewriter 改写并补齐结构化线索。"
    "不要改写问题，不要抽取案名、罪名、金额或字段意图。"
    "必须只输出一个 JSON 对象，不要 Markdown，不要解释。"
)

ANSWER_BY_LABEL = {
    "general_no_legal_retrieval": {
        "label": "general_no_legal_retrieval",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": False,
    },
    "criminal_law_general_direct": {
        "label": "criminal_law_general_direct",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": False,
    },
    "non_criminal_legal": {
        "label": "non_criminal_legal",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": False,
    },
    "criminal_law_law_only": {
        "label": "criminal_law_law_only",
        "needs_retrieval": True,
        "retrieval_targets": ["law_articles"],
        "case_required": False,
        "law_required": True,
        "clarify_required": False,
    },
    "criminal_law_case_only": {
        "label": "criminal_law_case_only",
        "needs_retrieval": True,
        "retrieval_targets": ["cases"],
        "case_required": True,
        "law_required": False,
        "clarify_required": False,
    },
    "criminal_law_case_and_law": {
        "label": "criminal_law_case_and_law",
        "needs_retrieval": True,
        "retrieval_targets": ["law_articles", "cases"],
        "case_required": True,
        "law_required": True,
        "clarify_required": False,
    },
    "criminal_law_multi_case": {
        "label": "criminal_law_multi_case",
        "needs_retrieval": True,
        "retrieval_targets": ["cases", "graph"],
        "case_required": True,
        "law_required": False,
        "clarify_required": False,
    },
    "criminal_law_advice": {
        "label": "criminal_law_advice",
        "needs_retrieval": True,
        "retrieval_targets": ["law_articles"],
        "case_required": False,
        "law_required": True,
        "clarify_required": False,
    },
    "unclear_need_clarification": {
        "label": "unclear_need_clarification",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": True,
    },
    "low_quality_or_incomplete": {
        "label": "low_quality_or_incomplete",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": False,
    },
}

TARGET_GENERATED_COUNTS = {
    "general_no_legal_retrieval": 520,
    "criminal_law_general_direct": 620,
    "non_criminal_legal": 560,
    "criminal_law_law_only": 780,
    "criminal_law_case_only": 760,
    "criminal_law_case_and_law": 820,
    "criminal_law_multi_case": 760,
    "criminal_law_advice": 620,
    "unclear_need_clarification": 360,
    "low_quality_or_incomplete": 320,
}

CRIMES = ["盗窃罪", "诈骗罪", "抢劫罪", "故意伤害罪", "故意杀人罪", "过失致人死亡罪", "寻衅滋事罪", "掩饰、隐瞒犯罪所得罪"]
AMOUNTS = ["4000元", "12000元", "30000元", "40000元", "68000元", "15万元", "28万元"]
CASE_TITLES = [
    "王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书",
    "刘方祥、刘礼芬盗窃一审刑事判决书",
    "吴必定盗窃罪一审刑事判决书",
    "张某诈骗罪一审刑事判决书",
    "李某故意伤害罪二审刑事判决书",
    "陈某抢劫罪一审刑事判决书",
]
CASE_SHORTS = ["王士东案", "刘方祥案", "吴必定案", "张某案", "李某案", "陈某案"]
CONTEXT_VARIANTS = [
    "请只判断检索范围",
    "先不要回答实体问题",
    "用于刑法问答系统",
    "我只需要系统先判断资料来源",
    "问题来自多轮对话后的当前轮",
    "用户希望得到简明结论",
    "不要扩展到民事责任",
    "重点区分法条和案例",
    "如果需要资料再进入检索",
    "保留当前问题原意",
    "这是检索前的路由判断",
    "需要避免过度检索",
    "不要把常识问题误判成类案检索",
    "注意是否有明确案例意图",
    "注意是否需要法条依据",
    "注意是否是多个案例比较",
]
FACT_VARIANTS = [
    "有自首情节",
    "已经退赔",
    "取得被害人谅解",
    "初犯偶犯",
    "存在共同犯罪",
    "被告人是从犯",
    "被告人认罪认罚",
    "没有退赔",
    "未成年人参与",
    "有前科劣迹",
    "犯罪未遂",
    "数额接近量刑档次边界",
    "被害人有过错",
    "证据主要是被告人供述",
    "存在电子数据证据",
    "一审后上诉",
]
PLACE_VARIANTS = [
    "基层法院",
    "中级法院",
    "东部地区",
    "西部地区",
    "同一法院",
    "不同法院",
    "近年判决",
    "一审判决",
    "二审裁判",
    "公开裁判文书",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def cycle_pick(values: list[str], index: int) -> str:
    return values[index % len(values)]


def answer_for(label: str) -> dict[str, Any]:
    answer = dict(ANSWER_BY_LABEL[label])
    answer["confidence"] = 0.88 if answer["needs_retrieval"] else 0.82
    answer["reason"] = label
    return answer


def router_input_from_rewriter(payload: dict[str, Any], raw_question: str | None = None) -> str:
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
    parts.append(f"Rewriter结构化线索：{compact_json(features)}")
    return "\n".join(parts)


def build_user_prompt(router_input: str) -> str:
    labels = "\n".join(f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in sorted(VALID_LABELS))
    return (
        "请根据 Rewriter 输出后的问题和结构化线索做 Router 分类。\n\n"
        f"可选标签：\n{labels}\n\n"
        "输出 JSON 字段固定为："
        "label, needs_retrieval, retrieval_targets, case_required, law_required, clarify_required, confidence, reason。\n\n"
        f"Router 输入：\n{router_input}"
    )


def build_messages(router_input: str, answer: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(router_input)},
        {"role": "assistant", "content": compact_json(answer)},
    ]


def messages_for_raw_question(raw_question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": raw_question}]


def multi_case_messages(user_question: str, first_index: int = 0) -> list[dict[str, str]]:
    first_title = cycle_pick(CASE_TITLES, first_index)
    second_title = cycle_pick(CASE_TITLES, first_index + 1)
    first_short = cycle_pick(CASE_SHORTS, first_index)
    second_short = cycle_pick(CASE_SHORTS, first_index + 1)
    return [
        {"role": "user", "content": f"我想找几个{cycle_pick(CRIMES, first_index)}案例参考"},
        {
            "role": "assistant",
            "content": (
                f"可以看两个参考案例。第一个参考案例是《{first_title}》，简称{first_short}，金额约{cycle_pick(AMOUNTS, first_index)}。"
                f"第二个参考案例是《{second_title}》，简称{second_short}，金额约{cycle_pick(AMOUNTS, first_index + 1)}。"
            ),
        },
        {"role": "user", "content": user_question},
    ]


def generated_question(label: str, index: int) -> tuple[list[dict[str, str]], str]:
    crime = cycle_pick(CRIMES, index)
    amount = cycle_pick(AMOUNTS, index)
    case_title = cycle_pick(CASE_TITLES, index)
    case_short = cycle_pick(CASE_SHORTS, index)
    variant = cycle_pick(CONTEXT_VARIANTS, index // 3)
    fact = cycle_pick(FACT_VARIANTS, index // 2)
    place = cycle_pick(PLACE_VARIANTS, index // 5)
    if label == "general_no_legal_retrieval":
        topics = ["晚餐菜单", "项目周报", "Python 列表去重", "高铁行程", "学习计划", "Excel 公式", "PPT 大纲", "健身安排", "旅游攻略", "邮件润色"]
        asks = ["怎么安排", "帮我写一下", "给个简短建议", "列三个选项", "做个模板", "用口语化表达", "整理成清单", "说明注意点"]
        qs = [f"{topic}{ask}，{variant}" for topic in topics for ask in asks]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_general"
    if label == "criminal_law_general_direct":
        concepts = ["刑法", "刑罚种类", "犯罪构成四要件", "刑法总则和分则", "自首和立功", "主刑和附加刑", "犯罪未遂", "共同犯罪", "单位犯罪", "正当防卫", "紧急避险", "追诉时效", "缓刑制度", "累犯", "数罪并罚"]
        asks = ["是什么", "有哪些基本含义", "用通俗话解释", "和相近概念有什么区别", "一般怎么理解", "在刑法体系里是什么位置", "能不能给个常识性说明", "不结合具体案例怎么解释"]
        qs = [f"{concept}{ask}，{variant}" for concept in concepts for ask in asks]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_criminal_general"
    if label == "non_criminal_legal":
        domains = ["离婚财产分割", "劳动仲裁材料", "租房押金退还", "商标抢注", "交通事故民事赔偿", "工伤认定", "合同违约金", "房屋买卖纠纷", "公司股权转让", "著作权侵权", "行政处罚复议", "遗产继承"]
        asks = ["怎么处理", "需要哪些材料", "一般看哪些规则", "能不能起诉", "怎么计算", "走什么程序", "有哪些风险点", "请给我法律方向"]
        qs = [f"{domain}{ask}，{variant}" for domain in domains for ask in asks]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_non_criminal_legal"
    if label == "criminal_law_law_only":
        asks = ["的构成要件是什么", f"{amount}对应什么量刑标准", "自首可以从轻到什么程度", "适用缓刑需要哪些条件", "的法条依据是哪几条", "既遂和未遂怎么区分", "共同犯罪中主从犯怎么认定", "数额较大和巨大怎么判断"]
        qs = [f"{crime}{ask}，补充语境：{fact}，{variant}" for crime in CRIMES for amount in AMOUNTS for ask in asks for fact in FACT_VARIANTS[:8]]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_law_only"
    if label == "criminal_law_case_only":
        asks = [f"找几个{crime}{amount}的类似案例", f"有没有{crime}判缓刑的判决书", f"{case_short}法院认定事实是什么", f"检索{case_title}的裁判理由", f"{crime}退赔后从轻的案例有哪些", f"{crime}有谅解书的判决结果", f"{case_short}证据采信情况", f"{crime}{amount}法院通常怎么判的案例"]
        qs = [f"{ask}，筛选语境：{fact}，{place}，{variant}" for ask in asks for fact in FACT_VARIANTS for place in PLACE_VARIANTS[:4]]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_case_only"
    if label == "criminal_law_case_and_law":
        asks = [f"{crime}{amount}一般怎么判，有没有案例和法条依据", f"我朋友涉嫌{crime}，想看量刑标准和类似判决", f"{case_short}为什么这么判，对应的刑法条文是什么", f"入户盗窃{amount}怎么定罪量刑，给我法条和案例", f"{crime}有自首退赔情节时，法条和类案怎么支持从轻", f"{crime}{amount}能不能缓刑，查法条也查案例", f"{case_title}适用了哪些条文，类案是否一致", f"{crime}主从犯量刑规则和判决样本都要看"]
        qs = [f"{ask}，补充语境：{fact}，{place}，{variant}" for ask in asks for fact in FACT_VARIANTS for place in PLACE_VARIANTS[:5]]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_case_and_law"
    if label == "criminal_law_multi_case":
        qs = [
            f"前面两个案例量刑差异在哪里，{variant}",
            f"这两个案子分别为什么这么判，{variant}",
            f"汇总几个{crime}{amount}案件的裁判理由差异，{fact}，{place}，{variant}",
            f"比较{case_short}和另一个类似案件的量刑因素，{fact}，{place}，{variant}",
            f"找多个{crime}类案，看法院通常怎么处理退赔，{fact}，{place}，{variant}",
            f"三个{crime}案例中自首和退赔对量刑影响有什么不同，{fact}，{place}，{variant}",
            f"同类{crime}{amount}判缓刑和实刑的案例差别，{fact}，{place}，{variant}",
            f"把几个{crime}判决书按裁判理由做对比，{fact}，{place}，{variant}",
        ]
        question = cycle_pick(qs, index)
        if "前面两个" in question or "这两个" in question:
            return multi_case_messages(question, index), "gpt_direct_multi_turn_multi_case"
        return messages_for_raw_question(question), "gpt_direct_multi_case"
    if label == "criminal_law_advice":
        qs = [
            f"我朋友涉嫌{crime}{amount}，现在应该重点看哪些刑法规定，{fact}，{variant}",
            f"家人因为{crime}被刑拘，想知道可能涉及哪些处罚规则，{fact}，{variant}",
            f"{crime}已经退赔并取得谅解，法律上可能怎么处理，{fact}，{variant}",
            f"被公安说涉嫌帮信罪，想了解刑法上风险点，{fact}，{variant}",
            f"取保候审期间又被叫去问话，刑事上要注意什么法律后果，{fact}，{variant}",
            f"亲属涉嫌{crime}，只想先知道刑法上可能的责任范围，{fact}，{variant}",
            f"{crime}{amount}如果主动投案，刑法上可能有哪些从宽规则，{fact}，{variant}",
            f"收到刑事传唤说涉及{crime}，先查哪些法律规则，{fact}，{variant}",
        ]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_criminal_advice"
    if label == "unclear_need_clarification":
        subjects = ["这个", "他这样", "前面那个", "这种情况", "案子", "我朋友这个事", "刚才说的", "那种行为", "这种钱", "那个结果"]
        asks = ["会不会判", "算不算", "严不严重", "有没有问题", "该怎么弄", "要不要查", "能不能处理", "是不是犯罪", "怎么判断"]
        qs = [f"{subject}{ask}，但事实还没说清，{variant}" for subject in subjects for ask in asks]
        return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_unclear"
    noise = ["？？？？", "判 判 判 什么", "刑法那个那个", "帮我看看", "123123案案案", "犯罪犯罪犯罪", "金额金额金额", "第几条第几条", "这个这个这个", "啊啊啊怎么判"]
    tails = ["", "。", "，，", "  ", "？？", f" {index}", f" {variant}"]
    qs = [f"{n}{tail}" for n in noise for tail in tails]
    return messages_for_raw_question(cycle_pick(qs, index)), "gpt_direct_low_quality"


def rewrite_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    if REWRITE_MODE == "lora":
        return rewrite_query(messages).payload
    return complete_rewriter_payload(rule_rewrite(messages), messages)


def build_row(
    *,
    row_id: str,
    label: str,
    source_messages: list[dict[str, str]],
    source: str,
    source_method: str,
) -> dict[str, Any]:
    rewrite_payload = rewrite_messages(source_messages)
    router_input = router_input_from_rewriter(rewrite_payload, raw_question)
    answer = answer_for(label)
    raw_question = next((m["content"] for m in reversed(source_messages) if m.get("role") == "user"), "")
    return {
        "id": row_id,
        "raw_question": raw_question,
        "source_messages": source_messages,
        "rewriter": rewrite_payload,
        "router_input": router_input,
        "label": label,
        "answer": answer,
        "messages": build_messages(router_input, answer),
        "source": source,
        "source_method": source_method,
        "text_hash": stable_hash(router_input),
    }


def build_row_from_rewriter_payload(
    *,
    row_id: str,
    label: str,
    raw_question: str,
    source_messages: list[dict[str, str]],
    rewrite_payload: dict[str, Any],
    source: str,
    source_method: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    rewrite_payload = complete_rewriter_payload(rewrite_payload, source_messages)
    router_input = router_input_from_rewriter(rewrite_payload, raw_question)
    answer = answer_for(label)
    row = {
        "id": row_id,
        "raw_question": raw_question,
        "source_messages": source_messages,
        "rewriter": rewrite_payload,
        "router_input": router_input,
        "label": label,
        "answer": answer,
        "messages": build_messages(router_input, answer),
        "source": source,
        "source_method": source_method,
        "text_hash": stable_hash(router_input),
    }
    if tags:
        row["tags"] = tags
    return row


def generate_gpt_authored_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, count in TARGET_GENERATED_COUNTS.items():
        for index in range(count):
            messages, method = generated_question(label, index)
            rows.append(
                build_row(
                    row_id=f"router_v2_gpt_{label}_{index:04d}",
                    label=label,
                    source_messages=messages,
                    source="gpt_direct_authored_router_templates_v1",
                    source_method=method,
                )
            )
    return rows


def load_old_router_rows(input_dir: Path, cap_per_label: int, seed: int) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ["train", "validation", "test"]:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            label = row.get("label") or (row.get("answer") or {}).get("label")
            question = normalize_text(row.get("question") or "")
            if label not in ANSWER_BY_LABEL or not question:
                continue
            candidates[label].append(row)
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    for label, label_rows in candidates.items():
        rng.shuffle(label_rows)
        for index, row in enumerate(label_rows[:cap_per_label]):
            output.append(
                build_row(
                    row_id=f"router_v2_old_{label}_{index:04d}",
                    label=label,
                    source_messages=messages_for_raw_question(row["question"]),
                    source="old_router_lora_dataset",
                    source_method="old_router_v1_vetted_label_rewriter_processed",
                )
            )
    return output


def load_subagent_seed_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scenario in data.get("scenarios") or []:
        label = scenario.get("intended_label")
        question = normalize_text(scenario.get("question") or "")
        if label not in ANSWER_BY_LABEL or not question:
            continue
        rewritten = normalize_text(scenario.get("rewritten_router_input") or "")
        source_messages = messages_for_raw_question(question)
        if rewritten:
            rewrite_payload = {
                "standalone_query": rewritten,
                "resolved_case_refs": [],
                "case_name_mentions": [],
                "crime_mentions": [],
                "amount_constraints": [],
                "field_intents": [],
                "rewrite_required": True,
                "confidence": 0.92,
                "warnings": ["subagent_rewritten_router_input"],
            }
            rows.append(
                build_row_from_rewriter_payload(
                    row_id=f"router_v2_subagent_{scenario.get('id')}",
                    label=label,
                    raw_question=question,
                    source_messages=source_messages,
                    rewrite_payload=rewrite_payload,
                    source="gpt_5_4_subagent_router_seed_proposal",
                    source_method="subagent_direct_authored_seed_with_rewriter_input",
                    tags=list(scenario.get("tags") or []),
                )
            )
        else:
            row = build_row(
                row_id=f"router_v2_subagent_{scenario.get('id')}",
                label=label,
                source_messages=source_messages,
                source="gpt_5_4_subagent_router_seed_proposal",
                source_method="subagent_direct_authored_seed",
            )
            row["tags"] = list(scenario.get("tags") or [])
            rows.append(row)
    return rows


def validate_answer(answer: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = answer.get("label")
    if label not in ANSWER_BY_LABEL:
        errors.append(f"unknown_label:{label}")
        return errors
    expected = ANSWER_BY_LABEL[label]
    for field in ["needs_retrieval", "retrieval_targets", "case_required", "law_required", "clarify_required"]:
        if answer.get(field) != expected[field]:
            errors.append(f"inconsistent:{field}")
    return errors


def validate_rows(rows: list[dict[str, Any]], min_count: int) -> dict[str, Any]:
    invalid: list[dict[str, Any]] = []
    conflicts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        errors = validate_answer(row.get("answer") or {})
        if not row.get("router_input"):
            errors.append("empty_router_input")
        if "standalone_query" not in (row.get("rewriter") or {}):
            errors.append("missing_rewriter_payload")
        if errors:
            invalid.append({"id": row.get("id"), "errors": errors})
        conflicts[row["text_hash"]].add(row["label"])
    conflict_rows = {key: sorted(value) for key, value in conflicts.items() if len(value) > 1}
    if len(rows) < min_count:
        raise SystemExit(f"dataset_too_small:{len(rows)} < {min_count}")
    if invalid:
        raise SystemExit(f"invalid_rows:{invalid[:5]}")
    if conflict_rows:
        raise SystemExit(f"conflicting_labels:{list(conflict_rows.items())[:5]}")
    return {"invalid_rows": 0, "conflict_count": 0}


def stratified_split(rows: list[dict[str, Any]], train_ratio: float, validation_ratio: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_n = int(total * train_ratio)
        validation_n = int(total * validation_ratio)
        if total >= 3:
            train_n = max(1, min(train_n, total - 2))
            validation_n = max(1, min(validation_n, total - train_n - 1))
        splits["train"].extend(label_rows[:train_n])
        splits["validation"].extend(label_rows[train_n : train_n + validation_n])
        splits["test"].extend(label_rows[train_n + validation_n :])
    for split, split_rows in splits.items():
        rng.shuffle(split_rows)
        for row in split_rows:
            row["split"] = split
    return splits


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Router-only v2 SFT data from GPT-authored scenarios plus vetted old router rows.")
    parser.add_argument("--old-router-dir", type=Path, default=DEFAULT_OLD_ROUTER_DATASET)
    parser.add_argument("--subagent-seed", type=Path, default=DEFAULT_SUBAGENT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-cap-per-label", type=int, default=220)
    parser.add_argument("--rewrite-mode", choices=["rules", "lora"], default="rules")
    parser.add_argument("--min-count", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.82)
    parser.add_argument("--validation-ratio", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.validation_ratio <= 0 or args.train_ratio + args.validation_ratio >= 1:
        raise SystemExit("--train-ratio and --validation-ratio must be positive and sum to less than 1.")

    global REWRITE_MODE
    REWRITE_MODE = args.rewrite_mode

    rows = generate_gpt_authored_rows()
    rows.extend(load_subagent_seed_rows(args.subagent_seed))
    rows.extend(load_old_router_rows(args.old_router_dir, args.old_cap_per_label, args.seed))
    validation = validate_rows(rows, args.min_count)
    splits = stratified_split(rows, args.train_ratio, args.validation_ratio, args.seed)

    counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for split, split_rows in splits.items():
        counts[split] = write_jsonl(args.output_dir / f"{split}.jsonl", split_rows)
        label_counts[split] = dict(Counter(row["label"] for row in split_rows))
        source_counts[split] = dict(Counter(row["source"] for row in split_rows))

    all_label_counts = dict(Counter(row["label"] for row in rows))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "old_router_dir": str(args.old_router_dir),
        "subagent_seed": str(args.subagent_seed),
        "seed": args.seed,
        "rewrite_mode": args.rewrite_mode,
        "total_count": len(rows),
        "counts": counts,
        "label_counts": label_counts,
        "all_label_counts": all_label_counts,
        "source_counts": source_counts,
        "validation": validation,
        "system_prompt": SYSTEM_PROMPT,
        "notes": [
            "Router-only dataset: assistant answers contain route fields only.",
            "Router input is the Rewriter-processed standalone query plus compact structured cues.",
            "GPT-authored rows are generated from GPT-designed scenario templates covering ordinary, edge, and multi-turn cases.",
            "Old router rows are capped per label and reprocessed through the current deterministic Rewriter completion path.",
            "Use processing/train_router_lora.py with --dataset-dir pointing here to train the Router LoRA.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "total_count": len(rows), "counts": counts, "all_label_counts": all_label_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
