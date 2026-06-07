from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common import OUT_DIR
from prepare_query_understanding_dataset import build_messages


DEFAULT_GPT_DATASET = OUT_DIR / "query_understanding_lora" / "gpt_authored_dataset"
DEFAULT_DIRECT_GENERAL_DATASET = OUT_DIR / "query_understanding_lora" / "gpt_direct_general_dataset"
DEFAULT_AUGMENTED_DATASET = OUT_DIR / "query_understanding_lora" / "augmented_dataset"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_understanding_lora" / "final_5000_dataset"
SCHEMA_VERSION = "query-understanding-final-5000-v1"
SYSTEM_PROMPT = (
    "你是刑法 GraphRAG 的 Query Understanding 模块，只服务于检索前处理。"
    "请把多轮问题改写成可检索的独立问题，并识别案名、罪名、金额、检索范围和字段意图。"
    "必须只输出一个 JSON 对象，不要 Markdown，不要解释。"
)
VALID_ROUTE_LABELS = {
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
REQUIRED_ANSWER_FIELDS = [
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
]
ROUTE_LABEL_MAP = {
    "guardrail_refusal_or_clarification": "criminal_law_advice",
    "guardrail_refusal": "criminal_law_advice",
    "guardrail_refuse": "criminal_law_advice",
    "case_law_retrieval": "criminal_law_case_and_law",
    "law_and_case_retrieval": "criminal_law_case_and_law",
    "criminal_law_article_lookup": "criminal_law_law_only",
    "comparative_case_reasoning": "criminal_law_multi_case",
    "non_criminal_civil_redirect": "non_criminal_legal",
    "non_criminal_or_out_of_scope": "non_criminal_legal",
    "out_of_scope": "general_no_legal_retrieval",
    "insufficient_information": "unclear_need_clarification",
    "no_answer": "unclear_need_clarification",
}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}:invalid_json:{exc}") from exc
    return rows


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        rows.extend(read_jsonl(path / name))
    return rows


def normalize_answer(answer: dict[str, Any]) -> dict[str, Any]:
    answer = dict(answer)
    for field in ["retrieval_targets", "case_name_mentions", "resolved_case_refs", "field_intents", "crime_mentions", "amount_constraints", "warnings"]:
        answer.setdefault(field, [])
    answer.setdefault("standalone_query", "")
    answer.setdefault("route_label", "low_quality_or_incomplete")
    answer["route_label"] = ROUTE_LABEL_MAP.get(answer["route_label"], answer["route_label"])
    answer.setdefault("needs_retrieval", False)
    answer.setdefault("case_required", False)
    answer.setdefault("law_required", False)
    answer.setdefault("confidence", 0.8)
    return answer


def normalize_row(row: dict[str, Any], *, category: str | None = None, source_method: str | None = None) -> dict[str, Any]:
    answer = normalize_answer(row["answer"])
    source_messages = row.get("source_messages") or []
    if not source_messages:
        question = row.get("question") or answer.get("standalone_query") or ""
        source_messages = [{"role": "user", "content": str(question)}]
    normalized = {
        "id": row.get("id", stable_hash(row)),
        "split": row.get("split", "train"),
        "category": category or row.get("category") or category_for_row(row),
        "source_messages": source_messages,
        "answer": answer,
        "messages": build_messages(source_messages, answer),
        "source": row.get("source") or "query_understanding_existing_dataset",
        "source_method": source_method or row.get("source_method") or "existing_dataset_row",
        "text_hash": row.get("text_hash") or stable_hash(source_messages),
    }
    return normalized


def category_for_row(row: dict[str, Any]) -> str:
    source_method = row.get("source_method")
    if source_method == "router_dataset_conversion_v1":
        return "router_conversion"
    if source_method == "history_boundary_conversion_v1":
        return "multi_turn_rewrite"
    return row.get("category") or "template_or_existing"


def answer_payload(
    *,
    standalone_query: str,
    route_label: str,
    retrieval_targets: list[str],
    case_required: bool,
    law_required: bool,
    case_name_mentions: list[str],
    resolved_case_refs: list[dict[str, Any]],
    field_intents: list[str],
    crime_mentions: list[str],
    amount_constraints: list[dict[str, Any]] | None = None,
    confidence: float = 0.9,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "standalone_query": standalone_query,
        "route_label": route_label,
        "needs_retrieval": route_label not in {"general_no_legal_retrieval", "criminal_law_general_direct", "non_criminal_legal", "low_quality_or_incomplete"},
        "retrieval_targets": retrieval_targets,
        "case_required": case_required,
        "law_required": law_required,
        "case_name_mentions": case_name_mentions,
        "resolved_case_refs": resolved_case_refs,
        "field_intents": field_intents,
        "crime_mentions": crime_mentions,
        "amount_constraints": amount_constraints or [],
        "confidence": confidence,
        "warnings": warnings or [],
    }


CASE_BANK = [
    ("王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书", "王士东案", "盗窃罪", 43852),
    ("刘方祥、刘礼芬盗窃一审刑事判决书", "刘方祥案", "盗窃罪", 39136),
    ("吴必定抢劫案", "吴必定案", "抢劫罪", None),
    ("张君抢劫、故意杀人案", "张君案", "抢劫罪", None),
    ("陈某诈骗罪一审刑事判决书", "陈某诈骗案", "诈骗罪", 40000),
    ("李某故意伤害罪二审刑事裁定书", "李某故意伤害案", "故意伤害罪", None),
    ("周某入户盗窃罪一审刑事判决书", "周某入户盗窃案", "盗窃罪", 3000),
    ("赵某寻衅滋事罪一审刑事判决书", "赵某寻衅滋事案", "寻衅滋事罪", None),
    ("孙某危险驾驶罪一审刑事判决书", "孙某危险驾驶案", "危险驾驶罪", None),
    ("何某合同诈骗罪一审刑事判决书", "何某合同诈骗案", "合同诈骗罪", 80000),
]

FOLLOWUPS = [
    ("第一个参考案例具体案情是什么", 1, "第一个参考案例", ["court_found_facts_text"], "案情 法院认定事实", "criminal_law_case_only", ["cases"]),
    ("第二个案例证据有哪些", 2, "第二个案例", ["evidence_text"], "证据 法院采信理由", "criminal_law_case_only", ["cases"]),
    ("刚才那个为什么判这么重", 1, "刚才那个", ["reasoning_text", "judgment_text"], "裁判理由 量刑", "criminal_law_case_only", ["cases"]),
    ("这个案子被告人怎么辩护的", 1, "这个案子", ["defense_text"], "辩护意见", "criminal_law_case_only", ["cases"]),
    ("前面第二个适用了哪些法条", 2, "前面第二个", ["legal_basis_text"], "适用法条 法律依据", "criminal_law_case_and_law", ["cases", "law_articles"]),
    ("哪个更接近我朋友的情况", 1, "比较前述案例", ["court_found_facts_text", "judgment_text"], "相似案例 对比 量刑", "criminal_law_multi_case", ["cases", "graph"]),
    ("有没有类似但判得轻一点的案例", 1, "前述案例", ["judgment_text", "reasoning_text"], "类似案例 轻判 裁判理由", "criminal_law_multi_case", ["cases", "graph"]),
    ("法院认定入户情节主要看什么", 1, "第一个参考案例", ["court_found_facts_text", "reasoning_text", "legal_basis_text"], "入户情节 法院认定 法条依据", "criminal_law_case_and_law", ["cases", "law_articles"]),
    ("这个金额对应哪一档量刑", 1, "金额追问", ["legal_basis_text", "judgment_text"], "金额 数额档次 量刑", "criminal_law_case_and_law", ["law_articles", "cases"]),
    ("证据够不够支撑法院认定", 1, "这个案子", ["evidence_text", "court_found_facts_text"], "证据 证明标准 法院认定", "criminal_law_case_only", ["cases"]),
]


def make_history(case_a: tuple[str, str, str, int | None], case_b: tuple[str, str, str, int | None], amount: int) -> list[dict[str, str]]:
    title_a, short_a, crime_a, amount_a = case_a
    title_b, short_b, crime_b, amount_b = case_b
    crime = crime_a if crime_a == crime_b else f"{crime_a}、{crime_b}"
    return [
        {"role": "user", "content": f"我想查{crime}相关案例，最好能对比案情、证据和量刑，金额大概{amount}元左右。"},
        {
            "role": "assistant",
            "content": (
                f"可以先看两个参考案例。第一个参考案例是《{title_a}》，"
                f"简称{short_a}，涉及{crime_a}"
                f"{'，金额约' + str(amount_a) + '元' if amount_a else ''}，可关注事实、证据和量刑理由。"
                f"第二个参考案例是《{title_b}》，简称{short_b}，涉及{crime_b}"
                f"{'，金额约' + str(amount_b) + '元' if amount_b else ''}。"
                "如果要查法条，可结合现行刑法条文、司法解释和案例裁判理由一起检索。"
            ),
        },
    ]


def generate_template_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    amounts = [3000, 40000, 43852, 39136, 80000, 120000]
    index = 0
    while len(rows) < count:
        case_a = rng.choice(CASE_BANK)
        case_b = rng.choice([case for case in CASE_BANK if case != case_a])
        followup = rng.choice(FOLLOWUPS)
        amount_value = rng.choice(amounts)
        text, case_index, ref_text, fields, suffix, route_label, targets = followup
        selected = case_a if case_index == 1 else case_b
        history = make_history(case_a, case_b, amount_value)
        source_messages = history + [{"role": "user", "content": text}]
        title, short_title, crime, case_amount = selected
        amount_constraints = []
        if "金额" in text or "量刑" in suffix or case_amount:
            amount_constraints.append({
                "value": int(case_amount or amount_value),
                "role": "theft_amount_total_or_single" if "盗窃" in crime else "unknown_amount",
                "raw_text": f"{int(case_amount or amount_value)}元",
            })
        answer = answer_payload(
            standalone_query=f"{title} {suffix}",
            route_label=route_label,
            retrieval_targets=targets,
            case_required="cases" in targets or "graph" in targets,
            law_required="law_articles" in targets,
            case_name_mentions=[title, short_title],
            resolved_case_refs=[{"ref": ref_text, "index": case_index, "case_title": title, "source": "assistant_history"}],
            field_intents=fields,
            crime_mentions=[crime],
            amount_constraints=amount_constraints,
            warnings=["historical_law_requires_version_check"] if "旧刑法" in text or "1979" in text else [],
        )
        row = {
            "id": f"template_multi_turn_{index:06d}",
            "split": "train",
            "category": "multi_turn_rewrite",
            "source_messages": source_messages,
            "answer": answer,
            "messages": build_messages(source_messages, answer),
            "source": "deterministic_multi_turn_templates",
            "source_method": "deterministic_multi_turn_template_v1",
            "text_hash": stable_hash(source_messages),
        }
        rows.append(row)
        index += 1
    return rows


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for row in rows:
        answer = row.get("answer") or {}
        key = stable_hash({
            "source_messages": row.get("source_messages"),
            "standalone_query": answer.get("standalone_query"),
            "route_label": answer.get("route_label"),
        })
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    answer = row.get("answer")
    if not isinstance(answer, dict):
        return ["answer_not_object"]
    for field in REQUIRED_ANSWER_FIELDS:
        if field not in answer:
            errors.append(f"missing_answer_field:{field}")
    if answer.get("route_label") not in VALID_ROUTE_LABELS:
        errors.append(f"invalid_route_label:{answer.get('route_label')}")
    messages = row.get("messages") or []
    if len(messages) < 3:
        errors.append("messages_too_short")
    else:
        try:
            if json.loads(messages[-1].get("content") or "") != answer:
                errors.append("assistant_json_not_equal_answer")
        except json.JSONDecodeError:
            errors.append("assistant_content_not_json")
    return errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def assign_final_ids_and_splits(rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    train_cutoff = int(target_count * 0.8)
    validation_cutoff = int(target_count * 0.9)
    final = []
    for index, row in enumerate(rows[:target_count], 1):
        row = dict(row)
        row["source_row_id"] = row.get("id")
        row["id"] = f"final_qu_{index:06d}"
        if index <= train_cutoff:
            row["split"] = "train"
        elif index <= validation_cutoff:
            row["split"] = "validation"
        else:
            row["split"] = "test"
        final.append(row)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 5000-row Query Understanding SFT dataset.")
    parser.add_argument("--gpt-dataset", type=Path, default=DEFAULT_GPT_DATASET)
    parser.add_argument("--direct-general-dataset", type=Path, default=DEFAULT_DIRECT_GENERAL_DATASET)
    parser.add_argument("--augmented-dataset", type=Path, default=DEFAULT_AUGMENTED_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--template-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gpt_rows = [normalize_row(row, source_method="gpt_5_5_direct_authored_jsonl_v1") for row in load_dataset_rows(args.gpt_dataset)]
    direct_general_rows = [
        normalize_row(row, category="criminal_law_general_direct", source_method="gpt_5_5_direct_general_jsonl_v1")
        for row in load_dataset_rows(args.direct_general_dataset)
    ]
    template_rows = generate_template_rows(args.template_count, args.seed)
    augmented = [normalize_row(row) for row in load_dataset_rows(args.augmented_dataset)]

    non_router = [row for row in augmented if row["source_method"] != "router_dataset_conversion_v1"]
    router_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmented:
        if row["source_method"] == "router_dataset_conversion_v1":
            router_by_label[row["answer"]["route_label"]].append(row)
    for values in router_by_label.values():
        rng.shuffle(values)

    router_target = args.target_count - len(gpt_rows) - len(direct_general_rows) - len(template_rows) - len(non_router)
    selected_router: list[dict[str, Any]] = []
    label_caps = {
        "general_no_legal_retrieval": 550,
        "criminal_law_law_only": 850,
        "non_criminal_legal": 350,
        "criminal_law_case_only": 350,
        "criminal_law_case_and_law": 450,
        "criminal_law_multi_case": 160,
        "criminal_law_advice": 120,
        "low_quality_or_incomplete": 40,
        "unclear_need_clarification": 40,
    }
    for label, cap in label_caps.items():
        selected_router.extend(router_by_label.get(label, [])[:cap])
    remaining_router = [
        row
        for label, values in router_by_label.items()
        for row in values[label_caps.get(label, 0):]
    ]
    rng.shuffle(remaining_router)
    if len(selected_router) < router_target:
        selected_router.extend(remaining_router[: router_target - len(selected_router)])
        remaining_router = remaining_router[router_target - len(selected_router):]
    else:
        selected_router = selected_router[:router_target]

    combined = dedupe_rows(gpt_rows + direct_general_rows + template_rows + non_router + selected_router)
    if len(combined) < args.target_count:
        combined = dedupe_rows(combined + remaining_router)
    if len(combined) < args.target_count:
        raise SystemExit(f"Only {len(combined)} unique rows available for target {args.target_count}.")
    selected = combined[:args.target_count]
    rng.shuffle(selected)
    final_rows = assign_final_ids_and_splits(selected, args.target_count)

    invalid = [{"id": row["id"], "errors": validate_row(row)} for row in final_rows if validate_row(row)]
    if invalid:
        (args.output_dir / "validation_report.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "validation_report.json").write_text(json.dumps({"invalid": invalid[:50]}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Invalid rows: {len(invalid)}")

    split_rows = {
        "train": [row for row in final_rows if row["split"] == "train"],
        "validation": [row for row in final_rows if row["split"] == "validation"],
        "test": [row for row in final_rows if row["split"] == "test"],
    }
    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)

    category_counts = Counter(row["category"] for row in final_rows)
    route_label_counts = Counter(row["answer"]["route_label"] for row in final_rows)
    source_method_counts = Counter(row["source_method"] for row in final_rows)
    multi_turn_count = category_counts.get("multi_turn_rewrite", 0)
    amount_rows = sum(1 for row in final_rows if row["answer"].get("amount_constraints"))
    now = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": now,
        "target_count": args.target_count,
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "category_counts": dict(category_counts),
        "route_label_counts": dict(route_label_counts),
        "source_method_counts": dict(source_method_counts),
        "multi_turn_count": multi_turn_count,
        "multi_turn_share": round(multi_turn_count / len(final_rows), 4),
        "amount_constraint_rows": amount_rows,
        "invalid_rows": 0,
        "inputs": {
            "gpt_dataset": str(args.gpt_dataset),
            "direct_general_dataset": str(args.direct_general_dataset),
            "augmented_dataset": str(args.augmented_dataset),
            "template_count": args.template_count,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
