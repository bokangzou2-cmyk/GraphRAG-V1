from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common import OUT_DIR, iter_jsonl
from prepare_query_understanding_dataset import (
    build_messages,
    convert_probe_case,
    convert_router_row,
)
from query_understanding_probe import default_eval_cases


DEFAULT_ROUTER_DATASET = OUT_DIR / "router_lora" / "dataset"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_understanding_lora" / "augmented_dataset"
SCHEMA_VERSION = "query-understanding-lora-augmented-dataset-v1"
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
VALID_ROUTE_LABELS = {
    "general_no_legal_retrieval",
    "non_criminal_legal",
    "criminal_law_law_only",
    "criminal_law_case_only",
    "criminal_law_case_and_law",
    "criminal_law_multi_case",
    "criminal_law_advice",
    "unclear_need_clarification",
    "low_quality_or_incomplete",
}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def answer(
    *,
    standalone_query: str,
    route_label: str,
    needs_retrieval: bool,
    retrieval_targets: list[str],
    case_required: bool = False,
    law_required: bool = False,
    case_name_mentions: list[str] | None = None,
    resolved_case_refs: list[dict[str, Any]] | None = None,
    field_intents: list[str] | None = None,
    crime_mentions: list[str] | None = None,
    amount_constraints: list[dict[str, Any]] | None = None,
    confidence: float = 0.9,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "standalone_query": standalone_query,
        "route_label": route_label,
        "needs_retrieval": needs_retrieval,
        "retrieval_targets": retrieval_targets,
        "case_required": case_required,
        "law_required": law_required,
        "case_name_mentions": case_name_mentions or [],
        "resolved_case_refs": resolved_case_refs or [],
        "field_intents": field_intents or [],
        "crime_mentions": crime_mentions or [],
        "amount_constraints": amount_constraints or [],
        "confidence": confidence,
        "warnings": warnings or [],
    }


def row_from_messages(
    *,
    row_id: str,
    split: str,
    question: str,
    source_messages: list[dict[str, Any]],
    answer_payload: dict[str, Any],
    category: str,
    source_method: str = "deterministic_qu_augmentation_v1",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "split": split,
        "question": question,
        "source_messages": source_messages,
        "answer": answer_payload,
        "messages": build_messages(source_messages, answer_payload),
        "source": "query_understanding_deterministic_templates",
        "source_method": source_method,
        "category": category,
        "text_hash": stable_hash(source_messages),
    }


def resolved(ref: str, title: str, index: int) -> list[dict[str, Any]]:
    return [{"ref": ref, "index": index, "case_title": title, "source": "assistant_history"}]


def amount(value: int, raw_text: str, role: str = "theft_amount_total_or_single") -> list[dict[str, Any]]:
    return [{"value": value, "role": role, "raw_text": raw_text}]


def history_contexts() -> list[dict[str, Any]]:
    return [
        {
            "key": "theft_wang_liu",
            "crime": "盗窃罪",
            "messages": [
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
            ],
            "cases": [
                "王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书",
                "刘方祥、刘礼芬盗窃一审刑事判决书",
            ],
        },
        {
            "key": "robbery_zhang_wu",
            "crime": "抢劫罪",
            "messages": [
                {"role": "user", "content": "有没有张君案和吴必定案的参考案例"},
                {
                    "role": "assistant",
                    "content": (
                        "以下为检索证据摘要。\n"
                        "1. 张君案：涉及抢劫、故意杀人等犯罪事实，裁判文书记录了犯罪事实、证据和量刑理由。\n"
                        "2. 吴必定案：涉及抢劫罪相关裁判，包含法院认定事实、辩护意见和判决结果。"
                    ),
                },
            ],
            "cases": ["张君案", "吴必定案"],
        },
    ]


def build_history_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    followups = [
        ("第一个参考案例具体案情是什么", 1, "第一个参考案例", ["court_found_facts_text", "reasoning_text", "judgment_text"], "案情 裁判理由 量刑"),
        ("第二个案例的法院认定事实和判决结果呢", 2, "第二个案例", ["court_found_facts_text", "judgment_text"], "法院认定事实 判决结果"),
        ("这个案子法院依据哪些证据认定的", 1, "这个案子", ["evidence_text", "court_found_facts_text"], "证据 法院认定"),
        ("为什么判这么重", 1, "implicit_previous_case", ["reasoning_text", "judgment_text"], "裁判理由 量刑"),
        ("被告人有什么辩护意见", 1, "implicit_previous_case", ["defense_text"], "辩护意见"),
        ("法院认定了哪些事实", 1, "implicit_previous_case", ["court_found_facts_text"], "法院认定事实"),
        ("裁判理由是什么", 1, "implicit_previous_case", ["reasoning_text"], "裁判理由"),
    ]
    for context in history_contexts():
        for text, case_index, ref_text, fields, suffix in followups:
            case_title = context["cases"][case_index - 1]
            source_messages = context["messages"] + [{"role": "user", "content": text}]
            payload = answer(
                standalone_query=f"{case_title} {suffix}",
                route_label="criminal_law_case_only",
                needs_retrieval=True,
                retrieval_targets=["cases"],
                case_required=True,
                case_name_mentions=[case_title],
                resolved_case_refs=resolved(ref_text, case_title, case_index),
                field_intents=fields,
                crime_mentions=[context["crime"]],
                confidence=0.92,
            )
            rows.append(row_from_messages(
                row_id=f"aug_history_{context['key']}_{stable_hash([text, case_title])}",
                split="train",
                question=text,
                source_messages=source_messages,
                answer_payload=payload,
                category="history_case_reference",
            ))
    return rows


def build_short_case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ("吴必定案", "抢劫罪"),
        ("张君案", "抢劫罪"),
        ("王士东案", "盗窃罪"),
        ("刘方祥案", "盗窃罪"),
    ]
    variants = [
        ("事实是什么", ["court_found_facts_text"], "法院认定事实"),
        ("证据有哪些", ["evidence_text"], "证据"),
        ("被告怎么辩护的", ["defense_text"], "辩护意见"),
        ("法院为什么这么判", ["reasoning_text", "judgment_text"], "裁判理由 判决结果"),
        ("判决结果是什么", ["judgment_text"], "判决结果"),
    ]
    for case_title, crime in cases:
        for suffix, fields, standalone_suffix in variants:
            question = f"{case_title}{suffix}"
            payload = answer(
                standalone_query=f"{case_title} {standalone_suffix}",
                route_label="criminal_law_case_only",
                needs_retrieval=True,
                retrieval_targets=["cases"],
                case_required=True,
                case_name_mentions=[case_title],
                field_intents=fields,
                crime_mentions=[crime],
                confidence=0.9,
            )
            rows.append(row_from_messages(
                row_id=f"aug_short_case_{stable_hash(question)}",
                split="train",
                question=question,
                source_messages=[{"role": "user", "content": question}],
                answer_payload=payload,
                category="short_case_name",
            ))
    return rows


def build_amount_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    places = ["入户", "入室"]
    values = [3000, 40000, 43852, 39136]
    suffixes = [
        ("怎么判，有案例参考吗", "criminal_law_case_and_law", ["law_articles", "cases"], True, True, ["judgment_text", "reasoning_text", "legal_basis_text"]),
        ("对应刑法哪条", "criminal_law_law_only", ["law_articles"], False, True, ["legal_basis_text"]),
        ("类似案例有哪些", "criminal_law_case_only", ["cases"], True, False, ["judgment_text", "reasoning_text"]),
    ]
    for place in places:
        for value in values:
            for suffix, label, targets, case_required, law_required, fields in suffixes:
                raw = f"{value}元"
                question = f"{place}盗窃{raw}{suffix}"
                payload = answer(
                    standalone_query=question,
                    route_label=label,
                    needs_retrieval=True,
                    retrieval_targets=targets,
                    case_required=case_required,
                    law_required=law_required,
                    field_intents=fields,
                    crime_mentions=["盗窃罪"],
                    amount_constraints=amount(value, raw),
                    confidence=0.91,
                )
                rows.append(row_from_messages(
                    row_id=f"aug_amount_theft_{stable_hash(question)}",
                    split="train",
                    question=question,
                    source_messages=[{"role": "user", "content": question}],
                    answer_payload=payload,
                    category="amount_constraint",
                ))
    return rows


def build_typo_and_scope_rows() -> list[dict[str, Any]]:
    specs = [
        ("故意伤害最判几年", "故意伤害罪最高刑期和量刑规则", "criminal_law_law_only", True, ["law_articles"], False, True, [], ["legal_basis_text", "judgment_text"], ["故意伤害罪"], []),
        ("盗窃对应哪条", "盗窃罪对应刑法条文", "criminal_law_law_only", True, ["law_articles"], False, True, [], ["legal_basis_text"], ["盗窃罪"], []),
        ("诈骗相关法条", "诈骗罪相关刑法条文", "criminal_law_law_only", True, ["law_articles"], False, True, [], ["legal_basis_text"], ["诈骗罪"], []),
        ("张君案和吴必定案判得有什么区别", "张君案 吴必定案 量刑差异 裁判理由", "criminal_law_multi_case", True, ["cases", "graph"], True, False, ["张君案", "吴必定案"], ["reasoning_text", "judgment_text"], ["抢劫罪"], []),
        ("王士东案和刘方祥案量刑为什么不一样", "王士东案 刘方祥案 量刑差异 裁判理由", "criminal_law_multi_case", True, ["cases", "graph"], True, False, ["王士东案", "刘方祥案"], ["reasoning_text", "judgment_text"], ["盗窃罪"], []),
        ("有没有根本查不到的火星刑法案例", "火星刑法案例", "low_quality_or_incomplete", False, ["none"], False, False, [], [], [], []),
        ("离婚财产怎么分", "离婚财产怎么分", "non_criminal_legal", False, ["none"], False, False, [], [], [], []),
        ("劳动仲裁时效多久", "劳动仲裁时效多久", "non_criminal_legal", False, ["none"], False, False, [], [], [], []),
        ("帮我规避盗窃罪的法律责任", "盗窃罪法律责任和风险", "criminal_law_advice", True, ["law_articles"], False, True, [], ["legal_basis_text"], ["盗窃罪"], []),
        ("怎么做才能诈骗后不被追责", "诈骗罪法律责任和追责风险", "criminal_law_advice", True, ["law_articles"], False, True, [], ["legal_basis_text"], ["诈骗罪"], []),
    ]
    rows: list[dict[str, Any]] = []
    for question, standalone, label, needs, targets, case_required, law_required, case_mentions, fields, crimes, amounts in specs:
        payload = answer(
            standalone_query=standalone,
            route_label=label,
            needs_retrieval=needs,
            retrieval_targets=targets,
            case_required=case_required,
            law_required=law_required,
            case_name_mentions=case_mentions,
            field_intents=fields,
            crime_mentions=crimes,
            amount_constraints=amounts,
            confidence=0.88 if needs else 0.8,
            warnings=["source_may_have_no_answer"] if "查不到" in question else [],
        )
        rows.append(row_from_messages(
            row_id=f"aug_scope_typo_{stable_hash(question)}",
            split="train",
            question=question,
            source_messages=[{"role": "user", "content": question}],
            answer_payload=payload,
            category="typo_scope_guardrail",
        ))
    return rows


def build_augmented_rows() -> list[dict[str, Any]]:
    rows = []
    rows.extend(build_history_rows())
    rows.extend(build_short_case_rows())
    rows.extend(build_amount_rows())
    rows.extend(build_typo_and_scope_rows())
    for index, row in enumerate(rows):
        if index % 10 == 0:
            row["split"] = "validation"
        elif index % 10 == 1:
            row["split"] = "test"
        row["messages"] = build_messages(row["source_messages"], row["answer"])
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_router_split(input_dir: Path, split: str) -> list[dict[str, Any]]:
    path = input_dir / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing router split: {path}")
    return list(iter_jsonl(path))


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    answer_payload = row.get("answer")
    if not isinstance(answer_payload, dict):
        return ["answer_not_object"]
    missing = [field for field in REQUIRED_ANSWER_FIELDS if field not in answer_payload]
    extra = [field for field in answer_payload if field not in REQUIRED_ANSWER_FIELDS]
    if missing:
        errors.append(f"missing_answer_fields:{','.join(missing)}")
    if extra:
        errors.append(f"extra_answer_fields:{','.join(extra)}")
    if answer_payload.get("route_label") not in VALID_ROUTE_LABELS:
        errors.append(f"invalid_route_label:{answer_payload.get('route_label')}")
    if not isinstance(answer_payload.get("retrieval_targets"), list):
        errors.append("retrieval_targets_not_list")
    if not isinstance(answer_payload.get("case_name_mentions"), list):
        errors.append("case_name_mentions_not_list")
    if not isinstance(answer_payload.get("resolved_case_refs"), list):
        errors.append("resolved_case_refs_not_list")
    if not isinstance(answer_payload.get("field_intents"), list):
        errors.append("field_intents_not_list")
    if not isinstance(answer_payload.get("crime_mentions"), list):
        errors.append("crime_mentions_not_list")
    if not isinstance(answer_payload.get("amount_constraints"), list):
        errors.append("amount_constraints_not_list")
    if not isinstance(answer_payload.get("warnings"), list):
        errors.append("warnings_not_list")
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        errors.append("missing_assistant_message")
    else:
        try:
            assistant_payload = json.loads(messages[-1].get("content") or "")
            if assistant_payload != answer_payload:
                errors.append("assistant_content_mismatch")
        except json.JSONDecodeError:
            errors.append("assistant_content_not_json")
    return errors


def validate_dataset(output_dir: Path) -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    totals = Counter()
    for split in ["train", "validation", "test"]:
        path = output_dir / f"{split}.jsonl"
        stats = {
            "rows": 0,
            "json_valid": 0,
            "row_error_count": 0,
            "missing_answer_fields": Counter(),
            "route_label_counts": Counter(),
            "source_method_counts": Counter(),
            "category_counts": Counter(),
            "history_reference_rows": 0,
            "amount_constraint_rows": 0,
            "case_name_rows": 0,
            "field_intent_rows": 0,
            "sample_errors": [],
        }
        if not path.exists():
            stats["sample_errors"].append(f"missing_split_file:{path}")
            split_stats[split] = stats
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                stats["rows"] += 1
                try:
                    row = json.loads(line)
                    stats["json_valid"] += 1
                except json.JSONDecodeError as exc:
                    stats["row_error_count"] += 1
                    if len(stats["sample_errors"]) < 10:
                        stats["sample_errors"].append(f"{line_number}:json_decode:{exc}")
                    continue
                errors = validate_row(row)
                answer_payload = row.get("answer") or {}
                for field in REQUIRED_ANSWER_FIELDS:
                    if field not in answer_payload:
                        stats["missing_answer_fields"][field] += 1
                if errors:
                    stats["row_error_count"] += 1
                    if len(stats["sample_errors"]) < 10:
                        stats["sample_errors"].append(f"{line_number}:{row.get('id')}:{';'.join(errors)}")
                stats["route_label_counts"][answer_payload.get("route_label")] += 1
                stats["source_method_counts"][row.get("source_method")] += 1
                stats["category_counts"][row.get("category") or "unclassified"] += 1
                stats["history_reference_rows"] += 1 if answer_payload.get("resolved_case_refs") else 0
                stats["amount_constraint_rows"] += 1 if answer_payload.get("amount_constraints") else 0
                stats["case_name_rows"] += 1 if answer_payload.get("case_name_mentions") else 0
                stats["field_intent_rows"] += 1 if answer_payload.get("field_intents") else 0
        for key in ["rows", "json_valid", "row_error_count", "history_reference_rows", "amount_constraint_rows", "case_name_rows", "field_intent_rows"]:
            totals[key] += stats[key]
        split_stats[split] = stats
    return {
        "output_dir": str(output_dir),
        "summary": dict(totals),
        "splits": split_stats,
    }


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    return value


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "validation", "test"]:
        rows = [convert_router_row(row, split) for row in load_router_split(args.input_dir, split)]
        for row in rows:
            row.setdefault("category", "router_conversion")
        if split == "train" and args.max_train:
            rng.shuffle(rows)
            rows = rows[: args.max_train]
        rows_by_split[split] = rows

    for index, case in enumerate(default_eval_cases()):
        split = "validation" if index % 3 == 0 else "train"
        row = convert_probe_case(case, split)
        row["category"] = case.get("category") or "probe_boundary"
        rows_by_split[split].append(row)

    for row in build_augmented_rows():
        rows_by_split[row["split"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {split: write_jsonl(args.output_dir / f"{split}.jsonl", rows) for split, rows in rows_by_split.items()}
    validation = validate_dataset(args.output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_router_dataset": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "counts": counts,
        "validation_summary": validation["summary"],
        "label_counts": {
            split: dict(Counter(row["answer"]["route_label"] for row in rows))
            for split, rows in rows_by_split.items()
        },
        "source_method_counts": {
            split: dict(Counter(row.get("source_method") for row in rows))
            for split, rows in rows_by_split.items()
        },
        "category_counts": {
            split: dict(Counter(row.get("category") or "unclassified" for row in rows))
            for split, rows in rows_by_split.items()
        },
        "notes": [
            "Dataset is deterministic and retrieval-only; no online API or frontend path is used.",
            "Old router rows are converted without modifying the original router dataset.",
            "Additional rows target history references, short case names, theft amount constraints, typos, multi-case comparison, no-answer, non-criminal, and responsibility-evasion queries.",
            "Assistant messages contain only the strict Query Understanding JSON schema.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(make_jsonable(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_report.json").write_text(json.dumps(make_jsonable(validation), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(args.output_dir), "counts": counts, "validation_summary": validation["summary"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment old router data into retrieval-only Query Understanding SFT data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROUTER_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap for quick dry-runs.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        report = validate_dataset(args.output_dir)
        print(json.dumps(make_jsonable(report), ensure_ascii=False, indent=2))
        if report["summary"].get("row_error_count"):
            raise SystemExit(1)
        return
    result = build_dataset(args)
    print(json.dumps(make_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
