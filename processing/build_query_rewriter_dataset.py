from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import OUT_DIR
from query_rewriter_common import build_rewriter_messages, normalize_rewriter_answer, validate_rewriter_answer


DEFAULT_QU_DATASET = OUT_DIR / "query_understanding_lora" / "final_6500_v3_dataset"
DEFAULT_GPT_SEED = OUT_DIR / "query_rewriter_lora" / "gpt_authored_seed.jsonl"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_rewriter_lora" / "final_dataset"
SCHEMA_VERSION = "query-rewriter-lora-dataset-v1"

REWRITER_KEEP_CATEGORIES = {
    "multi_turn_rewrite",
    "history_case_reference",
    "typo_fuzzy_crime_alias",
    "amount_constraint",
    "typo_fuzzy",
    "multi_case_compare",
    "law_lookup",
    "short_case_name",
    "case_lookup",
    "field_intent",
    "amount_case_law",
    "multi_case",
}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}:invalid_json:{exc}") from exc
    return rows


def load_dataset_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        rows.extend(read_jsonl(dataset_dir / f"{split}.jsonl"))
    return rows


def latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def infer_rewrite_required(source_messages: list[dict[str, Any]], answer: dict[str, Any]) -> bool:
    latest = latest_user_query(source_messages)
    standalone = str(answer.get("standalone_query") or "")
    if len(source_messages) > 1:
        return True
    if answer.get("resolved_case_refs"):
        return True
    if latest and standalone and latest != standalone:
        return True
    return False


def rewriter_answer_from_qu_answer(source_messages: list[dict[str, Any]], qu_answer: dict[str, Any]) -> dict[str, Any]:
    answer = {
        "standalone_query": qu_answer.get("standalone_query") or latest_user_query(source_messages),
        "resolved_case_refs": qu_answer.get("resolved_case_refs") or [],
        "case_name_mentions": qu_answer.get("case_name_mentions") or [],
        "crime_mentions": qu_answer.get("crime_mentions") or [],
        "amount_constraints": qu_answer.get("amount_constraints") or [],
        "field_intents": qu_answer.get("field_intents") or [],
        "rewrite_required": False,
        "confidence": qu_answer.get("confidence", 0.86),
        "warnings": qu_answer.get("warnings") or [],
    }
    answer["rewrite_required"] = infer_rewrite_required(source_messages, answer)
    return normalize_rewriter_answer(answer)


def normalize_row(row: dict[str, Any], *, source_method: str | None = None) -> dict[str, Any] | None:
    source_messages = row.get("source_messages") or []
    if not source_messages:
        question = row.get("question") or (row.get("answer") or {}).get("standalone_query")
        if question:
            source_messages = [{"role": "user", "content": str(question)}]
    if not source_messages:
        return None
    raw_answer = row.get("answer") or {}
    answer = rewriter_answer_from_qu_answer(source_messages, raw_answer)
    errors = validate_rewriter_answer(answer)
    if errors:
        return None
    return {
        "id": row.get("id") or stable_hash(row),
        "split": row.get("split", "train"),
        "category": row.get("category") or "rewriter_existing",
        "source_messages": source_messages,
        "answer": answer,
        "messages": build_rewriter_messages(source_messages, answer),
        "source": row.get("source") or row.get("source_dataset") or "query_understanding_existing_dataset",
        "source_method": source_method or row.get("source_method") or "query_understanding_to_rewriter_v1",
        "text_hash": row.get("text_hash") or stable_hash(source_messages),
    }


def normalize_gpt_seed_row(row: dict[str, Any]) -> dict[str, Any] | None:
    source_messages = row.get("source_messages") or []
    answer = normalize_rewriter_answer(row.get("answer") or {})
    if validate_rewriter_answer(answer) or not source_messages:
        return None
    return {
        "id": row.get("id") or stable_hash(row),
        "split": row.get("split", "train"),
        "category": row.get("category") or "gpt_authored_rewriter_seed",
        "source_messages": source_messages,
        "answer": answer,
        "messages": build_rewriter_messages(source_messages, answer),
        "source": row.get("source") or "gpt_authored_rewriter_seed",
        "source_method": row.get("source_method") or "gpt_5_5_rewriter_seed_v1",
        "text_hash": row.get("text_hash") or stable_hash(source_messages),
    }


def generate_local_rewriter_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases = [
        ("王士东、吉得才盗窃罪、掩饰、隐瞒犯罪所得、犯罪所得收益罪一审刑事判决书", "王士东案", "盗窃罪", 43852),
        ("刘方祥、刘礼芬盗窃一审刑事判决书", "刘方祥案", "盗窃罪", 39136),
        ("吴必定抢劫案", "吴必定案", "抢劫罪", None),
        ("张君抢劫、故意杀人案", "张君案", "抢劫罪", None),
        ("周某入户盗窃罪一审刑事判决书", "周某入户盗窃案", "盗窃罪", 3000),
        ("李某故意伤害罪二审刑事裁定书", "李某故意伤害案", "故意伤害罪", None),
    ]
    followups = [
        ("第一个参考案例具体情况是什么", 1, "案情 法院认定事实", ["court_found_facts_text"]),
        ("第二个案例证据有哪些", 2, "证据 法院采信", ["evidence_text"]),
        ("刚才那个为什么这么判", 1, "裁判理由 量刑", ["reasoning_text", "judgment_text"]),
        ("这个案子被告人怎么辩护", 1, "辩护意见", ["defense_text"]),
        ("前面两个案例量刑差异在哪里", 0, "多案例 量刑差异 对比", ["reasoning_text", "judgment_text"]),
        ("第一个案子适用了哪些法条", 1, "适用法条 法律依据", ["legal_basis_text"]),
    ]
    rows = []
    for idx in range(count):
        case_a = rng.choice(cases)
        case_b = rng.choice([item for item in cases if item != case_a])
        query, ordinal, suffix, fields = rng.choice(followups)
        source_messages = [
            {"role": "user", "content": f"我想找{case_a[2]}的类似案例，最好有金额、案情和量刑参考。"},
            {
                "role": "assistant",
                "content": (
                    f"可以看两个参考案例。第一个参考案例是《{case_a[0]}》，简称{case_a[1]}，涉及{case_a[2]}"
                    f"{'，金额约' + str(case_a[3]) + '元' if case_a[3] else ''}。"
                    f"第二个参考案例是《{case_b[0]}》，简称{case_b[1]}，涉及{case_b[2]}"
                    f"{'，金额约' + str(case_b[3]) + '元' if case_b[3] else ''}。"
                ),
            },
            {"role": "user", "content": query},
        ]
        selected_cases = [case_a, case_b] if ordinal == 0 else [case_a if ordinal == 1 else case_b]
        standalone_parts = [item[0] for item in selected_cases] + [suffix]
        amount_constraints = []
        for item in selected_cases:
            if item[3]:
                amount_constraints.append({"value": item[3], "role": "theft_amount_total_or_single" if "盗窃" in item[2] else "unknown_amount", "raw_text": f"{item[3]}元"})
        answer = normalize_rewriter_answer(
            {
                "standalone_query": " ".join(standalone_parts),
                "resolved_case_refs": [
                    {
                        "ref": query,
                        "index": ordinal or item_idx,
                        "case_title": item[0],
                        "short_title": item[1],
                        "source": "assistant_history",
                    }
                    for item_idx, item in enumerate(selected_cases, 1)
                ],
                "case_name_mentions": [item[0] for item in selected_cases] + [item[1] for item in selected_cases],
                "crime_mentions": sorted({item[2] for item in selected_cases}),
                "amount_constraints": amount_constraints,
                "field_intents": fields,
                "rewrite_required": True,
                "confidence": 0.9,
                "warnings": [],
            }
        )
        rows.append(
            {
                "id": f"local_rewriter_{idx:06d}",
                "split": "train",
                "category": "local_multi_turn_rewriter",
                "source_messages": source_messages,
                "answer": answer,
                "messages": build_rewriter_messages(source_messages, answer),
                "source": "local_curated_rewriter_templates",
                "source_method": "local_curated_rewriter_template_v1",
                "text_hash": stable_hash(source_messages),
            }
        )
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for row in rows:
        key = stable_hash({"source_messages": row.get("source_messages"), "answer": row.get("answer")})
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def assign_splits(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    train_cut = int(len(rows) * 0.82)
    validation_cut = int(len(rows) * 0.91)
    final = []
    for idx, row in enumerate(rows, 1):
        row = dict(row)
        row["source_row_id"] = row.get("id")
        row["id"] = f"rewriter_qu_{idx:06d}"
        row["split"] = "train" if idx <= train_cut else "validation" if idx <= validation_cut else "test"
        final.append(row)
    return final


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Query Rewriter-only LoRA SFT dataset.")
    parser.add_argument("--qu-dataset", type=Path, default=DEFAULT_QU_DATASET)
    parser.add_argument("--gpt-seed", type=Path, default=DEFAULT_GPT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-template-count", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for row in load_dataset_rows(args.qu_dataset):
        if row.get("category") in REWRITER_KEEP_CATEGORIES:
            normalized = normalize_row(row, source_method=f"{row.get('source_method', 'existing')}_rewriter_projection_v1")
            if normalized:
                rows.append(normalized)
    for row in read_jsonl(args.gpt_seed):
        normalized = normalize_gpt_seed_row(row)
        if normalized:
            rows.append(normalized)
    rows.extend(generate_local_rewriter_rows(args.local_template_count, args.seed))
    rows = assign_splits(dedupe_rows(rows), args.seed)

    invalid = [{"id": row["id"], "errors": validate_rewriter_answer(row["answer"])} for row in rows if validate_rewriter_answer(row["answer"])]
    if invalid:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "validation_report.json").write_text(json.dumps({"invalid": invalid[:50]}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Invalid rows: {len(invalid)}")

    split_rows = {split: [row for row in rows if row["split"] == split] for split in ["train", "validation", "test"]}
    for split, items in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", items)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_qu_dataset": str(args.qu_dataset),
        "input_gpt_seed": str(args.gpt_seed),
        "output_dir": str(args.output_dir),
        "counts": {split: len(items) for split, items in split_rows.items()},
        "total_count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "source_method_counts": dict(Counter(row["source_method"] for row in rows)),
        "rewrite_required_count": sum(1 for row in rows if row["answer"].get("rewrite_required")),
        "rows_with_case_refs": sum(1 for row in rows if row["answer"].get("resolved_case_refs")),
        "rows_with_amounts": sum(1 for row in rows if row["answer"].get("amount_constraints")),
        "invalid_rows": 0,
        "notes": [
            "Rewriter-only dataset. It intentionally excludes route_label, needs_retrieval and retrieval_targets from assistant answers.",
            "Rows are projected from previous Query Understanding data, GPT-authored seed data when available, and local curated multi-turn templates.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
