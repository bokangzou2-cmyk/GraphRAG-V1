from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import OUT_DIR


DEFAULT_DATASET_DIR = OUT_DIR / "query_understanding_lora" / "gpt_authored_dataset"
SYSTEM_PROMPT = (
    "你是刑法 GraphRAG 的 Query Understanding 模块，只服务于检索前处理。"
    "请把多轮问题改写成可检索的独立问题，并识别案名、罪名、金额、检索范围和字段意图。"
    "必须只输出一个 JSON 对象，不要 Markdown，不要解释。"
)
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
REQUIRED_QU_FIELDS = [
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
REQUIRED_ROW_FIELDS = [
    "id",
    "split",
    "category",
    "source_messages",
    "answer",
    "messages",
    "source",
    "source_method",
    "text_hash",
]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}:invalid_json:{exc}") from exc
        rows.append(row)
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = row["answer"]
    row["source"] = row.get("source") or "gpt_5_5_direct_authored"
    row["source_method"] = row.get("source_method") or "gpt_5_5_direct_authored_jsonl_v1"
    row["category"] = row.get("category") or answer.get("route_label") or "gpt_authored"
    source_messages = row.get("source_messages") or []
    if not row.get("messages"):
        row["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "最近对话：\n" + "\n".join(f"{m.get('role')}: {m.get('content')}" for m in source_messages)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
        ]
    else:
        row["messages"][-1]["content"] = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    row["text_hash"] = row.get("text_hash") or stable_hash(source_messages)
    return row


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            errors.append(f"missing_row_field:{field}")
    answer = row.get("answer")
    if not isinstance(answer, dict):
        errors.append("answer_not_object")
        return errors
    for field in REQUIRED_QU_FIELDS:
        if field not in answer:
            errors.append(f"missing_answer_field:{field}")
    if answer.get("route_label") not in VALID_LABELS:
        errors.append(f"invalid_route_label:{answer.get('route_label')}")
    messages = row.get("messages") or []
    if len(messages) < 3:
        errors.append("messages_too_short")
    else:
        try:
            parsed = json.loads(messages[-1].get("content") or "")
            if parsed != answer:
                errors.append("assistant_content_not_equal_answer")
        except json.JSONDecodeError:
            errors.append("assistant_content_not_json")
    return errors


def split_for_id(row_id: str, target_count: int) -> str:
    try:
        number = int(row_id.rsplit("_", 1)[-1])
        bucket = number / max(target_count, 1)
    except ValueError:
        bucket = int(stable_hash(row_id)[:8], 16) / 0xFFFFFFFF
    if bucket <= 0.8:
        return "train"
    if bucket <= 0.9:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge GPT-authored QU JSONL shards into train/validation/test files.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target-count", type=int, default=120)
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.dataset_dir / "shards"
    existing_main = []
    for name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        existing_main.extend(iter_jsonl(args.dataset_dir / name))
    shard_rows = []
    shard_paths = []
    if shard_dir.exists():
        shard_paths.extend(sorted(shard_dir.glob("*.jsonl")))
    shard_paths.extend(sorted(args.dataset_dir.glob("shard_*.jsonl")))
    for path in shard_paths:
        shard_rows.extend(iter_jsonl(path))

    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in existing_main + shard_rows:
        if "id" not in row:
            continue
        row = normalize_row(row)
        row["split"] = split_for_id(row["id"], args.target_count)
        rows_by_id[row["id"]] = row

    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    invalid = []
    for row in rows:
        errors = validate_row(row)
        if errors:
            invalid.append({"id": row.get("id"), "errors": errors})

    split_rows = {"train": [], "validation": [], "test": []}
    for row in rows:
        split_rows[row["split"]].append(row)
    for split, split_values in split_rows.items():
        path = args.dataset_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in split_values:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts = {split: len(values) for split, values in split_rows.items()}
    category_counts = Counter(row.get("category") for row in rows)
    multi_turn_count = category_counts.get("multi_turn_rewrite", 0)
    now = datetime.now(timezone.utc).isoformat()
    last_id = rows[-1]["id"] if rows else None
    progress = {
        "target_count": args.target_count,
        "generated_count": len(rows),
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "last_id": last_id,
        "completed_batches": len(rows) // 10,
        "updated_at": now,
        "resume_supported": True,
    }
    manifest = {
        "schema_version": "query_understanding_sft_v1",
        "source_method": "gpt_5_5_direct_authored_jsonl_v1",
        "target_count": args.target_count,
        "generated_count": len(rows),
        "split_counts": split_counts,
        "category_counts": dict(category_counts),
        "multi_turn_count": multi_turn_count,
        "multi_turn_share": round(multi_turn_count / len(rows), 4) if rows else 0.0,
        "created_at": now,
        "updated_at": now,
        "notes": "Merged from direct GPT-5.5-authored JSONL rows and shards. Scripts only merge/validate; content is authored by subagents.",
    }
    validation = {
        "jsonl_rows": len(rows),
        "invalid_rows": len(invalid),
        "missing_fields": {},
        "split_counts": split_counts,
        "category_counts": dict(category_counts),
        "multi_turn_count": multi_turn_count,
        "multi_turn_share": round(multi_turn_count / len(rows), 4) if rows else 0.0,
        "assistant_json_valid_count": len(rows) - sum(1 for item in invalid if "assistant_content_not_json" in item["errors"]),
        "sample_errors": invalid[:20],
    }
    (args.dataset_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.dataset_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.dataset_dir / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "invalid": len(invalid), "split_counts": split_counts, "category_counts": dict(category_counts)}, ensure_ascii=False, indent=2))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
