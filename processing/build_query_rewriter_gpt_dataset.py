from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import OUT_DIR
from query_rewriter_common import build_rewriter_messages, normalize_rewriter_answer, validate_rewriter_answer


DEFAULT_SHARD_DIR = OUT_DIR / "query_rewriter_lora" / "gpt_authored_v2_shards"
DEFAULT_OLD_QU_DATASET = OUT_DIR / "query_understanding_lora" / "final_6500_v3_dataset"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_rewriter_lora" / "final_gpt_dataset_v2"
SCHEMA_VERSION = "query-rewriter-gpt-authored-v3"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}:invalid_json:{exc}") from exc
    return rows


def load_dataset_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ["train", "validation", "test"]:
        rows.extend(read_jsonl(dataset_dir / f"{split}.jsonl"))
    return rows


def latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def source_messages_from_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_messages = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        source_messages.append({"role": message.get("role"), "content": str(message.get("content") or "")})
    return source_messages


def answer_from_chat_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            break
        return json.loads(content)
    return {}


def normalize_resolved_case_refs(items: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(items, list):
        return normalized
    for index, item in enumerate(items, 1):
        if isinstance(item, dict):
            normalized.append(item)
        elif str(item).strip():
            text = str(item).strip()
            normalized.append({"ref": text, "index": index, "case_title": text, "source": "dialogue_context"})
    return normalized


def normalize_amount_constraints(items: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value", item.get("normalized_amount_cny"))
        raw_text = item.get("raw_text", item.get("amount_text"))
        if value is None and raw_text:
            digits = re.sub(r"[^\d.]", "", str(raw_text))
            if digits:
                try:
                    value = int(float(digits))
                except ValueError:
                    value = None
        if value is None:
            continue
        normalized.append(
            {
                "value": value,
                "role": item.get("role") or "unknown_amount",
                "raw_text": raw_text or (f"{value}元" if value is not None else ""),
            }
        )
    return normalized


def normalize_field_intents(items: Any) -> list[str]:
    mapping = {
        "compare_sentencing": ["judgment_text", "reasoning_text"],
        "compare_facts": ["court_found_facts_text"],
        "compare_evidence": ["evidence_text"],
        "compare_defense": ["defense_text"],
        "compare_reasoning": ["reasoning_text"],
        "law_article_text": ["legal_basis_text"],
    }
    normalized = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        text = str(item)
        mapped = mapping.get(text, [text])
        for value in mapped:
            if value not in normalized:
                normalized.append(value)
    return normalized


def clean_gpt_answer(answer: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(answer)
    cleaned["resolved_case_refs"] = normalize_resolved_case_refs(cleaned.get("resolved_case_refs"))
    cleaned["amount_constraints"] = normalize_amount_constraints(cleaned.get("amount_constraints"))
    cleaned["field_intents"] = normalize_field_intents(cleaned.get("field_intents"))
    return cleaned


def normalize_gpt_row(row: dict[str, Any], source_file: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors = []
    source_messages = row.get("source_messages")
    answer = row.get("answer")
    if (not isinstance(source_messages, list) or not source_messages) and isinstance(row.get("messages"), list):
        try:
            source_messages = source_messages_from_chat_messages(row["messages"])
            answer = answer_from_chat_messages(row["messages"])
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"invalid_messages_answer:{exc}")
            return None, errors
    if not isinstance(source_messages, list) or not source_messages:
        errors.append("missing_source_messages")
        return None, errors
    answer = normalize_rewriter_answer(clean_gpt_answer(answer or {}))
    errors.extend(validate_rewriter_answer(answer))
    if errors:
        return None, errors
    normalized = {
        "id": row.get("id") or stable_hash(row),
        "split": "train",
        "category": row.get("category") or "gpt_authored_rewriter",
        "source_messages": source_messages,
        "answer": answer,
        "messages": build_rewriter_messages(source_messages, answer),
        "source": "gpt_direct_authored_rewriter_v2",
        "source_method": f"gpt_5_5_direct_rewriter_{source_file}",
        "text_hash": stable_hash(source_messages),
    }
    return normalized, []


def answer_from_old_qu(row: dict[str, Any]) -> dict[str, Any]:
    old = row.get("answer") or {}
    source_messages = row.get("source_messages") or [{"role": "user", "content": old.get("standalone_query") or ""}]
    answer = normalize_rewriter_answer(
        clean_gpt_answer(
        {
            "standalone_query": old.get("standalone_query") or latest_user_query(source_messages),
            "resolved_case_refs": old.get("resolved_case_refs") or [],
            "case_name_mentions": old.get("case_name_mentions") or [],
            "crime_mentions": old.get("crime_mentions") or [],
            "amount_constraints": old.get("amount_constraints") or [],
            "field_intents": old.get("field_intents") or [],
            "rewrite_required": bool(old.get("resolved_case_refs") or latest_user_query(source_messages) != old.get("standalone_query")),
            "confidence": old.get("confidence", 0.86),
            "warnings": old.get("warnings") or [],
        }
        )
    )
    return answer


def selected_old_rows(dataset_dir: Path, cap: int, rng: random.Random) -> list[dict[str, Any]]:
    categories = {"multi_turn_rewrite", "history_case_reference", "multi_case_compare", "short_case_name", "amount_case_law"}
    candidates = []
    for old in load_dataset_rows(dataset_dir):
        if old.get("category") not in categories:
            continue
        answer = answer_from_old_qu(old)
        if validate_rewriter_answer(answer):
            continue
        source_messages = old.get("source_messages") or []
        if not source_messages:
            continue
        if old.get("category") == "multi_turn_rewrite" and not answer.get("resolved_case_refs"):
            continue
        candidates.append(
            {
                "id": old.get("id") or stable_hash(old),
                "split": "train",
                "category": f"selected_old_{old.get('category')}",
                "source_messages": source_messages,
                "answer": answer,
                "messages": build_rewriter_messages(source_messages, answer),
                "source": "selected_old_qu_v3",
                "source_method": "selected_old_qu_v3_rewriter_projection",
                "text_hash": stable_hash(source_messages),
            }
        )
    rng.shuffle(candidates)
    return candidates[:cap]


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
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
    val_cut = int(len(rows) * 0.91)
    final = []
    for idx, row in enumerate(rows, 1):
        row = dict(row)
        row["source_row_id"] = row.get("id")
        row["id"] = f"rewriter_gpt_v3_{idx:06d}"
        row["split"] = "train" if idx <= train_cut else "validation" if idx <= val_cut else "test"
        final.append(row)
    return final


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge direct GPT-authored Query Rewriter shards into a training dataset.")
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--old-qu-dataset", type=Path, default=DEFAULT_OLD_QU_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-cap", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    shard_files = sorted(args.shard_dir.glob("shard_*.jsonl"))
    for shard in shard_files:
        for index, raw in enumerate(read_jsonl(shard), 1):
            normalized, errors = normalize_gpt_row(raw, shard.name)
            if errors:
                invalid.append({"file": str(shard), "line": index, "id": raw.get("id"), "errors": errors})
            elif normalized:
                rows.append(normalized)

    gpt_count_before_dedupe = len(rows)
    rows.extend(selected_old_rows(args.old_qu_dataset, args.old_cap, rng))
    rows = assign_splits(dedupe(rows), args.seed)

    forbidden_invalid = []
    for row in rows:
        errors = validate_rewriter_answer(row["answer"])
        if errors:
            forbidden_invalid.append({"id": row["id"], "errors": errors})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if invalid or forbidden_invalid:
        (args.output_dir / "validation_report.json").write_text(
            json.dumps({"invalid_source_rows": invalid[:100], "invalid_final_rows": forbidden_invalid[:100]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(f"Invalid rows: source={len(invalid)} final={len(forbidden_invalid)}")

    split_rows = {split: [row for row in rows if row["split"] == split] for split in ["train", "validation", "test"]}
    for split, items in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", items)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shard_dir": str(args.shard_dir),
        "shard_files": [str(path) for path in shard_files],
        "gpt_rows_before_dedupe": gpt_count_before_dedupe,
        "old_qu_cap": args.old_cap,
        "counts": {split: len(items) for split, items in split_rows.items()},
        "total_count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "source_method_counts": dict(Counter(row["source_method"] for row in rows)),
        "direct_gpt_rows_final": sum(1 for row in rows if row["source"] == "gpt_direct_authored_rewriter_v2"),
        "selected_old_rows_final": sum(1 for row in rows if row["source"] == "selected_old_qu_v3"),
        "rewrite_required_count": sum(1 for row in rows if row["answer"].get("rewrite_required")),
        "rows_with_case_refs": sum(1 for row in rows if row["answer"].get("resolved_case_refs")),
        "rows_with_amounts": sum(1 for row in rows if row["answer"].get("amount_constraints")),
        "invalid_rows": 0,
        "notes": [
            "Primary data source is direct GPT-authored JSONL shards.",
            "Selected old QU rows are capped and only retained for known-good multi-turn patterns.",
            "No router decision fields are present in assistant answers.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
