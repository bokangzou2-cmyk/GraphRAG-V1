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


DEFAULT_BASE_DATASET = OUT_DIR / "query_rewriter_lora" / "final_gpt_dataset_v3"
DEFAULT_EXTRA_SHARDS = OUT_DIR / "query_rewriter_lora" / "gpt_authored_v4_conservative_shards"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_rewriter_lora" / "final_gpt_dataset_v4_conservative"
SCHEMA_VERSION = "query-rewriter-gpt-authored-v4-conservative"
SPLITS = ("train", "validation", "test")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_messages_from_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_messages: list[dict[str, Any]] = []
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
        if content:
            return json.loads(content)
    return {}


def normalize_row(row: dict[str, Any], *, source_file: str, default_source: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    source_messages = row.get("source_messages")
    answer = row.get("answer")

    if (not isinstance(source_messages, list) or not source_messages) and isinstance(row.get("messages"), list):
        try:
            source_messages = source_messages_from_chat_messages(row["messages"])
            answer = answer_from_chat_messages(row["messages"])
        except (json.JSONDecodeError, TypeError) as exc:
            return None, [f"invalid_messages_answer:{exc}"]

    if not isinstance(source_messages, list) or not source_messages:
        return None, ["missing_source_messages"]
    if not isinstance(answer, dict):
        return None, ["missing_answer"]

    normalized_answer = normalize_rewriter_answer(answer)
    errors.extend(validate_rewriter_answer(normalized_answer))
    if errors:
        return None, errors

    source = str(row.get("source") or default_source)
    source_method = str(row.get("source_method") or f"{default_source}:{source_file}")
    normalized = {
        "id": row.get("id") or stable_hash({"source_messages": source_messages, "answer": normalized_answer}),
        "split": "train",
        "category": row.get("category") or "rewriter_v4_conservative",
        "source_messages": source_messages,
        "answer": normalized_answer,
        "messages": build_rewriter_messages(source_messages, normalized_answer),
        "source": source,
        "source_method": source_method,
        "text_hash": row.get("text_hash") or stable_hash(source_messages),
    }
    return normalized, []


def load_base_rows(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for split in SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        for index, raw in enumerate(read_jsonl(path), 1):
            normalized, errors = normalize_row(raw, source_file=path.name, default_source=str(raw.get("source") or "query_rewriter_v3_base"))
            if errors:
                invalid.append({"file": str(path), "line": index, "id": raw.get("id"), "errors": errors})
            elif normalized:
                normalized["source_dataset"] = "final_gpt_dataset_v3"
                rows.append(normalized)
    return rows, invalid


def load_extra_rows(shard_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    shard_files = sorted(shard_dir.glob("shard_*.jsonl"))
    for shard in shard_files:
        for index, raw in enumerate(read_jsonl(shard), 1):
            normalized, errors = normalize_row(raw, source_file=shard.name, default_source="gpt_5_4_direct_rewriter_v4_conservative")
            if errors:
                invalid.append({"file": str(shard), "line": index, "id": raw.get("id"), "errors": errors})
            elif normalized:
                normalized["source_dataset"] = "gpt_authored_v4_conservative_shards"
                rows.append(normalized)
    return rows, invalid, [str(path) for path in shard_files]


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
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
    final: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        item = dict(row)
        item["source_row_id"] = item.get("id")
        item["id"] = f"rewriter_gpt_v4_{index:06d}"
        item["split"] = "train" if index <= train_cut else "validation" if index <= val_cut else "test"
        final.append(item)
    return final


def conservative_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    no_rewrite = 0
    no_crime = 0
    general_like = 0
    for row in rows:
        answer = row["answer"]
        question = "\n".join(str(message.get("content") or "") for message in row.get("source_messages") or [] if message.get("role") == "user")
        if not answer.get("rewrite_required"):
            no_rewrite += 1
        if not answer.get("crime_mentions"):
            no_crime += 1
        if any(token in question for token in ["刑法主要", "刑法管", "刑法涉及", "违法行为", "犯罪一般", "刑事犯罪一般"]):
            general_like += 1
    return {"rewrite_required_false": no_rewrite, "empty_crime_mentions": no_crime, "general_like_questions": general_like}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Query Rewriter v4 conservative dataset from v3 plus GPT conservative shards.")
    parser.add_argument("--base-dataset-dir", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--extra-shards-dir", type=Path, default=DEFAULT_EXTRA_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-total-count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    base_rows, base_invalid = load_base_rows(args.base_dataset_dir)
    extra_rows, extra_invalid, shard_files = load_extra_rows(args.extra_shards_dir)
    all_rows = assign_splits(dedupe([*base_rows, *extra_rows]), args.seed)

    final_invalid = []
    for row in all_rows:
        errors = validate_rewriter_answer(row["answer"])
        if errors:
            final_invalid.append({"id": row["id"], "errors": errors})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = {
        "base_invalid_rows": base_invalid[:100],
        "extra_invalid_rows": extra_invalid[:100],
        "final_invalid_rows": final_invalid[:100],
    }
    if base_invalid or extra_invalid or final_invalid:
        (args.output_dir / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Invalid rows: base={len(base_invalid)} extra={len(extra_invalid)} final={len(final_invalid)}")
    if len(all_rows) < args.min_total_count:
        raise SystemExit(f"dataset_too_small:{len(all_rows)} < {args.min_total_count}")

    split_rows = {split: [row for row in all_rows if row["split"] == split] for split in SPLITS}
    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_dataset_dir": str(args.base_dataset_dir),
        "extra_shards_dir": str(args.extra_shards_dir),
        "extra_shard_files": shard_files,
        "base_rows_loaded": len(base_rows),
        "extra_rows_loaded": len(extra_rows),
        "total_count": len(all_rows),
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "category_counts": dict(Counter(row["category"] for row in all_rows)),
        "source_counts": dict(Counter(row["source"] for row in all_rows)),
        "source_dataset_counts": dict(Counter(row.get("source_dataset") for row in all_rows)),
        "conservative_stats": conservative_stats(all_rows),
        "invalid_rows": 0,
        "notes": [
            "v4 keeps all valid v3 rows and adds GPT-authored conservative no-overrewrite examples.",
            "The target behavior is to preserve clear/general questions and avoid adding user-unstated crimes, cases, amounts, or retrieval terms.",
            "Router decision fields remain forbidden from Rewriter answers.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "validation_report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
