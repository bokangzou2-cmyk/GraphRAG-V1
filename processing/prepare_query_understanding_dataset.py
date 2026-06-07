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
from query_understanding_probe import (
    amount_constraints_for,
    crime_mentions_for,
    default_eval_cases,
    extract_case_mentions,
    field_intents_for,
    query_understanding_prompt,
    rules_understand,
)


DEFAULT_ROUTER_DATASET = OUT_DIR / "router_lora" / "dataset"
DEFAULT_OUTPUT_DIR = OUT_DIR / "query_understanding_lora" / "dataset"
SCHEMA_VERSION = "query-understanding-lora-dataset-v1"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def qu_answer_from_router_row(row: dict[str, Any]) -> dict[str, Any]:
    question = row["question"]
    answer = row["answer"]
    messages = [{"role": "user", "content": question}]
    base = {
        "standalone_query": question,
        "route_label": row["label"],
        "needs_retrieval": bool(answer.get("needs_retrieval")),
        "retrieval_targets": list(answer.get("retrieval_targets") or ["none"]),
        "case_required": bool(answer.get("case_required")),
        "law_required": bool(answer.get("law_required")),
        "case_name_mentions": extract_case_mentions(question),
        "resolved_case_refs": [],
        "field_intents": field_intents_for(question, []),
        "crime_mentions": crime_mentions_for(question, messages),
        "amount_constraints": amount_constraints_for(question),
        "confidence": 0.86,
        "warnings": [],
    }
    if not base["needs_retrieval"]:
        base["confidence"] = 0.78
    return base


def build_messages(source_messages: list[dict[str, Any]], answer: dict[str, Any]) -> list[dict[str, str]]:
    prompt_messages = query_understanding_prompt(source_messages)
    return prompt_messages + [{"role": "assistant", "content": compact_json(answer)}]


def convert_router_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    answer = qu_answer_from_router_row(row)
    source_messages = [{"role": "user", "content": row["question"]}]
    return {
        "id": f"router_to_qu_{row['id']}",
        "split": split,
        "question": row["question"],
        "source_messages": source_messages,
        "answer": answer,
        "messages": build_messages(source_messages, answer),
        "source": row.get("source"),
        "source_method": "router_dataset_conversion_v1",
        "text_hash": stable_hash(row["question"]),
    }


def convert_probe_case(case: dict[str, Any], split: str) -> dict[str, Any]:
    answer = rules_understand(case["messages"])
    answer.pop("_raw_output", None)
    question = next((m["content"] for m in reversed(case["messages"]) if m.get("role") == "user"), "")
    return {
        "id": f"probe_to_qu_{case['id']}",
        "split": split,
        "question": question,
        "source_messages": case["messages"],
        "answer": answer,
        "messages": build_messages(case["messages"], answer),
        "source": "query_understanding_probe_eval_sample",
        "source_method": "history_boundary_conversion_v1",
        "text_hash": stable_hash(json.dumps(case["messages"], ensure_ascii=False)),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert old router LoRA data into retrieval-only Query Understanding SFT data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROUTER_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap for quick dry-runs.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    source_method_counts: dict[str, dict[str, int]] = {}

    for split in ["train", "validation", "test"]:
        rows = [convert_router_row(row, split) for row in load_router_split(args.input_dir, split)]
        if split == "train" and args.max_train:
            rng.shuffle(rows)
            rows = rows[: args.max_train]
        counts[split] = write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        label_counts[split] = dict(Counter(row["answer"]["route_label"] for row in rows))
        source_method_counts[split] = dict(Counter(row["source_method"] for row in rows))

    probe_cases = default_eval_cases()
    probe_rows = []
    for index, case in enumerate(probe_cases):
        split = "validation" if index % 3 == 0 else "train"
        probe_rows.append(convert_probe_case(case, split))
    for split in ["train", "validation"]:
        selected = [row for row in probe_rows if row["split"] == split]
        if not selected:
            continue
        path = args.output_dir / f"{split}.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] += len(selected)
        label_counts[split] = dict(Counter(label_counts[split]) + Counter(row["answer"]["route_label"] for row in selected))
        source_method_counts[split] = dict(Counter(source_method_counts[split]) + Counter(row["source_method"] for row in selected))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_router_dataset": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "counts": counts,
        "label_counts": label_counts,
        "source_method_counts": source_method_counts,
        "notes": [
            "This is a retrieval-only Query Understanding SFT dataset draft.",
            "Rows converted from the old router dataset preserve the old route label while adding standalone_query, field_intents, crime_mentions, case mentions, and amount constraints with deterministic rules.",
            "History-reference boundary rows are added from query_understanding_probe defaults.",
            "Use this as a seed dataset; add DeepSeek/subagent paraphrases before production finetuning if higher diversity is required.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
