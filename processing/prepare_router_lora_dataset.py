from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from common import OUT_DIR, iter_jsonl


QUESTION_DIR = OUT_DIR / "router_training_questions"
DEFAULT_INPUT = QUESTION_DIR / "router_questions_labeled_corrected.jsonl"
DEFAULT_OUTPUT_DIR = OUT_DIR / "router_lora" / "dataset"

SYSTEM_PROMPT = (
    "你是一个刑法问答系统的问题路由分类器。"
    "只根据用户问题判断是否需要法律检索以及需要检索什么。"
    "必须只输出一个 JSON 对象，不要解释，不要输出 Markdown。"
)

LABEL_SCHEMA = {
    "general_no_legal_retrieval": {
        "description": "普通非法律问题，不触发法律检索。",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
    },
    "non_criminal_legal": {
        "description": "非刑法法律问题，本刑法系统不触发检索。",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
    },
    "criminal_law_law_only": {
        "description": "刑法概念、构成要件、量刑规则、程序规则等，只需检索法条或规范。",
        "needs_retrieval": True,
        "retrieval_targets": ["law_articles"],
    },
    "criminal_law_case_only": {
        "description": "明确要求案例、判例、类案或裁判观点，主要检索案例。",
        "needs_retrieval": True,
        "retrieval_targets": ["cases"],
    },
    "criminal_law_case_and_law": {
        "description": "刑法问题同时需要法条依据和案例辅助。",
        "needs_retrieval": True,
        "retrieval_targets": ["law_articles", "cases"],
    },
    "criminal_law_multi_case": {
        "description": "要求多个案例、类案比较、趋势归纳或案例群分析。",
        "needs_retrieval": True,
        "retrieval_targets": ["cases", "graph"],
    },
    "criminal_law_advice": {
        "description": "具体个人刑事处境咨询，优先检索法条，必要时由后续链路再决定是否扩展案例。",
        "needs_retrieval": True,
        "retrieval_targets": ["law_articles"],
    },
    "unclear_need_clarification": {
        "description": "问题语义不清或缺少关键信息，需要澄清。",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
    },
    "low_quality_or_incomplete": {
        "description": "问题残缺、噪声过多或不可判断。",
        "needs_retrieval": False,
        "retrieval_targets": ["none"],
    },
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_user_prompt(question: str) -> str:
    labels = "\n".join(
        f"- {label}: {meta['description']}" for label, meta in LABEL_SCHEMA.items()
    )
    return (
        "请对下面的用户问题做路由分类。\n\n"
        f"可选标签：\n{labels}\n\n"
        "输出 JSON 字段固定为："
        "label, needs_retrieval, retrieval_targets, case_required, law_required, clarify_required。\n\n"
        f"用户问题：{question}"
    )


def build_answer_payload(label_payload: dict) -> dict:
    return {
        "label": label_payload["label"],
        "needs_retrieval": bool(label_payload["needs_legal_retrieval"]),
        "retrieval_targets": label_payload["retrieval_targets"],
        "case_required": bool(label_payload["case_required"]),
        "law_required": bool(label_payload["law_required"]),
        "clarify_required": bool(label_payload["clarify_required"]),
    }


def route_group(label: str) -> str:
    if label.startswith("criminal_law_"):
        return "criminal_law"
    if label == "non_criminal_legal":
        return "non_criminal_legal"
    return "general_or_other"


def build_messages(row: dict) -> list[dict]:
    answer = build_answer_payload(row["label_payload"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row["question"])},
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
    ]


def stratified_split(rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["label_payload"]["label"]].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_n = int(total * train_ratio)
        val_n = int(total * val_ratio)
        if total >= 3:
            train_n = max(1, min(train_n, total - 2))
            val_n = max(1, min(val_n, total - train_n - 1))
        splits["train"].extend(label_rows[:train_n])
        splits["validation"].extend(label_rows[train_n : train_n + val_n])
        splits["test"].extend(label_rows[train_n + val_n :])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_row(row: dict, split: str) -> dict:
    label_payload = row["label_payload"]
    label = label_payload["label"]
    return {
        "id": row["id"],
        "split": split,
        "question": row["question"],
        "label": label,
        "route_group": route_group(label),
        "answer": build_answer_payload(label_payload),
        "messages": build_messages(row),
        "source": row.get("source"),
        "source_method": row.get("source_method"),
        "text_hash": stable_hash(row["question"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare router classification data for Qwen LoRA SFT.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("--train-ratio and --val-ratio must be positive and sum to less than 1.")

    rows = list(iter_jsonl(args.input))
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    missing_payload = [row.get("id", "<missing-id>") for row in rows if "label_payload" not in row]
    if missing_payload:
        raise SystemExit(f"Rows missing label_payload: {missing_payload[:5]}")

    unknown_labels = sorted({row["label_payload"]["label"] for row in rows} - set(LABEL_SCHEMA))
    if unknown_labels:
        raise SystemExit(f"Unknown labels: {unknown_labels}")

    splits = stratified_split(rows, args.train_ratio, args.val_ratio, args.seed)
    counts = {}
    label_counts = {}
    route_group_counts = {}
    for split, split_rows in splits.items():
        converted = [convert_row(row, split) for row in split_rows]
        counts[split] = write_jsonl(args.output_dir / f"{split}.jsonl", converted)
        label_counts[split] = Counter(row["label"] for row in converted)
        route_group_counts[split] = Counter(row["route_group"] for row in converted)

    manifest = {
        "schema_version": "router-lora-dataset-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "counts": counts,
        "label_counts": label_counts,
        "route_group_counts": route_group_counts,
        "system_prompt": SYSTEM_PROMPT,
        "label_schema": LABEL_SCHEMA,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
