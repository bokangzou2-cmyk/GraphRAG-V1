from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import OUT_DIR


DEFAULT_DATASET_DIR = OUT_DIR / "query_understanding_lora" / "gpt_authored_dataset"
SYSTEM_PROMPT = (
    "你是刑法 GraphRAG 的 Query Understanding 模块，只服务于检索前处理。"
    "请把多轮问题改写成可检索的独立问题，并识别案名、罪名、金额、检索范围和字段意图。"
    "必须只输出一个 JSON 对象，不要 Markdown，不要解释。"
)
REQUIRED_LIST_FIELDS = [
    "retrieval_targets",
    "case_name_mentions",
    "resolved_case_refs",
    "field_intents",
    "crime_mentions",
    "amount_constraints",
    "warnings",
]
REQUIRED_SCALAR_DEFAULTS = {
    "standalone_query": "",
    "route_label": "low_quality_or_incomplete",
    "needs_retrieval": False,
    "case_required": False,
    "law_required": False,
    "confidence": 0.7,
}
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


def repair_answer(answer: dict[str, Any]) -> bool:
    changed = False
    for field in REQUIRED_LIST_FIELDS:
        if field not in answer:
            answer[field] = []
            changed = True
    for field, default in REQUIRED_SCALAR_DEFAULTS.items():
        if field not in answer:
            answer[field] = default
            changed = True
    label = answer.get("route_label")
    if label in ROUTE_LABEL_MAP:
        answer["route_label"] = ROUTE_LABEL_MAP[label]
        changed = True
    return changed


def repair_file(path: Path) -> int:
    rows = []
    changed_count = 0
    if not path.exists():
        return 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = row.get("answer")
        if isinstance(answer, dict) and repair_answer(answer):
            changed_count += 1
        messages = row.get("messages") or []
        if isinstance(answer, dict):
            assistant_content = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
            if len(messages) < 3:
                source_messages = row.get("source_messages") or []
                user_content = "最近对话：\n" + "\n".join(
                    f"{message.get('role')}: {message.get('content')}" for message in source_messages
                )
                row["messages"] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                changed_count += 1
            else:
                messages[-1]["content"] = assistant_content
        rows.append(row)
    if changed_count:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair schema-only issues in GPT-authored QU dataset shards.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    changed = {}
    for path in sorted((args.dataset_dir / "shards").glob("*.jsonl")):
        count = repair_file(path)
        if count:
            changed[str(path)] = count
    print(json.dumps({"changed": changed, "changed_rows": sum(changed.values())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
