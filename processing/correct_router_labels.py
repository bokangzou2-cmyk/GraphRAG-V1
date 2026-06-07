from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import OUT_DIR


QUESTION_DIR = OUT_DIR / "router_training_questions"
INPUT = QUESTION_DIR / "router_questions_labeled_deepseek.jsonl"
CORRECTED_OUT = QUESTION_DIR / "router_questions_labeled_corrected.jsonl"
REPORT_OUT = QUESTION_DIR / "router_label_correction_report.json"

CRIMINAL_PROCEDURE_RE = re.compile(r"犯罪嫌疑人|刑事诉讼|逮捕|取保候审|公诉|检察院|公安机关|侦查|辩护人|探监|无犯罪记录|判刑|拘役|缓刑")
CRIME_RE = re.compile(r"[\u4e00-\u9fff]{2,20}罪|犯罪|刑法|刑事责任|量刑|判刑|有期徒刑|拘役|缓刑|死刑")
GENERAL_LEGAL_RE = re.compile(r"工伤|劳动|合同|离婚|继承|法院|起诉|诉讼|法律|税率|仲裁|赔偿|房产|租赁|公司法|行政诉讼")
PERSONAL_SUBJECT_RE = re.compile(r"我|朋友|老公|老婆|女友|男友|家属|孩子|父亲|母亲|哥哥|姐姐|老板|员工|别人")
ADVICE_ACTION_RE = re.compile(r"怎么办|会不会|能不能|可以吗|如何争取|怎么处理|被抓|报警|坐牢|赔偿|开除|上学|查出来|影响")
ABSTRACT_RULE_RE = re.compile(r"标准|条件|规定|构成要件|特征|是什么|哪些|如何处罚|怎么规定|什么意思|立案标准|适用|能否适用")


LABEL_SHAPES = {
    "general_no_legal_retrieval": {
        "domain": "general",
        "needs_legal_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": False,
    },
    "non_criminal_legal": {
        "domain": "non_criminal_legal",
        "needs_legal_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": False,
    },
    "criminal_law_law_only": {
        "domain": "criminal_law",
        "needs_legal_retrieval": True,
        "retrieval_targets": ["law_articles"],
        "case_required": False,
        "law_required": True,
        "clarify_required": False,
    },
    "criminal_law_case_only": {
        "domain": "criminal_law",
        "needs_legal_retrieval": True,
        "retrieval_targets": ["cases"],
        "case_required": True,
        "law_required": False,
        "clarify_required": False,
    },
    "criminal_law_case_and_law": {
        "domain": "criminal_law",
        "needs_legal_retrieval": True,
        "retrieval_targets": ["cases", "law_articles"],
        "case_required": True,
        "law_required": True,
        "clarify_required": False,
    },
    "criminal_law_multi_case": {
        "domain": "criminal_law",
        "needs_legal_retrieval": True,
        "retrieval_targets": ["cases", "graph"],
        "case_required": True,
        "law_required": False,
        "clarify_required": False,
    },
    "criminal_law_advice": {
        "domain": "criminal_law",
        "needs_legal_retrieval": True,
        "retrieval_targets": ["law_articles"],
        "case_required": False,
        "law_required": True,
        "clarify_required": False,
    },
    "unclear_need_clarification": {
        "domain": "unclear",
        "needs_legal_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": True,
    },
    "low_quality_or_incomplete": {
        "domain": "low_quality",
        "needs_legal_retrieval": False,
        "retrieval_targets": ["none"],
        "case_required": False,
        "law_required": False,
        "clarify_required": False,
    },
}

MANUAL_OVERRIDES = {
    "disc_law_sft_criminal_66d8e17d819bbcbd": "criminal_law_law_only",
    "disc_law_sft_criminal_5a4a2898491226a4": "criminal_law_law_only",
    "disc_law_sft_criminal_d06e769a5a2fcc78": "criminal_law_law_only",
    "disc_law_sft_criminal_f8c3c858bb0afc52": "criminal_law_law_only",
    "disc_law_sft_criminal_7f78e79fffa3dc8f": "criminal_law_law_only",
    "disc_law_sft_criminal_f2daa0e94a1caf08": "criminal_law_law_only",
    "disc_law_sft_criminal_a0203e994777b1dd": "criminal_law_law_only",
    "disc_law_sft_criminal_67dfb943a80339a3": "criminal_law_law_only",
    "disc_law_sft_criminal_ec14d8a51d7217a9": "criminal_law_law_only",
    "disc_law_sft_criminal_62f052ece919a775": "criminal_law_law_only",
    "disc_law_sft_criminal_2f8fcfadcf4d43c5": "criminal_law_law_only",
    "disc_law_sft_non_criminal_legal_eae1adaf35602bb1": "criminal_law_law_only",
    "hc3_open_qa_general_9dc542478686f67d": "criminal_law_law_only",
    "disc_law_sft_criminal_abbee16d0599f6e5": "non_criminal_legal",
    "disc_law_sft_criminal_2feabb0eb5f33b25": "non_criminal_legal",
    "disc_law_sft_criminal_d2c2030176c210c6": "non_criminal_legal",
    "disc_law_sft_criminal_f0aa564d6291052c": "non_criminal_legal",
    "disc_law_sft_criminal_d68caf1c337c89c1": "criminal_law_law_only",
    "agieval_jec_qa_criminal_d80e7ec80fbfdb62": "criminal_law_law_only",
    "agieval_jec_qa_non_criminal_legal_9c73855f80a52714": "criminal_law_law_only",
    "hc3_law_non_criminal_legal_7c26b313292120cc": "non_criminal_legal",
    "agieval_jec_qa_non_criminal_legal_f94ce7e5209d2c13": "non_criminal_legal",
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def apply_shape(payload: dict, label: str) -> dict:
    updated = dict(payload)
    updated["label"] = label
    updated.update(LABEL_SHAPES[label])
    updated["quality"] = payload.get("quality") or "ok"
    updated["confidence"] = float(payload.get("confidence", 0.9))
    updated["reason"] = str(payload.get("reason") or "")[:300]
    return updated


def suggest_label(row: dict) -> tuple[str, str | None]:
    payload = row["label_payload"]
    label = payload["label"]
    question = row["question"]
    if row["id"] in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[row["id"]], "subagent_review_override"

    if label == "general_no_legal_retrieval" and GENERAL_LEGAL_RE.search(question):
        return "non_criminal_legal", "general_legal_terms"
    if label == "non_criminal_legal" and (CRIMINAL_PROCEDURE_RE.search(question) or CRIME_RE.search(question)):
        return "criminal_law_law_only", "noncriminal_criminal_terms"
    if label == "criminal_law_advice" and ABSTRACT_RULE_RE.search(question) and not (PERSONAL_SUBJECT_RE.search(question) or ADVICE_ACTION_RE.search(question)):
        return "criminal_law_law_only", "advice_is_abstract_rule"
    if (
        label == "criminal_law_law_only"
        and PERSONAL_SUBJECT_RE.search(question)
        and ADVICE_ACTION_RE.search(question)
        and not re.search(r"标准|构成要件|规定|刑法第|哪一条|区别|是什么", question)
    ):
        return "criminal_law_advice", "law_only_has_personal_context"
    return label, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic consistency fixes to router labels.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=CORRECTED_OUT)
    parser.add_argument("--report", type=Path, default=REPORT_OUT)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input))
    corrected = []
    changes = []
    for row in rows:
        original_payload = row["label_payload"]
        old_label = original_payload["label"]
        new_label, reason = suggest_label(row)
        new_payload = apply_shape(original_payload, new_label)
        new_row = dict(row)
        new_row["label_payload"] = new_payload
        new_row.pop("manual_retry_prompt", None)
        new_row.pop("schema_repaired_at_utc", None)
        if new_label != old_label or any(original_payload.get(key) != new_payload.get(key) for key in LABEL_SHAPES[new_label]):
            new_row["corrected_at_utc"] = datetime.now(timezone.utc).isoformat()
            new_row["correction"] = {
                "old_label": old_label,
                "new_label": new_label,
                "reason": reason or "shape_consistency",
            }
            changes.append(new_row)
        corrected.append(new_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in corrected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "router-label-correction-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(corrected),
        "unique_ids": len({row["id"] for row in corrected}),
        "changed_rows": len(changes),
        "label_counts_before": Counter(row["label_payload"]["label"] for row in rows),
        "label_counts_after": Counter(row["label_payload"]["label"] for row in corrected),
        "change_reasons": Counter((row.get("correction") or {}).get("reason") for row in changes),
        "changed_ids": [
            {
                "id": row["id"],
                "question": row["question"],
                "old_label": row["correction"]["old_label"],
                "new_label": row["correction"]["new_label"],
                "reason": row["correction"]["reason"],
            }
            for row in changes
        ],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "unique_ids", "changed_rows", "change_reasons")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
