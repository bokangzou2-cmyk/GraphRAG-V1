from __future__ import annotations

import json
from collections import Counter, defaultdict

from amount_utils import AMOUNT_ROLES, extract_amounts_from_case
from common import OUT_DIR, iter_jsonl, write_jsonl


INPUT = OUT_DIR / "cases_enriched_sample.jsonl"
OUTPUT = OUT_DIR / "case_amounts_sample.jsonl"
REPORT_JSON = OUT_DIR / "amount_quality_report_sample.json"
REPORT_TXT = OUT_DIR / "amount_quality_report_sample.txt"


def build_amounts() -> list[dict]:
    rows: list[dict] = []
    for case in iter_jsonl(INPUT):
        rows.extend(extract_amounts_from_case(case))
    return rows


def build_report(rows: list[dict]) -> dict:
    role_counts = Counter(row.get("role", "unknown_amount") for row in rows)
    field_counts = Counter(row.get("field", "") for row in rows)
    cases_by_role: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cases_by_role[row.get("role", "unknown_amount")].add(row.get("case_id", ""))
    return {
        "input": str(INPUT),
        "output": str(OUTPUT),
        "total_amounts": len(rows),
        "role_counts": {role: role_counts.get(role, 0) for role in sorted(AMOUNT_ROLES)},
        "field_counts": dict(field_counts.most_common()),
        "case_counts_by_role": {role: len(cases_by_role.get(role, set())) for role in sorted(AMOUNT_ROLES)},
        "unknown_amount_count": role_counts.get("unknown_amount", 0),
        "sample_by_role": {
            role: [
                {
                    "case_id": row.get("case_id"),
                    "value": row.get("value"),
                    "raw_text": row.get("raw_text"),
                    "field": row.get("field"),
                    "context": row.get("context"),
                    "confidence": row.get("confidence"),
                }
                for row in rows
                if row.get("role") == role
            ][:5]
            for role in sorted(AMOUNT_ROLES)
        },
    }


def write_report(report: dict) -> None:
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "Amount semantic quality report",
        f"total_amounts: {report['total_amounts']}",
        f"unknown_amount_count: {report['unknown_amount_count']}",
        "",
        "role_counts:",
    ]
    for role, count in report["role_counts"].items():
        lines.append(f"  {role}: {count}")
    lines.extend(["", "top field_counts:"])
    for field, count in list(report["field_counts"].items())[:20]:
        lines.append(f"  {field}: {count}")
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_amounts()
    count = write_jsonl(OUTPUT, rows)
    report = build_report(rows)
    write_report(report)
    print(f"wrote {count} case amount rows -> {OUTPUT}")
    print(f"wrote amount report -> {REPORT_JSON}")
    print(f"wrote amount report -> {REPORT_TXT}")


if __name__ == "__main__":
    main()
