from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import OUT_DIR
from hybrid_search_sample import search


DEFAULT_EVAL_PATH = OUT_DIR / "amount_semantic_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "amount_semantic_eval_report_sample.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "amount_semantic_eval_report_sample.txt"


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval file must contain a list: {path}")
    return data


def contains_any(value: Any, expected_values: list[str]) -> bool:
    text = "" if value is None else str(value)
    return any(expected in text for expected in expected_values)


def amount_role(row: dict) -> str:
    return str((row.get("amount_diagnostics") or {}).get("matched_amount_role") or "")


def amount_value(row: dict) -> float | None:
    value = (row.get("amount_diagnostics") or {}).get("matched_amount_value")
    return float(value) if value is not None else None


def check_case(rows: list[dict], expected: dict) -> list[dict]:
    top = rows[0] if rows else {}
    checks = []
    if expected.get("top1_title_contains_any"):
        values = expected["top1_title_contains_any"]
        checks.append({"name": "top1_title_contains_any", "passed": contains_any(top.get("title"), values), "expected": values})
    if expected.get("top3_title_contains_any"):
        values = expected["top3_title_contains_any"]
        checks.append({"name": "top3_title_contains_any", "passed": any(contains_any(row.get("title"), values) for row in rows[:3]), "expected": values})
    if expected.get("top1_amount_role_any"):
        values = set(expected["top1_amount_role_any"])
        checks.append({"name": "top1_amount_role_any", "passed": amount_role(top) in values, "expected": sorted(values)})
    if expected.get("top3_amount_role_any"):
        values = set(expected["top3_amount_role_any"])
        checks.append({"name": "top3_amount_role_any", "passed": any(amount_role(row) in values for row in rows[:3]), "expected": sorted(values)})
    if expected.get("top1_amount_value_between"):
        low, high = expected["top1_amount_value_between"]
        value = amount_value(top)
        checks.append({"name": "top1_amount_value_between", "passed": value is not None and low <= value <= high, "expected": [low, high], "actual": value})
    if expected.get("top3_no_title_contains_any"):
        values = expected["top3_no_title_contains_any"]
        checks.append({"name": "top3_no_title_contains_any", "passed": not any(contains_any(row.get("title"), values) for row in rows[:3]), "expected": values})
    return checks


def evaluate_case(case: dict) -> dict:
    result = search(case["query"], top_k=int(case.get("top_k", 8)), source_types=["case_chunk"])
    checks = check_case(result.get("selected_context", []), case.get("expected", {}))
    failed = [check for check in checks if not check["passed"]]
    return {
        "id": case.get("id", case["query"]),
        "category": case.get("category", "amount_semantic"),
        "query": case["query"],
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "selected_context": result.get("selected_context", []),
        "warnings": result.get("warnings", []),
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Amount semantic eval sample report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"pass: {report['pass_count']}",
        f"fail: {report['fail_count']}",
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['status']}] {item['id']}: {item['query']}")
        for row in item["selected_context"][:5]:
            diag = row.get("amount_diagnostics") or {}
            lines.append(
                f"  {row['rank']}. {row.get('title')} {row.get('field')} "
                f"score={row.get('hybrid_score')} amount={diag.get('matched_amount_value')}/{diag.get('matched_amount_role')}"
            )
        for check in item["checks"]:
            if not check["passed"]:
                lines.append(f"  fail. {check['name']} expected={json.dumps(check['expected'], ensure_ascii=False)} actual={check.get('actual')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run amount semantic role checks for sample hybrid retrieval.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT_PATH)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    results = [evaluate_case(case) for case in load_eval_cases(args.eval_file)]
    fail_count = sum(1 for item in results if item["status"] != "PASS")
    report = {
        "status": "PASS" if fail_count == 0 else "FAIL",
        "case_count": len(results),
        "pass_count": len(results) - fail_count,
        "fail_count": fail_count,
        "eval_file": str(args.eval_file),
        "results": results,
    }
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    write_text_report(args.text_report, report)
    print(f"status: {report['status']}")
    print(f"cases: {report['case_count']} pass: {report['pass_count']} fail: {report['fail_count']}")
    print(f"wrote {args.json_report}")
    print(f"wrote {args.text_report}")
    if args.fail_on_fail and fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
