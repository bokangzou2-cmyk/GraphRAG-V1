from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from answer_sample import answer
from common import OUT_DIR


DEFAULT_EVAL_PATH = OUT_DIR / "answer_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "answer_eval_report_sample.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "answer_eval_report_sample.txt"


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval file must contain a list: {path}")
    return data


def contains_any(value: Any, expected_values: list[str]) -> bool:
    text = "" if value is None else str(value)
    return any(expected in text for expected in expected_values)


def check_expected(result: dict, expected: dict) -> list[dict]:
    checks = []
    if expected.get("answer_type"):
        checks.append({
            "name": "answer_type",
            "passed": result.get("answer_type") == expected["answer_type"],
            "expected": expected["answer_type"],
        })

    for expected_text in expected.get("answer_contains_all") or []:
        checks.append({
            "name": "answer_contains",
            "passed": expected_text in result.get("answer", ""),
            "expected": expected_text,
        })

    citation_titles = expected.get("citation_title_contains_any") or []
    if citation_titles:
        checks.append({
            "name": "citation_title_contains_any",
            "passed": any(contains_any(citation.get("title"), citation_titles) for citation in result.get("citations", [])),
            "expected": citation_titles,
        })

    citation_fields = expected.get("citation_field_any") or []
    if citation_fields:
        checks.append({
            "name": "citation_field_any",
            "passed": any(citation.get("field") in citation_fields for citation in result.get("citations", [])),
            "expected": citation_fields,
        })

    citation_source_types = expected.get("citation_source_type_any") or []
    if citation_source_types:
        checks.append({
            "name": "citation_source_type_any",
            "passed": any(citation.get("source_type") in citation_source_types for citation in result.get("citations", [])),
            "expected": citation_source_types,
        })

    for expected_edge in expected.get("graph_edge") or []:
        checks.append({
            "name": "graph_edge",
            "passed": any(
                all(path.get(key) == value for key, value in expected_edge.items())
                for path in result.get("graph_paths", [])
            ),
            "expected": expected_edge,
        })

    return checks


def compact_result(result: dict) -> dict:
    return {
        "query": result.get("query"),
        "answer_type": result.get("answer_type"),
        "answer": result.get("answer"),
        "warnings": result.get("warnings", []),
        "citations": result.get("citations", []),
        "graph_paths": result.get("graph_paths", []),
        "search_summary": result.get("search_summary", {}),
    }


def evaluate_case(case: dict) -> dict:
    result = answer(
        case["query"],
        top_k=int(case.get("top_k", 8)),
        expansion_k=int(case.get("expansion_k", 12)),
        context_k=int(case.get("context_k", 12)),
    )
    checks = check_expected(result, case.get("expected", {}))
    failed = [check for check in checks if not check["passed"]]
    return {
        "id": case.get("id", case["query"]),
        "query": case["query"],
        "status": "PASS" if not failed else "WARN",
        "checks": checks,
        "result": compact_result(result),
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Answer eval sample report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"pass: {report['pass_count']}",
        f"warn: {report['warn_count']}",
        "",
    ]
    for item in report["results"]:
        result = item["result"]
        lines.append(f"[{item['status']}] {item['id']}: {item['query']}")
        lines.append(f"  answer_type: {result['answer_type']}")
        lines.append(f"  answer: {result['answer']}")
        for citation in result.get("citations", [])[:4]:
            lines.append(f"  citation. {citation.get('retrieval_id')} | {citation.get('field')} | {citation.get('title')}")
        for path_item in result.get("graph_paths", [])[:5]:
            lines.append(
                f"  graph. {path_item.get('edge_type')} {path_item.get('article')} "
                f"({path_item.get('law_version')}) -> {path_item.get('resolution')}"
            )
        for check in item["checks"]:
            if not check["passed"]:
                lines.append(f"  missing. {check['name']} expected={json.dumps(check['expected'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed answer checks against answer_sample.py.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT_PATH)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    eval_cases = load_eval_cases(args.eval_file)
    results = [evaluate_case(case) for case in eval_cases]
    warn_count = sum(1 for item in results if item["status"] != "PASS")
    report = {
        "status": "PASS" if warn_count == 0 else "WARN",
        "case_count": len(results),
        "pass_count": len(results) - warn_count,
        "warn_count": warn_count,
        "eval_file": str(args.eval_file),
        "results": results,
    }

    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    write_text_report(args.text_report, report)
    print(f"status: {report['status']}")
    print(f"cases: {report['case_count']} pass: {report['pass_count']} warn: {report['warn_count']}")
    print(f"wrote {args.json_report}")
    print(f"wrote {args.text_report}")

    if args.fail_on_warn and warn_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
