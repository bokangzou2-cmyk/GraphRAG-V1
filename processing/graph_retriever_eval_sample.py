from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import OUT_DIR
from graph_retriever_sample import retrieve


DEFAULT_EVAL_PATH = OUT_DIR / "graph_retriever_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "graph_retriever_eval_report_sample.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "graph_retriever_eval_report_sample.txt"


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval file must contain a list: {path}")
    return data


def contains_any(value: Any, expected_values: list[str]) -> bool:
    text = "" if value is None else str(value)
    return any(expected in text for expected in expected_values)


def check_candidates(candidates: list[dict], warnings: list[str], expected: dict) -> list[dict]:
    checks = []
    chunks = [item.get("chunk", {}) for item in candidates]
    if expected.get("candidate_type_any"):
        values = expected["candidate_type_any"]
        checks.append({"name": "candidate_type_any", "passed": any(item.get("candidate_type") in values for item in candidates), "expected": values})
    if expected.get("source_type_any"):
        values = expected["source_type_any"]
        checks.append({"name": "source_type_any", "passed": any(chunk.get("source_type") in values for chunk in chunks), "expected": values})
    if expected.get("field_any"):
        values = expected["field_any"]
        checks.append({"name": "field_any", "passed": any(chunk.get("field") in values for chunk in chunks), "expected": values})
    if expected.get("title_contains_any"):
        values = expected["title_contains_any"]
        checks.append({"name": "title_contains_any", "passed": any(contains_any(chunk.get("title"), values) for chunk in chunks), "expected": values})
    if expected.get("law_article_no_any"):
        values = expected["law_article_no_any"]
        checks.append({
            "name": "law_article_no_any",
            "passed": any((chunk.get("law_metadata") or {}).get("article_no") in values or chunk.get("article_no") in values for chunk in chunks),
            "expected": values,
        })
    for warning in expected.get("warnings_contains_all") or []:
        checks.append({"name": "warnings_contains", "passed": warning in warnings, "expected": warning})
    for path_expected in expected.get("graph_path") or []:
        checks.append({
            "name": "graph_path",
            "passed": any(
                all(path.get(key) == value for key, value in path_expected.items())
                for item in candidates
                for path in item.get("graph_paths", [])
            ),
            "expected": path_expected,
        })
    if expected.get("no_current_law_article"):
        checks.append({
            "name": "no_current_law_article",
            "passed": not any(chunk.get("source_type") == "law_article" and (chunk.get("law_metadata") or {}).get("law_version") == "current" for chunk in chunks),
            "expected": True,
        })
    return checks


def compact_candidate(item: dict) -> dict:
    chunk = item.get("chunk", {})
    return {
        "score": item.get("score"),
        "candidate_type": item.get("candidate_type"),
        "path_score": item.get("path_score"),
        "retrieval_id": item.get("retrieval_id"),
        "source_type": chunk.get("source_type"),
        "field": chunk.get("field"),
        "title": chunk.get("title"),
        "article_no": chunk.get("article_no") or (chunk.get("law_metadata") or {}).get("article_no"),
        "graph_paths": item.get("graph_paths", [])[:3],
    }


def evaluate_case(case: dict) -> dict:
    result = retrieve(case["query"], top_k=int(case.get("top_k", 10)))
    checks = check_candidates(result.get("candidates", []), result.get("warnings", []), case.get("expected", {}))
    failed = [check for check in checks if not check["passed"]]
    return {
        "id": case.get("id", case["query"]),
        "query": case["query"],
        "status": "PASS" if not failed else "WARN",
        "checks": checks,
        "warnings": result.get("warnings", []),
        "candidates": [compact_candidate(item) for item in result.get("candidates", [])],
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Graph retriever eval sample report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"pass: {report['pass_count']}",
        f"warn: {report['warn_count']}",
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['status']}] {item['id']}: {item['query']}")
        if item["warnings"]:
            lines.append(f"  warnings: {', '.join(item['warnings'])}")
        for candidate in item["candidates"][:5]:
            lines.append(f"  {candidate['candidate_type']} {candidate['source_type']} {candidate['field']} {candidate['title']} score={candidate['score']}")
        for check in item["checks"]:
            if not check["passed"]:
                lines.append(f"  missing. {check['name']} expected={json.dumps(check['expected'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sample graph retrieval checks.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT_PATH)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()
    results = [evaluate_case(case) for case in load_eval_cases(args.eval_file)]
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
