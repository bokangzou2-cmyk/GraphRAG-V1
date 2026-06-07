from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import OUT_DIR
from hybrid_search_sample import search


DEFAULT_EVAL_PATH = OUT_DIR / "hybrid_search_eval_sample.json"
DEFAULT_JSON_REPORT_PATH = OUT_DIR / "hybrid_search_eval_report_sample.json"
DEFAULT_TEXT_REPORT_PATH = OUT_DIR / "hybrid_search_eval_report_sample.txt"


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval file must contain a list: {path}")
    return data


def contains_any(value: Any, expected_values: list[str]) -> bool:
    text = "" if value is None else str(value)
    return any(expected in text for expected in expected_values)


def check_context(rows: list[dict], expected: dict) -> list[dict]:
    checks = []
    top = rows[0] if rows else {}
    if expected.get("title_contains_any"):
        values = expected["title_contains_any"]
        checks.append({"name": "title_contains_any", "passed": any(contains_any(row.get("title"), values) for row in rows), "expected": values})
    if expected.get("source_type_any"):
        values = expected["source_type_any"]
        checks.append({"name": "source_type_any", "passed": any(row.get("source_type") in values for row in rows), "expected": values})
    if expected.get("field_any"):
        values = expected["field_any"]
        checks.append({"name": "field_any", "passed": any(row.get("field") in values for row in rows), "expected": values})
    if expected.get("law_article_no_any"):
        values = expected["law_article_no_any"]
        checks.append({"name": "law_article_no_any", "passed": any((row.get("law_metadata") or {}).get("article_no") in values or row.get("article_no") in values for row in rows), "expected": values})
    if expected.get("graph_path_edge_any"):
        values = expected["graph_path_edge_any"]
        checks.append({"name": "graph_path_edge_any", "passed": any(path.get("edge_type") in values for row in rows for path in row.get("graph_paths", [])), "expected": values})
    if expected.get("top1_source_type"):
        checks.append({"name": "top1_source_type", "passed": top.get("source_type") == expected["top1_source_type"], "expected": expected["top1_source_type"]})
    if expected.get("top1_field_any"):
        values = expected["top1_field_any"]
        checks.append({"name": "top1_field_any", "passed": top.get("field") in values, "expected": values})
    if expected.get("top1_title_contains_any"):
        values = expected["top1_title_contains_any"]
        checks.append({"name": "top1_title_contains_any", "passed": contains_any(top.get("title"), values), "expected": values})
    if expected.get("top1_law_article_no_any"):
        values = expected["top1_law_article_no_any"]
        checks.append({"name": "top1_law_article_no_any", "passed": ((top.get("law_metadata") or {}).get("article_no") or top.get("article_no")) in values, "expected": values})
    if expected.get("rank_max"):
        rank_max = int(expected["rank_max"])
        field_values = expected.get("rank_field_any") or []
        title_values = expected.get("rank_title_contains_any") or []
        article_values = expected.get("rank_law_article_no_any") or []
        checks.append({
            "name": "rank_max",
            "passed": any(
                row.get("rank", 999999) <= rank_max
                and (not field_values or row.get("field") in field_values)
                and (not title_values or contains_any(row.get("title"), title_values))
                and (not article_values or ((row.get("law_metadata") or {}).get("article_no") or row.get("article_no")) in article_values)
                for row in rows
            ),
            "expected": {
                "rank_max": rank_max,
                "field_any": field_values,
                "title_contains_any": title_values,
                "law_article_no_any": article_values,
            },
        })
    if expected.get("citation_fields_required"):
        fields = expected["citation_fields_required"]
        checks.append({
            "name": "citation_fields_required",
            "passed": all(all(row.get(field) for field in fields) for row in rows[: int(expected.get("citation_required_top_k", 3))]),
            "expected": fields,
        })
    if expected.get("distinct_fields_min"):
        top_k = int(expected.get("distinct_fields_top_k", 5))
        values = {row.get("field") for row in rows[:top_k] if row.get("field")}
        checks.append({
            "name": "distinct_fields_min",
            "passed": len(values) >= int(expected["distinct_fields_min"]),
            "expected": {"min": expected["distinct_fields_min"], "top_k": top_k, "actual": sorted(values)},
        })
    for expected_path in expected.get("graph_path") or []:
        checks.append({
            "name": "graph_path",
            "passed": any(
                all(path.get(key) == value for key, value in expected_path.items())
                for row in rows
                for path in row.get("graph_paths", [])
            ),
            "expected": expected_path,
        })
    return checks


def evaluate_case(case: dict) -> dict:
    result = search(case["query"], top_k=int(case.get("top_k", 10)))
    checks = check_context(result["selected_context"], case.get("expected", {}))
    failed = [check for check in checks if not check["passed"]]
    return {
        "id": case.get("id", case["query"]),
        "query": case["query"],
        "status": "PASS" if not failed else "WARN",
        "checks": checks,
        "warnings": result.get("warnings", []),
        "selected_context": result["selected_context"],
    }


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Hybrid search eval sample report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"pass: {report['pass_count']}",
        f"warn: {report['warn_count']}",
        "",
    ]
    for item in report["results"]:
        lines.append(f"[{item['status']}] {item['id']}: {item['query']}")
        for row in item["selected_context"][:5]:
            lines.append(f"  {row['rank']}. {row['source_type']} {row['field']} {row['title']} score={row['hybrid_score']} sources={','.join(row['retrieval_sources'])}")
        for check in item["checks"]:
            if not check["passed"]:
                lines.append(f"  missing. {check['name']} expected={json.dumps(check['expected'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sample hybrid retrieval checks.")
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
